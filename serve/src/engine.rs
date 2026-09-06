//! The engine child process: one `serve/engine.mojo` running with
//! `BARO_SERVE=1`, driven over its stdin/stdout by a single worker task.
//! The engine handles one request at a time, so the worker's inbox is the
//! request queue.

use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::mpsc;

use crate::protocol::{parse_line, DoneStats, EngineMsg, Request};

/// What a request's consumer sees, in order: zero or more tokens, then
/// exactly one `Done` or `Error`.
#[derive(Debug, Clone)]
pub enum Event {
    Tok(u32),
    Done(DoneStats),
    Error(String),
}

#[derive(Debug, Clone)]
pub struct Limits {
    pub tmax: u32,
    pub mrows: u32,
    pub kmax: u32,
    pub spec_k: u32,
    pub pack: String,
}

struct Job {
    req: Request,
    out: mpsc::UnboundedSender<Event>,
}

pub struct Engine {
    tx: mpsc::Sender<Job>,
    next_id: AtomicU64,
    queued: Arc<AtomicUsize>,
    alive: Arc<AtomicBool>,
    pub limits: Limits,
    child: tokio::sync::Mutex<Option<Child>>,
}

/// How long the engine may take to print its ready line (pack load).
const READY_TIMEOUT: Duration = Duration::from_secs(600);
/// Requests waiting beyond this many are refused with 503.
const QUEUE_CAP: usize = 64;

impl Engine {
    /// Spawn the engine, wait for its ready line, start the worker.
    /// The child inherits this process's environment (BARO_* knobs pass
    /// through); `BARO_SERVE=1` and `BARO_PACK` are set here.
    pub async fn spawn(engine: &Path, pack: &Path) -> Result<Engine, String> {
        let mut child = tokio::process::Command::new(engine)
            .env("BARO_SERVE", "1")
            .env("BARO_PACK", pack)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", engine.display()))?;
        let stdin = child.stdin.take().ok_or("engine stdin not piped")?;
        let stdout = child.stdout.take().ok_or("engine stdout not piped")?;
        let mut lines = BufReader::new(stdout).lines();

        let limits = tokio::time::timeout(READY_TIMEOUT, async {
            loop {
                match lines.next_line().await {
                    Ok(Some(l)) => match parse_line(&l) {
                        EngineMsg::Ready {
                            tmax,
                            mrows,
                            kmax,
                            spec_k,
                            pack,
                        } => {
                            return Ok(Limits {
                                tmax,
                                mrows,
                                kmax,
                                spec_k,
                                pack,
                            })
                        }
                        EngineMsg::Log(s) => eprintln!("engine: {s}"),
                        other => eprintln!("engine: unexpected before ready: {other:?}"),
                    },
                    Ok(None) => return Err("engine exited before its ready line".to_string()),
                    Err(e) => return Err(format!("engine stdout: {e}")),
                }
            }
        })
        .await
        .map_err(|_| "engine did not become ready in time".to_string())??;

        let (tx, rx) = mpsc::channel::<Job>(QUEUE_CAP);
        let queued = Arc::new(AtomicUsize::new(0));
        let alive = Arc::new(AtomicBool::new(true));
        tokio::spawn(worker(rx, stdin, lines, queued.clone(), alive.clone()));
        Ok(Engine {
            tx,
            next_id: AtomicU64::new(1),
            queued,
            alive,
            limits,
            child: tokio::sync::Mutex::new(Some(child)),
        })
    }

    pub fn alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    /// Requests waiting or running.
    pub fn queue_depth(&self) -> usize {
        self.queued.load(Ordering::SeqCst)
    }

    /// Queue one request. The receiver yields its events; dropping it
    /// discards the rest of that request's output (the engine still runs
    /// it to completion — there is no cancel in the protocol).
    pub fn submit(&self, prompt: Vec<u32>, n: u32, spec: bool) -> Result<mpsc::UnboundedReceiver<Event>, String> {
        if !self.alive() {
            return Err("engine process has exited".into());
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (out, rx) = mpsc::unbounded_channel();
        let job = Job {
            req: Request { id, prompt, n, spec },
            out,
        };
        self.queued.fetch_add(1, Ordering::SeqCst);
        match self.tx.try_send(job) {
            Ok(()) => Ok(rx),
            Err(_) => {
                self.queued.fetch_sub(1, Ordering::SeqCst);
                Err(format!("queue full ({QUEUE_CAP} requests waiting)"))
            }
        }
    }

    /// Close the engine's stdin (it exits at EOF) and reap it.
    pub async fn shutdown(&self) {
        let child = self.child.lock().await.take();
        let Some(mut child) = child else { return };
        // The worker owns stdin; closing the queue drops it once the worker
        // finishes its current request.
        self.alive.store(false, Ordering::SeqCst);
        if tokio::time::timeout(Duration::from_secs(30), child.wait()).await.is_err() {
            eprintln!("engine did not exit after stdin EOF; killing");
            let _ = child.kill().await;
        }
    }
}

async fn worker(
    mut rx: mpsc::Receiver<Job>,
    mut stdin: ChildStdin,
    mut lines: tokio::io::Lines<BufReader<ChildStdout>>,
    queued: Arc<AtomicUsize>,
    alive: Arc<AtomicBool>,
) {
    while let Some(job) = rx.recv().await {
        let id = job.req.id;
        let result = run_one(&job, &mut stdin, &mut lines).await;
        queued.fetch_sub(1, Ordering::SeqCst);
        if let Err(e) = result {
            eprintln!("engine: request {id}: {e}");
            let _ = job.out.send(Event::Error(e));
            alive.store(false, Ordering::SeqCst);
            break;
        }
    }
    alive.store(false, Ordering::SeqCst);
    // Refuse whatever is still queued, then let stdin drop so the engine sees EOF.
    rx.close();
    while let Some(job) = rx.recv().await {
        queued.fetch_sub(1, Ordering::SeqCst);
        let _ = job.out.send(Event::Error("engine process has exited".into()));
    }
    let _ = stdin.shutdown().await;
}

/// Write one request, forward its lines until its Done/Error.
/// Only an I/O failure or engine EOF is an `Err` (the engine is gone);
/// a protocol-level rejection is delivered as `Event::Error`.
async fn run_one(
    job: &Job,
    stdin: &mut ChildStdin,
    lines: &mut tokio::io::Lines<BufReader<ChildStdout>>,
) -> Result<(), String> {
    let id = job.req.id;
    stdin
        .write_all(job.req.line().as_bytes())
        .await
        .map_err(|e| format!("write to engine stdin: {e}"))?;
    stdin.flush().await.map_err(|e| format!("flush engine stdin: {e}"))?;
    loop {
        let line = match lines.next_line().await {
            Ok(Some(l)) => l,
            Ok(None) => return Err("engine exited mid-request".into()),
            Err(e) => return Err(format!("engine stdout: {e}")),
        };
        match parse_line(&line) {
            EngineMsg::Tok { id: rid, tok } if rid == id => {
                let _ = job.out.send(Event::Tok(tok));
            }
            EngineMsg::Done { id: rid, stats } if rid == id => {
                let _ = job.out.send(Event::Done(stats));
                return Ok(());
            }
            EngineMsg::Error { id: rid, error } if rid == id => {
                let _ = job.out.send(Event::Error(error));
                return Ok(());
            }
            EngineMsg::Log(s) => eprintln!("engine: {s}"),
            other => eprintln!("engine: line for another request ignored: {other:?}"),
        }
    }
}
