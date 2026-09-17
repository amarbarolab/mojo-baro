//! The engine child process: one `serve/engine.mojo` running with
//! `BARO_SERVE=1`, driven over its stdin/stdout by a single worker task.
//! The engine advances one request at a time; the worker admits later lines
//! while routing each response event by request id.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::mpsc;

use crate::protocol::{cancel_line, parse_line, DoneStats, EngineMsg, Request, SampleParams};

/// What a request's consumer sees, in order: zero or more tokens, then
/// exactly one `Done` or `Error`.
#[derive(Debug, Clone)]
pub enum Event {
    Tok { tok: u32, logprob: Option<f64>, top_logprobs: Vec<(u32, f64)> },
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
    queued: Arc<AtomicUsize>,
    alive: Arc<AtomicBool>,
    /// The request id the engine is actively decoding, if any; set/cleared
    /// by the worker around `run_one`. `cancel` reads it to decide whether a
    /// cancel would land on anything, and the worker alone still owns
    /// `stdin` -- no sharing needed to write a cancel line, since the worker
    /// writes it itself once `cancel_tx` wakes its select loop.
    current: Arc<tokio::sync::Mutex<Option<u64>>>,
    cancel_tx: mpsc::UnboundedSender<u64>,
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
        let (cancel_tx, cancel_rx) = mpsc::unbounded_channel::<u64>();
        let queued = Arc::new(AtomicUsize::new(0));
        let alive = Arc::new(AtomicBool::new(true));
        let current = Arc::new(tokio::sync::Mutex::new(None));
        tokio::spawn(worker(rx, stdin, lines, queued.clone(), alive.clone(), cancel_rx, current.clone()));
        Ok(Engine {
            tx,
            queued,
            alive,
            current,
            cancel_tx,
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

    /// Queue one request under the given id (assigned by the caller --
    /// `EnginePool` owns a single id counter shared by every engine in the
    /// pool, so ids stay unique across engines). Returns a receiver of its
    /// events; dropping it without cancelling discards the rest of that
    /// request's output, the engine still runs it to completion.
    #[allow(clippy::too_many_arguments)]
    pub fn submit(
        &self, id: u64, prompt: Vec<u32>, n: u32, spec: bool, stop: Vec<Vec<u32>>, ckpt: Vec<u32>, sample: SampleParams,
        schema: Option<serde_json::Value>, reasoning: Option<bool>,
        state: (Option<String>, Option<String>),
    ) -> Result<mpsc::UnboundedReceiver<Event>, String> {
        if !self.alive() {
            return Err("engine process has exited".into());
        }
        let (out, rx) = mpsc::unbounded_channel();
        let job = Job {
            req: Request { id, prompt, n, spec, stop, ckpt, state_save: state.0, state_load: state.1, sample, schema, reasoning },
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

    /// Cancel `id` if the engine is actively decoding it right now. Returns
    /// `false` for a request that has not started (still queued), already
    /// finished, or does not exist -- cancelling a queued-but-not-yet-running
    /// request is not supported (brief CHAT/C1: not in this step).
    pub async fn cancel(&self, id: u64) -> bool {
        if *self.current.lock().await != Some(id) {
            return false;
        }
        self.cancel_tx.send(id).is_ok()
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

/// `BARO_POOL` engine processes (default 1 = today's single engine), each
/// with its own pack load and its own request queue. `submit` routes to the
/// engine with the fewest requests waiting or running (ties -> lowest
/// index), so at `BARO_POOL=1` this is exactly today's single-queue
/// behaviour; at `BARO_POOL=2` a second request starts on the other engine
/// instead of waiting behind the first. One id counter, owned by the pool
/// and never by an `Engine`, so ids stay unique across engines (`cancel`
/// and the API's `cmpl-<id>`/`chatcmpl-<id>` depend on that).
pub struct EnginePool {
    engines: Vec<Engine>,
    next_id: AtomicU64,
    pub limits: Limits,
}

impl EnginePool {
    /// No LLM engine (`--audio-only`, P3a): every text/decode route's existing
    /// `submit` error path answers 503 with no engine process spawned, no
    /// pack load, no VRAM held. `tmax` is `u32::MAX` so `check_and_submit`'s
    /// exceed-context check never intercepts the request first; `submit`
    /// itself is the thing that refuses.
    pub fn empty() -> EnginePool {
        EnginePool {
            engines: Vec::new(),
            next_id: AtomicU64::new(1),
            limits: Limits { tmax: u32::MAX, mrows: 0, kmax: 0, spec_k: 0, pack: "none (--audio-only)".into() },
        }
    }

    pub async fn spawn(engine: &Path, pack: &Path, pool_size: usize) -> Result<EnginePool, String> {
        let mut engines = Vec::with_capacity(pool_size.max(1));
        for _ in 0..pool_size.max(1) {
            engines.push(Engine::spawn(engine, pack).await?);
        }
        let limits = engines[0].limits.clone();
        Ok(EnginePool {
            engines,
            next_id: AtomicU64::new(1),
            limits,
        })
    }

    pub fn pool_size(&self) -> usize {
        self.engines.len()
    }

    pub fn alive(&self) -> bool {
        self.engines.iter().any(Engine::alive)
    }

    /// Requests waiting or running, summed over every engine in the pool.
    pub fn queue_depth(&self) -> usize {
        self.engines.iter().map(Engine::queue_depth).sum()
    }

    /// Per-engine breakdown of `queue_depth`: lets the C4 gate tell "two
    /// engines each running one request" apart from "one engine running
    /// both while the other sits idle," which the summed `queue_depth`
    /// alone cannot.
    pub fn queue_depths(&self) -> Vec<usize> {
        self.engines.iter().map(Engine::queue_depth).collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn submit(
        &self, prompt: Vec<u32>, n: u32, spec: bool, stop: Vec<Vec<u32>>, ckpt: Vec<u32>, sample: SampleParams,
        schema: Option<serde_json::Value>, reasoning: Option<bool>,
        state: (Option<String>, Option<String>),
    ) -> Result<(u64, mpsc::UnboundedReceiver<Event>), String> {
        if self.engines.is_empty() {
            return Err("no engine loaded (--audio-only)".into());
        }
        let mut best = 0;
        let mut best_q = self.engines[0].queue_depth();
        for (i, e) in self.engines.iter().enumerate().skip(1) {
            let q = e.queue_depth();
            if q < best_q {
                best = i;
                best_q = q;
            }
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let rx = self.engines[best].submit(id, prompt, n, spec, stop, ckpt, sample, schema, reasoning, state)?;
        Ok((id, rx))
    }

    /// Tries every engine (a request's id does not say which one it landed
    /// on); at most one will have it as its `current` decode.
    pub async fn cancel(&self, id: u64) -> bool {
        for e in &self.engines {
            if e.cancel(id).await {
                return true;
            }
        }
        false
    }

    pub async fn shutdown(&self) {
        for e in &self.engines {
            e.shutdown().await;
        }
    }
}

async fn worker(
    mut rx: mpsc::Receiver<Job>,
    mut stdin: ChildStdin,
    mut lines: tokio::io::Lines<BufReader<ChildStdout>>,
    queued: Arc<AtomicUsize>,
    alive: Arc<AtomicBool>,
    mut cancel_rx: mpsc::UnboundedReceiver<u64>,
    current: Arc<tokio::sync::Mutex<Option<u64>>>,
) {
    while let Some(job) = rx.recv().await {
        let mut active = HashMap::<u64, mpsc::UnboundedSender<Event>>::new();
        let mut order = VecDeque::<u64>::new();
        let mut rx_open = true;
        let mut error = None;
        let first_id = job.req.id;
        let first_line = job.req.line();
        let first_out = job.out;
        active.insert(first_id, first_out.clone());
        order.push_back(first_id);
        *current.lock().await = Some(first_id);
        if let Err(e) = write_line(&mut stdin, first_line).await {
            active.remove(&first_id);
            order.pop_front();
            let _ = first_out.send(Event::Error(e.clone()));
            queued.fetch_sub(1, Ordering::SeqCst);
            error = Some(e);
        }

        while error.is_none() && !active.is_empty() {
            tokio::select! {
                maybe_job = rx.recv(), if rx_open => match maybe_job {
                    Some(job) => {
                        let id = job.req.id;
                        let line = job.req.line();
                        let out = job.out;
                        active.insert(id, out.clone());
                        order.push_back(id);
                        if let Some(running) = *current.lock().await {
                            eprintln!("engine: wire admitted request {id} while request {running} is in flight");
                        }
                        if let Err(e) = write_line(&mut stdin, line).await {
                            active.remove(&id);
                            order.retain(|queued_id| *queued_id != id);
                            let _ = out.send(Event::Error(e.clone()));
                            queued.fetch_sub(1, Ordering::SeqCst);
                            error = Some(e);
                        }
                    }
                    None => rx_open = false,
                },
                line = lines.next_line() => match line {
                    Ok(Some(line)) => match parse_line(&line) {
                        EngineMsg::Tok { id, tok, logprob, top_logprobs } => {
                            if let Some(out) = active.get(&id) {
                                let _ = out.send(Event::Tok { tok, logprob, top_logprobs });
                            } else {
                                eprintln!("engine: line for unknown request ignored: {line}");
                            }
                        }
                        EngineMsg::Done { id, stats } => {
                            if let Some(out) = active.remove(&id) {
                                let _ = out.send(Event::Done(stats));
                                queued.fetch_sub(1, Ordering::SeqCst);
                                order.retain(|queued_id| *queued_id != id);
                                *current.lock().await = order.front().copied();
                            } else {
                                eprintln!("engine: terminal line for unknown request ignored: {line}");
                            }
                        }
                        EngineMsg::Error { id, error: message } => {
                            if let Some(out) = active.remove(&id) {
                                let _ = out.send(Event::Error(message));
                                queued.fetch_sub(1, Ordering::SeqCst);
                                order.retain(|queued_id| *queued_id != id);
                                *current.lock().await = order.front().copied();
                            } else {
                                eprintln!("engine: error line for unknown request ignored: {line}");
                            }
                        }
                        EngineMsg::Ready { .. } => eprintln!("engine: unexpected ready line after startup"),
                        EngineMsg::Log(s) => eprintln!("engine: {s}"),
                    },
                    Ok(None) => error = Some("engine exited mid-request".into()),
                    Err(e) => error = Some(format!("engine stdout: {e}")),
                },
                Some(cancel_id) = cancel_rx.recv() => {
                    if *current.lock().await == Some(cancel_id) {
                        if let Err(e) = write_line(&mut stdin, cancel_line(cancel_id)).await {
                            error = Some(e);
                        }
                    }
                }
            }
        }

        *current.lock().await = None;
        if let Some(e) = error {
            for (_, out) in active.drain() {
                let _ = out.send(Event::Error(e.clone()));
                queued.fetch_sub(1, Ordering::SeqCst);
            }
            eprintln!("engine: worker stopped: {e}");
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

async fn write_line(stdin: &mut ChildStdin, line: String) -> Result<(), String> {
    stdin.write_all(line.as_bytes()).await.map_err(|e| format!("write to engine stdin: {e}"))?;
    stdin.flush().await.map_err(|e| format!("flush engine stdin: {e}"))
}
