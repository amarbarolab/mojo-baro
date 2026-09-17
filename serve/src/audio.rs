//! P3a speech-in sidecar: `POST /v1/audio/transcriptions` proxied to a
//! `whisper-server` child, started on demand and stopped after idle.
//! whisper-server binds its port only after its model is loaded
//! (`examples/server/server.cpp`: `whisper_init_from_file_with_params`
//! before `listen_after_bind`), so a successful TCP connect is a valid
//! readiness check. The child is driven as a plain HTTP peer on loopback
//! (`POST /inference`, multipart in, JSON out), not over stdio like the LLM
//! engine in `engine.rs`.

use super::*;
use axum::extract::Multipart;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::process::{Child, Command};

fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.into())
}

fn env_num<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

pub struct AudioSidecar {
    bin: String,
    model: String,
    language: String,
    beam: u32,
    threads: u32,
    port: u16,
    idle_secs: u64,
    /// P3a preflight (`docs/PLATFORM-PLAN.md`): one clip through the
    /// endpoint on CPU with a small model, before the real GPU gate. When
    /// set, the sidecar is not wrapped in `gpu-wait` at all -- it is not a
    /// GPU workload.
    no_gpu: bool,
    /// Absolute path, not a bare name: a `gpu-wait run` job does not put
    /// `~/.local/bin` on the child's `PATH` (found live in the P3a timed
    /// gate, `bench/p3a-gate.sh` line 54's own nested call hit the same
    /// thing), so a bare `Command::new("gpu-wait")` fails inside one.
    gpu_wait_bin: String,
    child: tokio::sync::Mutex<Option<Child>>,
    last_used: AtomicU64,
}

impl AudioSidecar {
    pub fn from_env() -> AudioSidecar {
        AudioSidecar {
            bin: env_str("BARO_WHISPER_BIN", "$HOME/Models/whisper.cpp/build/bin/whisper-server"),
            model: env_str("BARO_WHISPER_MODEL", "$HOME/Models/whisper/ggml-large-v3-turbo-q5_0.bin"),
            language: env_str("BARO_WHISPER_LANGUAGE", "en"),
            beam: env_num("BARO_WHISPER_BEAM", 5u32),
            threads: env_num("BARO_WHISPER_THREADS", 8u32),
            port: env_num("BARO_WHISPER_PORT", 8090u16),
            idle_secs: env_num("BARO_WHISPER_IDLE_SECS", 300u64),
            no_gpu: env_str("BARO_WHISPER_NO_GPU", "0") == "1",
            gpu_wait_bin: env_str("BARO_GPU_WAIT", "$HOME/.local/bin/gpu-wait"),
            child: tokio::sync::Mutex::new(None),
            last_used: AtomicU64::new(0),
        }
    }
}

/// Start the sidecar if it is not already up (or has exited), wait for its
/// port to accept connections, and arm the idle reaper. A resident, not a
/// one-shot job: spawned through `gpu-wait run --shared`, the daemon's own
/// pattern for a long-lived service (`gpuwaitingroom` README, "Resident
/// service"). `gpu-wait run` drops the shell env, so a job's `PATH` does not
/// carry `~/.local/bin` -- `a.gpu_wait_bin` is always an absolute path,
/// never a bare name (found live: `bench/p3a-gate.sh`'s own nested snapshot
/// call hit the same PATH gap). When `GPU_WAITING_ROOM_JOB` is already set,
/// this process is itself running inside a job, so it spawns the sidecar
/// bare instead of calling `gpu-wait run` again -- nesting would queue the
/// sidecar behind the job admitting it.
async fn ensure_running(app: &Shared) -> Result<(), String> {
    let a = &app.audio;
    {
        let mut guard = a.child.lock().await;
        if let Some(child) = guard.as_mut() {
            if matches!(child.try_wait(), Ok(None)) {
                a.last_used.store(now(), Ordering::SeqCst);
                return Ok(());
            }
            *guard = None; // exited: fall through and respawn
        }
        let mut whisper_args = vec![
            a.bin.clone(),
            "-m".into(),
            a.model.clone(),
            "-l".into(),
            a.language.clone(),
            "-bs".into(),
            a.beam.to_string(),
            "-t".into(),
            a.threads.to_string(),
            // Matches the reference `whisper-cli -nt` in bench/p3a-gate.sh:
            // enabling timestamps changes whisper.cpp's segment text join,
            // dropping punctuation at segment boundaries (P3a CPU preflight,
            // isolated by diffing a direct call with and without this flag).
            "-nt".into(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            a.port.to_string(),
        ];
        if a.no_gpu {
            whisper_args.push("-ng".into());
        }
        let job_id = std::env::var("GPU_WAITING_ROOM_JOB").ok();
        let mut cmd = if a.no_gpu {
            eprintln!("audio: whisper sidecar running bare (--no-gpu, not a GPU workload)");
            let mut c = Command::new(&whisper_args[0]);
            c.args(&whisper_args[1..]);
            c
        } else if let Some(job) = &job_id {
            eprintln!("audio: whisper sidecar running bare, already admitted (GPU_WAITING_ROOM_JOB={job})");
            let mut c = Command::new(&whisper_args[0]);
            c.args(&whisper_args[1..]);
            c
        } else {
            eprintln!("audio: whisper sidecar spawned via {} run --shared", a.gpu_wait_bin);
            let mut gpu_wait_args = vec!["run".to_string(), "--shared".into(), "--vram".into(), "4".into(), "--".into()];
            gpu_wait_args.extend(whisper_args);
            let mut c = Command::new(&a.gpu_wait_bin);
            c.args(&gpu_wait_args);
            c
        };
        let child = cmd
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("spawn whisper sidecar: {e}"))?;
        *guard = Some(child);
    }
    a.last_used.store(now(), Ordering::SeqCst);

    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    loop {
        if TcpStream::connect(("127.0.0.1", a.port)).await.is_ok() {
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            return Err("whisper sidecar did not open its port in time".into());
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    let app = app.clone();
    tokio::spawn(reap_when_idle(app));
    Ok(())
}

async fn reap_when_idle(app: Shared) {
    loop {
        tokio::time::sleep(Duration::from_secs(15)).await;
        let idle = now().saturating_sub(app.audio.last_used.load(Ordering::SeqCst));
        if idle < app.audio.idle_secs {
            continue;
        }
        let mut guard = app.audio.child.lock().await;
        if let Some(mut child) = guard.take() {
            let _ = child.kill().await;
            eprintln!("audio: whisper sidecar stopped after {idle}s idle");
        }
        return;
    }
}

fn multipart_field(boundary: &str, name: &str, value: &str) -> Vec<u8> {
    format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").into_bytes()
}

fn multipart_file(boundary: &str, name: &str, filename: &str, content_type: &str, data: &[u8]) -> Vec<u8> {
    let mut out = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n"
    )
    .into_bytes();
    out.extend_from_slice(data);
    out.extend_from_slice(b"\r\n");
    out
}

fn parse_http_response(resp: &[u8]) -> Result<(StatusCode, String, Vec<u8>), String> {
    let sep = resp
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or("whisper sidecar: malformed response (no header terminator)")?;
    let head = std::str::from_utf8(&resp[..sep]).map_err(|_| "whisper sidecar: non-utf8 headers")?;
    let mut lines = head.split("\r\n");
    let status_line = lines.next().ok_or("whisper sidecar: empty response")?;
    let code: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or("whisper sidecar: bad status line")?;
    let status = StatusCode::from_u16(code).map_err(|_| "whisper sidecar: invalid status code")?;
    let content_type = lines
        .find_map(|l| l.to_ascii_lowercase().starts_with("content-type:").then(|| l[13..].trim().to_string()))
        .unwrap_or_else(|| "application/octet-stream".into());
    Ok((status, content_type, resp[sep + 4..].to_vec()))
}

/// One multipart request to whisper-server's `/inference`, `Connection:
/// close` so reading to EOF is a valid way to collect the whole response
/// (a local loopback peer, not a keep-alive pool worth building here).
async fn forward(app: &Shared, boundary: &str, body: Vec<u8>) -> Result<(StatusCode, String, Vec<u8>), String> {
    let port = app.audio.port;
    let mut stream = TcpStream::connect(("127.0.0.1", port)).await.map_err(|e| format!("connect whisper sidecar: {e}"))?;
    let mut head = format!(
        "POST /inference HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: multipart/form-data; boundary={boundary}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    head.extend_from_slice(&body);
    stream.write_all(&head).await.map_err(|e| format!("write to whisper sidecar: {e}"))?;
    let mut resp = Vec::new();
    stream.read_to_end(&mut resp).await.map_err(|e| format!("read from whisper sidecar: {e}"))?;
    parse_http_response(&resp)
}

pub async fn transcriptions(State(app): State<Shared>, mut multipart: Multipart) -> Result<Response, ApiError> {
    let mut file: Option<(String, String, Vec<u8>)> = None;
    let mut language: Option<String> = None;
    let mut response_format: Option<String> = None;
    let mut temperature: Option<String> = None;
    loop {
        let field = multipart.next_field().await.map_err(|e| bad(format!("multipart: {e}")))?;
        let Some(field) = field else { break };
        match field.name().unwrap_or("") {
            "file" => {
                let filename = field.file_name().unwrap_or("audio.wav").to_string();
                let content_type = field.content_type().unwrap_or("audio/wav").to_string();
                let data = field.bytes().await.map_err(|e| bad(format!("reading file field: {e}")))?.to_vec();
                file = Some((filename, content_type, data));
            }
            "language" => language = field.text().await.ok(),
            "response_format" => response_format = field.text().await.ok(),
            "temperature" => temperature = field.text().await.ok(),
            _ => {}
        }
    }
    let (filename, content_type, data) = file.ok_or_else(|| bad("multipart field \"file\" is required"))?;

    let boundary = format!("bAr0{}", now());
    let mut body = Vec::new();
    if let Some(l) = &language {
        body.extend(multipart_field(&boundary, "language", l));
    }
    if let Some(f) = &response_format {
        body.extend(multipart_field(&boundary, "response_format", f));
    }
    if let Some(t) = &temperature {
        body.extend(multipart_field(&boundary, "temperature", t));
    }
    body.extend(multipart_file(&boundary, "file", &filename, &content_type, &data));
    body.extend(format!("--{boundary}--\r\n").into_bytes());

    ensure_running(&app).await.map_err(|e| ApiError::Plain(StatusCode::SERVICE_UNAVAILABLE, e))?;
    let (status, resp_content_type, resp_body) =
        forward(&app, &boundary, body).await.map_err(|e| ApiError::Plain(StatusCode::BAD_GATEWAY, e))?;

    Ok((status, [(axum::http::header::CONTENT_TYPE, resp_content_type)], resp_body).into_response())
}
