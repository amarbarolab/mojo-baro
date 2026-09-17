//! CPU-only `baro-router` skeleton.
//!
//! The router owns placement and discovery; `baro-serve` remains the engine
//! process. This first slice keeps the P1 state-locality term out of the
//! placement decision. It provides the contracts around which the later proxy
//! and state moves fit, rather than making an empty locality field look done.

use std::collections::{BTreeMap, HashMap};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{Request, State};
use axum::http::{Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get};
use axum::{Json, Router};
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use serde::Serialize;
use serde_json::{json, Value};
use tokio::net::TcpStream;

const BARO_SERVICE: &str = "_baro-node._tcp";
const PAIR_SERVICE: &str = "_nvpair-node._tcp";

#[derive(Clone)]
struct Shared(Arc<RouterState>);

struct RouterState {
    node_id: String,
    http_port: u16,
    discovery: DiscoveryAdvertisement,
    _mdns: Option<MdnsRuntime>,
    engines: EngineRegistry,
    workloads: WorkloadCatalog,
    next_request: AtomicU64,
}

#[derive(Debug, Clone, Serialize)]
struct DiscoveryAdvertisement {
    instance: String,
    service: String,
    pair_service: String,
    port: u16,
    txt: BTreeMap<String, String>,
}

impl DiscoveryAdvertisement {
    fn new(node_id: &str, port: u16) -> Self {
        let mut txt = BTreeMap::new();
        txt.insert("v".into(), "1".into());
        txt.insert("ni".into(), port.to_string());
        // PAIR keys its directory by the UUID in the consolidated node record.
        txt.insert("uuid".into(), node_id.into());
        txt.insert("node_id".into(), node_id.into());
        Self {
            instance: format!("baro-{node_id}"),
            service: BARO_SERVICE.into(),
            pair_service: PAIR_SERVICE.into(),
            port,
            txt,
        }
    }

    /// The service records the mDNS adapter must advertise and browse.
    fn browse_services(&self) -> [&'static str; 2] {
        [BARO_SERVICE, PAIR_SERVICE]
    }
}

struct MdnsRuntime {
    // Holding the daemon handle keeps its worker and both registrations alive.
    _daemon: ServiceDaemon,
}

fn start_mdns(advertisement: &DiscoveryAdvertisement, node_id: &str) -> Result<MdnsRuntime, String> {
    let daemon = ServiceDaemon::new().map_err(|e| format!("mDNS daemon: {e}"))?;
    let host = format!("baro-{node_id}.local.");
    let mut properties = HashMap::new();
    for (key, value) in &advertisement.txt {
        properties.insert(key.clone(), value.clone());
    }
    for service in advertisement.browse_services() {
        let service_type = format!("{service}.local.");
        let info = ServiceInfo::new(
            &service_type,
            &advertisement.instance,
            &host,
            "127.0.0.1",
            advertisement.port,
            properties.clone(),
        )
        .map_err(|e| format!("mDNS service {service}: {e}"))?;
        daemon.register(info).map_err(|e| format!("mDNS register {service}: {e}"))?;
    }
    let browse_type = format!("{}.local.", advertisement.pair_service);
    let receiver = daemon.browse(&browse_type).map_err(|e| format!("mDNS browse: {e}"))?;
    std::thread::Builder::new()
        .name("baro-router-mdns".into())
        .spawn(move || {
            while let Ok(event) = receiver.recv() {
                if let ServiceEvent::ServiceResolved(service) = event {
                    eprintln!("mDNS discovered {}:{}", service.get_hostname(), service.get_port());
                }
            }
        })
        .map_err(|e| format!("mDNS browse thread: {e}"))?;
    Ok(MdnsRuntime { _daemon: daemon })
}

#[derive(Debug, Clone, Serialize)]
struct EngineInfo {
    id: String,
    address: String,
    port: u16,
    running: bool,
    healthy: bool,
    adopted: bool,
    pending: usize,
    models: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_timeout_ms: Option<u64>,
}

struct EngineRecord {
    info: EngineInfo,
    health: HealthHysteresis,
}

#[derive(Clone)]
struct EngineRegistry(Arc<RwLock<Vec<EngineRecord>>>);

impl EngineRegistry {
    fn from_env() -> Self {
        let records = std::env::var("BARO_ROUTER_ENGINES")
            .unwrap_or_default()
            .split(',')
            .filter_map(parse_engine)
            .map(|info| EngineRecord { info, health: HealthHysteresis::default() })
            .collect();
        Self(Arc::new(RwLock::new(records)))
    }

    fn snapshot(&self) -> Vec<EngineInfo> {
        self.0.read().expect("engine registry poisoned").iter().map(|e| e.info.clone()).collect()
    }

    fn choose(&self) -> Option<EngineChoice> {
        self.0
            .read()
            .expect("engine registry poisoned")
            .iter()
            .filter(|e| e.info.running && e.info.healthy)
            .min_by_key(|e| (e.info.pending, e.info.id.clone()))
            .map(|e| EngineChoice { id: e.info.id.clone(), address: e.info.address.clone() })
    }

    fn set_probe(&self, id: &str, ok: bool) {
        let mut engines = self.0.write().expect("engine registry poisoned");
        if let Some(engine) = engines.iter_mut().find(|e| e.info.id == id) {
            let healthy = engine.health.observe(ok);
            engine.info.running = ok;
            engine.info.adopted = ok;
            engine.info.healthy = healthy;
        }
    }

    fn bump_pending(&self, id: &str) {
        if let Some(engine) = self.0.write().expect("engine registry poisoned").iter_mut().find(|e| e.info.id == id) {
            engine.info.pending += 1;
        }
    }

    fn drop_pending(&self, id: &str) {
        if let Some(engine) = self.0.write().expect("engine registry poisoned").iter_mut().find(|e| e.info.id == id) {
            engine.info.pending = engine.info.pending.saturating_sub(1);
        }
    }
}

fn parse_engine(raw: &str) -> Option<EngineInfo> {
    let (id, endpoint) = raw.trim().split_once('=')?;
    let endpoint = endpoint.trim().trim_start_matches("http://").trim_end_matches('/');
    let port = endpoint.rsplit_once(':')?.1.parse().ok()?;
    Some(EngineInfo {
        id: id.trim().to_string(),
        address: endpoint.to_string(),
        port,
        running: false,
        healthy: false,
        adopted: false,
        pending: 0,
        models: Vec::new(),
        prefill_timeout_ms: Some(20_000),
    })
}

#[derive(Debug, Clone, Copy, Default)]
struct HealthHysteresis {
    healthy: bool,
    failures: u8,
    passes: u8,
}

impl HealthHysteresis {
    fn observe(&mut self, ok: bool) -> bool {
        if ok {
            self.failures = 0;
            self.passes = self.passes.saturating_add(1);
            if self.passes >= 2 {
                self.healthy = true;
            }
        } else {
            self.passes = 0;
            self.failures = self.failures.saturating_add(1);
            if self.failures >= 3 {
                self.healthy = false;
            }
        }
        self.healthy
    }
}

#[derive(Debug, Clone, Serialize)]
struct Workload {
    request_id: String,
    method: String,
    path: String,
    state: String,
    node_id: String,
    engine: Option<String>,
    started_ms: u128,
}

#[derive(Clone)]
struct WorkloadCatalog(Arc<Mutex<Vec<Workload>>>);

impl WorkloadCatalog {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(Vec::new())))
    }

    fn insert(&self, workload: Workload) {
        self.0.lock().expect("workload catalog poisoned").push(workload);
    }

    fn snapshot(&self) -> Vec<Workload> {
        self.0.lock().expect("workload catalog poisoned").clone()
    }
}

#[derive(Debug, Clone)]
struct FilteredRequest {
    method: Method,
    path: String,
    request_id: String,
}

#[derive(Debug, Clone)]
struct EngineChoice {
    id: String,
    address: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SendState {
    NotConnected,
    ConnectedNoBytes,
    FirstByteSent,
}

impl SendState {
    fn retry_allowed(self, cancel_ack: bool, cancel_timed_out: bool) -> bool {
        match self {
            Self::NotConnected => true,
            Self::ConnectedNoBytes => cancel_ack || cancel_timed_out,
            Self::FirstByteSent => false,
        }
    }
}

fn request_filter(request: &Request, request_id: String) -> Result<FilteredRequest, String> {
    let path = request.uri().path().to_string();
    if path == "/v1/node-info" || path == "/v1/workloads" || path == "/health" {
        return Err("control endpoint is not proxyable".into());
    }
    Ok(FilteredRequest { method: request.method().clone(), path, request_id })
}

fn choose_engine(registry: &EngineRegistry) -> Result<EngineChoice, String> {
    registry.choose().ok_or_else(|| "no healthy adopted engine".into())
}

async fn connected(choice: &EngineChoice) -> Result<TcpStream, String> {
    TcpStream::connect(&choice.address).await.map_err(|e| format!("{}: {e}", choice.address))
}

fn fail_to_connect(error: String) -> Response {
    error_while_proxy(format!("upstream connection failed: {error}"))
}

fn response_filter(status: StatusCode, body: Value) -> Response {
    (status, Json(body)).into_response()
}

fn error_while_proxy(error: String) -> Response {
    (StatusCode::BAD_GATEWAY, Json(json!({"error": {"type": "upstream", "message": error}}))).into_response()
}

fn log(catalog: &WorkloadCatalog, workload: Workload) {
    eprintln!("router request={} method={} path={} state={} engine={:?}", workload.request_id, workload.method, workload.path, workload.state, workload.engine);
    catalog.insert(workload);
}

async fn health(State(Shared(state)): State<Shared>) -> Json<Value> {
    Json(json!({"status": "ok", "node_id": state.node_id, "engines": state.engines.snapshot().len()}))
}

async fn node_info(State(Shared(state)): State<Shared>) -> Json<Value> {
    let engines = state.engines.snapshot();
    Json(json!({
        "schema": 1,
        "node_id": state.node_id,
        "hostUuid": state.node_id,
        "GPUs": [],
        "telemetryValid": false,
        "msSince": 0,
        "http_port": state.http_port,
        "engines": engines,
        "pending": engines.iter().map(|e| e.pending).sum::<usize>(),
        "discovery": &state.discovery,
        // P1's state-locality term is intentionally absent until its API lands.
    }))
}

async fn workloads(State(Shared(state)): State<Shared>) -> Json<Vec<Workload>> {
    Json(state.workloads.snapshot())
}

async fn proxy(State(Shared(state)): State<Shared>, request: Request) -> Response {
    let request_id = format!("router-{}", state.next_request.fetch_add(1, Ordering::Relaxed));
    let filtered = match request_filter(&request, request_id.clone()) {
        Ok(value) => value,
        Err(error) => return response_filter(StatusCode::NOT_FOUND, json!({"error": error})),
    };
    let choice = match choose_engine(&state.engines) {
        Ok(value) => value,
        Err(error) => return response_filter(StatusCode::SERVICE_UNAVAILABLE, json!({"error": error})),
    };
    state.engines.bump_pending(&choice.id);
    let workload = Workload {
        request_id: filtered.request_id.clone(),
        method: filtered.method.to_string(),
        path: filtered.path.clone(),
        state: "connected".into(),
        node_id: state.node_id.clone(),
        engine: Some(choice.id.clone()),
        started_ms: now_ms(),
    };
    let result = match connected(&choice).await {
        Ok(_socket) => {
            // Forwarding and streaming pass-through land after this CPU
            // skeleton. Never return a fake upstream answer here.
            let _send_state = SendState::ConnectedNoBytes;
            response_filter(StatusCode::NOT_IMPLEMENTED, json!({
                "error": "proxy transport pending",
                "method": filtered.method.as_str(),
                "path": filtered.path,
                "retry": "a connected request must be cancelled or time out before retry",
            }))
        }
        Err(error) => {
            let _ = SendState::NotConnected.retry_allowed(false, false);
            fail_to_connect(error)
        }
    };
    state.engines.drop_pending(&choice.id);
    log(&state.workloads, Workload { state: "done".into(), ..workload });
    result
}

fn now_ms() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis()
}

fn parse_args() -> Result<(String, String, u16), String> {
    let mut host = "127.0.0.1".to_string();
    let mut node_id = std::env::var("BARO_ROUTER_NODE_ID").unwrap_or_else(|_| "local".into());
    let mut port = 8090u16;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--host" => host = value()?,
            "--node-id" => node_id = value()?,
            "--port" => port = value()?.parse().map_err(|e| format!("--port: {e}"))?,
            "-h" | "--help" => {
                println!("usage: baro-router [--host HOST] [--node-id ID] [--port PORT]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok((host, node_id, port))
}

async fn probe_loop(registry: EngineRegistry) {
    loop {
        for engine in registry.snapshot() {
            let ok = TcpStream::connect(&engine.address).await.is_ok();
            registry.set_probe(&engine.id, ok);
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

#[tokio::main]
async fn main() {
    let (host, node_id, port) = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("baro-router: {error}");
            std::process::exit(2);
        }
    };
    let engines = EngineRegistry::from_env();
    let discovery = DiscoveryAdvertisement::new(&node_id, port);
    eprintln!("discovery advertise={} browse={:?} txt={:?}", discovery.service, discovery.browse_services(), discovery.txt);
    let mdns = match start_mdns(&discovery, &node_id) {
        Ok(runtime) => Some(runtime),
        Err(error) => {
            eprintln!("baro-router: mDNS unavailable: {error}");
            None
        }
    };
    let state = Arc::new(RouterState {
        node_id,
        http_port: port,
        discovery,
        _mdns: mdns,
        engines: engines.clone(),
        workloads: WorkloadCatalog::new(),
        next_request: AtomicU64::new(1),
    });
    tokio::spawn(probe_loop(engines));
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/node-info", get(node_info))
        .route("/v1/workloads", get(workloads))
        .route("/v1/{*path}", any(proxy))
        .route("/api/{*path}", any(proxy))
        .with_state(Shared(state));
    let address: SocketAddr = match format!("{host}:{port}").parse() {
        Ok(address) => address,
        Err(error) => {
            eprintln!("baro-router: invalid listen address: {error}");
            std::process::exit(2);
        }
    };
    let listener = match tokio::net::TcpListener::bind(address).await {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("baro-router: bind {address}: {error}");
            std::process::exit(1);
        }
    };
    println!("listening on http://{address}");
    if let Err(error) = axum::serve(listener, app).await {
        eprintln!("baro-router: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_requires_three_failures_and_two_passes() {
        let mut state = HealthHysteresis::default();
        assert!(!state.observe(true));
        assert!(state.observe(true));
        assert!(state.observe(false));
        assert!(state.observe(false));
        assert!(!state.observe(false));
    }

    #[test]
    fn sent_aware_retry_never_retries_after_first_byte() {
        assert!(SendState::NotConnected.retry_allowed(false, false));
        assert!(!SendState::ConnectedNoBytes.retry_allowed(false, false));
        assert!(SendState::ConnectedNoBytes.retry_allowed(true, false));
        assert!(SendState::ConnectedNoBytes.retry_allowed(false, true));
        assert!(!SendState::FirstByteSent.retry_allowed(true, true));
    }

    #[test]
    fn engine_parser_keeps_port_probe_address() {
        let info = parse_engine("ollama=http://127.0.0.1:11434/").unwrap();
        assert_eq!(info.address, "127.0.0.1:11434");
        assert_eq!(info.port, 11434);
        assert!(!info.adopted);
    }

    #[test]
    fn discovery_answers_both_service_shapes() {
        let advertisement = DiscoveryAdvertisement::new("test", 8090);
        assert_eq!(advertisement.browse_services(), [BARO_SERVICE, PAIR_SERVICE]);
        assert_eq!(advertisement.txt.get("ni"), Some(&"8090".to_string()));
    }
}
