"use strict";

const LS_CONV = "baro.conversation";
const LS_SETTINGS = "baro.settings";

const DEFAULT_SETTINGS = {
  serverUrl: location.origin,
  temperature: 0.7,
  maxTokens: 512,
  systemPrompt: "",
};

function loadJSON(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function saveJSON(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* private mode / quota: conversation just stops persisting */
  }
}

let settings = { ...DEFAULT_SETTINGS, ...loadJSON(LS_SETTINGS, {}) };
let conversation = loadJSON(LS_CONV, []);
let pendingAbort = null;
let pendingResponseId = null;

const messagesEl = document.getElementById("messages");
const emptyHint = document.getElementById("empty-hint");
const modelNameEl = document.getElementById("model-name");
const composer = document.getElementById("composer");
const promptEl = document.getElementById("message-input");
const sendBtn = document.getElementById("send-button");
const micBtn = document.getElementById("mic");
const settingsBtn = document.getElementById("settings-btn");
const newChatBtn = document.getElementById("new-chat-btn");
const settingsBackdrop = document.getElementById("settings-backdrop");
const settingsClose = document.getElementById("settings-close");
const settingsClear = document.getElementById("settings-clear");
const serverUrlEl = document.getElementById("server-address");
const temperatureEl = document.getElementById("temperature");
const temperatureVal = document.getElementById("temperature-val");
const maxTokensEl = document.getElementById("max-tokens");
const systemPromptEl = document.getElementById("system-prompt");

function renderMessages() {
  messagesEl.querySelectorAll(".msg").forEach((n) => n.remove());
  emptyHint.hidden = conversation.length > 0;
  for (const m of conversation) {
    messagesEl.appendChild(bubbleFor(m.role, m.content));
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function bubbleFor(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  if (role === "assistant") el.dataset.role = "assistant-message";
  el.textContent = text;
  return el;
}

function scrollToEnd() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setBusy(busy) {
  sendBtn.disabled = false;
  sendBtn.classList.toggle("stop", busy);
  sendBtn.textContent = busy ? "■" : "➤";
  promptEl.disabled = busy;
}

async function refreshModel() {
  try {
    const r = await fetch(new URL("/v1/models", settings.serverUrl));
    if (!r.ok) return;
    const j = await r.json();
    const id = j.data && j.data[0] && j.data[0].id;
    if (id) modelNameEl.textContent = id;
  } catch {
    modelNameEl.textContent = "baro (offline)";
  }
}

function buildMessages(userText) {
  const msgs = [];
  if (settings.systemPrompt) msgs.push({ role: "system", content: settings.systemPrompt });
  for (const m of conversation) msgs.push({ role: m.role, content: m.content });
  msgs.push({ role: "user", content: userText });
  return msgs;
}

async function sendMessage(userText) {
  conversation.push({ role: "user", content: userText });
  saveJSON(LS_CONV, conversation);
  renderMessages();

  const assistantBubble = bubbleFor("assistant", "");
  emptyHint.hidden = true;
  messagesEl.appendChild(assistantBubble);
  scrollToEnd();

  const controller = new AbortController();
  pendingAbort = controller;
  pendingResponseId = null;
  setBusy(true);

  let text = "";
  try {
    const resp = await fetch(new URL("/v1/chat/completions", settings.serverUrl), {
      method: "POST",
      headers: { "content-type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        messages: buildMessages(userText),
        stream: true,
        temperature: settings.temperature,
        max_tokens: Number(settings.maxTokens) || 512,
      }),
    });
    if (!resp.ok || !resp.body) throw new Error(`http ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const events = buf.split("\n\n");
      buf = events.pop() ?? "";
      for (const evt of events) {
        const line = evt.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") continue;
        let obj;
        try {
          obj = JSON.parse(payload);
        } catch {
          continue;
        }
        if (obj.id) pendingResponseId = obj.id;
        const delta = obj.choices && obj.choices[0] && obj.choices[0].delta;
        if (delta && typeof delta.content === "string") {
          text += delta.content;
          assistantBubble.textContent = text;
          scrollToEnd();
        }
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      assistantBubble.textContent = text || `(error: ${err.message})`;
    }
  } finally {
    pendingAbort = null;
    setBusy(false);
  }

  conversation.push({ role: "assistant", content: text });
  saveJSON(LS_CONV, conversation);
}

async function stopGeneration() {
  if (!pendingAbort) return;
  pendingAbort.abort();
  if (pendingResponseId) {
    try {
      await fetch(new URL("/v1/cancel", settings.serverUrl), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ id: pendingResponseId }),
      });
    } catch {
      /* best effort */
    }
  }
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  if (pendingAbort) {
    stopGeneration();
    return;
  }
  const text = promptEl.value.trim();
  if (!text) return;
  promptEl.value = "";
  autoGrow();
  sendMessage(text);
});

promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

function autoGrow() {
  promptEl.style.height = "auto";
  promptEl.style.height = `${Math.min(promptEl.scrollHeight, 120)}px`;
}
promptEl.addEventListener("input", autoGrow);

newChatBtn.addEventListener("click", () => {
  if (pendingAbort) stopGeneration();
  conversation = [];
  saveJSON(LS_CONV, conversation);
  renderMessages();
});

function openSettings() {
  serverUrlEl.value = settings.serverUrl;
  temperatureEl.value = settings.temperature;
  temperatureVal.textContent = settings.temperature;
  maxTokensEl.value = settings.maxTokens;
  systemPromptEl.value = settings.systemPrompt;
  settingsBackdrop.hidden = false;
}

function closeSettingsAndSave() {
  settings.serverUrl = serverUrlEl.value.trim() || DEFAULT_SETTINGS.serverUrl;
  settings.temperature = Number(temperatureEl.value);
  settings.maxTokens = Number(maxTokensEl.value) || 512;
  settings.systemPrompt = systemPromptEl.value;
  saveJSON(LS_SETTINGS, settings);
  settingsBackdrop.hidden = true;
  refreshModel();
}

settingsBtn.addEventListener("click", openSettings);
settingsClose.addEventListener("click", closeSettingsAndSave);
settingsBackdrop.addEventListener("click", (e) => {
  if (e.target === settingsBackdrop) closeSettingsAndSave();
});
temperatureEl.addEventListener("input", () => {
  temperatureVal.textContent = temperatureEl.value;
});
settingsClear.addEventListener("click", () => {
  conversation = [];
  saveJSON(LS_CONV, conversation);
  renderMessages();
  settingsBackdrop.hidden = true;
});

// Voice input (P3a): PCM WAV -> POST /v1/audio/transcriptions.
// Hidden outside a secure context, since getUserMedia requires one.
if (window.isSecureContext && navigator.mediaDevices && (window.AudioContext || window.webkitAudioContext)) {
  micBtn.hidden = false;
  let recording = null;

  function wavBlob(samples, rate) {
    const b = new ArrayBuffer(44 + samples.length * 2);
    const v = new DataView(b);
    const text = (at, s) => [...s].forEach((c, i) => v.setUint8(at + i, c.charCodeAt(0)));
    text(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); text(8, "WAVE");
    text(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    text(36, "data"); v.setUint32(40, samples.length * 2, true);
    samples.forEach((x, i) => v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, x)) * 0x7fff, true));
    return new Blob([b], { type: "audio/wav" });
  }

  micBtn.addEventListener("click", async () => {
    if (recording) {
      recording.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const Context = window.AudioContext || window.webkitAudioContext;
      const context = new Context();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);
      const sink = context.createGain();
      const chunks = [];
      sink.gain.value = 0;
      processor.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      source.connect(processor); processor.connect(sink); sink.connect(context.destination);
      recording = { stop() {
        const rate = context.sampleRate;
        source.disconnect(); processor.disconnect(); sink.disconnect(); stream.getTracks().forEach((t) => t.stop());
        context.close(); recording = null; micBtn.classList.remove("recording");
        const samples = new Float32Array(chunks.reduce((n, x) => n + x.length, 0));
        chunks.reduce((at, x) => (samples.set(x, at), at + x.length), 0);
        const form = new FormData();
        form.append("file", wavBlob(samples, rate), "voice.wav");
        form.append("response_format", "text");
        void (async () => { try {
          const r = await fetch(new URL("/v1/audio/transcriptions", settings.serverUrl), {
            method: "POST",
            body: form,
          });
          const text = await r.text();
          if (r.ok && text) {
            promptEl.value = (promptEl.value ? `${promptEl.value} ` : "") + text.trim();
            autoGrow();
            promptEl.focus();
          }
        } catch {
          /* transcription unavailable; the typed path still works */
        }})();
      }};
      micBtn.classList.add("recording");
    } catch {
      /* mic permission denied or unavailable */
    }
  });
}

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/web/sw.js", { scope: "/" }).catch(() => {});
}

renderMessages();
refreshModel();
