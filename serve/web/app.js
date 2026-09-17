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

// Voice input (P3a): MediaRecorder -> POST /v1/audio/transcriptions.
// Hidden outside a secure context, since getUserMedia requires one.
if (window.isSecureContext && navigator.mediaDevices && window.MediaRecorder) {
  micBtn.hidden = false;
  let recorder = null;
  let chunks = [];

  micBtn.addEventListener("click", async () => {
    if (recorder && recorder.state === "recording") {
      recorder.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunks = [];
      recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (e) => {
        if (e.data.size) chunks.push(e.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        micBtn.classList.remove("recording");
        const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
        const form = new FormData();
        form.append("file", blob, "voice.webm");
        form.append("response_format", "text");
        try {
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
        }
      };
      recorder.start();
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
