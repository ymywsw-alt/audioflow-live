// audioflow-live/server.js
// AudioFlow Engine v1 (stable): /make -> generates WAV + download_url
// Principles: fixed schema, JSON-only, rate limit, no partial patching.

import express from "express";
import crypto from "crypto";
import fs from "fs";
import path from "path";
import os from "os";

const app = express();

// ---------- Config ----------
const PORT = process.env.PORT || 3000;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";

// 비용/폭주 방지
const RATE_LIMIT_PER_MIN = parseInt(process.env.RATE_LIMIT_PER_MIN || "6", 10); // per IP
const COOLDOWN_MS = parseInt(process.env.COOLDOWN_MS || "0", 10); // optional
const MAX_MINUTES = parseFloat(process.env.MAX_MINUTES || "5"); // duration cap

// CORS (UI에서 호출 가능하도록)
const ALLOW_ORIGIN = process.env.ALLOW_ORIGIN || "*";

// Files
const OUT_DIR = path.join(os.tmpdir(), "audioflow");
if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

// ---------- Middleware ----------
app.use(express.json({ limit: "1mb" }));
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", ALLOW_ORIGIN);
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type,Authorization");
  if (req.method === "OPTIONS") return res.sendStatus(204);
  next();
});

// ---------- Fixed Schema Helpers ----------
function emptyData() {
  return {
    title: "",
    preset: "CALM_LOOP",
    duration_sec: 0,
    loopable: false,
    prompt: {
      topic: "",
      mood: "",
      tempo_bpm: 0,
      instruments: [],
      do_not: [],
    },
    audio: {
      format: "wav",
      sample_rate: 44100,
      bitrate_kbps: 1411,
      file_name: "",
      download_url: "",
    },
    notes: [],
  };
}

function okResponse(data) {
  return { ok: true, code: "OK", data };
}

function errResponse(code) {
  return { ok: false, code, data: emptyData() };
}

// ---------- Rate Limit (in-memory) ----------
const ipBuckets = new Map(); // ip -> { tsMinute, count, lastTs }

function getClientIp(req) {
  const xf = req.headers["x-forwarded-for"];
  if (typeof xf === "string" && xf.length > 0) return xf.split(",")[0].trim();
  return req.socket?.remoteAddress || "unknown";
}

function rateLimitCheck(ip) {
  const now = Date.now();
  const minute = Math.floor(now / 60000);
  const cur = ipBuckets.get(ip) || { tsMinute: minute, count: 0, lastTs: 0 };

  // reset per minute
  if (cur.tsMinute !== minute) {
    cur.tsMinute = minute;
    cur.count = 0;
  }

  // optional cooldown
  if (COOLDOWN_MS > 0 && cur.lastTs && now - cur.lastTs < COOLDOWN_MS) {
    return { ok: false, code: "E-RATE-001" };
  }

  cur.count += 1;
  cur.lastTs = now;
  ipBuckets.set(ip, cur);

  if (cur.count > RATE_LIMIT_PER_MIN) return { ok: false, code: "E-RATE-001" };
  return { ok: true };
}

// ---------- JSON extraction (defensive) ----------
function extractJsonObject(text) {
  if (typeof text !== "string") return null;
  const first = text.indexOf("{");
  const last = text.lastIndexOf("}");
  if (first === -1 || last === -1 || last <= first) return null;
  const slice = text.slice(first, last + 1);
  try {
    return JSON.parse(slice);
  } catch {
    return null;
  }
}

// ---------- OpenAI (JSON plan only) ----------
async function makeMusicPlan({ topic, preset, durationSec }) {
  // If no key, still run with deterministic fallback (no crash)
  if (!OPENAI_API_KEY) {
    return {
      title: `AudioFlow BGM: ${topic || "Untitled"}`,
      preset,
      duration_sec: durationSec,
      loopable: preset !== "UPBEAT_SHORTS",
      prompt: {
        topic: topic || "",
        mood: preset === "UPBEAT_SHORTS" ? "energetic, light, positive" : "calm, steady, unobtrusive",
        tempo_bpm: preset === "UPBEAT_SHORTS" ? 110 : 80,
        instruments: preset === "UPBEAT_SHORTS" ? ["soft pluck", "light percussion", "bass pad"] : ["warm pad", "soft piano", "air texture"],
        do_not: ["no artist imitation", "no recognizable melodies", "no lyrics", "no harsh distortion"],
      },
      notes: ["OPENAI_API_KEY missing: used fallback plan (audio generation still works)."],
    };
  }

  const system = [
    "You are an audio-for-video BGM planning engine.",
    "Return ONLY JSON. No markdown, no code fences.",
    "Goal: produce a safe, generic, non-infringing background music plan for monetizable videos.",
    "Do NOT imitate any artist or existing song. Avoid distinctive melodies.",
    "Keep music unobtrusive; prioritize watch-time and focus.",
  ].join(" ");

  const user = {
    topic: topic || "",
    preset,
    duration_sec: durationSec,
    required_keys: [
      "title",
      "preset",
      "duration_sec",
      "loopable",
      "prompt{topic,mood,tempo_bpm,instruments[],do_not[]}",
      "notes[]",
    ],
  };

  const body = {
    model: OPENAI_MODEL,
    temperature: 0.2,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: system },
      { role: "user", content: JSON.stringify(user) },
    ],
  };

  const resp = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${OPENAI_API_KEY}`,
    },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const t = await resp.text().catch(() => "");
    throw new Error(`openai_http_${resp.status}: ${t.slice(0, 200)}`);
  }

  const json = await resp.json();
  const content = json?.choices?.[0]?.message?.content || "";
  const plan = extractJsonObject(content);
  if (!plan) throw new Error("openai_parse_fail");
  return plan;
}

// ---------- WAV synthesis (no external deps) ----------
function writeWavMono16(filePath, sampleRate, samplesInt16) {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const dataSize = samplesInt16.length * 2;

  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + dataSize, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16); // PCM
  header.writeUInt16LE(1, 20); // format = PCM
  header.writeUInt16LE(numChannels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(bitsPerSample, 34);
  header.write("data", 36);
  header.writeUInt32LE(dataSize, 40);

  const data = Buffer.alloc(dataSize);
  for (let i = 0; i < samplesInt16.length; i++) {
    data.writeInt16LE(samplesInt16[i], i * 2);
  }

  fs.writeFileSync(filePath, Buffer.concat([header, data]));
}

function clamp16(x) {
  if (x > 32767) return 32767;
  if (x < -32768) return -32768;
  return x | 0;
}

function synthAmbientWav({ durationSec, preset }) {
  const sampleRate = 44100;
  const total = Math.max(1, Math.floor(durationSec * sampleRate));

  // Preset shaping
  const tempo =
    preset === "UPBEAT_SHORTS" ? 110 :
    preset === "DOCUMENTARY" ? 90 : 80;

  const baseFreq =
    preset === "UPBEAT_SHORTS" ? 220 :
    preset === "DOCUMENTARY" ? 196 : 174;

  const amp =
    preset === "UPBEAT_SHORTS" ? 0.22 :
    preset === "DOCUMENTARY" ? 0.18 : 0.16;

  // Deterministic-ish seed
  const seed = crypto.randomBytes(8).readUInt32LE(0);
  let rnd = seed;
  const rand = () => {
    // xorshift32
    rnd ^= rnd << 13; rnd ^= rnd >>> 17; rnd ^= rnd << 5;
    return (rnd >>> 0) / 4294967296;
  };

  // Simple pad: mixed sines + slow LFO + soft noise
  const samples = new Int16Array(total);

  const beatHz = tempo / 60;
  const lfoHz = preset === "UPBEAT_SHORTS" ? 0.35 : 0.18;

  const freqs = [
    baseFreq,
    baseFreq * 1.25,
    baseFreq * 1.5,
    baseFreq * 2.0,
  ];

  let phase = freqs.map(() => rand() * Math.PI * 2);
  let lfoPhase = rand() * Math.PI * 2;
  let beatPhase = rand() * Math.PI * 2;

  for (let i = 0; i < total; i++) {
    const t = i / sampleRate;

    // LFO for movement
    lfoPhase += (2 * Math.PI * lfoHz) / sampleRate;
    const lfo = 0.6 + 0.4 * Math.sin(lfoPhase);

    // Gentle pulse to give structure (still BGM)
    beatPhase += (2 * Math.PI * beatHz) / sampleRate;
    const pulse = 0.75 + 0.25 * Math.max(0, Math.sin(beatPhase));

    // ADSR-ish fade in/out
    const fadeIn = Math.min(1, t / 1.2);
    const fadeOut = Math.min(1, (durationSec - t) / 1.2);
    const env = Math.max(0, Math.min(fadeIn, fadeOut));

    let x = 0;

    // Sine stack
    for (let k = 0; k < freqs.length; k++) {
      phase[k] += (2 * Math.PI * freqs[k]) / sampleRate;
      x += Math.sin(phase[k]) * (k === 0 ? 1.0 : 0.55 / (k));
    }

    // Soft noise texture
    const noise = (rand() * 2 - 1) * 0.08;

    // UPBEAT: slightly more rhythmic click-ish layer (still safe)
    const click = preset === "UPBEAT_SHORTS"
      ? (Math.sin(beatPhase * 4) > 0.98 ? 0.35 : 0)
      : 0;

    const y = (x * 0.55 + noise + click) * amp * lfo * pulse * env;

    samples[i] = clamp16(y * 32767);
  }

  return { sampleRate, samples };
}

// ---------- Routes ----------
app.get("/health", (req, res) => {
  const hasKey = !!OPENAI_API_KEY;
  res.json({ ok: true, status: "healthy", hasOpenAIKey: hasKey });
});

app.get("/", (req, res) => {
  // 최소 UI (엔진 상태 확인용)
  res.setHeader("Content-Type", "text/html; charset=utf-8");
  res.end(`
    <html>
      <head><meta charset="utf-8"/><title>AudioFlow Engine</title></head>
      <body style="font-family:Arial; padding:24px;">
        <h2>AudioFlow Engine v1</h2>
        <p>POST <code>/make</code> to generate a WAV and get download_url.</p>
        <p>GET <code>/health</code> for healthcheck.</p>
      </body>
    </html>
  `);
});

app.get("/download/:file", (req, res) => {
  try {
    const file = req.params.file || "";
    const safe = path.basename(file);
    const full = path.join(OUT_DIR, safe);
    if (!fs.existsSync(full)) return res.status(404).json(errResponse("E-SERVER-001"));
    res.setHeader("Content-Type", "audio/wav");
    res.setHeader("Content-Disposition", `attachment; filename="${safe}"`);
    fs.createReadStream(full).pipe(res);
  } catch {
    return res.status(500).json(errResponse("E-SERVER-001"));
  }
});

app.post("/make", async (req, res) => {
  const ip = getClientIp(req);
  const rl = rateLimitCheck(ip);
  if (!rl.ok) return res.status(429).json(errResponse(rl.code));

  try {
    const topic = String(req.body?.topic || "").slice(0, 200);
    const presetRaw = String(req.body?.preset || "CALM_LOOP").toUpperCase();
    const preset =
      presetRaw === "UPBEAT_SHORTS" ? "UPBEAT_SHORTS" :
      presetRaw === "DOCUMENTARY" ? "DOCUMENTARY" : "CALM_LOOP";

    let durationSec = parseInt(req.body?.duration_sec || "90", 10);
    if (!Number.isFinite(durationSec) || durationSec <= 0) durationSec = 90;

    // cap duration
    const cap = Math.max(10, Math.floor(MAX_MINUTES * 60));
    durationSec = Math.min(durationSec, cap);

    // 1) OpenAI로 "플랜(JSON)"만 생성 (서비스 흔들림 방지)
    let plan;
    try {
      plan = await makeMusicPlan({ topic, preset, durationSec });
    } catch (e) {
      // OpenAI 실패해도 서비스는 유지: plan fallback
      plan = {
        title: `AudioFlow BGM: ${topic || "Untitled"}`,
        preset,
        duration_sec: durationSec,
        loopable: preset !== "UPBEAT_SHORTS",
        prompt: {
          topic: topic || "",
          mood: preset === "UPBEAT_SHORTS" ? "energetic, light, positive" : "calm, steady, unobtrusive",
          tempo_bpm: preset === "UPBEAT_SHORTS" ? 110 : 80,
          instruments: preset === "UPBEAT_SHORTS" ? ["soft pluck", "light percussion", "bass pad"] : ["warm pad", "soft piano", "air texture"],
          do_not: ["no artist imitation", "no recognizable melodies", "no lyrics", "no harsh distortion"],
        },
        notes: [`OpenAI failed -> fallback plan. (${String(e?.message || "").slice(0, 80)})`],
      };
    }

    // 2) 서버에서 WAV 합성 (외부 의존성 없이 안정적으로)
    const { sampleRate, samples } = synthAmbientWav({ durationSec, preset });

    const stamp = new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
    const fileName = `audioflow_${stamp}_${crypto.randomBytes(3).toString("hex")}.wav`;
    const fullPath = path.join(OUT_DIR, fileName);
    writeWavMono16(fullPath, sampleRate, samples);

    // 3) 고정 스키마로 응답
    const data = emptyData();
    data.title = String(plan?.title || `AudioFlow BGM: ${topic || "Untitled"}`).slice(0, 120);
    data.preset = preset;
    data.duration_sec = durationSec;
    data.loopable = preset !== "UPBEAT_SHORTS";
    data.prompt.topic = topic;
    data.prompt.mood = String(plan?.prompt?.mood || "").slice(0, 80);
    data.prompt.tempo_bpm = Number(plan?.prompt?.tempo_bpm || (preset === "UPBEAT_SHORTS" ? 110 : 80)) || 80;
    data.prompt.instruments = Array.isArray(plan?.prompt?.instruments) ? plan.prompt.instruments.slice(0, 8) : [];
    data.prompt.do_not = Array.isArray(plan?.prompt?.do_not) ? plan.prompt.do_not.slice(0, 8) : [];

    data.audio.format = "wav";
    data.audio.sample_rate = 44100;
    data.audio.bitrate_kbps = 1411;
    data.audio.file_name = fileName;
    data.audio.download_url = `/download/${fileName}`;
    data.notes = Array.isArray(plan?.notes) ? plan.notes.slice(0, 6) : [];

    return res.json(okResponse(data));
  } catch (e) {
    // 최종 보호: 어떤 예외든 고정 스키마로만 반환
    return res.status(500).json(errResponse("E-SERVER-001"));
  }
});

// ---------- Start ----------
app.listen(PORT, () => {
  console.log(`AudioFlow Engine v1 listening on :${PORT}`);
});
