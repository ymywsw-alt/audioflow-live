/* AudioFlow v1 (FinishFlow 동일 스택)
 * - Render Web Service (Node)
 * - GitHub Auto Deploy
 * - OpenAI API: "파라미터(안전한 사운드 레시피 JSON)"만 생성
 * - 실제 음원은 서버에서 절차적(모방 없는) 합성 → WAV 반환
 *
 * 목표: "수익형 영상 배경 부품" (튀지 않음, 멜로디 없음, 저작권 리스크 최소)
 */

const express = require("express");
const crypto = require("crypto");

const app = express();
app.use(express.json({ limit: "1mb" }));

const PORT = process.env.PORT || 3000;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini"; // 없으면 기본값
const MAX_MINUTES = Number(process.env.MAX_MINUTES || 5); // Render 즉시생성은 1~5분 권장

// --------------------------
// Utility
// --------------------------
function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}
function nowStamp() {
  const d = new Date();
  const p = (x) => String(x).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}
function sha256(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}
function randSeed() {
  return crypto.randomBytes(8).readBigUInt64BE(0);
}

// --------------------------
// WAV writer (PCM16 mono)
// --------------------------
function writeWavMono16(samples, sampleRate) {
  const numFrames = samples.length;
  const bytesPerSample = 2;
  const dataSize = numFrames * bytesPerSample;
  const buffer = Buffer.alloc(44 + dataSize);

  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write("WAVE", 8);
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16); // PCM chunk size
  buffer.writeUInt16LE(1, 20); // PCM
  buffer.writeUInt16LE(1, 22); // mono
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * bytesPerSample, 28);
  buffer.writeUInt16LE(bytesPerSample, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(dataSize, 40);

  let o = 44;
  for (let i = 0; i < numFrames; i++) {
    let s = samples[i];
    if (s > 1) s = 1;
    if (s < -1) s = -1;
    const pcm = Math.round(s * 32767);
    buffer.writeInt16LE(pcm, o);
    o += 2;
  }
  return buffer;
}

function softClip(x, drive) {
  const t = Math.tanh(x * drive);
  const d = Math.tanh(drive);
  return t / d;
}

// 1-pole lowpass stateful
function makeOnePoleLP(cutoffHz, sampleRate) {
  const a = Math.exp(-2 * Math.PI * cutoffHz / sampleRate);
  let y = 0;
  return (x) => {
    y = a * y + (1 - a) * x;
    return y;
  };
}

function rmsNormalize(samples, targetDb) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  const rms = Math.sqrt(sum / samples.length + 1e-12);
  const target = Math.pow(10, targetDb / 20);
  let gain = target / (rms + 1e-12);
  gain = Math.min(gain, 6);

  let peak = 0;
  for (let i = 0; i < samples.length; i++) {
    samples[i] *= gain;
    const a = Math.abs(samples[i]);
    if (a > peak) peak = a;
  }
  if (peak > 0.99) {
    const g2 = 0.99 / peak;
    for (let i = 0; i < samples.length; i++) samples[i] *= g2;
  }
  return samples;
}

// --------------------------
// OpenAI: 안전한 "사운드 레시피(JSON)" 생성
// (음원 자체를 OpenAI에서 만들지 않음 → 저작권/유사도 리스크 최소화)
// --------------------------
async function fetchSoundRecipe({ country, minutes }) {
  // API 키 없으면 내부 기본 레시피 사용 (테스트는 가능)
  if (!OPENAI_API_KEY) {
    return {
      preset_name: country === "JP" ? "JP:DarkAmbientMinimal" : "KR:MinimalTension",
      noise_cutoff_hz: country === "JP" ? 1600 : 1800,
      sub_hz: country === "JP" ? 42 : 45,
      pulse_density: country === "JP" ? 0.12 : 0.15,
      intro_sec: 2,
      outro_sec: 3,
      target_db: -16
    };
  }

  const system = `
너는 음악 예술 AI가 아니라 '수익형 영상에 사용되는 저작권 무리스크 음원 생산 엔진'의 파라미터 설계자다.
멜로디/감정 과잉/모방 금지. 오직 '튀지 않는 배경 부품' 기준으로 사운드 파라미터만 결정한다.
반드시 JSON만 출력한다. 키는 아래만 허용:
preset_name, noise_cutoff_hz, sub_hz, pulse_density, intro_sec, outro_sec, target_db
값 범위:
noise_cutoff_hz: 900~3200
sub_hz: 35~70
pulse_density: 0.06~0.20
intro_sec: 1~3
outro_sec: 2~4
target_db: -18~-14
`;

  const user = `
국가=${country}, 길이(분)=${minutes}.
목표: 설명형/판단형/정보형 영상에 깔아도 메시지를 방해하지 않는 배경음.
요구: 멜로디 없음, 저작권/유사도 리스크 최소, 장시간에서도 피로 낮음.
`;

  // Chat Completions 사용 (FinishFlow 스타일 유지)
  const resp = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "authorization": `Bearer ${OPENAI_API_KEY}`,
      "content-type": "application/json"
    },
    body: JSON.stringify({
      model: OPENAI_MODEL,
      temperature: 0.2,
      messages: [
        { role: "system", content: system.trim() },
        { role: "user", content: user.trim() }
      ],
      response_format: { type: "json_object" }
    })
  });

  if (!resp.ok) {
    const t = await resp.text();
    throw new Error(`OpenAI error: ${t}`);
  }
  const data = await resp.json();
  const txt = data?.choices?.[0]?.message?.content || "{}";
  let parsed = {};
  try { parsed = JSON.parse(txt); } catch { parsed = {}; }

  // 안전 기본값 + 범위 클램프
  const recipe = {
    preset_name: String(parsed.preset_name || (country === "JP" ? "JP:DarkAmbientMinimal" : "KR:MinimalTension")),
    noise_cutoff_hz: clamp(Number(parsed.noise_cutoff_hz || (country === "JP" ? 1600 : 1800)), 900, 3200),
    sub_hz: clamp(Number(parsed.sub_hz || (country === "JP" ? 42 : 45)), 35, 70),
    pulse_density: clamp(Number(parsed.pulse_density || (country === "JP" ? 0.12 : 0.15)), 0.06, 0.20),
    intro_sec: clamp(Number(parsed.intro_sec || 2), 1, 3),
    outro_sec: clamp(Number(parsed.outro_sec || 3), 2, 4),
    target_db: clamp(Number(parsed.target_db || -16), -18, -14)
  };

  return recipe;
}

// --------------------------
// Procedural synthesis (no melody)
// - noise texture + sub drone + very subtle pulse energy
// - intro/outro envelope
// - micro macro movement to avoid "static loop" feel
// --------------------------
function synthWav({ country, minutes, recipe }) {
  const sampleRate = 44100;
  minutes = clamp(minutes, 1, MAX_MINUTES);
  const totalSec = minutes * 60;

  const introSec = clamp(recipe.intro_sec ?? 2, 1, 3);
  const outroSec = clamp(recipe.outro_sec ?? 3, 2, 4);

  const n = totalSec * sampleRate;
  const out = new Float32Array(n);

  const lp = makeOnePoleLP(recipe.noise_cutoff_hz, sampleRate);

  const subHz = recipe.sub_hz;
  const density = recipe.pulse_density;

  // deterministic-ish seed-based randomness for pulses
  const seed = randSeed();
  let r = Number(seed % 2147483647n);
  function prng() {
    // simple LCG
    r = (r * 48271) % 2147483647;
    return r / 2147483647;
  }

  const introS = introSec * sampleRate;
  const outroS = outroSec * sampleRate;

  for (let i = 0; i < n; i++) {
    const t = i / sampleRate;

    // noise -> LP
    const w = prng() * 2 - 1;
    const noise = lp(w) * 0.18;

    // sub drone with slow drift (no melody)
    const drift = 0.8 + 0.4 * Math.sin(2 * Math.PI * 0.03 * t);
    const sub = Math.sin(2 * Math.PI * (subHz * drift) * t) * 0.10;

    // subtle pulse energy (not a beat)
    const hit = (prng() < (density / sampleRate)) ? 1 : 0;
    const pulse = hit * 0.06;

    // slow amp movement
    const amp = 0.65 + 0.35 * Math.sin(2 * Math.PI * 0.015 * t);

    // macro drift to reduce fatigue + repetition feel
    const macro = 0.9 + 0.1 * Math.sin(2 * Math.PI * 0.007 * t);

    let s = (noise + sub + pulse) * amp * macro;

    // intro/outro
    if (i < introS) {
      const x = i / introS;
      s *= Math.pow(x, 1.6);
    } else if (i > n - outroS) {
      const x = (n - i) / outroS;
      s *= Math.pow(x, 1.6);
    }

    out[i] = softClip(s, 1.3);
  }

  rmsNormalize(out, recipe.target_db ?? -16);
  const wav = writeWavMono16(out, sampleRate);
  const hash = sha256(wav);

  const license_proof = {
    engine: "AudioFlow v1",
    country,
    minutes,
    sample_rate: sampleRate,
    policy: {
      no_melody: true,
      purpose: "revenue-video background component",
      copyright_risk: "minimized (procedural synthesis; no imitation)"
    },
    recipe,
    wav_sha256: hash,
    created_at: nowStamp()
  };

  return { wav, license_proof };
}

// --------------------------
// UI (최소 화면) + API
// --------------------------
app.get("/", (req, res) => {
  res.setHeader("content-type", "text/html; charset=utf-8");
  res.end(`
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>AudioFlow v1</title>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto;max-width:720px;margin:40px auto;padding:16px}
    .card{border:1px solid #ddd;border-radius:14px;padding:16px}
    button{padding:12px 14px;border-radius:12px;border:1px solid #111;background:#111;color:#fff;font-size:16px;cursor:pointer}
    input{padding:10px;font-size:16px;width:120px}
    small{opacity:.75}
  </style>
</head>
<body>
  <h1>AudioFlow v1</h1>
  <p><small>수익형 영상용 · 저작권 무리스크 · 배경음(멜로디 없음)</small></p>

  <div class="card">
    <h3>국가 선택</h3>
    <label><input type="radio" name="country" value="KR" checked> 한국(KR)</label>
    <label style="margin-left:12px;"><input type="radio" name="country" value="JP"> 일본(JP)</label>

    <h3 style="margin-top:18px;">길이(분)</h3>
    <p><small>Render 즉시 생성은 1~${MAX_MINUTES}분 권장 (장시간 60~180분은 다음 단계에서 별도 처리)</small></p>
    <input id="min" type="number" min="1" max="${MAX_MINUTES}" value="1" />

    <div style="margin-top:16px;">
      <button id="go">음원 생성 → WAV 다운로드</button>
    </div>

    <p id="msg" style="margin-top:12px;"></p>
  </div>

<script>
  const btn = document.getElementById('go');
  const msg = document.getElementById('msg');
  btn.onclick = async () =>代表 {};
</script>

<script>
  const go = document.getElementById('go');
  const msgEl = document.getElementById('msg');

  function getCountry(){
    const el = document.querySelector('input[name="country"]:checked');
    return el ? el.value : 'KR';
  }

  go.onclick = async () => {
    msgEl.textContent = '생성 중...';
    go.disabled = true;
    try{
      const minutes = Number(document.getElementById('min').value || 1);
      const country = getCountry();
      const res = await fetch('/make', {
        method:'POST',
        headers:{'content-type':'application/json'},
        body: JSON.stringify({ country, minutes })
      });
      if(!res.ok){
        const t = await res.text();
        throw new Error(t || 'failed');
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'audioflow_' + country + '_' + minutes + 'm.wav';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      msgEl.textContent = '완료: WAV 다운로드 시작됨';
    }catch(e){
      msgEl.textContent = '오류: ' + (e && e.message ? e.message : 'unknown');
    }finally{
      go.disabled = false;
    }
  };
</script>
</body>
</html>
  `);
});

app.get("/health", (req, res) => res.json({ ok: true }));

app.post("/make", async (req, res) => {
  try {
    const country = (req.body?.country === "JP") ? "JP" : "KR";
    const minutes = clamp(Number(req.body?.minutes || 1), 1, MAX_MINUTES);

    const recipe = await fetchSoundRecipe({ country, minutes });
    const { wav, license_proof } = synthWav({ country, minutes, recipe });

    // license proof는 헤더로도 남김(짧게)
    res.setHeader("x-audioflow-proof", license_proof.wav_sha256.slice(0, 16));
    res.setHeader("content-type", "audio/wav");
    res.setHeader("content-disposition", `attachment; filename="audioflow_${country}_${minutes}m.wav"`);
    res.setHeader("cache-control", "no-store");
    res.status(200).send(wav);
  } catch (e) {
    res.status(500).send(String(e?.message || e));
  }
});

app.get("/proof", async (req, res) => {
  // 최근 증빙은 서버 저장을 안 하므로, “어떻게 생성되는지” 정책만 노출
  res.json({
    engine: "AudioFlow v1",
    policy: {
      no_melody: true,
      purpose: "revenue-video background component",
      stack: "ChatGPT → GitHub → Render → OpenAI API(recipe JSON) → Render WAV"
    }
  });
});

app.listen(PORT, () => {
  console.log("AudioFlow live on port", PORT);
  if (!OPENAI_API_KEY) {
    console.log("WARNING: OPENAI_API_KEY is not set. Using fallback recipe (still generates WAV).");
  }
});
