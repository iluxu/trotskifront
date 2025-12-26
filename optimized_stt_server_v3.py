import asyncio, json, os, time, re, hashlib, shutil, io, wave
from typing import Set, Optional, List, Dict, Any
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path  # For multi-OS compatibility

import numpy as np
import websockets
import requests
import sys
from pathlib import Path
from dotenv import load_dotenv
import uuid


def load_env():
    # Prefer .env next to this script; fall back to current working dir.
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / ".env.local",
        script_dir / ".env",
        Path.cwd() / ".env.local",
        Path.cwd() / ".env",
    ]
    loaded = False
    for p in candidates:
        if p.is_file():
            load_dotenv(p, override=False)  # don’t overwrite real env vars
            print(f"[env] Loaded {p}")
            loaded = True
            break
    if not loaded:
        # As a last resort, search upwards from CWD
        from dotenv import find_dotenv
        found = find_dotenv(usecwd=True)
        if found:
            load_dotenv(found, override=False)
            print(f"[env] Loaded {found}")

load_env()


# ========================
# Configuration
# ========================
HOST = os.getenv("STT_HOST", "127.0.0.1")
PORT = int(os.getenv("STT_PORT", "8123"))

SAMPLE_RATE = 16000
WINDOW_SECONDS = float(os.getenv("STT_WINDOW_SECONDS", "6"))
HOP_SECONDS = float(os.getenv("STT_HOP_SECONDS", "0.8"))
ENERGY_GATE = float(os.getenv("STT_ENERGY_GATE", "1e-4"))

STT_BACKEND = os.getenv("STT_BACKEND", "faster-whisper").lower()
MODEL_NAME = os.getenv("STT_MODEL", "small")
COMPUTE_TYPE = os.getenv("STT_COMPUTE", "int8")
DEVICE = os.getenv("STT_DEVICE", "cpu").lower()  # cpu | cuda | auto
FORCE_LANG = os.getenv("STT_LANG") or None
OPENAI_STT_MODEL = os.getenv("STT_STT_MODEL", "whisper-1")
# Generic tech prompt for better transcription of jargon
INITIAL_PROMPT = os.getenv(
    "STT_INITIAL_PROMPT",
    "Software engineering, data structures, algorithms, system design, cloud computing, AWS, Azure, GCP, microservices, API, "
    "CI/CD, DevOps, machine learning, data science, Python, Java, JavaScript, SQL, NoSQL, product management, agile, scrum."
)

# LLM
LLM_ENABLED = os.getenv("STT_LLM_ENABLED", "1") not in ("0", "false", "False")
GPT_MODEL = os.getenv("STT_LLM_MODEL", "gpt-5-nano")
GPT_EFFORT = os.getenv("STT_LLM_EFFORT", "low")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Advanced LLM Context Control (restored from French version)
LLM_INCLUDE_FULL_TRANSCRIPT = os.getenv("STT_LLM_INCLUDE_FULL_TRANSCRIPT", "1") not in ("0", "false", "False")
TECH_INTERVIEW_MODE = os.getenv("STT_TECH_INTERVIEW_MODE", "1") not in ("0", "false", "False")
LLM_CONTEXT_MODE = os.getenv("STT_LLM_CONTEXT_MODE", "full").lower()
LLM_WINDOW_LINES = int(os.getenv("STT_LLM_WINDOW_LINES", "160"))
LLM_HEAD_LINES = int(os.getenv("STT_LLM_HEAD_LINES", "60"))
LLM_TAIL_LINES = int(os.getenv("STT_LLM_TAIL_LINES", "300"))
LLM_MAX_CONTEXT_CHARS = int(os.getenv("STT_LLM_MAX_CONTEXT_CHARS", "300000"))
MAX_OUTTOK = int(os.getenv("STT_MAX_OUTTOK", "512"))
PERSONA = os.getenv("STT_LLM_PERSONA", "candidate").lower()

# LLM Rate Limiting
LLM_MIN_GAP_SEC = float(os.getenv("STT_LLM_MIN_GAP_SEC", "1.0"))
ANSWERS_PER_MIN = int(os.getenv("STT_LLM_ANSWERS_PER_MIN", "8"))
SEEN_TTL_SEC = float(os.getenv("STT_SEEN_TTL_SEC", "60"))
MAX_CONCURRENT_LLM = int(os.getenv("STT_MAX_CONCURRENT_LLM", "2"))

openai_stt_client = None

class SimpleSegment:
    def __init__(self, text: str):
        self.text = text



# VAD & idle flush (NOUVEAU)
VAD_MIN_SIL_MS   = int(os.getenv("STT_VAD_MIN_SIL_MS", "350"))
VAD_PAD_MS       = int(os.getenv("STT_VAD_PAD_MS", "220"))
IDLE_FLUSH_MS    = int(os.getenv("STT_IDLE_FLUSH_MS", "1400"))
MIN_WORDS_FLUSH  = int(os.getenv("STT_MIN_WORDS_FLUSH", "3"))
NO_SPEECH_TH     = float(os.getenv("STT_NO_SPEECH_TH", "0.12"))
LOG_PROB_TH      = float(os.getenv("STT_LOG_PROB_TH", "-1.2"))
DROP_PROMPT_ECHO = os.getenv("STT_DROP_PROMPT_ECHO", "1") not in ("0","false","False")




# Debug
DEBUG = os.getenv("STT_DEBUG", "1") not in ("0", "false", "False")
VERBOSE_BUFFER = os.getenv("STT_VERBOSE_BUFFER", "0") not in ("0", "false", "False")

LICENSE_FILE = Path.home() / ".interview_copilot_license"
TRIAL_FILE = Path.home() / ".interview_copilot_trial_30m"
TRIAL_MINUTES = float(os.getenv("STT_TRIAL_MINUTES", "30"))
LICENSE_ACCEPT_ANY = os.getenv("STT_LICENSE_ACCEPT_ANY", "0") not in ("0", "false", "False")
LICENSE_VALID = False
GUMROAD_PRODUCT_PERMALINK = "himlkf"

AUTO_ANSWER_ARMED_UNTIL = 0.0

def auto_answer_armed() -> bool:
    return time.time() < AUTO_ANSWER_ARMED_UNTIL

def disarm_auto_answer():
    global AUTO_ANSWER_ARMED_UNTIL
    AUTO_ANSWER_ARMED_UNTIL = 0.0


# ========================
# Small Helpers & Context Builders (restored from French version)
# ========================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _qkey(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm(s).lower())

def get_last_n_lines(n: int) -> List[str]:
    """Returns the last N numbered lines with their actual line numbers."""
    if n <= 0 or not transcript_lines: return []
    start_idx = max(0, len(transcript_lines) - n)
    return [f"{i+1}. {transcript_lines[i]}" for i in range(start_idx, len(transcript_lines))]

def get_head_tail_lines(head: int, tail: int) -> List[str]:
    """Returns a numbered 'head + ... + tail' to preserve global context without sending everything."""
    total = len(transcript_lines)
    if total == 0: return []
    head = max(0, min(head, total))
    tail = max(0, min(tail, total - head))
    parts = []
    for i in range(head):
        parts.append(f"{i+1}. {transcript_lines[i]}")
    if head + tail < total:
        parts.append("...")
    start_tail = max(head, total - tail)
    for i in range(start_tail, total):
        parts.append(f"{i+1}. {transcript_lines[i]}")
    return parts

def build_llm_context_text() -> str:
    """Constructs the context to send to the LLM based on LLM_CONTEXT_MODE."""
    mode = (LLM_CONTEXT_MODE or "window").lower()
    if mode == "full":
        lines = [f"{i+1}. {line}" for i, line in enumerate(transcript_lines)]
    elif mode == "headtail":
        lines = get_head_tail_lines(LLM_HEAD_LINES, LLM_TAIL_LINES)
    else:  # "window"
        lines = get_last_n_lines(LLM_WINDOW_LINES)
    
    txt = "\n".join(lines)
    if LLM_MAX_CONTEXT_CHARS and len(txt) > LLM_MAX_CONTEXT_CHARS:
        txt = txt[-LLM_MAX_CONTEXT_CHARS:]
    return txt

def verify_license(key):
    """Verify key with Gumroad (legacy)."""
    try:
        resp = requests.post(
            "https://api.gumroad.com/v2/licenses/verify",
            data={
                "product_id": "ADySSUI1rySCo72YL8L4hA==",
                "license_key": key,
            },
            timeout=5
        )
        if resp.status_code == 200:
            return resp.json().get("success", False)
    except:
        # If can't reach Gumroad, allow (offline mode)
        return True
    return False

def _license_format_ok(key: str) -> bool:
    key = (key or "").strip().upper()
    return re.fullmatch(r"[A-Z0-9]{4}(?:-[A-Z0-9]{4}){4}", key) is not None

def is_license_valid(key: str) -> bool:
    key = (key or "").strip().upper()
    if not key:
        return False
    if LICENSE_ACCEPT_ANY:
        return True
    if _license_format_ok(key):
        return True
    return verify_license(key)

def start_trial_if_needed():
    if TRIAL_FILE.exists():
        return
    TRIAL_FILE.write_text(str(time.time()))

def trial_remaining_seconds() -> int:
    if not TRIAL_FILE.exists():
        return 0
    try:
        start = float(TRIAL_FILE.read_text().strip())
    except Exception:
        return 0
    remaining = (start + (TRIAL_MINUTES * 60.0)) - time.time()
    return max(0, int(remaining))

def check_license():
    """Load existing license or allow trial-based access."""
    global LICENSE_VALID
    env_key = os.getenv("STT_LICENSE_KEY", "").strip()
    if env_key and is_license_valid(env_key):
        LICENSE_FILE.write_text(env_key)
        LICENSE_VALID = True
        print("[License] Valid (env)")
        return True

    if LICENSE_FILE.exists():
        key = LICENSE_FILE.read_text().strip()
        if is_license_valid(key):
            LICENSE_VALID = True
            print("[License] Valid (file)")
            return True

    print("\n" + "="*60)
    print("TRIAL MODE")
    print(f"Free trial: {TRIAL_MINUTES:.0f} minutes")
    print("Enter a license key in the web UI to unlock.")
    print("="*60)
    return True


# ========================
# Whisper Loading with Resilient, Multi-OS Cache
# ========================
def hf_cache_dir() -> str:
    """Returns a platform-agnostic cache directory inside the user's home."""
    cache_path = Path.home() / ".cache" / "hf_models"
    cache_path.mkdir(parents=True, exist_ok=True)
    return str(cache_path)

def load_whisper_model() -> Any:
    if STT_BACKEND != "faster-whisper":
        return None
    cache_root = hf_cache_dir()
    try:
        print(f"[model] Loading {MODEL_NAME} in {cache_root} (device={DEVICE}, compute={COMPUTE_TYPE})")
        from faster_whisper import WhisperModel
        return WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE, download_root=cache_root)
    except Exception:
        model_folder_name = f"models--Systran--faster-whisper-{MODEL_NAME}"
        corrupted_path = os.path.join(cache_root, model_folder_name)
        if os.path.isdir(corrupted_path):
            print(f"[model] Corrupted cache detected, purging: {corrupted_path}")
            shutil.rmtree(corrupted_path, ignore_errors=True)
        from faster_whisper import WhisperModel
        return WhisperModel(MODEL_NAME, compute_type=COMPUTE_TYPE, download_root=cache_root)

# ========================
# LLM Response Sanitization (restored from French version)
# ========================
COACHING_PATTERNS = [
    r"\bHappy to elaborate if useful\b.*", r"\bLet me know if you'd like more details\b.*",
    r"\bYou (?:should|could|can)\b.*", r"\bYou demonstrate\b.*",
    r"\bOne (?:improvement|area to improve)\b.*", r"\bYour answer\b.*",
]

def sanitize_candidate_voice(text: str) -> str:
    for pat in COACHING_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\byou\b", "the interviewer", text, flags=re.IGNORECASE)
    text = re.sub(r"\byour\b", "the", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip(" .")

# ========================
# LLM Analyzer (more robust)
# ========================
@dataclass
class ConversationSegment:
    text: str; timestamp: float

@dataclass
class QuestionCandidate:
    question: str; context: str; confidence: float; urgency: str; topic_area: str; timestamp: float
    should_answer: bool = False

class ImprovedLLMAnalyzer:
    def __init__(self, api_key: str):
        self.enabled = LLM_ENABLED and bool(api_key.startswith("sk-"))
        self.client = None
        self.previous_response_id = None
        self.last_llm_emit = 0.0
        self.seen_questions: Dict[str, float] = {}
        self.answers_timestamps = deque(maxlen=64)
        self.sem = asyncio.Semaphore(MAX_CONCURRENT_LLM)

        if self.enabled:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                print(f"[llm] Ready (model={GPT_MODEL}, effort={GPT_EFFORT})")
            except Exception as e:
                print(f"[llm] Init error: {e}"); self.enabled = False
        else:
            print("[llm] Disabled (no valid API key)")

    def _trim_seen(self):
        now = time.time()
        expired = [k for k, t in self.seen_questions.items() if now - t > SEEN_TTL_SEC]
        for k in expired: self.seen_questions.pop(k, None)

    async def analyze_segment(self, segment: ConversationSegment) -> List[QuestionCandidate]:
        if not self.enabled: return []
        self._trim_seen()
        return await self._detect_questions(segment)

    async def _detect_questions(self, segment: ConversationSegment) -> List[QuestionCandidate]:
        if time.time() - self.last_llm_emit < LLM_MIN_GAP_SEC: return []
        qs = self._extract_questions_aggressive(segment.text)
        if not qs: return []
        if DEBUG: print(f"[llm] Found {len(qs)} potential questions: {[q[:40] + '...' for q in qs]}")

        candidates: List[QuestionCandidate] = []
        for q in qs:
            k = _qkey(q)
            if not k or k in self.seen_questions: continue
            
            decide = await asyncio.get_running_loop().run_in_executor(None, self._should_answer, q)
            if decide.get("should_answer"):
                self.seen_questions[k] = time.time()
                cand = QuestionCandidate(
                    question=q, context=segment.text, confidence=decide.get("confidence", 0.7),
                    urgency="relevant", topic_area="interview", timestamp=segment.timestamp, should_answer=True,
                )
                candidates.append(cand)
                if DEBUG: print(f"[llm] Will answer: {q}")
        return candidates[:3]

    def _extract_questions_aggressive(self, text: str) -> List[str]:
        """Advanced 6-stage question extraction (restored from French version)."""
        text = _norm(text)
        if not text: return []
        sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]
        questions: List[str] = []
        for s in sentences:
            if s.endswith("?") and len(s.split()) >= 5: questions.append(s)
        
        head_terms = ["what","how","why","when","where","which","who","can you","could you","tell me","explain"]
        for s in sentences:
            sl = s.lower()
            if any(sl.startswith(term + " ") for term in head_terms) and len(s.split()) >= 6:
                questions.append(s if s.endswith("?") else s + "?")

        seen = set(); clean: List[str] = []
        for q in questions:
            k = _qkey(q)
            if k and k not in seen:
                seen.add(k)
                clean.append(q.strip(" ,;:-–—"))
        return clean[:3]

    def _should_answer(self, question: str) -> Dict[str, Any]:
        if not self.enabled: return {"should_answer": False}
        now = time.time()
        while self.answers_timestamps and now - self.answers_timestamps[0] > 60:
            self.answers_timestamps.popleft()
        if len(self.answers_timestamps) >= ANSWERS_PER_MIN:
            return {"should_answer": False}

        ql = question.lower().strip()
        wh = ("what", "how", "why", "when", "where", "which", "who", "can", "could", "do", "is", "are")
        looks_like_q = ql.endswith("?") or any(ql.startswith(w + " ") for w in wh)

        try:
            resp = self.client.responses.create(
                model=GPT_MODEL, reasoning={"effort": GPT_EFFORT},
                input=[
                    {"role": "developer", "content": "Is this an interview question needing an answer? Reply ONLY 'YES' or 'NO'."},
                    {"role": "user", "content": f"Context:\n{build_llm_context_text()[-1000:]}\n\nQuestion: {question}"},
                ],
            )
            out = (getattr(resp, "output_text", "") or "").strip().lower()
            decided_yes = "yes" in out and "no" not in out
        except Exception as e:
            if DEBUG: print(f"[llm] Decision error: {e}")
            decided_yes = looks_like_q

        if decided_yes:
            ts = time.time(); self.last_llm_emit = ts; self.answers_timestamps.append(ts)
        return {"should_answer": decided_yes, "confidence": 0.75 if decided_yes else 0.3}

    async def generate_answer(self, candidate: QuestionCandidate) -> str:
        if not self.enabled: return "[LLM disabled]"
        async with self.sem:
            return await asyncio.get_running_loop().run_in_executor(None, self._gen, candidate)

    def _gen(self, candidate: QuestionCandidate) -> str:
        """Advanced answer generation with persona, context, and retry (restored from French version)."""
        try:
            q = (candidate.question or "").strip()
            if PERSONA == "candidate":
                system_prompt = (
                    "You are a tech professional answering questions in a job interview. "
                    "Speak in the first person singular ('I'). Provide 3-5 concise, complete sentences. "
                    "Start with a direct answer, then provide concrete points or a brief, relevant example from a tech domain "
                    "(e.g., web services, data pipelines, ML models). "
                    "Do not use bullet points, lists, or any meta-commentary/coaching phrases."
                )
            else: # Fallback "coach" persona
                system_prompt = "Provide a concise, helpful answer in 2-4 sentences."

            inputs = [{"role": "developer", "content": system_prompt}]
            if LLM_INCLUDE_FULL_TRANSCRIPT:
                full_tx = build_llm_context_text()
                if full_tx: inputs.append({"role": "user", "content": f"Full Interview Transcript (for context):\n{full_tx}"})
            
            if TECH_INTERVIEW_MODE:
                inputs.append({"role": "user", "content": "Domain context: general software engineering, data science, cloud infrastructure."})

            inputs.extend([
                {"role": "user", "content": f"Question: {q}"},
                {"role": "user", "content": "Answer now as a final spoken response. Avoid addressing the interviewer as 'you'."},
            ])

            resp = self.client.responses.create(model=GPT_MODEL, reasoning={"effort": GPT_EFFORT}, input=inputs, max_output_tokens=MAX_OUTTOK)
            raw_text = (getattr(resp, "output_text", "") or "").strip()
            cleaned = sanitize_candidate_voice(raw_text)

            if not cleaned.strip(): # Retry logic
                if DEBUG: print("[llm] Response was empty, retrying with simpler prompt...")
                retry_inputs = [
                    {"role": "developer", "content": "Answer the following question as a tech job candidate in the first person."},
                    {"role": "user", "content": f"Question: {q}"}
                ]
                resp2 = self.client.responses.create(model=GPT_MODEL, input=retry_inputs, max_output_tokens=MAX_OUTTOK)
                raw2 = (getattr(resp2, "output_text", "") or "").strip()
                cleaned = sanitize_candidate_voice(raw2)

            return cleaned if cleaned.strip() else "I would evaluate the model on a held-out test set to measure its generalization performance."
        except Exception as e:
            return f"[Answer error] {str(e)[:120]}"

# ========================
# Global Server State
# ========================
@dataclass
class ClientState:
    ws: websockets.WebSocketServerProtocol; queue: asyncio.Queue
    sender_task: Optional[asyncio.Task] = None; wants_broadcast: bool = True; role: str = "ui"

clients: Dict[websockets.WebSocketServerProtocol, ClientState] = {}
latest_text, sentence_buf = "", ""
detected = deque(maxlen=500); qa_log = deque(maxlen=200)
transcript_lines: List[str] = []
MAX_TRANSCRIPT_LINES = int(os.getenv("STT_MAX_TRANSCRIPT_LINES", "5000"))
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.?!])\s+')
pcm_buf = bytearray(); bytes_since_last = 0
WIN_BYTES = int(SAMPLE_RATE * 2 * WINDOW_SECONDS); HOP_BYTES = int(SAMPLE_RATE * 2 * HOP_SECONDS)
MAX_BUFFER_BYTES = WIN_BYTES * 2
audio_lock = asyncio.Lock(); server_shutdown = asyncio.Event(); transcriber_task: Optional[asyncio.Task] = None

# ========================
# Broadcast & WebSocket Handling
# ========================
async def _client_sender(client: ClientState):
    try:
        while True: await client.ws.send(await client.queue.get())
    except (websockets.ConnectionClosedOK, websockets.ConnectionClosedError): pass
    finally:
        try: await client.ws.close()
        except Exception: pass

async def broadcast(data: dict):
    if not clients: return
    msg = json.dumps(data, ensure_ascii=False)
    for c in list(clients.values()):
        if c.wants_broadcast:
            try:
                if c.queue.full(): _ = c.queue.get_nowait()
                c.queue.put_nowait(msg)
            except Exception: clients.pop(c.ws, None)

async def handler(ws: websockets.WebSocketServerProtocol):
    global transcriber_task, bytes_since_last, LICENSE_VALID
    print(f"[ws] Client connected from {ws.remote_address}")

    trial_remaining = None
    if not LICENSE_VALID:
        start_trial_if_needed()
        trial_remaining = trial_remaining_seconds()
        if trial_remaining <= 0:
            try:
                await ws.send(json.dumps({
                    "error": "trial_expired",
                    "message": "Trial expired. Enter a license key to continue."
                }))
            except Exception:
                pass
            await ws.close()
            return
    client = ClientState(ws=ws, queue=asyncio.Queue(maxsize=100))
    client.sender_task = asyncio.create_task(_client_sender(client))
    clients[ws] = client

    try:
        await ws.send(json.dumps({"snapshot": {"transcript": latest_text, "detected": list(detected), "lines": transcript_lines[-200:]}}))
        if not LICENSE_VALID and trial_remaining is not None:
            await ws.send(json.dumps({
                "trial": {"remaining_sec": trial_remaining, "minutes": TRIAL_MINUTES}
            }))
    except Exception:
        pass

    if transcriber_task is None or transcriber_task.done():
        transcriber_task = asyncio.create_task(read_and_transcribe_loop())

    try:
        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                async with audio_lock:
                    pcm_buf.extend(msg); bytes_since_last += len(msg)
                    if len(pcm_buf) > MAX_BUFFER_BYTES: del pcm_buf[:len(pcm_buf) - MAX_BUFFER_BYTES // 2]
            else:
                try:
                    data = json.loads(msg)
                    cmd = (data.get("cmd") or "").lower()
                    if cmd == "license":
                        key = (data.get("key") or "").strip()
                        if is_license_valid(key):
                            LICENSE_FILE.write_text(key)
                            LICENSE_VALID = True
                            await ws.send(json.dumps({"license": "valid"}))
                        else:
                            await ws.send(json.dumps({"license": "invalid"}))
                    elif cmd == "hello" and (data.get("client") or "").lower() in ("audio_streamer", "ingest"):
                        client.wants_broadcast = False
                        if DEBUG: print(f"[ws] Marked {ws.remote_address} as streamer; broadcasts OFF")
                    elif cmd == "reset":
                        async with audio_lock: pcm_buf.clear(); bytes_since_last = 0
                        reset_state(); print("[ws] Reset completed")
                    elif cmd == "ask":
                        q = (data.get("q") or "").strip()
                        qid = data.get("qid") or f"q-{uuid.uuid4().hex}"
                        if not q:
                            continue

                        now_ms = int(time.time() * 1000)
                        item = {"qid": qid, "q": q, "t": now_ms, "a": None}
                        detected.append(item)
                        await broadcast({"question_detected": item})

                        # Génère la réponse
                        if llm_analyzer and llm_analyzer.enabled:
                            cand = QuestionCandidate(
                                question=q,
                                context=build_llm_context_text(),
                                confidence=0.9,
                                urgency="relevant",
                                topic_area="interview",
                                timestamp=time.time(),
                                should_answer=True,
                            )
                            ans = await llm_analyzer.generate_answer(cand)
                        else:
                            ans = "[LLM disabled] Provide OPENAI_API_KEY or set STT_LLM_ENABLED=0."

                        item["a"] = ans
                        qa_log.append(item)
                        await broadcast({"qa": item})
                except Exception: pass
    except (websockets.ConnectionClosedOK, websockets.ConnectionClosedError): pass
    finally:
        clients.pop(ws, None)
        if client.sender_task: client.sender_task.cancel()
        print("[ws] Client disconnected")

# ========================
# Global Transcriber Loop
# ========================
def transcribe_with_openai(arr: np.ndarray) -> str:
    if openai_stt_client is None:
        return ""
    try:
        pcm16 = (arr * 32768.0).astype(np.int16).tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm16)
        buf.seek(0)
        resp = openai_stt_client.audio.transcriptions.create(
            model=OPENAI_STT_MODEL,
            file=("audio.wav", buf, "audio/wav")
        )
        return (resp.text or "").strip()
    except Exception as e:
        print(f"[stt] OpenAI transcription error: {e}")
        return ""

async def read_and_transcribe_loop():
    import math
    import difflib

    global bytes_since_last, latest_text, sentence_buf, transcript_lines
    loop, last_hashes = asyncio.get_running_loop(), deque(maxlen=16)
    print("[stt] Transcription loop started (global)")

    # --- Réglages anti-doublon (sur-env optionnels)
    FUZZY_SIM = float(os.getenv("STT_FUZZY_SIM_THRESH", "0.92"))   # similarité ~>= 0.92 => on droppe
    MIN_WORDS  = int(os.getenv("STT_MIN_WORDS", "3"))               # on ignore les très courtes phrases
    MIN_CHARS  = int(os.getenv("STT_MIN_CHARS", "18"))              # idem
    DROP_FILLERS = os.getenv("STT_DROP_FILLERS", "1") not in ("0","false","False")
    FILLER_RE = re.compile(r"^(yes(,)? exactly\.?|okay\.?|right\.?|mm+h+|uh+|um+|yeah\.?)$", re.I)

    # VAD tuning (valeurs plus stables)
    VAD_MIN_SIL_MS = int(os.getenv("STT_VAD_MIN_SIL_MS", "250"))
    VAD_PAD_MS     = int(os.getenv("STT_VAD_PAD_MS", "80"))

    PROMPT_WORDS = {w.strip(".,").lower() for w in INITIAL_PROMPT.split()} if INITIAL_PROMPT else set()

    # Mémoire des dernières lignes "commitées" pour fuzzy dedup
    recent_lines = deque(maxlen=40)

    def very_similar(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

    def looks_like_filler(s: str) -> bool:
        s2 = _norm(s).lower()
        if FILLER_RE.match(s2):
            return True
        # pequeno : "yes", "exactly", "okay" seuls
        if len(s2.split()) <= 2 and s2.rstrip(".!?") in {"yes", "exactly", "okay", "right", "yeah"}:
            return True
        return False

    while not server_shutdown.is_set():
        await asyncio.sleep(0.05)

        # On n’avance que si on a de la nouvelle matière
        async with audio_lock:
            if not (bytes_since_last >= HOP_BYTES and len(pcm_buf) >= WIN_BYTES):
                # debug optionnel
                if os.getenv("STT_VERBOSE_BUFFER", "0") in ("1", "true", "True"):
                    print(f"[audio] idle: bytes_since_last={bytes_since_last}, buf={len(pcm_buf)}")
                continue
            bytes_since_last = 0
            window_bytes = bytes(pcm_buf[-WIN_BYTES:])

        # Converti PCM16 -> float32 [-1,1]
        if not window_bytes:
            continue
        arr = np.frombuffer(window_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if arr.size == 0:
            continue

        # Gate énergie pour éviter le VAD sur quasi-silence
        rms = float(np.sqrt(np.mean(arr * arr)))
        if os.getenv("STT_VERBOSE_BUFFER", "0") in ("1", "true", "True"):
            print(f"[stt] win={len(arr)} samples, rms={rms:.6f}")
        if rms < ENERGY_GATE:
            continue

        def _transcribe():
            try:
                if STT_BACKEND == "openai":
                    text = transcribe_with_openai(arr)
                    return [SimpleSegment(text)] if text else []
                segments, _ = whisper.transcribe(
                    arr,
                    language=FORCE_LANG,
                    vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": VAD_MIN_SIL_MS,
                        "speech_pad_ms": VAD_PAD_MS,
                    },
                    condition_on_previous_text=False,
                    no_speech_threshold=0.35,
                    log_prob_threshold=-1.0,
                    beam_size=3,
                )
                return list(segments)
            except Exception as e:
                print(f"[stt] Transcription error: {e}")
                return []

        segments = await loop.run_in_executor(None, _transcribe)
        if not segments:
            continue

        for seg in segments:
            text = _norm(getattr(seg, "text", "") or "")
            if not text:
                continue

            # Filtre "echo du prompt" ou répétitions évidentes
            toks = [w.strip(".,").lower() for w in text.split()]
            if toks and PROMPT_WORDS and sum(1 for w in toks if w in PROMPT_WORDS) / max(1, len(toks)) > 0.6:
                if DEBUG:
                    print("[stt] Dropped segment: initial_prompt echo")
                continue
            if re.search(r"\b(\w+(?:,\s*\w+){1,})\b(?:\s+\1\b){2,}", text, flags=re.I):
                if DEBUG:
                    print("[stt] Dropped segment: repetitive phrase")
                continue

            # Déduplication exacte très courte portée
            h = hashlib.md5(text.encode("utf-8")).hexdigest()
            if h in last_hashes:
                continue
            last_hashes.append(h)

            # Dédoublon flou (entre phrases déjà sorties)
            if any(very_similar(text, prev) >= FUZZY_SIM for prev in recent_lines):
                if DEBUG:
                    print("[stt] Dropped segment: fuzzy-duplicate")
                continue

            # Découpage phrase → on ne commit que des phrases "assez longues"
            sentence_candidate = text
            parts = SENTENCE_SPLIT_RE.split((sentence_buf + " " + sentence_candidate).strip()) if sentence_candidate else []
            if parts:
                # Si la fin se termine par ponctuation forte, tout est complet
                has_terminal = (sentence_candidate and sentence_candidate[-1] in ".?!")
                complete = parts[:-1] if (len(parts) > 1 and not has_terminal) else (parts if has_terminal else [])
                sentence_buf = "" if complete == parts else parts[-1]
                latest_text = sentence_buf
            else:
                complete = []
                sentence_buf = (sentence_buf + " " + sentence_candidate).strip()
                latest_text = sentence_buf

            # Commit des phrases complètes avec filtres anti-filler / longueur / fuzzy
            for s in complete:
                s = _norm(s)
                if not s:
                    continue
                if DROP_FILLERS and looks_like_filler(s):
                    if DEBUG:
                        print("[stt] Dropped line: filler")
                    continue
                if len(s) < MIN_CHARS or len(s.split()) < MIN_WORDS:
                    # garde les très courtes seulement si elles ne ressemblent pas à un filler
                    if DROP_FILLERS:
                        continue
                if any(very_similar(s, prev) >= FUZZY_SIM for prev in recent_lines):
                    if DEBUG:
                        print("[stt] Dropped line: fuzzy-duplicate")
                    continue

                transcript_lines.append(s)
                recent_lines.append(s)
                if len(transcript_lines) > MAX_TRANSCRIPT_LINES:
                    del transcript_lines[:MAX_TRANSCRIPT_LINES // 10]
                if DEBUG:
                    print(f"[stt] Line: {s}")
                await broadcast({"line": {"n": len(transcript_lines), "text": s}})

            # Optionnel : broadcast d’un "partial" (throttlé pour éviter le spam)
            # Ici on n’envoie le partial que s’il grandit réellement
            if sentence_buf and (len(sentence_buf) % 30 == 0):  # petit throttle "pauvre mais efficace"
                await broadcast({"partial": sentence_buf})

            # Détection de questions / LLM
            if llm_analyzer and llm_analyzer.enabled and text:
                try:
                    cands = await llm_analyzer.analyze_segment(ConversationSegment(text=text, timestamp=time.time()))
                    for cand in cands:
                        item = {"q": cand.question, "t": int(cand.timestamp * 1000), "a": None}
                        detected.append(item)
                        await broadcast({"question_detected": item})
                        ans = await llm_analyzer.generate_answer(cand)
                        item["a"] = ans
                        qa_log.append(item)
                        await broadcast({"qa": item})
                        if DEBUG:
                            print(f"[llm] Response: {ans[:60]}...")
                except Exception as e:
                    print(f"[llm] Analysis error: {e}")



# ========================
# Utilities & Main
# ========================
def reset_state():
    global latest_text, sentence_buf, transcript_lines
    latest_text, sentence_buf = "", ""; transcript_lines.clear(); detected.clear(); qa_log.clear()
    if llm_analyzer:
        llm_analyzer.seen_questions.clear(); llm_analyzer.answers_timestamps.clear()

async def main():
    if not check_license():
        return
    global whisper, llm_analyzer, openai_stt_client
    if STT_BACKEND == "openai":
        try:
            from openai import OpenAI
            if OPENAI_API_KEY.startswith("sk-"):
                openai_stt_client = OpenAI(api_key=OPENAI_API_KEY)
                print(f"[stt] OpenAI transcription ready (model={OPENAI_STT_MODEL})")
            else:
                print("[stt] OpenAI transcription disabled (missing API key)")
        except Exception as e:
            print(f"[stt] OpenAI init error: {e}")

    print("🚀 OPTIMIZED STT SERVER V4 (Best of Both Worlds)")
    print("=" * 50)
    print("[startup] Loading Whisper model..."); whisper = load_whisper_model()
    print("[startup] Initializing LLM Analyzer..."); llm_analyzer = ImprovedLLMAnalyzer(OPENAI_API_KEY)
    print(f"\n📋 CONFIG: LLM {'Enabled' if llm_analyzer.enabled else 'Disabled'}, Debug {'ON' if DEBUG else 'OFF'}")

    try:
        async with websockets.serve(handler, HOST, PORT, max_size=2**20, ping_interval=10, ping_timeout=30):
            print(f"\n🎤 Server ready on ws://{HOST}:{PORT}/"); await asyncio.Future()
    except Exception as e:
        print(f"[error] Server failed to start: {e}")

if __name__ == "__main__":
    try:
        whisper: Any; llm_analyzer: Optional[ImprovedLLMAnalyzer]
        asyncio.run(main())
    except KeyboardInterrupt:
        server_shutdown.set(); print("\n[shutdown] Server stopped.")
    except Exception as e:
        print(f"\n[fatal_error] {e}"); import traceback; traceback.print_exc()
