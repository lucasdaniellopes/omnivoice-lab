import json
import math
import os
import re
import queue
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from collections import deque
from pathlib import Path

import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

print("Iniciando OmniVoice Lab... carregando bibliotecas (alguns segundos)", flush=True)

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import omnivoice.models.omnivoice as ovmod
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import VoiceClonePrompt
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

DATA = ROOT / "data"
VOICES_DIR = DATA / "voices"
OUTPUTS_DIR = DATA / "outputs"
HISTORY_FILE = DATA / "history.jsonl"
for d in (VOICES_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL_ID = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32
SR = 24000


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class GpuMonitor:
    def __init__(self, interval=0.5, history=240):
        self.interval = interval
        self.history = deque(maxlen=history)
        self.latest = {}
        self.name = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nv = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(self.name, bytes):
                self.name = self.name.decode()
        except Exception as e:
            print(f"[gpu] NVML indisponível: {e}")
            self.nv = None
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            if self.nv:
                try:
                    self.latest = self._sample()
                    self.history.append(self.latest)
                except Exception as e:
                    print(f"[gpu] erro NVML: {e}")
            time.sleep(self.interval)

    def _sample(self):
        nv, h = self.nv, self.handle
        util = nv.nvmlDeviceGetUtilizationRates(h)
        mem = nv.nvmlDeviceGetMemoryInfo(h)
        s = {
            "t": time.time(),
            "util": util.gpu,
            "mem_util": util.memory,
            "mem_used": mem.used / 2**20,
            "mem_total": mem.total / 2**20,
        }
        for key, fn in (
            ("temp", lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)),
            ("power", lambda: nv.nvmlDeviceGetPowerUsage(h) / 1000),
            ("power_limit", lambda: nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1000),
            ("clock_sm", lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)),
            ("clock_mem", lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM)),
            ("fan", lambda: nv.nvmlDeviceGetFanSpeed(h)),
        ):
            try:
                s[key] = fn()
            except Exception:
                s[key] = None
        if torch.cuda.is_available():
            s["torch_alloc"] = torch.cuda.memory_allocated() / 2**20
            s["torch_reserved"] = torch.cuda.memory_reserved() / 2**20
        return s


class JobCancelled(Exception):
    pass


class Job:
    def __init__(self, params: dict):
        self.id = uuid.uuid4().hex[:10]
        self.params = params
        self.status = "queued"
        self.phase = "na fila"
        self.created = time.time()
        self.started = None
        self.finished = None
        self.error = None
        self.cancel = False
        self.step = 0
        self.num_step = params.get("num_step", 32)
        self.chunk = 0
        self.n_chunks = 1
        self.audio_tokens_done = 0.0
        self.ctx_tokens_done = 0
        self.gen_time = 0.0
        self.gen_t0 = None
        self.it_s = 0.0
        self.tok_s = 0.0
        self.ctx_tok_s = 0.0
        self.target_frames = 0
        self.last_step_t = None
        self.output = None
        self.audio_seconds = None
        self.rtf = None
        self.total_time = None
        self.peak_vram = None
        self.ref_text_used = None

    @property
    def progress(self):
        if self.status == "done":
            return 100.0
        total = max(1, self.num_step * self.n_chunks)
        done = self.chunk * self.num_step + self.step
        return min(99.0, 100.0 * done / total) if self.status == "running" else 0.0

    def to_dict(self):
        now = time.time()
        return {
            "id": self.id,
            "status": self.status,
            "phase": self.phase,
            "progress": round(self.progress, 1),
            "step": self.step,
            "num_step": self.num_step,
            "chunk": self.chunk,
            "n_chunks": self.n_chunks,
            "it_s": round(self.it_s, 2),
            "tok_s": round(self.tok_s, 1),
            "ctx_tok_s": round(self.ctx_tok_s, 1),
            "audio_tokens": int(self.audio_tokens_done),
            "target_frames": self.target_frames,
            "created": self.created,
            "elapsed": round((self.finished or now) - self.started, 2) if self.started else 0,
            "output": self.output,
            "audio_seconds": self.audio_seconds,
            "rtf": self.rtf,
            "peak_vram": self.peak_vram,
            "error": self.error,
            "ref_text_used": self.ref_text_used,
            "params": self.params,
        }


ASR_ENGINE = os.environ.get("OMNIVOICE_ASR", "parakeet").lower()

PARAKEET_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
PARAKEET_DIR = ROOT / "hf_cache" / "parakeet-tdt-0.6b-v3-onnx"
PARAKEET_FILES = ["config.json", "vocab.txt", "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx"]


class ParakeetPipe:
    def __init__(self, path: Path):
        import onnx_asr

        self.model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v3", str(path), quantization="int8",
            providers=["CPUExecutionProvider"],
        )

    def __call__(self, audio):
        import torchaudio

        if isinstance(audio, str):
            wav, sr = sf.read(audio, dtype="float32", always_2d=True)
            wav, sr = wav.mean(axis=1), sr
        else:
            wav, sr = np.asarray(audio["array"], dtype=np.float32), audio["sampling_rate"]
        if sr != 16000:
            wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, 16000).numpy()
        return {"text": self.model.recognize(wav, sample_rate=16000)}


ASR_REPO = "openai/whisper-large-v3-turbo"
ASR_DIR = ROOT / "hf_cache" / "whisper-large-v3-turbo"
ASR_FILES = [
    "config.json", "generation_config.json", "preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "added_tokens.json", "special_tokens_map.json", "normalizer.json",
    "model.safetensors",
]


def _with_retry(fn, job=None, label="", tries=6):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            transient = "429" in msg or "Too Many" in msg or "timed out" in msg or "Connection" in msg
            if not transient or attempt == tries - 1:
                raise
            wait = min(60, 5 * 2**attempt)
            log(f"HuggingFace recusou ({msg[:80]}), tentando de novo em {wait}s")
            for left in range(wait, 0, -1):
                if job:
                    if job.cancel:
                        raise JobCancelled()
                    job.phase = f"{label} — HF limitou (429), nova tentativa em {left}s"
                time.sleep(1)


def _local_or_download(repo_id: str) -> str:
    if os.path.isdir(repo_id):
        return repo_id
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id, local_files_only=True)
    except Exception:
        return _with_retry(lambda: snapshot_download(repo_id))


class Engine:
    def __init__(self):
        self.model = None
        self.status = "carregando"
        self.load_error = None
        self.load_seconds = None
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.q: "queue.Queue[Job]" = queue.Queue()
        self.current: Job | None = None
        self._load_history()
        threading.Thread(target=self._worker, daemon=True).start()

    def _load_history(self):
        if not HISTORY_FILE.exists():
            return
        for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("output") and not (OUTPUTS_DIR / d["output"]).exists():
                continue
            j = Job(d.get("params", {}))
            for k in ("id", "status", "created", "output", "audio_seconds", "rtf",
                      "peak_vram", "error", "ref_text_used", "it_s", "tok_s",
                      "ctx_tok_s", "n_chunks", "num_step", "target_frames"):
                if k in d:
                    setattr(j, k, d[k])
            j.phase = "concluído" if j.status == "done" else j.status
            j.step = j.num_step
            j.chunk = max(0, j.n_chunks - 1)
            j.started = d.get("created")
            j.finished = (j.started or 0) + d.get("elapsed", 0)
            self.jobs[j.id] = j
            self.order.append(j.id)

    def _persist(self, job: Job):
        with HISTORY_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(job.to_dict(), ensure_ascii=False) + "\n")

    def _load_model(self):
        t0 = time.time()
        self.status = "carregando"

        log(f"Carregando modelo {MODEL_ID} em {DEVICE} ({DTYPE})... (1ª vez baixa ~3 GB)")
        model = OmniVoice.from_pretrained(_local_or_download(MODEL_ID), device_map=DEVICE, dtype=DTYPE)
        self._instrument(model, ovmod)
        self.model = model
        self.load_seconds = round(time.time() - t0, 1)
        self.status = "pronto"
        log(f"Modelo pronto em {self.load_seconds}s — pode usar a página.")

    def _instrument(self, model, ovmod):
        engine = self

        orig_chunk = ovmod.chunk_text_punctuation

        def chunk_wrapper(*a, **kw):
            chunks = orig_chunk(*a, **kw)
            if engine.current and not getattr(engine.current, "_split", False):
                engine.current.n_chunks = max(1, len(chunks))
            return chunks

        ovmod.chunk_text_punctuation = chunk_wrapper

        orig_iter = model._generate_iterative

        def gen_iter_wrapper(task, gen_config):
            job = engine.current
            if job:
                job.phase = "gerando"
                job.step = 0
                job.num_step = gen_config.num_step
                job._call_tokens = sum(task.target_lens) * model.config.num_audio_codebook
                job.target_frames += sum(task.target_lens)
                job.last_step_t = time.perf_counter()
            out = orig_iter(task, gen_config)
            if job:
                job.chunk += 1
                job.step = 0
            return out

        model._generate_iterative = gen_iter_wrapper

        def fwd_hook(module, args, kwargs, output):
            job = engine.current
            if not job:
                return
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            now = time.perf_counter()
            dt = now - (job.last_step_t or now)
            job.last_step_t = now
            job.gen_time += dt
            ids = kwargs.get("input_ids")
            if ids is None and args:
                ids = args[0]
            if ids is not None:
                job.ctx_tokens_done += int(ids.shape[0] * ids.shape[-1])
            job.audio_tokens_done += getattr(job, "_call_tokens", 0) / max(1, job.num_step)
            job.step += 1
            if job.gen_time > 0:
                steps_total = job.chunk * job.num_step + job.step
                job.it_s = steps_total / job.gen_time
                job.tok_s = job.audio_tokens_done / job.gen_time
                job.ctx_tok_s = job.ctx_tokens_done / job.gen_time
            if job.cancel:
                raise JobCancelled()

        model.register_forward_hook(fwd_hook, with_kwargs=True)

        orig_decode = model._decode_and_post_process

        def decode_wrapper(*a, **kw):
            if engine.current:
                engine.current.phase = "decodificando áudio"
            return orig_decode(*a, **kw)

        model._decode_and_post_process = decode_wrapper

    def submit(self, params: dict) -> Job:
        job = Job(params)
        self.jobs[job.id] = job
        self.order.append(job.id)
        self.q.put(job)
        return job

    def _worker(self):
        try:
            self._load_model()
        except Exception as e:
            traceback.print_exc()
            self.status = "erro"
            self.load_error = f"{type(e).__name__}: {e}"
            log(f"ERRO ao carregar o modelo: {self.load_error}")
            log("Feche esta janela e rode o run.bat de novo. Se repetir, mande o erro acima.")
            return
        while True:
            job = self.q.get()
            if job.cancel:
                job.status, job.phase = "cancelled", "cancelado"
                continue
            self.current = job
            p = job.params
            who = {"clone": f"voz '{p.get('voice_name')}'", "design": f"design '{p.get('instruct')}'"}.get(p.get("mode"), "auto")
            log(f"Job {job.id} começou: {who}, {len(p.get('text', ''))} caracteres, {p.get('num_step')} passos")
            try:
                self._run(job)
                log(f"Job {job.id} pronto: {job.audio_seconds}s de áudio em {job.total_time:.1f}s "
                    f"(RTF {job.rtf}, {job.it_s:.1f} it/s, pico VRAM {job.peak_vram} MB)")
            except JobCancelled:
                job.status, job.phase = "cancelled", "cancelado"
                log(f"Job {job.id} cancelado")
            except Exception as e:
                traceback.print_exc()
                job.status, job.phase = "error", "erro"
                job.error = f"{type(e).__name__}: {e}"
                log(f"Job {job.id} deu ERRO: {job.error}")
            finally:
                job.finished = time.time()
                self.current = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._persist(job)

    def _ensure_asr(self, job: Job):
        if self.model._asr_pipe is not None:
            return
        parakeet = ASR_ENGINE == "parakeet"
        label = "Parakeet v3" if parakeet else "Whisper"
        repo, local, files = (
            (PARAKEET_REPO, PARAKEET_DIR, PARAKEET_FILES) if parakeet else (ASR_REPO, ASR_DIR, ASR_FILES)
        )
        missing = [f for f in files if not (local / f).exists()]
        if missing:
            from huggingface_hub import hf_hub_download

            for i, fname in enumerate(missing, 1):
                job.phase = f"baixando {label} ({i}/{len(missing)}: {fname})"
                _with_retry(
                    lambda: hf_hub_download(repo, fname, local_dir=str(local)),
                    job,
                    job.phase,
                )
        job.phase = f"carregando {label}"
        if parakeet:
            self.model._asr_pipe = ParakeetPipe(local)
        else:
            self.model.load_asr_model(model_name=str(local))

    def _voice_prompt(self, job: Job, voice_id: str):
        vdir = VOICES_DIR / voice_id
        meta = json.loads((vdir / "meta.json").read_text(encoding="utf-8"))
        cache = vdir / "prompt.pt"
        if cache.exists():
            prompt = VoiceClonePrompt.load(str(cache))
        else:
            ref_text = (meta.get("ref_text") or "").strip() or None
            if ref_text is None:
                self._ensure_asr(job)
                job.phase = f"transcrevendo referência ({'Parakeet v3' if ASR_ENGINE == 'parakeet' else 'Whisper'})"
            else:
                job.phase = "codificando áudio de referência"
            prompt = self.model.create_voice_clone_prompt(
                ref_audio=str(vdir / "ref.wav"), ref_text=ref_text
            )
            prompt.save(str(cache))
            if not meta.get("ref_text"):
                meta["ref_text"] = prompt.ref_text
                meta["ref_text_auto"] = True
                (vdir / "meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        job.ref_text_used = prompt.ref_text
        return prompt

    def _run(self, job: Job):
        p = job.params
        job.status = "running"
        job.started = time.time()
        job.phase = "preparando"
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        kwargs = dict(
            text=p["text"],
            language=p.get("language") or None,
            num_step=int(p.get("num_step", 32)),
            guidance_scale=float(p.get("guidance_scale", 2.0)),
            t_shift=float(p.get("t_shift", 0.1)),
            class_temperature=float(p.get("class_temperature", 0.0)),
            position_temperature=float(p.get("position_temperature", 5.0)),
            denoise=bool(p.get("denoise", True)),
            postprocess_output=bool(p.get("postprocess", True)),
            normalize_text=bool(p.get("normalize_text", False)),
        )
        if p.get("speed"):
            kwargs["speed"] = float(p["speed"])
        if p.get("duration"):
            kwargs["duration"] = float(p["duration"])
        job.num_step = kwargs["num_step"]

        mode = p.get("mode", "clone")
        if mode == "clone":
            kwargs["voice_clone_prompt"] = self._voice_prompt(job, p["voice_id"])
        elif mode == "design":
            kwargs["instruct"] = p.get("instruct") or None

        seed = p.get("seed")
        if seed not in (None, ""):
            torch.manual_seed(int(seed))

        job.phase = "gerando"
        t0 = time.perf_counter()
        if p.get("pause_mode") == "split":
            audio = self._generate_split(job, kwargs, p)
        else:
            audio = np.asarray(self.model.generate(**kwargs)[0], dtype=np.float32)
        total = time.perf_counter() - t0

        name = f"{job.id}.wav"
        sf.write(str(OUTPUTS_DIR / name), audio, self.model.sampling_rate or SR)
        job.output = name
        job.audio_seconds = round(len(audio) / (self.model.sampling_rate or SR), 2)
        job.total_time = total
        job.rtf = round(total / max(job.audio_seconds, 1e-6), 4)
        if torch.cuda.is_available():
            job.peak_vram = round(torch.cuda.max_memory_allocated() / 2**20)
        job.status, job.phase = "done", "concluído"

    def _generate_split(self, job: Job, kwargs: dict, p: dict) -> np.ndarray:
        sr = self.model.sampling_rate or SR
        sentence_ms = max(0, int(p.get("pause_sentence_ms", 350)))
        comma_ms = max(0, int(p.get("pause_comma_ms", 120)))
        segs = split_for_pauses(kwargs.pop("text"), split_commas=comma_ms > 0)
        kwargs.pop("duration", None)
        if not segs:
            raise ValueError("texto vazio após dividir na pontuação")

        kwargs.update(pad_duration=0.0, fade_duration=0.02)
        texts = [s for s, _ in segs]
        job._split = True
        job.n_chunks = math.ceil(len(texts) / SPLIT_BATCH)
        outs: list[np.ndarray] = []

        if "voice_clone_prompt" not in kwargs:
            job.n_chunks += 1
            first = np.asarray(self.model.generate(text=texts[0], **kwargs)[0], dtype=np.float32)
            outs.append(first)
            job.phase = "fixando a voz do 1º trecho"
            kwargs["voice_clone_prompt"] = self.model.create_voice_clone_prompt(
                ref_audio=(torch.from_numpy(first), sr), ref_text=texts[0]
            )
            kwargs.pop("instruct", None)
            texts = texts[1:]
            job.phase = "gerando"

        prompt = kwargs.pop("voice_clone_prompt")
        speed = float(kwargs.pop("speed", None) or 1.0)
        frame_rate = self.model.audio_tokenizer.config.frame_rate
        total_tokens = self.model._estimate_target_tokens(
            " ".join(texts), prompt.ref_text, prompt.ref_audio_tokens.size(-1), speed=speed
        )
        weights = [max(self.model.duration_estimator.calculate_total_weight(t), 1e-3) for t in texts]
        durations = [total_tokens * w / sum(weights) / frame_rate for w in weights]

        for i in range(0, len(texts), SPLIT_BATCH):
            batch = texts[i : i + SPLIT_BATCH]
            res = self.model.generate(
                text=batch, voice_clone_prompt=[prompt] * len(batch),
                duration=durations[i : i + SPLIT_BATCH], **kwargs,
            )
            outs.extend(np.asarray(a, dtype=np.float32) for a in res)
            job.phase = "gerando"

        gaps = {
            "comma": comma_ms,
            "mid": (comma_ms + sentence_ms) // 2,
            "sentence": sentence_ms,
            "paragraph": int(sentence_ms * 1.75),
        }
        pieces = [np.zeros(int(0.1 * sr), np.float32)]
        for (_, kind), a in zip(segs, outs):
            pieces.append(_trim_edges(a, sr))
            pieces.append(np.zeros(int(sr * gaps.get(kind, 0) / 1000), np.float32))
        pieces[-1] = np.zeros(int(0.1 * sr), np.float32)
        return np.concatenate(pieces)


SPLIT_BATCH = 4


def _trim_edges(a: np.ndarray, sr: int, keep_ms: int = 15, below_peak_db: float = 40) -> np.ndarray:
    hop = sr // 100
    n = len(a) // hop
    if n < 3:
        return a
    rms = np.sqrt((a[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9
    db = 20 * np.log10(rms)
    loud = np.nonzero(db > db.max() - below_peak_db)[0]
    if not len(loud):
        return a
    keep = int(sr * keep_ms / 1000)
    start = max(0, loud[0] * hop - keep)
    end = min(len(a), (loud[-1] + 1) * hop + keep)
    out = a[start:end].copy()
    f = min(len(out) // 2, sr // 100)
    ramp = np.linspace(0, 1, f, dtype=np.float32)
    out[:f] *= ramp
    out[-f:] *= ramp[::-1]
    return out
_SPLIT_RE = re.compile(r"(\n\s*\n+|\n|(?:[!?…]|\.(?!\d))+[\"'”»)]*|[;:]|,(?!\d)|(?<!\d),)")


def split_for_pauses(text: str, split_commas: bool = True) -> list[tuple[str, str]]:
    parts = _SPLIT_RE.split(text)
    segs: list[tuple[str, str]] = []
    buf = ""
    for i in range(0, len(parts), 2):
        chunk = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        if sep == "," and not split_commas:
            buf += chunk + sep
            continue
        if sep.strip() == "" and sep:
            kind = "paragraph" if sep.count("\n") > 1 else "sentence"
            seg = (buf + chunk).strip()
        else:
            kind = "comma" if sep == "," else "mid" if sep in (";", ":") else "sentence" if sep else "end"
            seg = (buf + chunk + sep).strip()
        buf = ""
        if segs and len(re.sub(r"\W", "", seg)) < 2:
            prev, _ = segs[-1]
            segs[-1] = ((prev + " " + seg).strip(), kind)
            continue
        if seg:
            segs.append((seg, kind))
    return segs


gpu = GpuMonitor()
engine = Engine()
app = FastAPI(title="OmniVoice Lab")


def _probe_duration(path: Path) -> float:
    try:
        return round(sf.info(str(path)).duration, 2)
    except Exception:
        return 0.0


def _voice_list():
    out = []
    for vdir in sorted(VOICES_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        mf = vdir / "meta.json"
        if mf.exists():
            m = json.loads(mf.read_text(encoding="utf-8"))
            m["cached"] = (vdir / "prompt.pt").exists()
            out.append(m)
    return out


@app.get("/api/voices")
def list_voices():
    return _voice_list()


@app.post("/api/voices")
async def add_voices(
    files: list[UploadFile] = File(...),
    names: str = Form("[]"),
    ref_texts: str = Form("[]"),
):
    names_l = json.loads(names)
    texts_l = json.loads(ref_texts)
    created = []
    for i, f in enumerate(files):
        vid = uuid.uuid4().hex[:10]
        vdir = VOICES_DIR / vid
        vdir.mkdir(parents=True)
        src = vdir / f"original{Path(f.filename or 'a.wav').suffix or '.bin'}"
        with src.open("wb") as fh:
            shutil.copyfileobj(f.file, fh)
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1",
             "-ar", str(SR), str(vdir / "ref.wav")],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            shutil.rmtree(vdir, ignore_errors=True)
            raise HTTPException(400, f"ffmpeg falhou em {f.filename}: {r.stderr[-400:]}")
        name = (names_l[i] if i < len(names_l) else "") or Path(f.filename or vid).stem
        meta = {
            "id": vid,
            "name": name,
            "filename": f.filename,
            "ref_text": (texts_l[i] if i < len(texts_l) else "") or "",
            "duration": _probe_duration(vdir / "ref.wav"),
            "created": time.time(),
        }
        (vdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        created.append(meta)
    return created


@app.patch("/api/voices/{vid}")
async def edit_voice(vid: str, payload: dict):
    vdir = VOICES_DIR / vid
    mf = vdir / "meta.json"
    if not mf.exists():
        raise HTTPException(404)
    m = json.loads(mf.read_text(encoding="utf-8"))
    if "name" in payload:
        m["name"] = payload["name"]
    if "ref_text" in payload and payload["ref_text"] != m.get("ref_text"):
        m["ref_text"] = payload["ref_text"]
        m["ref_text_auto"] = False
        (vdir / "prompt.pt").unlink(missing_ok=True)
    mf.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    return m


@app.delete("/api/voices/{vid}")
def delete_voice(vid: str):
    vdir = VOICES_DIR / vid
    if not (vdir / "meta.json").exists():
        raise HTTPException(404)
    shutil.rmtree(vdir)
    return {"ok": True}


@app.get("/api/voices/{vid}/audio")
def voice_audio(vid: str):
    f = VOICES_DIR / vid / "ref.wav"
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="audio/wav")


@app.post("/api/generate")
async def generate(payload: dict):
    if not (payload.get("text") or "").strip():
        raise HTTPException(400, "texto vazio")
    mode = payload.get("mode", "clone")
    if mode == "clone":
        vids = payload.get("voice_ids") or []
        if not vids:
            raise HTTPException(400, "selecione ao menos uma voz")
        jobs = []
        for vid in vids:
            p = dict(payload, voice_id=vid)
            p.pop("voice_ids", None)
            meta = json.loads((VOICES_DIR / vid / "meta.json").read_text(encoding="utf-8"))
            p["voice_name"] = meta["name"]
            jobs.append(engine.submit(p).to_dict())
        return jobs
    return [engine.submit(payload).to_dict()]


@app.post("/api/jobs/{jid}/cancel")
def cancel(jid: str):
    job = engine.jobs.get(jid)
    if not job:
        raise HTTPException(404)
    job.cancel = True
    if job.status == "queued":
        job.status, job.phase = "cancelled", "cancelado"
    return {"ok": True}


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    job = engine.jobs.pop(jid, None)
    if not job:
        raise HTTPException(404)
    engine.order.remove(jid)
    if job.output:
        (OUTPUTS_DIR / job.output).unlink(missing_ok=True)
        for conv in EXPORT_DIR.glob(f"{Path(job.output).stem}.*"):
            conv.unlink(missing_ok=True)
    return {"ok": True}


EXPORT_FORMATS = {
    "wav": ("wav", "audio/wav", None),
    "mp3": ("mp3", "audio/mpeg", ["-c:a", "libmp3lame", "-q:a", "2"]),
    "opus": ("ogg", "audio/ogg", ["-c:a", "libopus", "-b:a", "64k"]),
    "vorbis": ("ogg", "audio/ogg", ["-c:a", "libvorbis", "-q:a", "5"]),
    "flac": ("flac", "audio/flac", ["-c:a", "flac"]),
}
EXPORT_DIR = OUTPUTS_DIR / "export"


@app.get("/api/outputs/{name}")
def output(name: str, fmt: str = "wav", download: bool = False):
    f = OUTPUTS_DIR / Path(name).name
    if not f.exists():
        raise HTTPException(404)
    if fmt not in EXPORT_FORMATS:
        raise HTTPException(400, f"formato inválido: {fmt}")
    ext, mime, args = EXPORT_FORMATS[fmt]
    if args:
        EXPORT_DIR.mkdir(exist_ok=True)
        out = EXPORT_DIR / f"{f.stem}.{fmt}.{ext}"
        if not out.exists():
            r = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(f), *args, str(out)],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                out.unlink(missing_ok=True)
                raise HTTPException(500, f"ffmpeg: {r.stderr[-300:]}")
        f = out
    job = engine.jobs.get(Path(name).stem)
    label = (job.params.get("voice_name") or job.params.get("mode")) if job else None
    fname = f"{label}_{Path(name).stem}.{ext}" if label else f"{Path(name).stem}.{ext}"
    return FileResponse(f, media_type=mime, filename=fname if download else None)


@app.get("/api/languages")
def languages():
    try:
        return sorted(lang_display_name(n) for n in LANG_NAMES)
    except Exception:
        return []


def _snapshot():
    return {
        "model": {
            "id": MODEL_ID,
            "status": engine.status,
            "error": engine.load_error,
            "load_seconds": engine.load_seconds,
            "device": DEVICE,
            "dtype": str(DTYPE).replace("torch.", ""),
            "torch": torch.__version__,
            "gpu_name": gpu.name,
            "queue": engine.q.qsize(),
            "asr": ASR_ENGINE,
        },
        "gpu": gpu.latest,
        "jobs": [engine.jobs[j].to_dict() for j in reversed(engine.order[-60:])],
    }


@app.get("/api/state")
def state():
    s = _snapshot()
    s["gpu_history"] = list(gpu.history)
    return s


@app.get("/api/stream")
async def stream():
    import asyncio

    async def gen():
        while True:
            yield f"data: {json.dumps(_snapshot(), ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")

def _announce(url: str):
    import socket
    import webbrowser

    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", int(url.rsplit(":", 1)[1])), timeout=0.5).close()
            break
        except OSError:
            time.sleep(0.2)
    log(f"Servidor no ar: {url}  (o modelo continua carregando em segundo plano)")
    if os.environ.get("OMNIVOICE_NO_BROWSER") != "1":
        webbrowser.open(url)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    url = f"http://127.0.0.1:{port}"
    import socket

    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
        log(f"Já existe um servidor rodando em {url} — abrindo o navegador nele.")
        import webbrowser

        webbrowser.open(url)
        sys.exit(0)
    except OSError:
        pass
    threading.Thread(target=_announce, args=(url,), daemon=True).start()
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    log("Servidor desligado.")
