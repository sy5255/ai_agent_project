import json
import logging
import os
import sys
import uuid
import base64
import subprocess
import math
import time
import re
from pathlib import Path
from urllib.parse import unquote, quote
from io import BytesIO
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, Request, Body, Query, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates

try:
    from PIL import Image
    PIL_OK = True
except Exception:
    Image = None
    PIL_OK = False

# --- [NEW] Set CUDA device visibility globally --- 
# PyTorch/Transformers가 로드되기 전에 환경 변수를 설정하여 3개의 GPU(0, 1, 2)를 인식시킵니다.
# _lazy_load_qwen의 device_map="auto"가 이 설정을 사용합니다. 
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2") 
_log_msg_gpu = f"CUDA_VISIBLE_DEVICES = '{os.environ.get('CUDA_VISIBLE_DEVICES')}'" 
# # --- End [NEW] ---

# -----------------------------
# 기본 설정 / 경로
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("app")

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"

# 원본 merge 이미지 폴더
IMAGE_DIR = Path(os.getenv("IMAGE_DIR", r"D:\aip-expert\dataset\dataset_updated\mask_merge")).resolve()
# 라벨(.py) 저장 폴더
LABEL_DIR = Path(os.getenv("LABEL_DIR", r"D:\aip-expert\dataset\dataset_updated\measure_label")).resolve()
# 실행/overlay 보관 폴더
RUN_DIR = Path(os.getenv("RUN_DIR", str(BASE_DIR / "runs"))).resolve()
# 메타 폴더
META_ROOT = Path(os.getenv("META_ROOT", r"D:\aip-expert\dataset\dataset_updated\meta")).resolve()
# Few-shot 저장 폴더
FEWSHOT_DIR = Path(os.getenv("FEWSHOT_DIR", str(BASE_DIR / "fewshot_repo"))).resolve()
# Scribble 폴더 (검은 배경에 빨강/초록 선만 있는 GT)
SCRIBBLE_DIR = Path(os.getenv("SCRIBBLE_DIR", r"D:\aip-expert\dataset\dataset_updated\scribble")).resolve()

# Gemma3 API 환경변수 (없으면 None)
GEMMA_BASE_URL = os.getenv("GEMMA_BASE_URL", "http://gemma3/v1") 
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "google/gemma-3-27b-it")  # 예: "Qwen/Qwen-VL-8B-Thinking" 대체 가능
GEMMA_X_DEP_TICKET = os.getenv("GEMMA_X_DEP_TICKET", "12345") 
GEMMA_SEND_SYSTEM_NAME = os.getenv("GEMMA_SEND_SYSTEM_NAME", "AutoMeasure")
GEMMA_USER_ID = os.getenv("GEMMA_USER_ID", "ss") 
GEMMA_USER_TYPE = os.getenv("GEMMA_USER_TYPE", "AD_ID")

# -----------------------------
# Llama-4 / gpt-oss 사내 API (선택적)
# -----------------------------
LLAMA4_BASE_URL = os.getenv("LLAMA4_BASE_URL", "http://llama-4/maverick/v1")  # v1까지 
LLAMA4_MODEL = os.getenv("LLAMA4_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
#LLAMA_MODEL_SCOUT    = os.getenv("LLAMA_MODEL_SCOUT", "meta-llama/llama-4-scout-17b-16e-instruct")
LLAMA4_X_DEP_TICKET = os.getenv("LLAMA4_X_DEP_TICKET", "12345") 
LLAMA4_SEND_SYSTEM_NAME = os.getenv("LLAMA4_SEND_SYSTEM", "AutoMeasure")
LLAMA4_USER_ID = os.getenv("LLAMA4_USER_ID", "ss") 
LLAMA4_USER_TYPE = os.getenv("LLAMA4_USER_TYPE", "AD_ID")

GPTOSS_BASE_URL = os.getenv("GPTOSS_BASE_URL", "http://gpt-oss-120b/v1")
GPTOSS_MODEL     = os.getenv("GPTOSS_MODEL", "openai/gpt-oss-120b")
GPTOSS_X_DEP_TICKET = os.getenv("GPTOSS_X_DEP_TICKET", "12345") 
GPTOSS_SEND_SYSTEM_NAME = os.getenv("GPTOSS_SEND_SYSTEM", "AutoMeasure")
GPTOSS_USER_ID   = os.getenv("GPTOSS_USER_ID", "ss") 
GPTOSS_USER_TYPE = os.getenv("GPTOSS_USER_TYPE", "ss") 

# 최대 이미지 수 제한 (OpenAI 호환: 5장)
MAX_IMAGES_PER_PROMPT = 5

# OpenAI 호환 키(사내 게이트웨이에서 요구될 수 있음)
#os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "dummy-key"))
os.environ['OPENAI_API_KEY'] = 'api_key'

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# 대화/스레드 메모리: 파일별(history)
HISTORY_FILE = "chat_history.json"

# 메모리: GPT-OSS 대화 세션/버전관리 (파일별 유지)
SESSIONS: Dict[str, List[Dict]] = {}         # image_name -> list of turns [{"role":"user/assistant","content":"..."}]
CODE_BACKUPS: Dict[str, List[str]] = {}      # image_name -> [code_snapshot_0, code_snapshot_1, ...]
STRUCTURED_DIFF_CACHE: Dict[str, Dict] = {}  # image_name -> latest structured diff json-like

# --------- 실행 상태(프로그레스 폴링용) ----------
RUN_STATUS: Dict[str, Dict] = {}  # key=stem, value={'phase':str,'attempt':int,'progress':int,'label':str}
def _set_status(stem: str, phase: str, attempt: int, progress: int, label: str):
    RUN_STATUS[stem] = {'phase': phase, 'attempt': attempt, 'progress': int(max(0,min(100,progress))), 'label': label, 'ts': time.time()}

def _clear_status(stem: str):
    try: RUN_STATUS.pop(stem, None)
    except Exception: pass

# 표준 CSV 헤더 (요청 6)
STD_CSV_HEADERS = [
    "measure_item","group_id","index","value_nm",
    "sx","sy","ex","ey",
    "meta_tag","component_label","image_name","run_id","note"
]

# -----------------------------
# [NEW] Qwen3-VL 로컬(HuggingFace) 설정/캐시
# -----------------------------
QWEN_ENABLE = os.getenv("QWEN_ENABLE", "1")  # "1"이면 사용 시도
QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
QWEN_DEVICE = os.getenv("QWEN_DEVICE", "auto")  # "cuda"|"cpu"|"auto"
QWEN_DTYPE = os.getenv("QWEN_DTYPE", "auto")    # "auto"|"bfloat16"|"float16"|"float32"
_qwen_loaded = False
_qwen_pipe = None

# -----------------------------
# [NEW] CSV 표준 헤더 normalize (단일 파일용)
# -----------------------------


def _lazy_load_qwen() -> bool:
    """
    Qwen3-VL / Qwen2-VL 멀티모달 로더 (GPU 우선, bf16/fp16, flash-attn 가능시 자동 사용).
    - 모델은 HF 캐시를 사용하므로 최초 1회 이후엔 로딩 로그만 보이고 빠르게 재사용됨.
    - 어떤 오류가 나도 예외 올리지 않고 False만 반환(상위에서 원문 그대로 사용하도록).
    """
    global _qwen_loaded, _qwen_pipe
    if _qwen_loaded:
        return _qwen_pipe is not None
    try:
        if QWEN_ENABLE != "1":
            log.info("[QWEN] disabled by env")
            _qwen_loaded = True
            _qwen_pipe = None
            return False

        import torch
        from transformers import AutoProcessor, AutoConfig

        # --- Device/precision 결정
        has_cuda = torch.cuda.is_available()
        dev = "cuda" if has_cuda else "cpu"
        # dtype: GPU면 bf16>fp16>fp32 순, CPU면 float32
        if dev == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        # flash-attn 가능하면 사용 시도 (transformers>=4.43)
        attn_impl = None
        try:
            import flash_attn  # noqa:F401
            attn_impl = "flash_attention_2"
        except Exception:
            attn_impl = None

        log.info(f"[QWEN] loading model={QWEN_MODEL_ID} device={dev} dtype={str(dtype)} cache_dir=(default)")
        cfg = AutoConfig.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
        cfg_name = cfg.__class__.__name__.lower()
        model_type = (getattr(cfg, "model_type", "") or "").lower()
        is_mm = any(k in (cfg_name + " " + model_type) for k in ["qwen3vl", "qwen3_vl", "qwen2vl", "qwen2_vl", "vision"])

        from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForVision2Seq, AutoModelForCausalLM
        processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)

        model = None
        meta = {"device": dev, "dtype": str(dtype), "attn": attn_impl}

        # 1순위: ImageTextToText (신규)
        if is_mm and model is None:
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    QWEN_MODEL_ID,
                    trust_remote_code=True,
                    device_map="auto" if dev == "cuda" else None,
                    torch_dtype=dtype,
                    attn_implementation=attn_impl,
                    low_cpu_mem_usage=True,
                )
                _qwen_pipe = (processor, model, True, meta)
                _qwen_loaded = True
                log.info("[QWEN] loaded (ImageTextToText, multimodal).")
                return True
            except Exception as e1:
                log.warning(f"[QWEN] ImageTextToText load failed: {e1}")

        # 2순위: Vision2Seq (구형)
        if is_mm and model is None:
            try:
                model = AutoModelForVision2Seq.from_pretrained(
                    QWEN_MODEL_ID,
                    trust_remote_code=True,
                    device_map="auto" if dev == "cuda" else None,
                    torch_dtype=dtype,
                    attn_implementation=attn_impl,
                    low_cpu_mem_usage=True,
                )
                _qwen_pipe = (processor, model, True, meta)
                _qwen_loaded = True
                log.info("[QWEN] loaded (Vision2Seq, multimodal).")
                return True
            except Exception as e2:
                log.warning(f"[QWEN] Vision2Seq load failed: {e2}")

        # 3순위: 텍스트 전용 (최후백업 — 그래도 예외는 안 올리고 원문 흐름 유지)
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_ID,
            trust_remote_code=True,
            device_map="auto" if dev == "cuda" else None,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )
        _qwen_pipe = (processor, model, False, meta)
        _qwen_loaded = True
        log.info("[QWEN] loaded (CausalLM, text-only).")
        return True

    except Exception as e:
        log.exception("[QWEN] load failed: %s", e)
        _qwen_loaded = True
        _qwen_pipe = None
        return False

# -----------------------------
# 유틸 (기존 유지)
# -----------------------------
def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in ALLOWED_EXTS

def list_image_files() -> list[Path]:
    if not IMAGE_DIR.exists(): return []
    files = [p for p in IMAGE_DIR.iterdir() if _is_image_file(p)]
    files.sort(key=lambda x: x.name.lower())
    return files

def _normalize_name(raw: str) -> str:
    name = unquote(raw).strip()
    while name.startswith(("./", ".\\", "/", "\\", ".")):
        if name.startswith("./") or name.startswith(".\\"): name = name[2:]
        elif name.startswith("/") or name.startswith("\\"): name = name[1:]
        elif name.startswith("."): name = name[1:]
        else: break
    return name

def _mask_name_from_merge(name: str) -> str:
    key = "_gray_merge.png"
    if key in name:
        base = name.split(key)[0]
        return f"{base}_gray.tif"
    return ""

def _scribble_path_from_merge(name: str) -> Optional[Path]:
    if "_gray_merge.png" in name:
        base = name.split("_gray_merge.png")[0]
        cand = SCRIBBLE_DIR / f"{base}.png"
        if cand.exists(): return cand
    stem = Path(name).stem
    cand2 = SCRIBBLE_DIR / f"{stem}.png"
    return cand2 if cand2.exists() else None

def _code_label_name(name: str) -> str:
    key = "_gray_merge.png"
    if key in name:
        base = name.split(key)[0]
        return f"{base}.py"
    return f"{Path(name).stem}.py"

def _run_dir_for(image_name: str) -> Path:
    return RUN_DIR / Path(image_name).stem

def _overlay_path_for(image_name: str) -> Path:
    rd = _run_dir_for(image_name); rd.mkdir(parents=True, exist_ok=True); return rd / "overlay.png"

def _python_executable() -> str:
    return sys.executable or "python"

def _ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    
def _sdiff_to_recipe(sdiff: dict) -> dict:
    """
    scribble_v2_lite 기반 SDIFF에서 '측정 레시피'를 추출한다.
    - guide_y_levels: 빨간 수평 가이드라인의 y 레벨(중복 제거/정렬)
    - guide_x_refs  : 빨간 수직 기준선 x 값(선택)
    - measure_class : 기본 측정에 사용할 마스크 클래스(필요시 후단에서 바꿔도 됨)
    - segment_length_px / segment_gap_px: 초록 세그먼트 타일링 기본값
    - snap_tolerance_px: red/green 접촉 시 살짝 비켜주는 픽셀
    - offset_nm: raw_text에서 '… nm' 오프셋 힌트가 있으면 추출
    """
    import re, json

    recipe = {
        "guide_y_levels": [],
        "guide_x_refs": [],
        "measure_class": 50,
        "segment_length_px": 24,
        "segment_gap_px": 16,
        "snap_tolerance_px": 3,
        "offset_nm": None,
        "notes": {}
    }
    if not isinstance(sdiff, dict):
        return recipe

    red = (sdiff.get("red") or {})
    red_lines = red.get("lines") or []

    # 1) 빨간 수평선 → guide_y_levels
    for ln in red_lines:
        ep = ln.get("endpoints")
        if isinstance(ep, list) and len(ep) == 2:
            (x1, y1), (x2, y2) = ep
            if isinstance(x1, (int, float)) and isinstance(y1, (int, float)) and \
               isinstance(x2, (int, float)) and isinstance(y2, (int, float)):
                # 수평성 느슨 판정
                if abs(y2 - y1) <= abs(x2 - x1):
                    recipe["guide_y_levels"].append(int(round((y1 + y2) / 2)))

    # 2) 빨간 수직선 → guide_x_refs
    for ln in red_lines:
        ep = ln.get("endpoints")
        if isinstance(ep, list) and len(ep) == 2:
            (x1, y1), (x2, y2) = ep
            if isinstance(x1, (int, float)) and isinstance(y1, (int, float)) and \
               isinstance(x2, (int, float)) and isinstance(y2, (int, float)):
                if abs(x2 - x1) <= abs(y2 - y1):
                    recipe["guide_x_refs"].append(int(round((x1 + x2) / 2)))

    # 중복 제거/정렬
    recipe["guide_y_levels"] = sorted({y for y in recipe["guide_y_levels"]})
    recipe["guide_x_refs"]   = sorted({x for x in recipe["guide_x_refs"]})

    # 3) raw_text에서 offset nm 추출
    raw_text = (((sdiff.get("notes") or {}).get("raw_text")) or "")
    try:
        m = re.search(r"(?:offset|오프셋)\D+(\d+(?:\.\d+)?)\s*nm", raw_text, re.I)
        if m:
            recipe["offset_nm"] = float(m.group(1))
    except Exception:
        pass

    recipe["notes"]["source"] = (sdiff.get("notes") or {}).get("source")
    return recipe

import math
def _fmt1(v):
    return f"{v:.1f}" if isinstance(v, (int, float)) else "n/a"

def _angle_stats_from_lines(lines: list):
    """lines[*].angle_deg 값으로 통계. 없으면 전부 None."""
    vals = []
    for ln in (lines or []):
        ang = ln.get("angle_deg")
        if isinstance(ang, (int, float)):
            vals.append(float(ang))
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None}
    import math
    n = len(vals)
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n
    std = math.sqrt(var)
    return {"mean": mean, "std": std, "min": min(vals), "max": max(vals)}

def _orientation_percent(lines: list):
    """lines[*].orientation in {'h','v','mixed'} 비율. 없으면 빈 dict."""
    tot = 0
    cnt = {"h": 0, "v": 0, "mixed": 0}
    for ln in (lines or []):
        o = ln.get("orientation")
        if o in cnt:
            cnt[o] += 1
            tot += 1
    if tot == 0:
        return {}
    return {k: round(100.0 * v / tot, 1) for k, v in cnt.items() if v > 0}

def _safe_count(bucket: dict):
    """
    count가 None이면 lines 길이로 대체.
    bucket 예: {"count": int|None, "lines": [...]}
    """
    if not isinstance(bucket, dict):
        return 0
    c = bucket.get("count")
    if isinstance(c, int):
        return c
    lines = bucket.get("lines") or []
    return len(lines)

def _sdiff_to_summary_text(sdiff: dict) -> str:
    """
    scribble_v2_lite 요약 문자열 생성 (널-세이프).
    각도/지표가 없으면 'n/a'로 채움.
    """
    if not isinstance(sdiff, dict):
        return "[SDIFF SUMMARY]\n- (invalid sdiff)"

    # buckets
    red = sdiff.get("red", {}) or {}
    grn = sdiff.get("green", {}) or {}
    notes = sdiff.get("notes", {}) or {}
    rules = sdiff.get("rules", {}) or {}

    rg = _safe_count(red)
    gg = _safe_count(grn)

    rstats = _angle_stats_from_lines(red.get("lines") or [])
    gstats = _angle_stats_from_lines(grn.get("lines") or [])

    ror = _orientation_percent(red.get("lines") or [])
    gor = _orientation_percent(grn.get("lines") or [])

    src = notes.get("source") or "unknown"
    mask_name  = notes.get("mask_name") or "-"
    merge_name = notes.get("merge_name") or "-"
    scrib_name = notes.get("scribble_name") or "-"

    txt = []
    txt.append("[SDIFF SUMMARY]")
    txt.append(f"- source: {src}")
    txt.append(f"- inputs: mask={mask_name}, merge={merge_name}, scribble={scrib_name}")
    txt.append(
        f"- red_guides: {rg} lines "
        f"(angle mean={_fmt1(rstats['mean'])}°, std={_fmt1(rstats['std'])}°, "
        f"min={_fmt1(rstats['min'])}°, max={_fmt1(rstats['max'])}°) "
        f"| orientation%: {ror if ror else '{}'}"
    )
    txt.append(
        f"- green_measures: {gg} lines "
        f"(angle mean={_fmt1(gstats['mean'])}°, std={_fmt1(gstats['std'])}°, "
        f"min={_fmt1(gstats['min'])}°, max={_fmt1(gstats['max'])}°) "
        f"| orientation%: {gor if gor else '{}'}"
    )
    if rules:
        txt.append(f"- rules: {rules}")
    return "\n".join(txt)


def _structured_diff_semantic_summary(structured: dict) -> str:
    """
    red/green 라인의 시맨틱 힌트를 읽기 쉬운 텍스트로 요약한다.
    좌표 대신 `detected_class_hint*`, `paired_red_id`, `orientation` 등만 노출한다.
    """

    def _norm_orientation(val: Any) -> str:
        if not isinstance(val, str):
            return "unknown"
        v = val.strip().lower()
        if v in {"h", "horizontal"}:
            return "horizontal"
        if v in {"v", "vertical"}:
            return "vertical"
        if v in {"diag", "diagonal"}:
            return "diagonal"
        return val.strip() or "unknown"

    def _sem_val(line: Dict[str, Any], fallback: str) -> str:
        for key in ("semantic", "semantic_role"):
            val = line.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return fallback

    def _fmt_hint(val: Any) -> str:
        if val is None:
            return "?"
        return str(val)

    if not isinstance(structured, dict):
        return "- (invalid structured_diff)"

    red_bucket = structured.get("red") or {}
    green_bucket = structured.get("green") or {}

    red_lines = red_bucket.get("lines") or []
    green_lines = green_bucket.get("lines") or []

    lines: List[str] = []

    if red_lines:
        lines.append("### Red Guides")
        for ln in red_lines:
            if not isinstance(ln, dict):
                continue
            rid = str(ln.get("id") or "red")
            semantic = _sem_val(ln, "guide")
            orientation = _norm_orientation(ln.get("orientation"))
            details: List[str] = []
            if "detected_class_hint" in ln:
                details.append(f"class_hint={_fmt_hint(ln.get('detected_class_hint'))}")
            details.append(f"orientation={orientation}")
            lines.append(f"- {rid} ({semantic}): " + ", ".join(details))

    if green_lines:
        lines.append("### Green Measures")
        for ln in green_lines:
            if not isinstance(ln, dict):
                continue
            gid = str(ln.get("id") or "green")
            semantic = _sem_val(ln, "measure")
            orientation = _norm_orientation(ln.get("orientation"))
            details = []
            start_hint = ln.get("detected_class_hint_start")
            end_hint = ln.get("detected_class_hint_end")
            if start_hint is not None or end_hint is not None:
                details.append(
                    f"start_hint={_fmt_hint(start_hint)} → end_hint={_fmt_hint(end_hint)}"
                )
            anchor = ln.get("paired_red_id")
            if anchor:
                details.append(f"anchor={anchor}")
            details.append(f"orientation={orientation}")
            lines.append(f"- {gid} ({semantic}): " + ", ".join(details))

    if not lines:
        return "- (no structured diff hints)"

    return "\n".join(lines)

def _deg_norm(a):
    a = float(a) % 360.0
    if a < 0: a += 360.0
    return a

def _orientation_from_angle(a_deg, tol=15.0): # [FIX] 허용치를 8.0 -> 15.0으로 늘림
    a = float(a_deg) % 360.0
    if a < 0: a += 360.0 # [NEW] 각도 정규화
    if min(abs(a-0), abs(a-180), abs(a-360)) <= tol: return "horizontal" # 0, 180, 360
    if min(abs(a-90), abs(a-270)) <= tol: return "vertical" # 90, 270
    return "diagonal"

def _classify_measure_semantics(ln):
    ori = _orientation_from_angle(ln.get("angle_deg", 0.0))
    if ori == "vertical":
        return "height"
    if ori == "horizontal":
        return "width"
    return "oblique"

def _angle_stats(lines):
    if not lines:
        return {"mean": None, "std": None, "min": None, "max": None}
    arr = np.array([_deg_norm(l.get("angle_deg", 0.0)) for l in lines], dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max())
    }

def _sdiff_counts_detail(structured: dict):
    c = structured.get("counts") or {}
    rg = int(c.get("red_guides") or len(structured.get("red_guides") or []))
    gm = int(c.get("green_measures") or len(structured.get("green_measures") or []))
    return rg, gm


def _sdiff_to_verbose_text(structured: dict, max_list:int=200) -> str:
    reds  = structured.get("red_lines_detail")   or []
    greens= structured.get("green_lines_detail") or []
    lines = []
    lines.append("[SDIFF LINES DETAIL]")
    lines.append("## RED GUIDES (role=guide)")
    for i, ln in enumerate(reds[:max_list], 1):
        lines.append(
            f"{i:02d}. id={ln.get('id','')}, p1=({ln['x1']},{ln['y1']}), p2=({ln['x2']},{ln['y2']}), "
            f"length_px={ln.get('length_px'):.1f}, angle_deg={ln.get('angle_deg'):.1f}, thickness_px={ln.get('thickness_px')}"
        )
    if len(reds) > max_list:
        lines.append(f"... (+{len(reds)-max_list} more)")
    lines.append("")
    lines.append("## GREEN MEASURES (role=measure)")
    for i, ln in enumerate(greens[:max_list], 1):
        sem = _classify_measure_semantics(ln)
        lines.append(
            f"{i:02d}. id={ln.get('id','')}, paired_red_id={ln.get('paired_red_id')}, "
            f"p1=({ln['x1']},{ln['y1']}), p2=({ln['x2']},{ln['y2']}), length_px={ln.get('length_px'):.1f}, "
            f"angle_deg={ln.get('angle_deg'):.1f}, thickness_px={ln.get('thickness_px')}, semantic={sem}"
        )
    if len(greens) > max_list:
        lines.append(f"... (+{len(greens)-max_list} more)")
    return "\n".join(lines)

def _cv_count_scribble_lines(scribble_path: Path) -> Dict[str, Any]:
    """
    스크리블 이미지에서 빨강/초록 선분의 '개수'를 대략 추정하고,
    각 색상별 대표 endpoints 1~2개를 수집한다. (점선/겹침 보정용)
    - 출력 예:
      {
        "red":   {"count": 6,  "endpoints": [ [[x1,y],[x2,y]], ... ]},
        "green": {"count": 12, "endpoints": [ [[x1,y],[x2,y]], ... ]}
      }
    """
    out = {"red":{"count":0,"endpoints":[]}, "green":{"count":0,"endpoints":[]}}
    img = cv2.imread(str(scribble_path), cv2.IMREAD_COLOR)
    if img is None:
        return out

    H, W = img.shape[:2]

    def color_mask(bgr):
        return ((img == np.array(bgr, np.uint8)).all(axis=2)).astype(np.uint8)*255

    def count_and_pick(mask, prefer_horizontal=True):
        # 얇은 점선 보존 + 연결강화
        k = max(1, int(round(min(H, W) * 0.002)))  # 이미지 크기 비례 커널
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1 if prefer_horizontal else k))
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 짧은 dash들도 잡아내기 위해 minLineLength 낮춤
        lines = cv2.HoughLinesP(m, 1, np.pi/180, threshold=40, minLineLength=20, maxLineGap=10)
        segs = []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:,0]:
                # 화면 밖 값 제거
                x1 = int(np.clip(x1, 0, W-1)); x2 = int(np.clip(x2, 0, W-1))
                y1 = int(np.clip(y1, 0, H-1)); y2 = int(np.clip(y2, 0, H-1))
                segs.append((x1,y1,x2,y2))

        # 비슷한 선분(겹침/연결)을 수평/수직 기준으로 병합(개수 과대 방지)
        def merge_collinear(segments, axis='h', d_tol=6, gap_tol=12):
            if not segments: return []
            merged = []
            used = [False]*len(segments)

            if axis=='h':
                # y가 비슷하고 x-구간이 겹치면 병합
                segments = sorted(segments, key=lambda s:( (s[1]+s[3])//2, min(s[0],s[2]) ))
                for i,s in enumerate(segments):
                    if used[i]: continue
                    x1,y1,x2,y2 = s
                    y = int(round((y1+y2)/2))
                    xmin, xmax = min(x1,x2), max(x1,x2)
                    used[i]=True
                    for j,t in enumerate(segments[i+1:], start=i+1):
                        if used[j]: continue
                        a1,b1,a2,b2 = t
                        yb = int(round((b1+b2)/2))
                        if abs(y - yb) <= d_tol:
                            txmin, txmax = min(a1,a2), max(a1,a2)
                            # 구간이 이어지거나 살짝 겹치면 확장
                            if not (txmax < xmin-gap_tol or txmin > xmax+gap_tol):
                                xmin = min(xmin, txmin); xmax = max(xmax, txmax)
                                used[j]=True
                    merged.append((xmin,y,xmax,y))
            else:
                # x가 비슷하고 y-구간이 겹치면 병합(수직)
                segments = sorted(segments, key=lambda s:( (s[0]+s[2])//2, min(s[1],s[3]) ))
                for i,s in enumerate(segments):
                    if used[i]: continue
                    x1,y1,x2,y2 = s
                    x = int(round((x1+x2)/2))
                    ymin, ymax = min(y1,y2), max(y1,y2)
                    used[i]=True
                    for j,t in enumerate(segments[i+1:], start=i+1):
                        if used[j]: continue
                        a1,b1,a2,b2 = t
                        xb = int(round((a1+a2)/2))
                        if abs(x - xb) <= d_tol:
                            tymin, tymax = min(b1,b2), max(b1,b2)
                            if not (tymax < ymin-gap_tol or tymin > ymax+gap_tol):
                                ymin = min(ymin, tymin); ymax = max(ymax, tymax)
                                used[j]=True
                    merged.append((x,y,ymin,ymax))  # 보관용
                # 포맷 통일
                merged = [(x, ymin, x, ymax) for (x,_,ymin,ymax) in merged]

            return merged

        # 수평/수직 두 번 병합 후 더 많은 쪽을 채택(점선 형태에 강함)
        mh = merge_collinear(segs, axis='h')
        mv = merge_collinear(segs, axis='v')
        final = mh if len(mh) >= len(mv) else mv

        # 대표 1~2개 endpoints 추출
        reps = []
        for i,(a,b,c,d) in enumerate(final[:2]):
            reps.append([[int(a),int(b)],[int(c),int(d)]])

        return len(final), reps

    red_mask   = color_mask((0,0,255))
    green_mask = color_mask((0,255,0))

    rc, rreps = count_and_pick(red_mask,   prefer_horizontal=True)
    gc, greps = count_and_pick(green_mask, prefer_horizontal=True)

    out["red"]["count"] = int(rc);   out["red"]["endpoints"] = rreps
    out["green"]["count"] = int(gc); out["green"]["endpoints"] = greps
    return out

CSV_HEADER_STANDARD = (
    'CSV columns (strict): ["measure_item","group_id","index","value_nm",'
    '"sx","sy","ex","ey","meta_tag","component_label","image_name","run_id","note"]'
)

PROMPT_GUARD_HEAD = (
    "[EXECUTION RULES]\n"
    "1) Draw overlay strictly on the MASK image (do NOT draw on merge).\n"
    "2) Produce measurements.csv with the EXACT header:\n"
    f"   {CSV_HEADER_STANDARD}\n"
    "3) Keep ALL existing CLI args intact and write outputs into the provided out_dir.\n"
    "## MANDATORY CHECKLIST (MASK-ONLY COMPUTATION)\n"
    "- Use ONLY `--mask_path` to derive geometry; DO NOT read colored overlays from `--mask_merge_path`.\n"
    "- Interpret segmentation classes (e.g., 30/50/70) via connected components / contours / PCA axes / skeleton / morphology.\n"
    "- Apply STRUCTURED_DIFF(lite) as high-level hints:\n"
    "  * red.lines[i]: {role:'guide', orientation:'h|v|mixed', angle_deg, offset_nm?}\n"
    "  * green.lines[i]: {role:'measure', orientation:'h|v|mixed', angle_deg, overlaps_red_id?}\n"
    "- Compute final line coordinates from the mask geometry and offsets; do not trust any pre-drawn coordinates.\n"
    "- Draw overlay **on the mask image** using BGR pure colors only (red=(0,0,255), green=(0,255,0)), `cv2.LINE_8`, no alpha.\n"
    "- Save 'overlay.png' and 'measurements.csv' with the exact standard header.\n"
)

import re

# -----------------------------
# [NEW] merge 이름 → scribble 경로 매핑(견고)
# -----------------------------
def _scribble_path_for(merge_name: str) -> Path:
    """
    예) merge: C2_gray_merge.png  →  scribble: SCRIBBLE_DIR/C2.png
    """
    stem = Path(merge_name).stem
    base = re.sub(r"(?:_gray_merge|_merge|_gray)$", "", stem, flags=re.IGNORECASE)
    candidates = [
        SCRIBBLE_DIR / f"{base}.png",
        SCRIBBLE_DIR / f"{base}.jpg",
        SCRIBBLE_DIR / f"{base}.jpeg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return SCRIBBLE_DIR / f"{base}.png"

def _save_thumb(in_path: Path, out_path: Path, max_side: int = 1280, quality: int = 85):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if PIL_OK:
        img = Image.open(in_path).convert("RGB")  # type: ignore
        w, h = img.size  # type: ignore
        scale = min(1.0, float(max_side)/float(max(w,h)))
        if scale < 1.0:
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)  # type: ignore
        img.save(out_path, format="JPEG", quality=quality, optimize=True)  # type: ignore
    else:
        bgr = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
        if bgr is None: raise FileNotFoundError(in_path)
        h, w = bgr.shape[:2]
        scale = min(1.0, float(max_side)/float(max(w,h)))
        if scale < 1.0:
            bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])

def _prep_for_vision_llm(path: Path, max_side: int = 1280, quality: int = 80) -> str:
    if PIL_OK:
        img = Image.open(path).convert("RGB")  # type: ignore
        w, h = img.size  # type: ignore
        scale = min(1.0, float(max_side)/float(max(w,h)))
        if scale < 1.0:
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)  # type: ignore
        bio = BytesIO()
        img.save(bio, format="JPEG", quality=quality, optimize=True)  # type: ignore
        b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    else:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None: raise FileNotFoundError(path)
        h, w = bgr.shape[:2]
        scale = min(1.0, float(max_side)/float(max(w,h)))
        if scale < 1.0:
            bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok: raise RuntimeError("encode fail")
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

# pHash / similarity helpers (기존)
def _phash64_from_path(path: Path) -> int:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None: return 0
    img = cv2.resize(img, (32,32), interpolation=cv2.INTER_AREA)
    img = np.float32(img)
    dct = cv2.dct(img)
    dct_low = dct[:8,:8]
    med = np.median(dct_low)
    bits = (dct_low > med).astype(np.uint64)
    v=0
    for i in range(8):
        for j in range(8):
            v = (v<<1) | int(bits[i,j])
    return int(v)

def _phash_hex(v: int) -> str:
    return f"{v:016x}"

def _hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def _sim_phash(a_hex: Optional[str], b_hex: Optional[str]) -> float:
    if not a_hex or not b_hex: return 0.0
    a = int(a_hex,16); b = int(b_hex,16)
    return 1.0 - _hamming64(a,b)/64.0

_STOP = {"", " ", "-", "_", ".", "#", "@", "on", "off", "and", "or"}
def _name_tokens(name: str) -> List[str]:
    s = Path(name).stem.lower()
    toks = re.split(r"[^a-zA-Z0-9]+", s)
    return [t for t in toks if t not in _STOP]

def _meta_path_from_mask(mask_path: Path) -> Path:
    base = mask_path.name.split("_gray.tif")[0] + ".json"
    return META_ROOT / base

def _meta_summary(mask_path: Optional[Path]) -> Dict:
    out = {"meta_path": None, "um_per_px_x": None, "um_per_px_y": None, "classes": None}
    if not mask_path: return out
    mpath = _meta_path_from_mask(mask_path)
    out["meta_path"] = str(mpath)
    if mpath.exists():
        try:
            meta = json.loads(mpath.read_text(encoding="utf-8"))
            out["um_per_px_x"] = float(meta.get("pixel_scale_um_x")) if "pixel_scale_um_x" in meta else None
            out["um_per_px_y"] = float(meta.get("pixel_scale_um_y")) if "pixel_scale_um_y" in meta else None
            if "mask_info" in meta and "classes_present" in meta["mask_info"]:
                out["classes"] = meta["mask_info"]["classes_present"]
        except Exception as e:
            log.warning(f"[meta] read fail {mpath}: {e}")
    return out

def _code_summary_text(code_text: str) -> Dict:
    args = re.findall(r'add_argument\(\s*["\'](--[^"\']+)["\']', code_text)
    funcs = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\):', code_text)
    outputs=[]
    if re.search(r'overlay\.png', code_text, re.IGNORECASE): outputs.append("overlay.png")
    if re.search(r'\.to_csv', code_text): outputs.append("measurements.csv")
    return {"cli_args": args, "functions": [{"name":f[0],"params":f[1]} for f in funcs[:15]], "outputs": outputs}

# -----------------------------
# 라우트(템플릿/자산)
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index_codecheck.html", {"request": request})

@app.get("/health")
async def health():
    return JSONResponse({"ok": True})

@app.get("/get_images")
async def get_images():
    if not IMAGE_DIR.exists():
        return JSONResponse({"error":"IMAGE_DIR not found","dir":str(IMAGE_DIR)}, status_code=500)
    files = list_image_files()
    items=[]
    for p in files:
        items.append({
            "name": p.name,
            "url": f"/get_image/{quote(p.name, safe='')}",
            "labeled": (LABEL_DIR / _code_label_name(p.name)).exists()
        })
    return JSONResponse(items)

@app.get("/get_image/{image_name}")
async def get_image(image_name: str):
    normalized = _normalize_name(image_name)
    path = IMAGE_DIR / normalized
    if not path.exists() or not path.is_file():
        return JSONResponse({"error":"Image not found","requested": image_name}, status_code=404)
    return FileResponse(path)

@app.get("/get_mask/{image_name}")
async def get_mask(image_name: str):
    normalized = _normalize_name(image_name)
    mask_name = _mask_name_from_merge(normalized)
    if not mask_name:
        return JSONResponse({"error":"Mask name rule failed","requested": image_name}, status_code=404)
    mask_path = Path(IMAGE_DIR).parent / "mask" / mask_name
    if not mask_path.exists() or not mask_path.is_file():
        return JSONResponse({"error":"Mask file not found","mask": mask_name,"searched":str(mask_path)}, status_code=404)
    return FileResponse(mask_path)

@app.get("/get_scribble/{image_name}")
async def get_scribble(image_name: str):
    normalized = _normalize_name(image_name)
    p = _scribble_path_from_merge(normalized)
    if not p or not p.exists():
        return JSONResponse({"error":"Scribble not found"}, status_code=404)
    return FileResponse(p)

@app.get("/overlay")
async def get_overlay_by_query(name: str = Query(..., description="URL-encoded filename")):
    return await _serve_overlay(name)

@app.get("/label/status/{image_name}")
async def label_status(image_name: str):
    try:
        normalized = _normalize_name(image_name)
        py_name = _code_label_name(normalized)
        labeled = (LABEL_DIR / py_name).exists()
        return JSONResponse({"ok": True, "labeled": labeled, "filename": py_name})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=200)

async def _serve_overlay(image_name: str):
    requested_raw = image_name
    normalized = _normalize_name(image_name)
    stem = Path(normalized).stem

    candidates: list[Path] = []
    root = RUN_DIR / stem
    if root.exists():
        p0 = root / "overlay.png"
        if p0.exists():
            candidates.append(p0)
        for sub in root.rglob("overlay.png"):
            if sub.is_file():
                candidates.append(sub)

    if RUN_DIR.exists():
        for sub in RUN_DIR.iterdir():
            try:
                if sub.is_dir() and Path(sub.name).name.startswith(stem):
                    cand = sub / "overlay.png"
                    if cand.exists() and cand not in candidates:
                        candidates.append(cand)
            except Exception:
                pass

    if not candidates:
        log.warning(f"[overlay] not found for '{requested_raw}'")
        return JSONResponse({"error": "overlay.png not found", "requested": requested_raw}, status_code=404)

    picked = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        if picked.stat().st_size == 0:
            return JSONResponse({"error": "overlay.png is empty", "path": str(picked)}, status_code=500)
    except Exception:
        pass
    return FileResponse(picked, media_type="image/png")

# -----------------------------
# Scribble → 구조 라인 추출 (fallback + Qwen assist)
# -----------------------------
def _extract_color_lines_componentwise(bgr: np.ndarray, color: str) -> List[Dict[str, int]]:
    """
    [FIXED] 109/51 노이즈 검출 버그 해결.
    - [REVERT] 불안정한 HoughLinesP 로직을 폐기하고, 안정적인 'fitLine (컨투어)' 로직으로 원복.
    - [NEW] 'cv2.contourArea > 30' 필터를 추가하여 1~10px 크기의 노이즈 컨투어를 사전 제거.
    - [NEW] 'line_length > 10' 필터를 추가하여 최종 결과물에서 '점'을 제거.
    - [KEPT] '조건부 복원' 로직(red/green 분리)은 유지.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    # --- 1. 빨간/초록 마스크를 먼저 둘 다 생성
    m_red_1 = cv2.inRange(hsv, (0, 70, 70), (10, 255, 255))
    m_red_2 = cv2.inRange(hsv, (170, 70, 70), (180, 255, 255))
    red_mask = cv2.bitwise_or(m_red_1, m_red_2)
    
    green_mask = cv2.inRange(hsv, (38, 60, 60), (85, 255, 255))
    
    mask = None # 최종 처리할 마스크
    kernel_3x3 = np.ones((3,3), np.uint8)
    kernel_5x5 = np.ones((5,5), np.uint8)

    if color == "red":
        # --- [KEPT] 2. '빨간색' 처리: 조건부 복원 (사용자 3가지 시나리오 고려)
        dilated_red_mask = cv2.dilate(red_mask, kernel_3x3, iterations=1)
        filling_green_mask = cv2.bitwise_and(green_mask, dilated_red_mask)
        restored_red_mask = cv2.bitwise_or(red_mask, filling_green_mask)
        mask = cv2.morphologyEx(restored_red_mask, cv2.MORPH_CLOSE, kernel_5x5, iterations=2)
            
    elif color == "green":
        # --- [KEPT] 2. '초록색' 처리: 순수 초록선만 추출
        dilated_red_mask = cv2.dilate(red_mask, kernel_3x3, iterations=1)
        pure_green_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(dilated_red_mask))
        mask = cv2.morphologyEx(pure_green_mask, cv2.MORPH_CLOSE, kernel_5x5, iterations=2)

    else: # helper
        mask = cv2.inRange(hsv, (10,10,10),(30,60,255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_5x5, iterations=2)

    # --- [REVERT] 3. 컨투어 기반 'fitLine' 로직 (HoughLinesP 폐기) ---
    H, W = bgr.shape[:2]
    lines = []

    # [FIX] 노이즈 제거를 위해 RETR_EXTERNAL 사용
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        # [NEW] 1차 노이즈 필터: 면적이 30px 미만인 작은 컨투어(점)는 무시
        if cv2.contourArea(c) < 30:
            continue
            
        pts = c.reshape(-1,2).astype(np.float32)
        try:
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = _as_scalar(vx), _as_scalar(vy), _as_scalar(x0), _as_scalar(y0)
        except Exception:
            continue
            
        # 컨투어의 양 끝점을 찾아 라인을 그림
        pts_vec = pts - np.array([x0, y0])
        projections = pts_vec[:, 0] * vx + pts_vec[:, 1] * vy
        
        t_min = np.min(projections)
        t_max = np.max(projections)

        sx = int(np.clip(round(x0 + vx*t_min), 0, W-1))
        sy = int(np.clip(round(y0 + vy*t_min), 0, H-1))
        ex = int(np.clip(round(x0 + vx*t_max), 0, W-1))
        ey = int(np.clip(round(y0 + vy*t_max), 0, H-1))
        
        # [NEW] 2차 노이즈 필터: 최종 라인 길이가 10px 미만인 '점'은 무시
        if _line_length_px(sx,sy,ex,ey) < 10:
            continue
            
        lines.append({"sx": sx, "sy": sy, "ex": ex, "ey": ey})
    
    return lines

def _orientation_from_angle(a_deg, tol=8.0):
    a = float(a_deg) % 360.0
    if min(abs(a-0), abs(a-180)) <= tol: return "horizontal"
    if min(abs(a-90), abs(a-270)) <= tol: return "vertical"
    return "diagonal"

def _semantic_from_orientation(ori: str) -> str:
    if ori == "vertical": return "height"
    if ori == "horizontal": return "width"
    return "oblique"

# [REPLACE] _enrich_lines 함수
def _enrich_lines(lines: List[Dict[str,int]], role: str, um_per_px_x: float = 0.0, mask_img: Optional[np.ndarray] = None) -> List[Dict[str,object]]:
    """ 
    [CHANGED] VLM의 추론을 위해 모든 CV '사실' 힌트를 주입합니다.
    - 'red' (가이드): '선 전체'를 샘플링하여 `detected_class_hint` (위치) 추가.
    - 'green' (측정): '양 끝점'을 샘플링하여 `detected_class_hint_start`와 `detected_class_hint_end` (양쪽 위치) 추가.
    """
    out=[]
    # (Heuristics) 이 값은 튜닝이 필요할 수 있습니다.
    REF_GUIDE_MIN_LEN_PX = 200 # 200px 이상 긴 라인은 'reference'로 간주
    OFFSET_GUIDE_MAX_LEN_UM = 0.1 # 0.1um (100nm) 이하의 짧은 라인은 'offset'으로 간주
    DIAGONAL_TOLERANCE = 15.0 # 15도까지 수평/수직으로 간주

    for i,ln in enumerate(lines, 1):
        x1,y1,x2,y2 = ln["sx"],ln["sy"],ln["ex"],ln["ey"]
        L = _line_length_px(x1,y1,x2,y2)
        ang = _line_angle_deg(x1,y1,x2,y2)
        ori = _orientation_from_angle(ang, tol=DIAGONAL_TOLERANCE)

        length_um = (L * um_per_px_x) if um_per_px_x > 0 else 0.0
        semantic_role = "unknown"

        out_item = {
            "id": f"{role}_{i}",
            "role": role,
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "length_px": float(L),
            "angle_deg": float(ang),
            "orientation": ori,
            "thickness_px": 1,
            "length_um": float(length_um),
            # 'semantic_role'은 나중에 채움
        }

        # [NEW] 클래스 힌트: 역할(role)에 따라 다른 전략 적용
        if role == "red":
            # [RED] 가이드라인은 '선 전체'를 샘플링하여 *위치* 힌트 제공
            class_hint = _sample_class_at_line(mask_img, x1, y1, x2, y2)
            out_item["detected_class_hint"] = class_hint # (e.g., 50)

            # (Red 의미론 추론)
            if (ori == 'v' or ori == 'diagonal') and length_um > 0 and length_um < OFFSET_GUIDE_MAX_LEN_UM:
                semantic_role = f"offset_guide_{length_um*1000.0:.0f}nm"
            elif (ori == 'h' or ori == 'diagonal') and L > REF_GUIDE_MIN_LEN_PX:
                semantic_role = "reference_guide_h"
            elif (ori == 'v' or ori == 'diagonal') and L > REF_GUIDE_MIN_LEN_PX:
                semantic_role = "reference_guide_v"
            else:
                semantic_role = f"{ori}_guide"
        
        elif role == "green":
            # [GREEN] 측정선은 '양 끝점'을 모두 샘플링
            class_hint_start = _sample_class_near_point(mask_img, x1, y1, radius=15)
            class_hint_end = _sample_class_near_point(mask_img, x2, y2, radius=15)
            
            # [RESTORED] VLM 추론을 위해 두 힌트를 모두 주입
            out_item["detected_class_hint_start"] = class_hint_start # (e.g., 50)
            out_item["detected_class_hint_end"] = class_hint_end   # (e.g., 10 or 30)
            
            # (Green 의미론 추론)
            semantic_role = _semantic_from_orientation(ori)

        out_item["semantic_role"] = semantic_role
        out.append(out_item)
        
    return out

def _pair_green_to_nearest_red(greens: List[Dict], reds: List[Dict]):
    # 가장 가까운 교차/말단 거리 기반 단순 매칭
    if not greens or not reds: return
    def _endpoints(g): return [(g["x1"],g["y1"]),(g["x2"],g["y2"])]
    for g in greens:
        best=None; bestd=1e9
        for r in reds:
            cand = min(_line_length_px(ex[0],ex[1], r["x1"],r["y1"]) for ex in _endpoints(g))
            cand = min(cand, _line_length_px(g["x2"],g["y2"], r["x2"],r["y2"]))
            if cand < bestd:
                bestd=cand; best=r
        if best: g["paired_red_id"] = best["id"]
        g["semantic"] = _semantic_from_orientation(g.get("orientation","diagonal"))

def _structured_from_scribble(scribble_path: Optional[Path], um_per_px_x: float = 0.0, mask_path: Optional[Path] = None) -> Dict:
    """ [CHANGED] mask_path를 인자로 받아 _enrich_lines로 mask_img 전달 """
    if not scribble_path or not scribble_path.exists():
        return {}
    bgr = cv2.imread(str(scribble_path), cv2.IMREAD_COLOR)
    if bgr is None: return {}

    # [NEW] 마스크 이미지 로드 (class hint 샘플링용)
    mask_img = None
    if mask_path and mask_path.exists():
        try:
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        except Exception:
            mask_img = None # 실패해도 SDIFF 생성은 계속

    H, W = bgr.shape[:2]
    red_lines    = _extract_color_lines_componentwise(bgr, "red")
    green_lines  = _extract_color_lines_componentwise(bgr, "green")
    helper_lines = []

    # [CHANGED] 픽셀 스케일과 mask_img 전달
    reds_detail   = _enrich_lines(red_lines, "red", um_per_px_x=um_per_px_x, mask_img=mask_img)
    greens_detail = _enrich_lines(green_lines, "green", um_per_px_x=um_per_px_x, mask_img=mask_img)

    _pair_green_to_nearest_red(greens_detail, reds_detail)

    sdiff = {
        "schema": "scribble_v1", # (v1으로 생성, v2_lite는 업그레이드 함수가 담당)
        "image_size": {"width": W, "height": H},
        "counts": {"red": len(red_lines), "green": len(green_lines), "helper": len(helper_lines)},
        "red": red_lines,
        "green": green_lines,
        "helper": helper_lines,
        # 디테일 + 통계에 쓰일 필드
        "red_lines_detail": reds_detail,
        "green_lines_detail": greens_detail,
        "notes": {"source": "hybrid_cv_geometry_qwen_notes"} # (이 함수는 CV 담당)
    }
    return sdiff

# ---------- Few-shot 텍스트(슬림화) ----------
def _load_all_fewshots() -> List[Dict]:
    if not FEWSHOT_DIR.exists(): return []
    out=[]
    for sub in FEWSHOT_DIR.iterdir():
        if not sub.is_dir(): continue
        meta = sub/"item.json"
        if meta.exists():
            try:
                j = json.loads(meta.read_text(encoding="utf-8"))
                j["_dir"] = str(sub)
                out.append(j)
            except Exception:
                pass
    return out

def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars: return s
    head = s[:max_chars-200]
    return head + "\n\n...[truncated]..."

def _fewshot_block_text_only(image_name: str, topk: int = 2, *, slim: bool = True) -> List[str]:
    items = _select_topk_fewshots_for(image_name, k=topk)
    out=[]
    for it in items:
        d = Path(it["_dir"])
        try:
            for nm in ("code_summary.json","meta_summary.json","item.json"):
                p = d / nm
                if p.exists():
                    out.append(f"[FEWSHOT:{nm}]\n"+_truncate(p.read_text(encoding="utf-8"), 8000))
            if not slim:
                pcode = d / "code.py"
                if pcode.exists():
                    out.append("[FEWSHOT:code.py]\n"+_truncate(pcode.read_text(encoding="utf-8"), 12000))
        except Exception:
            pass
    return out

# ---------- MMR 기반 다양성 선택(그대로 유지) ----------
def _select_topk_fewshots_for(image_name: str, k: int = 2) -> List[Dict]:
    def _recency_score(ts: str, lam_days: float = 90.0) -> float:
        try:
            t = time.mktime(time.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S"))
        except Exception:
            return 0.5
        days = max(0.0, (time.time()-t)/86400.0)
        return math.exp(-days/lam_days)
    def _jaccard(a: set, b: set) -> float:
        if not a and not b: return 1.0
        return len(a & b)/max(1, len(a | b))
    def _scale_sim(q: Optional[float], x: Optional[float], tau: float = 0.05) -> float:
        if q is None or x is None: return 0.5
        diff = abs(q - x)
        return math.exp(-diff / tau)

    normalized = _normalize_name(image_name)
    merge_path = (IMAGE_DIR / normalized).resolve()
    q_merge_ph=None
    try: q_merge_ph=_phash_hex(_phash64_from_path(merge_path))
    except Exception: pass
    q_tokens = _name_tokens(normalized)

    mask_name = _mask_name_from_merge(normalized)
    q_um=None; q_classes=[]
    if mask_name:
        msum = _meta_summary((Path(IMAGE_DIR).parent/"mask"/mask_name).resolve())
        q_um = msum.get("um_per_px_x")
        if msum.get("classes"): q_classes=list(msum["classes"])

    cands = _load_all_fewshots()
    scored=[]
    for it in cands:
        sim_img=0.0
        if it.get("hashes",{}).get("merge_phash"):
            sim_img=_sim_phash(q_merge_ph, it["hashes"]["merge_phash"])
        sim_name   = _jaccard(set(q_tokens), set(it.get("name_tokens",[])))
        sim_scale  = _scale_sim(q_um, it.get("meta",{}).get("um_per_px_x"), tau=0.05)
        sim_cls    = _jaccard(set(q_classes), set(it.get("meta",{}).get("classes") or []))
        rec        = _recency_score(it.get("created_at",""))
        ppt_bonus  = 0.05 if it.get("has_ppt") else 0.0
        score = (0.45*sim_img + 0.20*sim_name + 0.15*sim_scale +
                 0.05*sim_cls + 0.05*rec + ppt_bonus + 0.05*0.0)
        it2 = dict(it); it2["score"]=float(score)
        it2["score_breakdown"] = {
            "image": float(sim_img), "name": float(sim_name), "scale": float(sim_scale),
            "classes": float(sim_cls), "tags": float(0.0), "recency": float(rec),
            "ppt_bonus": float(ppt_bonus),
        }
        scored.append(it2)

    if not scored: return []

    LAMBDA = 0.65
    def _pair_similarity(a: Dict, b: Dict) -> float:
        ah = a.get("hashes",{}).get("merge_phash")
        bh = b.get("hashes",{}).get("merge_phash")
        sim_img = _sim_phash(ah, bh) if (ah and bh) else 0.0
        ja = set(a.get("name_tokens",[])); jb=set(b.get("name_tokens",[]))
        sim_name = (len(ja & jb)/max(1,len(ja|jb))) if (ja or jb) else 0.0
        return 0.7*sim_img + 0.3*sim_name

    scored.sort(key=lambda x: x["score"], reverse=True)
    selected=[scored[0]]
    Remaining = scored[1:]
    while len(selected) < min(k, len(scored)) and Remaining:
        best=None; best_mmr=-1e9
        for cand in Remaining:
            rel = cand["score"]
            if not selected:
                mmr = rel
            else:
                max_sim = max(_pair_similarity(cand, s) for s in selected)
                mmr = LAMBDA*rel - (1.0-LAMBDA)*max_sim
            if mmr > best_mmr:
                best_mmr = mmr; best=cand
        if best is None: break
        selected.append(best)
        Remaining = [x for x in Remaining if x is not best]
    return selected[:k]

# ---------------- GPT-OSS 보조 유틸 (안전 호출 & 안전 파싱) ----------------
def _build_gptoss():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url=GPTOSS_BASE_URL, openai_proxy=GPTOSS_BASE_URL, model=GPTOSS_MODEL,
        default_headers={
            "Content-Type":"application/json",
            "x-dep-ticket":GPTOSS_X_DEP_TICKET,
            "Send-System-Name":GPTOSS_SEND_SYSTEM_NAME,
            "User-Id":GPTOSS_USER_ID,
            "User-Type":GPTOSS_USER_TYPE,
            "Prompt-Msg-Id": str(uuid.uuid4()),
            "Completion-Msg-Id": str(uuid.uuid4()),
        },
        temperature=0.1, max_tokens=4096, timeout=240, max_retries=1,
    )

def _extract_text_from_aimessage(resp_obj: Any) -> str:
    if resp_obj is None:
        return ""
    if isinstance(resp_obj, str):
        return resp_obj
    c = getattr(resp_obj, "content", None)
    if isinstance(c, str) and c.strip():
        return c
    if isinstance(c, list):
        try:
            texts = []
            for blk in c:
                if isinstance(blk, dict):
                    t = blk.get("text") or blk.get("content") or ""
                    if isinstance(t, str) and t.strip():
                        texts.append(t)
            if texts:
                return "\n".join(texts)
        except Exception:
            pass
        if c:
            return "\n".join(str(x) for x in c if x)
    ak = getattr(resp_obj, "additional_kwargs", {}) or {}
    if isinstance(ak, dict):
        fc = ak.get("function_call")
        if fc and isinstance(fc, dict):
            arg = fc.get("arguments")
            if isinstance(arg, str) and arg.strip():
                return arg
        tcs = ak.get("tool_calls") or []
        if isinstance(tcs, list):
            chunks=[]
            for tc in tcs:
                if isinstance(tc, dict):
                    f = tc.get("function") or {}
                    arg = f.get("arguments")
                    if isinstance(arg, str) and arg.strip():
                        chunks.append(arg)
            if chunks:
                return "\n".join(chunks)
    return str(c or "")

def _strip_code_fence(code: str) -> str:
    if code.startswith("```"):
        code = re.sub(r"^```[a-zA-Z0-9]*\s*", "", code, flags=re.DOTALL).strip()
        code = re.sub(r"\s*```$", "", code, flags=re.DOTALL).strip()
    return code

_GUARD_HEADER = (
    "중요 규칙:\n"
    "1) overlay.png는 반드시 'mask' 좌표계(입력 tif) 위에 그린다. merge 위에 그리지 말 것.\n"
    "2) measurements.csv는 다음 표준 헤더와 순서를 강제한다:\n"
    "   [measure_item,group_id,index,value_nm,sx,sy,ex,ey,meta_tag,component_label,image_name,run_id,note]\n"
    "3) STRUCTURED_DIFF의 count/clip/border 규칙을 위반하지 말 것.\n"
)

# [KEEP] Green Line의 '끝점' 힌트를 위한 '끝점 주변 샘플링' 함수
def _sample_class_near_point(mask_img: np.ndarray, x: int, y: int, radius: int = 15) -> Optional[int]:
    """(x,y) 좌표 주변 (radius*2+1)x(radius*2+1) 영역을 샘플링하여 가장 흔한 non-zero 클래스 반환"""
    if mask_img is None:
        return None
    try:
        H, W = mask_img.shape
        # 좌표 범위 클리핑
        x_min = max(0, x - radius)
        y_min = max(0, y - radius)
        x_max = min(W, x + radius + 1)
        y_max = min(H, y + radius + 1)

        if x_min >= x_max or y_min >= y_max:
            return None # 영역이 없음

        # 영역 추출 및 샘플링
        patch = mask_img[y_min:y_max, x_min:x_max]
        non_zero_samples = patch[patch > 0]
        
        if non_zero_samples.size > 0:
            # 가장 빈번하게 나타나는 클래스 값 반환
            counts = np.bincount(non_zero_samples)
            return int(np.argmax(counts))
        return None
    except Exception:
        return None # 샘플링 실패
    
# [ADD or RESTORE] Red Line의 '위치' 힌트를 위한 '선 전체 샘플링' 함수
def _sample_class_at_line(mask_img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[int]:
    """cv2.line_iterator를 사용해 라인 위의 픽셀을 샘플링하고, 가장 흔한 non-zero 클래스 반환"""
    if mask_img is None:
        return None
    try:
        # Bresenham's line algorithm
        num_points = int(np.hypot(x2 - x1, y2 - y1))
        if num_points == 0:
            # 점 하나만 샘플링
            if 0 <= y1 < mask_img.shape[0] and 0 <= x1 < mask_img.shape[1]:
                val = mask_img[y1, x1]
                return int(val) if val > 0 else None
            return None
        
        x_coords = np.linspace(x1, x2, num_points).astype(int)
        y_coords = np.linspace(y1, y2, num_points).astype(int)
        
        # 유효한 좌표만 필터링
        H, W = mask_img.shape
        valid_idx = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)
        if not np.any(valid_idx):
            return None
            
        x_coords, y_coords = x_coords[valid_idx], y_coords[valid_idx]
        
        # 픽셀 값 샘플링 (0이 아닌 값만)
        samples = mask_img[y_coords, x_coords]
        non_zero_samples = samples[samples > 0]
        
        if non_zero_samples.size > 0:
            # 가장 빈번하게 나타나는 클래스 값 반환
            counts = np.bincount(non_zero_samples)
            return int(np.argmax(counts))
        return None # 0(배경) 또는 유효 샘플 없음
    except Exception:
        return None # 샘플링 실패
    
def _safe_invoke_gptoss_for_code(guide_text: str, fewshot_texts: List[str]) -> str:
    """
    - 시스템/유저 메시지로 안전하게 호출
    - 응답이 tool_calls/function_call 형태여도 텍스트를 최대한 복구
    - 코드펜스 제거, 빈 응답 시 한 번 재시도
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    llm = _build_gptoss()

    # [FIX] 시스템 프롬프트 수정: 'classes_present' 사용을 강제
    sys_msg = (
        "너는 이미지를 분석하는 파이썬 측정 스크립트 생성 전문가다. "
        "너의 유일한 임무는 '지시 프롬프트'에 서술된 **측정 로직(Logic)**을 "
        "`mask_path` 이미지 위에서 실행하는 **알고리즘 코드(예: cv2.findContours, cv2.PCACompute)**로 구현하는 것이다."
        "마크다운 코드펜스는 절대 사용하지 말 것.\\n\\n"
        "## 중요 규칙:\\n"
        "1. '지시 프롬프트'는 너의 **가장 중요한 명령**이다. (예: '수직 높이 측정' 로직)\\n"
        "2. '지시 프롬프트'에 포함된 **[MASK METADATA]** 블록을 반드시 확인하라.\\n"
        "3. 로직을 구현할 때, **절대 `class_val=10` 같은 임의의 값을 추측하지 마라.**\\n"
        "4. '지시 프롬프트'가 'class_val'을 명시했다면 그 값을 사용하고, 명시하지 않았다면 `[MASK METADATA]`의 'classes' 리스트(예: `[30, 50, 70]`)에서 '지시 프롬프트'의 '손가락' 같은 설명과 가장 일치하는 **올바른 `class_val`을 선택**하여 코드에 사용해야 한다.\\n"
        "5. 'few-shot 텍스트'는 **스타일과 구문 참조용**이며, '지시 프롬프트'와 충돌 시 무시해야 한다.\\n"
        "6. [STRUCTURED_DIFF HINT] 섹션은 좌표가 아닌 알고리즘 힌트이며, 실제 기하는 반드시 `mask_path` 데이터에서 재계산해야 한다."
    )
    
    # [CRITICAL FIX] 
    # 호출할 함수를 원본(load_pixel_scale_um)이 아닌,
    # 3/4-value 호환성을 보장하는 래퍼(__wrap_load_pixel_scale_um)로 변경합니다.
    user_msg = (
        "아래 지시 프롬프트와 few-shot 텍스트를 바탕으로 전체 파이썬 코드(measure.py)만 출력하세요.\\n"
        "- 필수 인자: --mask_path, --mask_merge_path, --meta_root, --out_dir\\n"
        "- out_dir에 overlay.png, measurements.csv 생성\\n"
        "- meta_utils.__wrap_load_pixel_scale_um(mask_path, meta_root)는 (umx, umy, classes, meta_path) 4개 값 반환으로 가정\\n" # <-- 수정됨
        "- draw_text 기본 False, line_thickness 기본 5 등 사용자 옵션 반영\\n"
        "- [경고] [STRUCTURED_DIFF HINT] 섹션은 좌표가 아닌 알고리즘 힌트이며, 실제 기하는 반드시 mask_path에서 복원하세요.\\n"
        "- 절대 설명 문장 없이, 순수 파이썬 코드만 출력\\n\\n"
        "=== 지시 프롬프트 (Primary Command: Implement this Logic + Data Hints) ===\\n"
        f"{guide_text}\\n\\n"
        "=== few-shot 텍스트 (Style Reference Only) ===\\n"
        + ("\\n\\n".join(fewshot_texts) if fewshot_texts else "(없음)")
    )

    resp = llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=user_msg)])
    code = _extract_text_from_aimessage(resp).strip()
    code = _strip_code_fence(code)

    if code.strip():
        return code

    # 1회 재시도
    retry_user = user_msg + "\\n\\n[중요] 지금 바로 '파이썬 코드 본문'만 출력하세요. 주석은 허용, 설명문 금지."
    resp2 = llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=retry_user)])
    code2 = _extract_text_from_aimessage(resp2).strip()
    return _strip_code_fence(code2)

# -----------------------------
# Scribble 요약(Qwen) + Fallback
# -----------------------------
def _qwen_mm_generate_chat(processor, model, pil_images, prompt: str) -> str:
    """
    Qwen2/3-VL 멀티모달 안전 호출:
    - chat 템플릿으로 input_ids 생성
    - images는 별도로 encode (features)
    """
    import torch
    messages = [{
        "role": "user",
        "content": [
            *({"type": "image", "image": im} for im in pil_images),
            {"type": "text", "text": prompt}
        ]
    }]
    # 1) 토큰 생성 (text ids)
    text_inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    # 2) 이미지 features
    vision_inputs = processor(images=pil_images, return_tensors="pt")

    # 디바이스 이동
    text_inputs = {k: v.to(model.device) for k, v in ({"input_ids": text_inputs}).items()}
    vision_inputs = {k: v.to(model.device) for k, v in vision_inputs.items()}

    # generate
    with torch.no_grad():
        out = model.generate(**text_inputs, **vision_inputs, max_new_tokens=256)
    return processor.batch_decode(out, skip_special_tokens=True)[0]

def _qwen_summarize_scribble(mask_path: Path, merge_path: Path, scribble_path: Path) -> Dict[str, Any]:
    """
    [CHANGED] 3개 이미지(Mask, Scribble, Merge)를 모두 로드하여 Qwen-VL에 전달합니다.
    - [NEW] 프롬프트를 수정하여 'JSON 형식'의 구조화된 분석을 강제함 (추론 속도 향상).
    - [NEW] max_new_tokens를 1024 -> 512로 줄임 (JSON 출력이 짧으므로).
    - [REMOVED] raw_text에서 count/endpoint를 파싱하던 레거시 로직 제거.
    """
    import re, json
    from PIL import Image

    # --- 3개 이미지 로드를 위한 PIL 헬퍼 함수 ---
    def _load_pil(p: Optional[Path], max_side: int = 1280) -> Optional[Image.Image]:
        if not p or not p.exists():
            return None
        try:
            img = Image.open(p).convert("RGB")
            w, h = img.size
            ms = max(w, h)
            if ms > max_side:
                scale = float(max_side) / ms
                img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            return img
        except Exception as e:
            log.warning(f"[QWEN] PIL load failed for {p}: {e}")
            return None
    # --- 헬퍼 함수 끝 ---

    pil_mask = _load_pil(mask_path)
    pil_scribble = _load_pil(scribble_path)
    pil_merge = _load_pil(merge_path)

    pil_images = [img for img in [pil_mask, pil_scribble, pil_merge] if img is not None]

    # 결과 골격
    sdiff: Dict[str, Any] = {
        "schema": "scribble_v2_lite",
        "red":   {"count": None, "lines": []}, # 카운트는 CV가 채울 예정
        "green": {"count": None, "lines": []}, # 카운트는 CV가 채울 예정
        "rules": {"green_on_red": True, "nudge_if_touching": True}, # 기본 규칙
        "notes": {
            "source": "qwen_json_analysis", # [CHANGED] 소스 이름 변경
            "mask_name": mask_path.name if mask_path else None,
            "merge_name": merge_path.name if merge_path else None,
            "scribble_name": scribble_path.name if scribble_path else None,
            "raw_text": None,
        }
    }

    # Qwen 호출
    raw_text = ""
    try:
        _lazy_load_qwen()
        if _qwen_pipe is not None:
            import torch
            processor, model, is_mm, meta = _qwen_pipe

            # [FIX] 1. 이미지 '데이터'가 아닌 '플레이스홀더'를 포함하는 messages 리스트 구성
            content = []
            prompt_parts = []
            
            # --- [PROMPT CHANGE START] ---
            prompt_parts.append("You are an expert vision analyst. Analyze the provided images:")
            
            img_idx_counter = 0
            if pil_mask:
                content.append({"type": "image"}) # <--- 플레이스홀더
                prompt_parts.append(f"- Image {img_idx_counter+1}: 'Mask' (segmentation input, e.g., class 10, 30, 50)")
                img_idx_counter += 1
            if pil_scribble:
                content.append({"type": "image"}) # <--- 플레이스홀더
                prompt_parts.append(f"- Image {img_idx_counter+1}: 'Scribble' (RED guides, GREEN measures)")
                img_idx_counter += 1
            if pil_merge:
                content.append({"type": "image"}) # <--- 플레이스홀더
                prompt_parts.append(f"- Image {img_idx_counter+1}: 'Merged Example' (desired output)")
                img_idx_counter += 1

            prompt_parts.append("\nAnalyze the 'Scribble' in context of the 'Mask'.")
            prompt_parts.append("Provide your analysis STRICTLY in the following JSON format. Do NOT add any other text, explanations, or 'thinking' steps.")
            
            # [NEW] JSON 구조화 프롬프트
            # Red Line의 'relationship_to_mask'가 Red Line의 의도를 파악하는 핵심
            json_prompt_structure = """
{
  "analysis_summary": "One-sentence summary of the measurement task.",
  "red_guides": [
    {
      "id": 1,
      "purpose": "Describe the guide line's purpose (e.g., 'Horizontal reference line').",
      "relationship_to_mask": "CRITICAL: Describe its alignment with 'Mask' classes (e.g., 'Fitted to the top-most points of class 10', 'Positioned on class 50', 'Spans class 10 and 30')."
    }
  ],
  "green_measures": [
    {
      "id": 1,
      "purpose": "Describe the measurement line's purpose (e.g., 'Vertical height measurement').",
      "measured_classes": "CRITICAL: Describe which mask class(es) this line appears to measure (e.g., 'Measures class 30', 'Measures from red guide to class 10')."
    }
  ]
}
"""
            prompt_parts.append(json_prompt_structure)
            # --- [PROMPT CHANGE END] ---

            user_text = "\n".join(prompt_parts)

            content.append({"type": "text", "text": user_text})
            messages = [{"role": "user", "content": content}]

            dev = model.device

            if is_mm and pil_images:
                # [FIX] 2. apply_chat_template을 호출하여 <|image_1|> 토큰이 포함된 '프롬프트 문자열' 생성
                prompt_string = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # [FIX] 3. processor를 '단일 호출' (텍스트 문자열 + 이미지 리스트)
                inputs = processor(
                    text=prompt_string,
                    images=pil_images,
                    return_tensors="pt"
                ).to(dev)
            else:
                # 텍스트 전용
                prompt_string = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = processor(
                    text=prompt_string,
                    images=None,
                    return_tensors="pt"
                ).to(dev)

            # [SPEEDUP] max_new_tokens=512로 수정 (JSON 출력은 짧음)
            gen_kwargs = dict(max_new_tokens=512)
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)

            # (이하 디코딩 및 파싱 로직)
            if hasattr(processor, "batch_decode"):
                raw_text = processor.batch_decode(out, skip_special_tokens=True)[0]
                
                # 프롬프트 텍스트 제거 (기존 로직)
                if raw_text.strip().startswith(prompt_string.strip()):
                     raw_text = raw_text.strip()[len(prompt_string.strip()):].strip()
                elif "ASSISTANT:" in raw_text:
                     raw_text = raw_text.split("ASSISTANT:")[-1].strip()
                elif raw_text.strip().startswith(user_text.strip()):
                     raw_text = raw_text.strip()[len(user_text.strip()):].strip()

                # [NEW] JSON이 아닌 불필요한 텍스트(e.g., "Here is the JSON:") 제거
                if raw_text.startswith("{") and raw_text.endswith("}"):
                    pass # Good JSON
                else:
                    # JSON 블록만 추출
                    m = re.search(r"\{.*\}", raw_text, re.DOTALL)
                    if m:
                        raw_text = m.group(0)
                        
                _log_qwen_raw(mask_path, raw_text)
            else:
                tok = getattr(processor, "tokenizer", None)
                raw_text = tok.decode(out[0], skip_special_tokens=True) if tok is not None else str(out)
        else:
            raw_text = "{\"error\": \"Qwen not available\"}"
    except Exception as e:
        log.exception(f"[QWEN] call failed: {e}")
        raw_text = f"(Qwen call failed: {e})"

    sdiff["notes"]["raw_text"] = (raw_text or "")

    # [REMOVED]
    # 기존의 raw_text에서 'count', 'endpoint' 등을 파싱하던 로직 (qwen_red_count, m_grn, epats 등)
    # 모두 제거합니다.
    # Qwen은 이제 '의미 분석(JSON)'만 담당하고,
    # '카운트/좌표'는 CV(_structured_from_scribble)가 담당합니다.

    return sdiff

def _qwen_describe(mask_path: Path, merge_path: Path, scribble_path: Optional[Path]) -> Dict[str, Any]:
    base = _structured_from_scribble(scribble_path)
    red = base.get("red_guides", []) if isinstance(base, dict) else []
    green = base.get("green_measures", []) if isinstance(base, dict) else []

    red_cnt = int(len(red))
    green_cnt = int(len(green))

    sdiff = {
        "version": "2",
        "canvas": "mask",
        "counts": {"red_expected": red_cnt, "green_expected": green_cnt},
        "red_guides": [
            {"x1": r["x1"], "y1": r["y1"], "x2": r["x2"], "y2": r["y2"], "tolerance_px": 4, "role": "guide"}
            for r in red
        ],
        "green_measures": [
            {"x1": g["x1"], "y1": g["y1"], "x2": g["x2"], "y2": g["y2"], "clip_to": {"class": [30,50,70]}, "border_exclude": True, "role": "distance"}
            for g in green
        ],
        "consistency_checks": {"enforce_counts": True, "enforce_clip": True, "enforce_border_exclude": True},
        "notes": "auto-assembled from scribble; Qwen may refine"
    }

    vision_text = ""
    if QWEN_ENABLE == "1":
        ok = _lazy_load_qwen()
        if ok and _qwen_pipe:
            try:
                from PIL import Image as _PIL
                processor, model = _qwen_pipe
                imgs = []
                for p in [mask_path, merge_path, scribble_path]:
                    if p and Path(p).exists():
                        imgs.append(_PIL.open(p).convert("RGB"))
                prompt = (
                    "You are a vision analyst. Compare images: [mask], [mask-merge], [scribble]. "
                    "Count red lines and green lines in scribble, describe their vertical/horizontal roles, "
                    "and state any measurement intent. Answer in concise Korean bullets within 6 lines."
                )
                inputs = processor(text=prompt, images=imgs, return_tensors="pt").to(model.device)
                out = model.generate(**inputs, max_new_tokens=180)
                vision_text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            except Exception as e:
                vision_text = f"(fallback) 빨강 {red_cnt}개, 초록 {green_cnt}개. 스크리블 라인 기반 측정 의도 추정."
        else:
            vision_text = f"(fallback) 빨강 {red_cnt}개, 초록 {green_cnt}개. 스크리블 라인 기반 측정 의도 추정."
    else:
        vision_text = f"(disabled) 빨강 {red_cnt}개, 초록 {green_cnt}개."
    return {"vision_text": vision_text, "sdiff_v2": sdiff}

def _as_scalar(x) -> float:
    import numpy as _np
    return float(_np.asarray(x).ravel()[0])

def _line_angle_deg(x1,y1,x2,y2) -> float:
    return float(np.degrees(np.arctan2((y2 - y1), (x2 - x1))))

def _line_length_px(x1,y1,x2,y2) -> float:
    return float(np.hypot(x2 - x1, y2 - y1))

def _dedup_lines(lines: List[Dict[str,int]], angle_tol=3.0, dist_tol=5.0) -> List[Dict[str,int]]:
    """
    비슷한 각도 & 시작/끝점이 가까운 선분 병합(간단 버전).
    """
    out=[]
    for ln in lines:
        ax = _line_angle_deg(ln["sx"],ln["sy"],ln["ex"],ln["ey"])
        merged=False
        for e in out:
            bx = _line_angle_deg(e["sx"],e["sy"],e["ex"],e["ey"])
            if abs(((ax-bx+180)%360)-180) <= angle_tol:
                d1 = _line_length_px(ln["sx"],ln["sy"], e["sx"],e["sy"])
                d2 = _line_length_px(ln["ex"],ln["ey"], e["ex"],e["ey"])
                if min(d1,d2) <= dist_tol:
                    # 단순: 더 긴 쪽으로 확장
                    cand = [ (e["sx"],e["sy"]), (e["ex"],e["ey"]), (ln["sx"],ln["sy"]), (ln["ex"],ln["ey"]) ]
                    xs = [c[0] for c in cand]; ys=[c[1] for c in cand]
                    e["sx"],e["sy"] = int(min(xs)), int(min(ys))
                    e["ex"],e["ey"] = int(max(xs)), int(max(ys))
                    merged=True
                    break
        if not merged:
            out.append(dict(ln))
    return out

# ----------------- VL(설명) 프롬프트 블록 -----------------
def _vl_available(model: str) -> bool:
    if model == "gemma":
        return bool(GEMMA_BASE_URL and GEMMA_MODEL and GEMMA_X_DEP_TICKET)
    return bool(LLAMA4_BASE_URL and LLAMA4_MODEL and LLAMA4_X_DEP_TICKET)

def _build_llm_for(model: str):
    from langchain_openai import ChatOpenAI
    if model == "gemma":
        return ChatOpenAI(
            base_url=GEMMA_BASE_URL, openai_proxy=GEMMA_BASE_URL, model=GEMMA_MODEL,
            default_headers={
                "Content-Type":"application/json",
                "x-dep-ticket":GEMMA_X_DEP_TICKET,
                "Send-System-Name":GEMMA_SEND_SYSTEM_NAME,
                "User-Id":GEMMA_USER_ID,
                "User-Type":GEMMA_USER_TYPE,
                "Prompt-Msg-Id": str(uuid.uuid4()),
                "Completion-Msg-Id": str(uuid.uuid4()),
            },
            temperature=0.2, max_tokens=4096, timeout=240, max_retries=1,
        )
    else:
        return ChatOpenAI(
            base_url=LLAMA4_BASE_URL, openai_proxy=LLAMA4_BASE_URL, model=LLAMA4_MODEL,
            default_headers={
                "Content-Type":"application/json",
                "x-dep-ticket":LLAMA4_X_DEP_TICKET,
                "Send-System-Name":LLAMA4_SEND_SYSTEM_NAME,
                "User-Id":LLAMA4_USER_ID,
                "User-Type":LLAMA4_USER_TYPE,
                "Prompt-Msg-Id": str(uuid.uuid4()),
                "Completion-Msg-Id": str(uuid.uuid4()),
            },
            temperature=0.2, max_tokens=4096, timeout=240, max_retries=1,
        )

# [REPLACE] _vl_prompt_blocks 함수
# [REPLACE] _vl_prompt_blocks 함수
def _vl_prompt_blocks(mask_path: Path, merge_path: Path, scribble_path: Optional[Path], ppt_path: Optional[Path], user_text: str, structured: Dict, fewshot_texts: List[str], meta_summary: Dict):
    """ 
    [CHANGED] 
    - Llama4 프롬프트에 '0) IMAGE SUMMARY' 섹션을 추가.
    - SDIFF.notes.raw_text (Qwen의 JSON 분석 결과)를 이 섹션에 주입.
    - sys_prompt를 수정하여 Llama4가 '0) SUMMARY'와 '[SDIFF]'를 조합하여 추론하도록 지시.
    """
    blocks=[]
    for label, p in [("INPUT(mask)", mask_path), ("EXPECTED(mask-merge)", merge_path)]:
        try: uri = _prep_for_vision_llm(p, 1280, 80)
        except Exception: continue
        blocks.append({"type":"text","text":label})
        blocks.append({"type":"image_url","image_url":{"url":uri}})
    if scribble_path and scribble_path.exists():
        try:
            uri = _prep_for_vision_llm(scribble_path, 1280, 80)
            blocks.append({"type":"text","text":"SCRIBBLE(red/green only)"})
            blocks.append({"type":"image_url","image_url":{"url":uri}})
        except Exception: pass
    if ppt_path and ppt_path.exists():
        try:
            uri = _prep_for_vision_llm(ppt_path, 1280, 80)
            blocks.append({"type":"text","text":"PPT capture"})
            blocks.append({"type":"image_url","image_url":{"url":uri}})
        except Exception: pass

    # --- [NEW] 0) IMAGE SUMMARY (Qwen 분석) 블록 추가 ---
    # structured 딕셔너리에서 notes.raw_text를 추출
    raw_text = (structured.get("notes") or {}).get("raw_text") or "(Qwen raw_text not found)"
    
    # 새 블록으로 추가
    blocks.append({
        "type": "text", 
        "text": "0) IMAGE SUMMARY (Qwen's Analysis)\n" + raw_text
    })
    # --- [NEW] 끝 ---

    # SDIFF(CV 사실)와 MASK METADATA(클래스 목록) 주입
    blocks.append({"type":"text","text":"[STRUCTURED_DIFF (CV Facts)]\n"+json.dumps(structured, ensure_ascii=False, indent=2)})
    blocks.append({"type":"text","text":"[MASK METADATA (Available Classes)]\n"+json.dumps(meta_summary, ensure_ascii=False, indent=2)})

    if fewshot_texts:
        blocks.append({"type":"text","text":"[FEW-SHOT TEXTS]\n"+"\\n\\n".join(fewshot_texts)})

    # [CHANGED] VLM(Llama4)의 시스템 프롬프트 (0번 섹션 인지하도록 수정)
    sys_prompt = (
        "너는 '코드 LLM에게 줄 지시 프롬프트'를 작성하는 비서다. "
        "너는 방금 '0) IMAGE SUMMARY' (Qwen 분석)와 '[STRUCTURED_DIFF (CV Facts)]'를 받았다." # [CHANGED]
        "다음 섹션 헤더를 반드시 그대로 사용: "
        "0) IMAGE SUMMARY\\n1) TASK SUMMARY\\n2) INPUTS\\n3) OUTPUTS\\n4) PIPELINE STEPS\\n"
        "5) PARAMETERS\\n6) CONSTRAINTS\\n7) PSEUDOCODE\\n8) E2E CHECKLIST\\n9) CODE STYLE\\n"
        "overlay.png/measurements.csv/μm 변환/CLI 인자/기본값을 명확히 하라.\\n\\n"
        
        "## [매우 중요] SDIFF/JSON 로드 절대 금지 (보안 규칙):\\n"
        "1. 너의 **유일한 임무**는 '0) IMAGE SUMMARY'와 '[STRUCTURED_DIFF]'를 **'해석'**하여, **'측정 알고리즘(PIPELINE STEPS)'**으로 변환하는 것이다.\\n" # [CHANGED]
        "2. 너의 최종 출력(PIPELINE STEPS)에는 **`SDIFF`, `.json`을 읽거나 로드하라는 지시를 **절대 포함하지 마라.**\\n"
        "3. 최종 코드는 SDIFF JSON 파일의 존재를 몰라야 하며, **반드시 `--mask_path`와 `meta_utils`만을 사용**하여 `cv2.findContours`, `cv2.inRange`, `np.where` 같은 순수 CV/Numpy 알고리즘으로 모든 기하학적 계산을 수행해야 한다.\\n"
        "5. (나쁜 예): '2. SDiff 데이터를 해석합니다.' -> [금지]\\n"
        "6. (좋은 예): '2. `class_val=10`인 마스크 영역을 `cv2.inRange`로 찾습니다.'\\n"
        
        "## [매우 중요] 측정 알고리즘 추론 규칙 (CV 힌트 우선):\\n"
        "### 1. RED 가이드라인 (SDIFF.red):\\n"
        "1. **(의도)** '0) IMAGE SUMMARY' (Qwen JSON)에서 `red_guides.relationship_to_mask` (예: 'Fitted to... class 10')를 확인하여 **'생성 기준 클래스'**를 파악하라.\\n" # [CHANGED]
        "2. **(사실)** `SDIFF.red.lines[i].detected_class_hint` (예: 50)를 확인하여 이 선의 **'실제 위치 클래스'**를 파악하라.\\n"
        "3. **(지시)** 'PIPELINE STEPS'에 이 두 정보를 **'알고리즘 로직'**으로 변환하라. (예: '`class_val=10`의 최상단 점들을 찾아 `class 50` 영역에 가이드라인을 그린다.')\\n"

        "### 2. GREEN 측정선 (SDIFF.green):\\n"
        "1. **(사실)** `SDIFF.green.lines` 목록을 분석하라.\\n"
        "2. **(사실)** `detected_class_hint_start` (예: 50)와 `detected_class_hint_end` (예: 10)를 확인하라.\\n"
        "3. **(사실)** `paired_red_id` (예: 'red_1')와 `SDIFF.red.lines[0].detected_class_hint` (예: 50)를 확인하라.\\n"
        "4. **(추론)** 만약 Green의 `start_hint` (50)가 Red의 `hint` (50)와 *일치*하고 `paired_red_id`가 존재하면, 이 측정은 **'위에서 계산한 Red 가이드라인'에서 시작**한 것이다.\\n"
        "5. **(추론)** 만약 Green의 `start_hint` (예: 10)가 Red의 `hint` (50)와 *불일치*하거나 `paired_red_id`가 없다면, 이 측정은 **'Mask Class 10'에서 시작**한 것이다.\\n"
        "6. **(지시)** 'PIPELINE STEPS'에 이 추론 결과를 **'알고리즘 로직'**으로 변환하라. (예: 'red_1 가이드라인에서 class 10 영역까지 측정', 또는 'class 10 영역에서 class 30 영역까지 측정')"
    )
    # [변경 끝]
    
    blocks = [{"type":"text","text":_GUARD_HEADER+"\\n\\n"+sys_prompt}] + blocks
    blocks.append({"type":"text","text": user_text})
    return blocks
# -----------------------------
# VL (Llama4/Gemma3) — 4장 입력 + STRUCTURED_DIFF
# -----------------------------
@app.post("/qwen/describe")
async def qwen_describe(
    selected_name: str = Form(...),
):
    try:
        normalized = _normalize_name(selected_name)
        merge_path = (IMAGE_DIR/normalized).resolve()
        if not merge_path.exists():
            return JSONResponse({"ok": False, "error": "merge not found"}, status_code=200)
        mask_name = _mask_name_from_merge(normalized)
        mask_path = (Path(IMAGE_DIR).parent/"mask"/mask_name).resolve() if mask_name else None
        if not mask_path or not mask_path.exists():
            return JSONResponse({"ok": False, "error": "mask not found"}, status_code=200)
        scribble_path = _scribble_path_from_merge(normalized)

        out = _qwen_describe(mask_path, merge_path, scribble_path)
        STRUCTURED_DIFF_CACHE[selected_name] = out.get("sdiff_v2", {})
        return JSONResponse({"ok": True, **out})
    except Exception as e:
        log.exception("[qwen/describe] failed")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=200)

# [기존 app_codecheck.py 파일 내용 중 vl_describe4 함수 부분만 찾아서 교체]

@app.post("/vl/describe4")
async def vl_describe4(
    vl_model: str = Form("llama4"),
    selected_name: str = Form(...),
    prompt: str = Form("이미지를 상세히 한글로 설명해주세요."),
    image: UploadFile = File(None)
):
    """
    [CHANGED] Llama4 설명 생성 API.
    - SDIFF 재생성 로직을 '완전히 제거'했습니다.
    - 이 API는 더 이상 SDIFF를 덮어쓰지 않고, 'STRUCTURED_DIFF_CACHE'를 읽기만 합니다.
    - SDIFF 생성은 /vl/sdiff_qwen API가 전담합니다.
    """
    try:
        if not _vl_available(vl_model):
            return JSONResponse({"ok": False, "error": f"{vl_model} not available"}, status_code=200)

        # --- 입력 경로 준비 ---
        normalized = _normalize_name(selected_name)
        stem = Path(normalized).stem
        merge_path = (IMAGE_DIR/normalized).resolve()
        if not merge_path.exists():
            return JSONResponse({"ok": False, "error": "merge not found"}, status_code=200)

        mask_name = _mask_name_from_merge(normalized)
        mask_path = (Path(IMAGE_DIR).parent/"mask"/mask_name).resolve() if mask_name else None
        if not mask_path or not mask_path.exists():
            log.warning(f"Mask not found for {selected_name}, proceeding without it.")
            mask_path = None # None으로 설정하고 계속 진행
        
        scribble_path = _scribble_path_for(normalized)
        
        # (PPT 로직 ...)
        ppt_path = None
        ppt_id = None
        if image:
            tmp_dir = BASE_DIR/"ppt_uploads"; _ensure_dir(tmp_dir)
            ext = Path(image.filename or "ppt.png").suffix or ".png"
            ppt_path = tmp_dir / f"ppt_{uuid.uuid4().hex}{ext}"
            ppt_path.write_bytes(await image.read())
            ppt_id = ppt_path.name

        llm = _build_llm_for(vl_model)

        # --- [NEW] 1. 픽셀 스케일 *및* 클래스 정보 조회 (Llama4 프롬프트용)
        meta_sum = {}
        if mask_path:
            try:
                meta_sum = _meta_summary(mask_path)
            except Exception:
                meta_sum = {}

        # --- [REFACTORED] SDIFF '읽기' ---
        # SDIFF (재)생성 로직을 모두 제거합니다.
        # 오직 메모리 캐시에서 읽어오기만 합니다.
        # /vl/sdiff_qwen API가 SDIFF 생성을 미리 수행했어야 합니다.
        structured = STRUCTURED_DIFF_CACHE.get(selected_name)
        if not structured:
            log.warning(f"[vl/describe4] SDIFF not found in cache for '{selected_name}'. Using empty SDIFF.")
            structured = {} # 캐시에 없으면 빈 객체 사용 (재생성 금지)
        
        # --- Few-shot 텍스트(슬림) & VL 블록 빌드 ---
        few_texts = _fewshot_block_text_only(selected_name, topk=2, slim=True)
        blocks = _vl_prompt_blocks(mask_path, merge_path, scribble_path, ppt_path, prompt, structured, few_texts, meta_summary=meta_sum)

        # --- SDIFF 요약/상세 프리펜드 ---
        try:
            sdiff_summary = _sdiff_to_summary_text(structured) if structured else "[SDIFF SUMMARY]\n- (none)"
            sdiff_detail  = _sdiff_to_verbose_text(structured, max_list=200) if structured else "[SDIFF LINES DETAIL]\n- (none)"
        except Exception:
            sdiff_summary = "[SDIFF SUMMARY]\n- (build failed)"
            sdiff_detail  = "[SDIFF LINES DETAIL]\n- (build failed)"
        preface = PROMPT_GUARD_HEAD + "\\n" + sdiff_summary + "\\n\\n" + sdiff_detail + "\\n\\n"

        if isinstance(blocks, list):
            blocks = [{"type":"text","text": preface}] + blocks
        elif isinstance(blocks, str):
            blocks = preface + blocks
        else:
            blocks = preface + str(blocks)

        # --- (2단계) Llama4/Gemma3 입력 프롬프트 전체 로깅 ---
        if os.getenv("DEBUG_GPTOSS_PROMPT", "1") == "1":
            try:
                debug_path = (RUN_DIR / stem / "vl_describe.debug.txt")
                debug_content = []
                debug_content.append(f"[vl_describe4 log @ {time.strftime('%Y-%m-%d %H:%M:%S')}]\\n")
                debug_content.append("--- SDIFF Read from Cache ---\\n") # 'Received' -> 'Read from Cache'
                debug_content.append(json.dumps(structured, ensure_ascii=False, indent=2))
                debug_content.append("\\n\\n--- Preface (for text blocks) ---\\n")
                debug_content.append(preface)
                debug_content.append("\\n\\n--- Prompt Blocks (text only, no truncation) ---\\n")
                for i, block in enumerate(blocks):
                    if block.get("type") == "text":
                        debug_content.append(f"[Block {i}: Text]\\n{block.get('text', '')}\\n")
                debug_path.write_text("\\n".join(debug_content), encoding="utf-8")
            except Exception as e:
                log.warning(f"[vl_describe4] debug log failed: {e}")

        # --- LLM 호출 ---
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content=blocks)])
        text = (getattr(resp, "content", "") or "").strip()

        # --- Few-shot UI 메타 ---
        used=[]
        sels = _select_topk_fewshots_for(selected_name, k=2)
        for it in sels:
            item_id = it["id"]; d = Path(it["_dir"])
            parts=[]
            for nm in ("mask.jpg","merge.jpg","ppt.jpg"):
                if (d/nm).exists():
                    parts.append({"name": nm, "url": f"/fewshot/asset?item_id={quote(item_id, safe='')}&name={quote(nm, safe='')}"})
            used.append({
                "id": item_id, "score": it["score"], "score_breakdown": it.get("score_breakdown", {}),
                "meta": it.get("meta", {}), "parts": parts, "created_at": it.get("created_at")
            })

        # 이 API는 더 이상 SDIFF를 덮어쓰지 않으므로, 캐시에서 읽은 'structured'를 그대로 반환합니다.
        return JSONResponse({"ok": True, "text": text, "ppt_id": ppt_id, "used_fewshots": used, "structured_diff": structured})
    except Exception as e:
        log.exception("[vl/describe4] failed")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=200)

# [기존 app_codecheck.py 파일 내용 중 vl_continue4 함수 부분만 찾아서 교체]

@app.post("/vl/continue4")
async def vl_continue4(
    vl_model: str = Form("llama4"),
    selected_name: str = Form(...),
    prev_text: str = Form(...),
    user_followup: str = Form("이어서 보강 및 수정해 주세요."),
    ppt_id: str = Form(None)
):
    try:
        if not _vl_available(vl_model):
            return JSONResponse({"ok": False, "error": f"{vl_model} not available"}, status_code=200)

        normalized = _normalize_name(selected_name)
        merge_path = (IMAGE_DIR/normalized).resolve()
        mask_name = _mask_name_from_merge(normalized)
        mask_path = (Path(IMAGE_DIR).parent/"mask"/mask_name).resolve() if mask_name else None
        if not merge_path.exists() or not mask_path or not mask_path.exists():
            return JSONResponse({"ok": False, "error": "inputs not found"}, status_code=200)

        scribble_path = _scribble_path_for(normalized)

        ppt_path = (BASE_DIR/"ppt_uploads"/ppt_id).resolve() if ppt_id else None
        if ppt_path and not ppt_path.exists():
            ppt_path=None

        llm = _build_llm_for(vl_model)

        # [NEW] 메타데이터(클래스 정보) 로드
        meta_sum = {}
        if mask_path:
            try:
                meta_sum = _meta_summary(mask_path)
            except Exception:
                meta_sum = {}
        
        structured = STRUCTURED_DIFF_CACHE.get(selected_name) or _structured_from_scribble(scribble_path)
        few_texts = _fewshot_block_text_only(selected_name, topk=2, slim=True)

        follow_text = f"[후속 요청]\n{user_followup}\n\n이전 초안을 중복 없이 보강/수정하라."
        base_text = follow_text + "\n\n[이전 초안]\n" + (prev_text or "")

        # [FIX] meta_sum 전달
        blocks = _vl_prompt_blocks(mask_path, merge_path, scribble_path, ppt_path, base_text, structured, few_texts, meta_summary=meta_sum)

        try:
            sdiff_summary = _sdiff_to_summary_text(structured) if structured else "[SDIFF SUMMARY]\n- (none)"
            sdiff_detail  = _sdiff_to_verbose_text(structured, max_list=200) if structured else "[SDIFF LINES DETAIL]\n- (none)"
        except Exception:
            sdiff_summary = "[SDIFF SUMMARY]\n- (build failed)"
            sdiff_detail  = "[SDIFF LINES DETAIL]\n- (build failed)"
        preface = PROMPT_GUARD_HEAD + "\n" + sdiff_summary + "\n\n" + sdiff_detail + "\n\n"
        if isinstance(blocks, list):
            blocks = [{"type":"text","text": preface}] + blocks
        else:
            blocks = preface + (blocks if isinstance(blocks, str) else str(blocks))

        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content=blocks)])
        text = (getattr(resp, "content", "") or "").strip()

        used=[]
        sels = _select_topk_fewshots_for(selected_name, k=2)
        for it in sels:
            item_id = it["id"]; d = Path(it["_dir"])
            parts=[]
            for nm in ("mask.jpg","merge.jpg","ppt.jpg"):
                if (d/nm).exists():
                    parts.append({"name": nm, "url": f"/fewshot/asset?item_id={quote(item_id, safe='')}&name={quote(nm, safe='')}"})
            used.append({
                "id": item_id, "score": it["score"], "score_breakdown": it.get("score_breakdown", {}),
                "meta": it.get("meta", {}), "parts": parts, "created_at": it.get("created_at")
            })

        return JSONResponse({"ok": True, "text": text, "used_fewshots": used, "structured_diff": structured})
    except Exception as e:
        log.exception("[vl/continue4] failed")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=200)
    
# ------------------------------------------
# [NEW] SDIFF 캐시/로딩 헬퍼
# ------------------------------------------
def _load_latest_structured_diff_for(image_stem: str) -> Dict:
    """
    우선 메모리 캐시 → 없으면 scribble에서 즉시 추출 → 그래도 없으면 {}
    """
    # 캐시에 파일명 기준으로 저장되므로 stem 또는 풀네임 모두 검사
    for k, v in STRUCTURED_DIFF_CACHE.items():
        if Path(k).stem == image_stem or k == image_stem:
            return v or {}
    # 파일명으로부터 scribble 추론
    # MERGE 파일명 복원 시도 (stem + '_gray_merge.png' 관례 우선)
    candidate_merge = IMAGE_DIR / f"{image_stem}_gray_merge.png"
    merge_path = candidate_merge if candidate_merge.exists() else None
    if not merge_path:
        # fallback: IMAGE_DIR에서 stem이 포함된 첫 파일
        for p in IMAGE_DIR.iterdir():
            if p.is_file() and p.stem == image_stem:
                merge_path = p; break
    if merge_path:
        scribble_path = _scribble_path_for(merge_path.name)
        if scribble_path.exists():
            # mask도 추적
            mn = _mask_name_from_merge(merge_path.name)
            mask_path = (Path(IMAGE_DIR).parent/"mask"/mn) if mn else None
            return _qwen_summarize_scribble(mask_path, merge_path, scribble_path)
    return {}

# ------------------------------------------
# [NEW] Qwen으로 STRUCTURED_DIFF 생성
# ------------------------------------------
@app.post("/vl/sdiff_qwen")
async def vl_sdiff_qwen(data: Dict = Body(...)):
    """
    [CHANGED] CV(Geometry) + Qwen(Semantics) 하이브리드 SDIFF 생성
    1.  [NEW] 픽셀 스케일(meta)을 먼저 조회
    2.  OpenCV 기반(_structured_from_scribble)으로 픽셀 스케일을 전달하여 정확한 기하학/의미 정보 추출
    3.  Qwen 기반(_qwen_summarize_scribble)으로 '의미' 설명(raw_text) 추출
    4.  두 정보를 병합하여 가장 신뢰도 높은 SDIFF를 생성
    """
    try:
        name = (data or {}).get("image_name")
        if not name:
            return JSONResponse({"ok": False, "error": "image_name required"})
        merge_path = (IMAGE_DIR / name).resolve()

        mask_name = _mask_name_from_merge(name)
        mask_path = (Path(IMAGE_DIR).parent/"mask"/mask_name).resolve() if mask_name else None

        scribble_path = _scribble_path_for(name)

        if not merge_path.exists():
            return JSONResponse({"ok": False, "error": f"merge not found: {merge_path}"})
        if not scribble_path.exists():
            return JSONResponse({"ok": False, "error": f"scribble not found: {scribble_path}"})
        
        # [FIX] 마스크 경로는 null일 수 있지만, 존재하지 않으면 오류 대신 None으로 처리
        if mask_path and not mask_path.exists():
            log.warning(f"Mask file not found at {mask_path}, proceeding without it.")
            mask_path = None # CV 힌트가 null이 되더라도 Qwen 분석은 계속 진행

        # 1. [NEW] 픽셀 스케일 조회 (의미 추론용)
        um_per_px_x = 0.0
        if mask_path:
            try:
                meta_sum = _meta_summary(mask_path)
                um_per_px_x = meta_sum.get("um_per_px_x") or 0.0
            except Exception:
                um_per_px_x = 0.0

        # 2. [CV] OpenCV 기반으로 정확한 기하학 정보(좌표, 각도, 개수, 의미) 추출
        # [CRITICAL FIX] 
        # _structured_from_scribble 함수에 'mask_path'를 전달하여
        # CV 힌트(detected_class_hint)가 'null'이 되지 않도록 수정합니다.
        sdiff_final = _structured_from_scribble(
            scribble_path, 
            um_per_px_x=um_per_px_x, 
            mask_path=mask_path # <--- 이 인자가 누락되어 SDIFF 힌트가 null이 되었습니다.
        )
        
        if not isinstance(sdiff_final, dict):
            sdiff_final = {} # 실패 시 뼈대만이라도 확보
        sdiff_final.setdefault("notes", {})

        # 3. [Qwen] VLM으로 '의미' 설명(raw_text) 추출
        try:
            _lazy_load_qwen()
            # [FIX] mask_path가 None일 경우에도 Qwen이 실패하지 않도록 처리
            sdiff_qwen = _qwen_summarize_scribble(mask_path, merge_path, scribble_path)

            qwen_raw_text = (sdiff_qwen.get("notes") or {}).get("raw_text")
            if qwen_raw_text:
                sdiff_final["notes"]["raw_text"] = qwen_raw_text

            sdiff_final["notes"]["source"] = "hybrid_cv_geometry_qwen_notes"
            sdiff_final["notes"]["mask_name"] = mask_path.name if mask_path else None
            sdiff_final["notes"]["merge_name"] = merge_path.name
            sdiff_final["notes"]["scribble_name"] = scribble_path.name

        except Exception as qe:
            log.warning(f"Qwen semantic enrichment failed, using CV-only: {qe}")
            sdiff_final["notes"]["source"] = "cv_geometry_only (qwen_failed)"

        # 4. v1 -> v2_lite로 업그레이드 시도 (이전 답변의 _upgrade_sdiff_to_lite 함수가 수정됨)
        try:
            sdiff_final = _upgrade_sdiff_to_lite(sdiff_final)
        except Exception as ue:
            log.warning(f"SDIFF v2_lite upgrade failed: {ue}")

        STRUCTURED_DIFF_CACHE[name] = sdiff_final
        return JSONResponse({"ok": True, "structured_diff": sdiff_final})

    except Exception as e:
        log.exception("/vl/sdiff_qwen error: %s", e)
        return JSONResponse({"ok": False, "error": str(e)})

# ---------- few-shot 자산 ----------
@app.get("/fewshot/select")
async def fewshot_select(name: str = Query(...), k: int = Query(2)):
    try:
        items = _select_topk_fewshots_for(name, k=k)
        return JSONResponse({"ok": True, "items": items})
    except Exception as e:
        log.exception("[fewshot/select] failed")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/fewshot/asset")
async def fewshot_asset(item_id: str = Query(...), name: str = Query(...)):
    safe = {"merge.jpg","overlay.jpg","ppt.jpg","mask.jpg"}
    if name not in safe: return JSONResponse({"error":"invalid asset name"}, status_code=400)
    p = (FEWSHOT_DIR / item_id / name).resolve()
    if not p.exists() or not p.is_file() or FEWSHOT_DIR not in p.parents:
        return JSONResponse({"error":"asset not found"}, status_code=404)
    return FileResponse(p)

# ------------------------------------------
# [CHANGED] Top-1 few-shot 코드 (요청 이미지 기준)
# ------------------------------------------
@app.get("/fewshot/top1_code")
async def fewshot_top1_code(image_name: str = Query(...)):
    try:
        sels = _select_topk_fewshots_for(image_name, k=1)
        if not sels:
            return JSONResponse({"ok": False, "error": "no few-shot candidate found"})
        best = Path(sels[0]["_dir"]) / "code.py"
        if not best.exists():
            return JSONResponse({"ok": False, "error": "few-shot has no code.py"})
        code = best.read_text(encoding="utf-8", errors="ignore")
        return JSONResponse({"ok": True, "code": code, "item_id": sels[0]["id"], "score": sels[0]["score"]})
    except Exception as e:
        log.exception("/fewshot/top1_code error: %s", e)
        return JSONResponse({"ok": False, "error": str(e)})

# -----------------------------
# GPT-OSS (코드 생성/수정 대화) — 프롬프트
# -----------------------------
def _gptoss_prompt_for_code(guide_text: str, fewshot_texts: List[str]) -> str:
    # [CRITICAL FIX] 
    # 호출할 함수를 래퍼(__wrap_load_pixel_scale_um)로 변경합니다.
    return (
        _GUARD_HEADER + "\\n\\n" +
        "아래의 지시 프롬프트와 few-shot 텍스트를 바탕으로, 완전한 파이썬 측정 스크립트(measure.py)를 출력하세요.\\n"
        "- 실행 인자: --mask_path, --mask_merge_path, --meta_root, --out_dir (필수)\\n"
        "- overlay.png와 measurements.csv를 out_dir에 생성\\n"
        "- meta_utils.__wrap_load_pixel_scale_um(mask_path, meta_root)는 (umx,umy,classes,meta_path) 4개 값 반환 가정\\n" # <-- 수정됨
        "- draw_text 기본 False, 선 두께 기본 5\\n"
        "- 절대 마크다운 코드펜스 없이 순수 파이썬 코드만 출력\\n\\n"
        "=== 지시 프롬프트 ===\\n"
        f"{guide_text}\\n\\n"
        "=== few-shot 텍스트 ===\\n"
        + ("\\n\\n".join(fewshot_texts) if fewshot_texts else "(없음)")
    )

def _gptoss_prompt_for_fix(current_code: str, error_text: str) -> str:
    return (
        "다음 파이썬 스크립트(measure.py)를 실행했더니 에러가 발생했습니다.\n"
        "에러 메시지의 원인을 분석하고, 전체 스크립트를 수정하여 재출력해 주세요.\n"
        "- 기존 기능은 절대 깨지면 안 됩니다.\n"
        "- overlay.png와 measurements.csv를 out_dir에 출력해야 합니다.\n"
        "- meta_utils.load_pixel_scale_um(mask_path, meta_root)는 (umx,umy,classes,meta_path) 4개 값 반환 사용\n"
        "- 절대 마크다운 코드펜스 없이 순수 파이썬 코드만 출력\n\n"
        "=== 현재 코드 ===\n"
        f"{current_code}\n\n"
        "=== 이번 에러 메시지 ===\n"
        f"{error_text}\n"
    )
    
def _load_pil_resized_keep_scale(p: Path, max_side: int):
    """이미지를 max_side로 다운스케일하여 PIL Image와 (sx, sy) 스케일을 반환.
    원본(W,H) -> 축소(w',h'), scale = (w'/W, h'/H). 파일 없으면 (None, (1,1))."""
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        return None, (1.0, 1.0)
    H, W = bgr.shape[:2]
    ms = max(H, W)
    if ms > max_side:
        s = float(max_side) / ms
        bgr = cv2.resize(bgr, (max(1, int(W*s)), max(1, int(H*s))), interpolation=cv2.INTER_AREA)
        sx, sy = (bgr.shape[1]/W, bgr.shape[0]/H)
    else:
        sx, sy = (1.0, 1.0)
    pil = PIL.Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return pil, (sx, sy)


def _rescale_endpoints_to_original(endpoints, scale_xy):
    """[[x1,y1],[x2,y2]]를 원본 좌표로 역스케일. scale_xy=(sx,sy)는 축소비율(<=1)."""
    try:
        (sx, sy) = scale_xy
        def _rr(v, s, maxv=None):
            if s <= 0: return int(v)
            out = int(round(float(v) / s))
            if maxv is not None:
                out = max(0, min(out, maxv-1))
            return out
        (x1,y1),(x2,y2) = endpoints
        return [[_rr(x1, sx), _rr(y1, sy)], [_rr(x2, sx), _rr(y2, sy)]]
    except Exception:
        return endpoints

def _reject_merge_copying(code: str) -> str | None:
    """
    'merge 픽셀 복사'를 유발하는 위험 패턴을 정규식으로 차단.
    발견 시 설명 문자열을 반환(없으면 None).
    """
    import re
    patterns = [
        r'cv2\.imread\([^)]*mask_merge',            # merge 이미지를 직접 읽음
        r'\binRange\s*\([^)]*(0\s*,\s*255\s*,\s*0|0\s*,\s*0\s*,\s*255)',  # 색상 inRange
        r'==\s*\(\s*0\s*,\s*255\s*,\s*0\s*\)',      # 녹색(BGR) 등가 비교
        r'==\s*\(\s*0\s*,\s*0\s*,\s*255\s*\)',      # 빨강(BGR) 등가 비교
        r'\bextract_lines\s*\(.*(merge|mask_merge).*?\)',  # merge 기반 라인 추출 함수 호출
        r'\bfindContours\s*\([^)]*merge[^)]*\)',    # merge 마스크에서 윤곽선 추출 시도
    ]
    for pat in patterns:
        if re.search(pat, code):
            return f"Forbidden pattern detected: {pat}"
    return None


# [REPLACE] _upgrade_sdiff_to_lite 함수
def _upgrade_sdiff_to_lite(sdiff: dict | None) -> dict | None:
    """
    [CHANGED] v1 → v2_lite 업그레이드.
    - [NEW] 'red'에는 `detected_class_hint` (위치) 필드를 전달.
    - [NEW] 'green'에는 `detected_class_hint_start`와 `detected_class_hint_end` (양쪽 위치) 필드를 전달.
    """
    if not sdiff or not isinstance(sdiff, dict):
        return sdiff
    schema = sdiff.get("schema") or ""

    # 이미 v2_lite 형식이면 추가 작업 없이 반환
    if schema == "scribble_v2_lite":
        # 만약 v2_lite에도 힌트가 이미 있다면 (또는 새 형식이라면) 그대로 반환
        if sdiff.get("green",{}).get("lines",[]):
            first_green = sdiff["green"]["lines"][0]
            # 새 형식을 사용 중인지(start/end) 확인
            if "detected_class_hint_start" in first_green and "detected_class_hint_end" in first_green:
                 return sdiff
        # (v2_lite지만 힌트가 없다면 v1 필드가 남아있길 기대하며 v1->v2 변환 로직 실행)
        pass

    # 기본 뼈대
    out = {
        "schema": "scribble_v2_lite",
        "red":   {"count": 0, "lines": []},
        "green": {"count": 0, "lines": []},
        "rules": sdiff.get("rules", {"green_on_red": True, "nudge_if_touching": True}), # 규칙 보존
        "notes": dict(sdiff.get("notes", {}))
    }

    # CV가 `_structured_from_scribble`에서 생성한 *상세* 목록을 가져옴
    raw_red_lines = sdiff.get("red_lines_detail") or []
    raw_green_lines = sdiff.get("green_lines_detail") or []

    # --- 1. RED (guide) 라인 처리 ---
    lite_red_lines = []
    for ln in raw_red_lines:
        if not isinstance(ln, dict): continue
        lite_red_lines.append({
            "id": ln.get("id"),
            "role": "guide",
            "orientation": ln.get("orientation"),
            "angle_deg": ln.get("angle_deg"),
            "length_px": ln.get("length_px"),
            "endpoints": [ln.get("x1"), ln.get("y1"), ln.get("x2"), ln.get("y2")],
            # [NEW] Red line의 '위치' 힌트 복사
            "detected_class_hint": ln.get("detected_class_hint") 
        })

    # --- 2. GREEN (measure) 라인 처리 ---
    lite_green_lines = []
    for ln in raw_green_lines:
        if not isinstance(ln, dict): continue
        lite_green_lines.append({
            "id": ln.get("id"),
            "role": "measure",
            "orientation": ln.get("orientation"),
            "angle_deg": ln.get("angle_deg"),
            "length_px": ln.get("length_px"),
            "semantic": ln.get("semantic"), 
            "paired_red_id": ln.get("paired_red_id"), # (e.g., "red_1")
            "endpoints": [ln.get("x1"), ln.get("y1"), ln.get("x2"), ln.get("y2")],
            # [RESTORED] Green line의 '양쪽 위치' 힌트 복사
            "detected_class_hint_start": ln.get("detected_class_hint_start"),
            "detected_class_hint_end": ln.get("detected_class_hint_end")
        })

    # --- 3. 최종 SDIFF v2_lite 객체 생성 ---
    out["red"]["lines"] = lite_red_lines
    out["green"]["lines"] = lite_green_lines
    
    out["red"]["count"] = len(lite_red_lines)
    out["green"]["count"] = len(lite_green_lines)

    return out

# -------- CSV 표준화/정합성 검사 (run_dir 단위) --------
_STD_COLS = ["measure_item","group_id","index","value_nm","sx","sy","ex","ey","meta_tag","component_label","image_name","run_id","note"]

def _normalize_measurements_csv(run_dir: "Path", image_name: str):
    """
    measurements.csv 존재/비어있음 방어 + 표준화.
    실패해도 (ok=False, msg) 반환만 하고 예외는 올리지 않는다.
    """
    try:
        from pathlib import Path as _P
        import pandas as pd
        csv_path = _P(run_dir) / "measurements.csv"
        if not csv_path.exists():
            return False, "csv not found"
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return False, f"csv read failed: {e}"
        if df is None or df.empty:
            return False, "csv empty"

        # 표준 헤더 강제(없으면 추가)
        want_cols = ["measure_item","group_id","index","value_nm",
                     "sx","sy","ex","ey","meta_tag","component_label",
                     "image_name","run_id","note"]
        for c in want_cols:
            if c not in df.columns:
                df[c] = "" if c not in ("value_nm","index","group_id") else 0
        df = df[want_cols]
        df.to_csv(csv_path, index=False)
        return True, "normalized"
    except Exception as e:
        return False, f"normalize failed: {e}"
    

# 과거 이름으로도 호출되는 경우 대비(존재 시 교체하지 않아도 됨)
def normalize_measurements_csv(csv_path: "Path"):
    """
    (구버전 시그니처 호환) csv_path가 파일 전체 경로로 넘어오는 경우를 처리.
    """
    try:
        import pandas as pd
        from pathlib import Path as _P
        csv_path = _P(csv_path)
        if not csv_path.exists():
            return False, "csv not found"
        df = pd.read_csv(csv_path)
        if df is None or df.empty:
            return False, "csv empty"
        want_cols = ["measure_item","group_id","index","value_nm",
                     "sx","sy","ex","ey","meta_tag","component_label",
                     "image_name","run_id","note"]
        for c in want_cols:
            if c not in df.columns:
                df[c] = "" if c not in ("value_nm","index","group_id") else 0
        df = df[want_cols]
        df.to_csv(csv_path, index=False)
        return True, "normalized"
    except Exception as e:
        return False, f"normalize failed: {e}"
    
# 원래 버전
"""
def normalize_measurements_csv(csv_path: Path):
    try:
        import pandas as pd
        if not csv_path.exists():
            df = pd.read_csv(csv_path)
        for col in STD_CSV_HEADERS:
            if col not in df.columns:
                df[col] = "" if col not in ("sx","sy","ex","ey","index","value_nm") else 0
        cols = [c for c in STD_CSV_HEADERS] + [c for c in df.columns if c not in STD_CSV_HEADERS]
        df = df[cols]
        df.to_csv(csv_path, index=False)
    except Exception as e:
        log.warning("normalize_measurements_csv failed: %s", e)"""
    
def _log_qwen_raw(mask_path: Path, text: str):
    try:
        stem = Path(mask_path).stem if mask_path else "unknown"
        d = (RUN_DIR / stem)
        d.mkdir(parents=True, exist_ok=True)
        (d / "qwen_raw.txt").write_text(text or "", encoding="utf-8")
    except Exception as e:
        log.warning("[QWEN] failed to write raw: %s", e)

# -----------------------------
# ------- 실행 로그/코드 조회/상태 API (변경 없음)
# -----------------------------
@app.get("/run/log")
async def get_run_log(name: str = Query(..., description="merge 파일명")):
    try:
        normalized = _normalize_name(name)
        stem = Path(normalized).stem
        p = RUN_DIR / stem / "run.log"
        if not p.exists():
            cand = None
            root = RUN_DIR / stem
            if root.exists():
                logs = list(root.rglob("run.log"))
                if logs:
                    cand = max(logs, key=lambda x: x.stat().st_mtime)
            if cand is None:
                return JSONResponse({"ok": False, "error": "run.log not found"}, status_code=200)
            p = cand
        text = p.read_text(encoding="utf-8", errors="ignore")
        return JSONResponse({"ok": True, "text": text, "path": str(p)})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=200)

@app.get("/run/status")
async def run_status(name: str = Query(...)):
    stem = Path(_normalize_name(name)).stem
    s = RUN_STATUS.get(stem)
    if not s:
        return JSONResponse({"ok": False})
    return JSONResponse({"ok": True, **s})

@app.get("/code/get")
async def code_get(image_name: str = Query(...)):
    try:
        normalized = _normalize_name(image_name)
        py_name = _code_label_name(normalized)
        p_label = (LABEL_DIR / py_name).resolve()
        if p_label.exists():
            return JSONResponse({"ok": True, "filename": py_name, "source": "label", "code": p_label.read_text(encoding="utf-8", errors="ignore")})

        stem = Path(normalized).stem
        p_run = RUN_DIR / stem / "measure.py"
        if p_run.exists():
            return JSONResponse({"ok": True, "filename": str(p_run.name), "source": "run", "code": p_run.read_text(encoding="utf-8", errors="ignore")})

        root = RUN_DIR / stem
        if root.exists():
            ms = list(root.rglob("measure.py"))
            if ms:
                mlatest = max(ms, key=lambda x: x.stat().st_mtime)
                return JSONResponse({"ok": True, "filename": str(mlatest), "source": "run", "code": mlatest.read_text(encoding="utf-8", errors="ignore")})

        return JSONResponse({"ok": False, "error": "no code found"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})

# -----------------------------
# VL 설명 엔드포인트(위에서 정의) 이후 — 코드 실행/저장/되돌리기
# -----------------------------
@app.post("/code/run")
async def code_run(payload: dict = Body(...)):
    image_name = payload.get("image_name")
    code_text  = payload.get("code","")
    auto_fix   = bool(payload.get("auto_fix", True))
    max_fixes  = int(payload.get("max_fixes", 5))
    if not image_name or not code_text:
        return JSONResponse({"error":"image_name and code are required"}, status_code=400)

    mask_merge_path = IMAGE_DIR / _normalize_name(image_name)
    if not mask_merge_path.exists():
        return JSONResponse({"error": f"mask_merge image not found: {image_name}"}, status_code=404)

    mask_name = _mask_name_from_merge(_normalize_name(image_name))
    if not mask_name:
        return JSONResponse({"error":"Mask name rule failed"}, status_code=400)
    mask_path = Path(IMAGE_DIR).parent / "mask" / mask_name
    if not mask_path.exists():
        return JSONResponse({"error": f"mask image not found: {mask_name}", "searched": str(mask_path)}, status_code=404)

    run_dir = _run_dir_for(image_name); _ensure_dir(run_dir)
    measure_py = run_dir / "measure.py"
    measure_py.write_text(code_text, encoding="utf-8")

    # meta_utils.py 준비 (기존 유지)
    meta_utils_dest = run_dir / "meta_utils.py"
    candidates = []
    envp = os.getenv("META_UTILS_PATH")
    if envp: candidates.append(Path(envp))
    candidates += [BASE_DIR/"meta_utils.py", Path(IMAGE_DIR).parent/"meta_utils.py", Path.cwd()/"meta_utils.py"]
    picked = next((c for c in candidates if c.exists() and c.is_file()), None)
    if picked:
        meta_utils_dest.write_text(picked.read_text(encoding="utf-8"), encoding="utf-8")
        txt = meta_utils_dest.read_text(encoding="utf-8")
        if "def __wrap_load_pixel_scale_um" not in txt:
            txt += """

# --- compatibility wrapper: (mask_path, meta_root) -> (umx,umy,classes,meta_path)
def __wrap_load_pixel_scale_um(mask_path: str, meta_root: str):
    res = load_pixel_scale_um(mask_path, meta_root)
    # 다양한 구현 호환(3~4개)
    if isinstance(res, (tuple,list)):
        if len(res)==4:
            return res[0],res[1],res[2],res[3]
        if len(res)==3:
            return res[0],res[1],res[2], find_meta_path(mask_path, meta_root)
    raise RuntimeError("load_pixel_scale_um incompatible return")
"""
            meta_utils_dest.write_text(txt, encoding="utf-8")
    else:
        meta_utils_dest.write_text(f'''# -*- coding: utf-8 -*-
import json, os
from pathlib import Path
from typing import Optional, List, Tuple
DEFAULT_META_ROOT = r"{str(META_ROOT)}"
def find_meta_path(mask_path: str, meta_root: str = DEFAULT_META_ROOT) -> Path:
    base = mask_path.split("_gray.tif")[0] + ".json"
    return Path(meta_root) / os.path.basename(base)
def load_pixel_scale_um(mask_path: str, meta_root: str = DEFAULT_META_ROOT):
    mp = find_meta_path(mask_path, meta_root)
    if not mp.exists(): raise FileNotFoundError(f"Meta JSON not found: {mp}")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    if "pixel_scale_um_x" not in meta or "pixel_scale_um_y" not in meta:
        raise KeyError("pixel_scale_um_x / pixel_scale_um_y not found in meta JSON")
    classes=None
    if "mask_info" in meta and "classes_present" in meta["mask_info"]:
        classes = meta["mask_info"]["classes_present"]
    return float(meta["pixel_scale_um_x"]), float(meta["pixel_scale_um_y"]), classes
# wrapper
def __wrap_load_pixel_scale_um(mask_path: str, meta_root: str):
    res = load_pixel_scale_um(mask_path, meta_root)
    if isinstance(res,(tuple,list)) and len(res)==3:
        from pathlib import Path as _P
        def find_meta_path(mask_path, meta_root):
            base = mask_path.split("_gray.tif")[0] + ".json"
            return _P(meta_root) / os.path.basename(base)
        return res[0],res[1],res[2], find_meta_path(mask_path, meta_root)
    return res
''', encoding="utf-8")

    overlay_path = _overlay_path_for(image_name)
    if overlay_path.exists():
        try: overlay_path.unlink()
        except Exception: pass

    stem = Path(_normalize_name(image_name)).stem

    # --- [NEW] Ensure structured-diff JSON exists next to the merge image ---
    try:
        sdiff_path = mask_merge_path.with_suffix(".json")

        need_write = True
        if sdiff_path.exists():
            try:
                j = json.loads(sdiff_path.read_text(encoding="utf-8"))
                need_write = False if isinstance(j, dict) and j else True
            except Exception:
                need_write = True

        if need_write:
            sdiff = None

            # (a) 캐시 우선
            try:
                sdiff = STRUCTURED_DIFF_CACHE.get(image_name)
            except Exception:
                sdiff = None

            # (b) 없으면 Qwen 또는 색 임계값 기반으로 생성
            if not isinstance(sdiff, dict) or not sdiff:
                try:
                    scr_path = _scribble_path_for(_normalize_name(image_name))
                    if scr_path and scr_path.exists():
                        try:
                            _lazy_load_qwen()
                        except Exception:
                            pass
                        if (_qwen_pipe is not None):
                            sdiff = _qwen_summarize_scribble(mask_path, mask_merge_path, scr_path)
                        else:
                            sdiff = _structured_from_scribble(scr_path)
                    else:
                        sdiff = {
                            "schema":"scribble_v1",
                            "counts":{"red":0,"green":0,"helper":0},
                            "red":[], "green":[], "helper":[],
                            "notes":{
                                "source":"server_autofill_no_scribble",
                                "mask_name": mask_path.name if mask_path else None,
                                "merge_name": mask_merge_path.name
                            }
                        }
                except Exception as _e:
                    sdiff = {
                        "schema":"scribble_v1",
                        "counts":{"red":0,"green":0,"helper":0},
                        "red":[], "green":[], "helper":[],
                        "notes":{
                            "source":f"server_autofill_error:{_e}",
                            "mask_name": mask_path.name if mask_path else None,
                            "merge_name": mask_merge_path.name
                        }
                    }

            # 캐시 동기화 & 메타
            try:
                sdiff.setdefault("notes", {})
                sdiff["notes"]["_server_injected"] = True
                sdiff["notes"]["_expected_path"] = str(sdiff_path)
            except Exception:
                pass

            # 최종 저장 (merge와 같은 폴더)
            sdiff_path.write_text(json.dumps(sdiff, ensure_ascii=False, indent=2), encoding="utf-8")

            # 캐시에도 반영
            try:
                STRUCTURED_DIFF_CACHE[image_name] = sdiff
            except Exception:
                pass

    except Exception as e:
        log.warning(f"[run] failed to ensure structured diff JSON: {e}")

    def _exec_once():
        cmd = [
            _python_executable(), str(measure_py),
            "--mask_path", str(mask_path),
            "--mask_merge_path", str(mask_merge_path),
            "--out_dir", str(run_dir),
            "--meta_root", str(META_ROOT),
        ]
        log.info(f"[RUN] {' '.join(cmd)} (cwd={run_dir})")
        try:
            proc = subprocess.run(cmd, cwd=str(run_dir), capture_output=True, text=True, timeout=int(os.getenv("MEASURE_TIMEOUT_SEC","300")))
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            return 124, "", f"Timeout: {e}"
        except Exception as e:
            return 125, "", f"Exec failed: {e}"

    _set_status(stem, phase="exec", attempt=0, progress=10, label="실행 중…")

    rc, stdout, stderr = _exec_once()

    try:
        log_path = _run_dir_for(image_name) / "run.log"
        with open(log_path, "w", encoding="utf-8", newline="") as f:
            # run.log 기록 직후, 단일 CSV 정리도 시도
            try:
                out_csv = _run_dir_for(image_name) / "measurements.csv"
                normalize_measurements_csv(out_csv)  # (기존 호출 그대로 유지)
            except Exception as _e:
                log.warning(f"[run] csv normalize failed: {_e}")

            f.write(f"[cmd] {_python_executable()} {measure_py.name} --mask_path ...\n")
            f.write(f"[returncode] {rc}\n\n[stdout]\n${{stdout}}\n\n[stderr]\n${{stderr}}\n".replace("${stdout}", stdout or "").replace("${stderr}", stderr or ""))
    except Exception as e:
        log.warning(f"[run] failed to write run.log: {e}")

    exists = (_run_dir_for(image_name) / "overlay.png").exists() or False

    # [NEW] measurements.csv 표준화 시도(성공/실패와 무관)
    norm_ok, norm_msg = _normalize_measurements_csv(_run_dir_for(image_name), image_name)

    if rc==0 and exists:
        _set_status(stem, phase="done", attempt=0, progress=100, label="완료")
        _clear_status(stem)
        return JSONResponse({
            "status": "ok", "returncode": rc, "stdout": stdout, "stderr": stderr,
            "overlay_exists": exists, "overlay_url": f"/overlay?name={quote(image_name, safe='')}",
            "csv_normalized": norm_ok, "csv_note": norm_msg
        })

    # 자동 수정 루프
    auto_log = []
    auto_fixed = False
    if auto_fix:
        llm = _build_gptoss()
        auto_log.append({"attempt": 0, "returncode": rc, "stdout": stdout, "stderr": stderr})
        for i in range(1, max_fixes+1):
            _set_status(stem, phase="fix", attempt=i, progress=10+int(80*(i/(max_fixes+1))), label=f"에러 {i} 해결 중")
            cur_code = measure_py.read_text(encoding="utf-8", errors="ignore")
            fix_prompt = _gptoss_prompt_for_fix(cur_code, (stderr or stdout or f"returncode {rc}"))
            try:
                resp = llm.invoke(fix_prompt)
                new_code = _extract_text_from_aimessage(resp).strip()
                new_code = _strip_code_fence(new_code)
                if not new_code.strip():
                    auto_log.append({"attempt": i, "fix_error": "LLM returned empty"})
                    continue
                measure_py.write_text(new_code, encoding="utf-8")
            except Exception as e:
                auto_log.append({"attempt": i, "fix_error": str(e)})
                continue

            rc2, stdout2, stderr2 = _exec_once()
            try:
                with open(_run_dir_for(image_name)/"run.log","a",encoding="utf-8",newline="") as f:
                    f.write(f"\n[auto-fix attempt {i}] returncode={rc2}\n\n[stdout]\n{stdout2 or ''}\n\n[stderr]\n{stderr2 or ''}\n")
            except Exception:
                pass

            exists2 = (_run_dir_for(image_name) / "overlay.png").exists() or False
            _normalize_measurements_csv(_run_dir_for(image_name), image_name)

            auto_log.append({"attempt": i, "returncode": rc2, "stdout": stdout2, "stderr": stderr2, "overlay": exists2})
            if rc2==0 and exists2:
                auto_fixed=True
                rc, stdout, stderr = rc2, stdout2, stderr2
                exists = exists2
                break

        _set_status(stem, phase="final", attempt=len(auto_log)-1, progress=100, label="완료")
        _clear_status(stem)

    overlay_url = f"/overlay?name={quote(image_name, safe='')}" if exists else None
    return JSONResponse({
        "status": "ok" if (rc==0 and exists) else "error",
        "returncode": rc, "stdout": stdout, "stderr": stderr,
        "overlay_exists": exists, "overlay_url": overlay_url,
        "auto_fixed": auto_fixed,
        "first_returncode": auto_log[0]["returncode"] if auto_log else None,
        "first_stdout": auto_log[0].get("stdout") if auto_log else None,
        "first_stderr": auto_log[0].get("stderr") if auto_log else None
    })

def _dump_debug(stem: str, fname: str, content: str):
    """RUN_DIR/<stem>/<fname>에 안전하게 디버그 텍스트를 기록."""
    try:
        d = RUN_DIR / stem
        d.mkdir(parents=True, exist_ok=True)
        p = d / fname
        # 이전 내용이 있다면 이어붙임
        if p.exists():
            prev = p.read_text(encoding="utf-8", errors="ignore")
        else:
            prev = ""
        p.write_text(prev + content, encoding="utf-8")
    except Exception as e:
        log.warning(f"[debug] write failed {fname}: {e}")

# [NEW] SDIFF -> GPT-OSS 전용 Invoker
def _safe_invoke_gptoss_for_code_from_sdiff(guide_text_with_sdiff: str, fewshot_texts: List[str]) -> str:
    """
    SDIFF를 '힌트'로 직접 받는 GPT-OSS 호출기.
    - '좌표 복사'를 방지하는 강력한 시스템 프롬프트를 사용합니다.
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    llm = _build_gptoss()

    # [CRITICAL] SDIFF 직접 주입용 시스템 프롬프트 (방화벽)
    sys_msg = (
        "너는 이미지를 분석하는 파이썬 측정 스크립트 생성 전문가다. "
        "너의 유일한 임무는 '지시 프롬프트'(guide_text)에 서술된 **[STRUCTURED_DIFF HINT]**를 **'해석'**하여, "
        "`--mask_path` 이미지 위에서 실행하는 **알고리즘 코드(예: cv2.findContours, np.where)**로 구현하는 것이다.\\n\\n"
        
        "## [매우 중요] SDIFF 사용 규칙 (보안):\\n"
        "1. [STRUCTURED_DIFF HINT]는 **'시맨틱 힌트'**로만 제공된다.\\n"
        "2. 너는 `detected_class_hint_start/end`, `paired_red_id`, `semantic` 같은 **'힌트'**만 참조해야 한다.\\n"
        "3. **`endpoints`, `length_px`, `angle_deg`** 같은 기하학적 정보는 **절대로 사용해서는 안 된다.** (치팅 금지)\\n"
        "4. 너의 임무는 이 '힌트'를 바탕으로 `cv2.findContours`, `np.where` 등을 사용해 기하학적 정보를 **'처음부터 재계산(re-calculate)'**하는 것이다.\\n"
        "5. 'Few-Shot 코드'에 `load_sdiff`가 있더라도 무시하고, 이 규칙을 최우선으로 하라.\\n"
        "6. 이 [STRUCTURED_DIFF HINT] 섹션은 좌표가 아닌 알고리즘 힌트이며, 실제 기하는 반드시 `mask_path`에서 복원해야 한다.\\n\\n"

        "## [매우 중요] 클래스 값(class_val) 처리 규칙:\\n"
        "1. '지시 프롬프트'의 **[MASK METADATA]** 블록을 반드시 확인하라.\\n"
        "2. `SDIFF`의 힌트(예: 10, 30, 50)가 `[MASK METADATA]`의 'classes' 리스트에 있는지 확인하고, **올바른 `class_val`을 선택**하여 코드에 사용해야 한다."
    )
    
    # User message (guide_text_with_sdiff는 SDIFF JSON + MASK METADATA를 포함)
    user_msg = (
        "아래 지시 프롬프트와 few-shot 텍스트를 바탕으로 전체 파이썬 코드(measure.py)만 출력하세요.\\n"
        "- 필수 인자: --mask_path, --mask_merge_path, --meta_root, --out_dir\\n"
        "- out_dir에 overlay.png, measurements.csv 생성\\n"
        "- meta_utils.__wrap_load_pixel_scale_um(mask_path, meta_root)는 (umx, umy, classes, meta_path) 4개 값 반환으로 가정\\n"
        "- draw_text 기본 False, line_thickness 기본 5 등 사용자 옵션 반영\\n"
        "- [경고] [STRUCTURED_DIFF HINT] 섹션은 좌표가 아닌 알고리즘 힌트이며, 실제 기하는 반드시 mask_path에서 복원하세요.\\n"
        "- 절대 설명 문장 없이, 순수 파이썬 코드만 출력\\n\\n"
        "=== 지시 프롬프트 (SDIFF HINTS + METADATA) ===\\n"
        f"{guide_text_with_sdiff}\\n\\n"
        "=== few-shot 텍스트 (Style Reference Only) ===\\n"
        + ("\\n\\n".join(fewshot_texts) if fewshot_texts else "(없음)")
    )

    resp = llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=user_msg)])
    code = _extract_text_from_aimessage(resp).strip()
    code = _strip_code_fence(code)

    if code.strip():
        return code

    # 1회 재시도
    retry_user = user_msg + "\\n\\n[중요] 지금 바로 '파이썬 코드 본문'만 출력하세요. 주석은 허용, 설명문 금지."
    resp2 = llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=retry_user)])
    code2 = _extract_text_from_aimessage(resp2).strip()
    return _strip_code_fence(code2)
        
# [NEW] SDIFF -> GPT-OSS 직접 생성 엔드포인트
@app.post("/gptoss/generate_from_sdiff")
async def gptoss_generate_from_sdiff(payload: dict = Body(...)):
    """
    [NEW WORKFLOW] Llama4를 건너뛰고 SDIFF를 GPT-OSS에 '힌트'로 직접 전달합니다.
    - SDIFF의 `endpoints` 등 기하 정보는 '무시'하고
    - SDIFF의 `class_hints` 등 '시맨틱' 정보만 사용하도록 프롬프트 엔지니어링
    """
    try:
        import json, os
        from pathlib import Path

        image_name = payload.get("image_name")
        # [NEW] Llama4 프롬프트 대신 SDIFF(json 텍스트)를 받음
        structured_diff_text = payload.get("structured_diff_text", "") 

        if not image_name or not structured_diff_text:
            return JSONResponse({"ok": False, "error": "image_name and structured_diff_text required"}, status_code=200)

        stem = Path(_normalize_name(image_name)).stem

        # --- 1. `meta_summary` (클래스 정보) 재조회 (Failsafe용) ---
        meta_sum = {}
        if image_name:
            try:
                mask_name = _mask_name_from_merge(_normalize_name(image_name))
                if mask_name:
                    mask_path = (Path(IMAGE_DIR).parent/"mask"/mask_name).resolve()
                    if mask_path.exists():
                         meta_sum = _meta_summary(mask_path)
            except Exception as e:
                log.warning(f"Failed to re-fetch meta for GPT-OSS: {e}")

        # --- 2. Top-1 Few-shot '전체 코드' 로드 (스타일 참조용) ---
        sels = []
        fewshot_texts = []
        try:
            sels = _select_topk_fewshots_for(image_name, k=1)
            if sels:
                best_dir = Path(sels[0]["_dir"])
                code_path = best_dir / "code.py"
                if code_path.exists():
                    full_code_str = code_path.read_text(encoding="utf-8", errors="ignore")
                    preamble = (
                        "아래는 가장 유사한 측정 코드 예시입니다. "
                        "이 스타일과 구조를 참고하여 현재 요청에 맞는 코드를 작성하세요: "
                        "(단, 이 예제에 `load_sdiff`나 `json.load`가 있더라도 절대 따라하지 마세요.)"
                    )
                    fewshot_texts = [f"{preamble}\n\n```python\n{full_code_str}\n```"]
        except Exception as e:
            log.warning(f"[gptoss_generate_sdiff] fewshot load failed: {e}")

        # --- 3. [NEW] SDIFF + MASK METADATA 결합 ---
        full_prompt_list = []
        
        # (A) SDIFF 힌트 주입
        full_prompt_list.append("Generate a Python script based on the [STRUCTURED_DIFF] hint below.")
        full_prompt_list.append("Your code MUST be a traditional OpenCV/Numpy algorithm (e.g., cv2.findContours, np.where).")
        full_prompt_list.append("\n## [STRUCTURED_DIFF HINT]\n")
        full_prompt_list.append(structured_diff_text) # SDIFF JSON 텍스트

        # (B) MASK METADATA 주입 (Failsafe 힌트)
        if meta_sum:
            try:
                meta_json_str = json.dumps(meta_sum, ensure_ascii=False, indent=2)
                full_prompt_list.append("\n\n### MASK METADATA (Data/Hints for Algorithm)\n")
                full_prompt_list.append("IMPORTANT: Use the 'classes' list (e.g., [10, 30, 50]) to select the correct `class_val` for your algorithm.")
                full_prompt_list.append(meta_json_str)
            except Exception as e:
                log.warning(f"Failed to serialize Meta Summary for prompt: {e}")

        full_prompt = "\n".join(full_prompt_list)
        
        # --- 4. 로깅 ---
        if os.getenv("DEBUG_GPTOSS_PROMPT", "1") == "1":
            try:
                # (로그 파일명을 다르게 하여 충돌 방지)
                debug_path = (RUN_DIR / stem / "gptoss_gen_sdiff.debug.txt") 
                debug_content = []
                debug_content.append(f"[gptoss_generate_from_sdiff log @ {time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
                debug_content.append("--- 1. SDIFF Hint + MASK METADATA (Combined) ---\n")
                debug_content.append(full_prompt)
                debug_content.append("\n\n--- 2. Few-Shot Code (Injected) ---\n")
                if fewshot_texts:
                    debug_content.append(fewshot_texts[0])
                else:
                    debug_content.append("(No few-shot code injected)")
                debug_path.write_text("\n".join(debug_content), encoding="utf-8")
            except Exception as e:
                log.warning(f"[gptoss_generate_sdiff] debug log failed: {e}")

        # --- 5. [NEW] 전용 Invoker 호출 ---
        code = _safe_invoke_gptoss_for_code_from_sdiff(full_prompt, fewshot_texts).strip()
        code = _strip_code_fence(code)

        # (이하 로깅 및 반환은 gptoss_generate와 동일)
        if os.getenv("DEBUG_GPTOSS_PROMPT", "1") == "1":
            _dump_debug(
                stem,
                "gptoss_gen_sdiff.debug.txt", # 로그 파일명 통일
                "\n--- 3. Model Output (Head) ---\n" + (code[:4000] if code else "(empty)") + "\n"
            )

        if not code.strip():
            return JSONResponse({"ok": False, "error": "LLM returned empty code."}, status_code=200)

        # (치팅/복사 방지 룰은 동일하게 적용)
        violation = _reject_merge_copying(code)
        if violation:
            # (violation 피드백 및 재시도 로직은 gptoss_generate와 동일하게 유지)
            feedback = (
                "You violated NO-COLOR-COPY rule: " + violation + "\n"
                "Rewrite the entire script STRICTLY following the SDIFF HINTS and MASK METADATA:\n"
                "- Use ONLY `mask_path` and implement the logic (PCA/contours).\n"
                "- Use the correct 'class_val' from MASK METADATA and SDIFF HINTS.\n"
            )
            full_prompt_2 = full_prompt + "\n\n[FEEDBACK]\n" + feedback + "\n"
            code2 = _safe_invoke_gptoss_for_code_from_sdiff(full_prompt_2, fewshot_texts).strip()
            code2 = _strip_code_fence(code2)

            if code2.strip():
                violation2 = _reject_merge_copying(code2)
                if not violation2:
                    code = code2 # 재시도 성공 시 코드 교체
            
            # (재시도 후에도 위반 시, 원본 코드를 반환)

        return JSONResponse({"ok": True, "code": code})
    except Exception as e:
        log.exception("[gptoss/generate_from_sdiff] failed")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=200)

@app.post("/gptoss/generate")
async def gptoss_generate(payload: dict = Body(...)):
    """
    - [CHANGED] Llama4가 요약한 'guide_text'와 SDIFF의 'raw_text'를 조합하여
    -           '0) IMAGE SUMMARY' (Qwen 원본) + 'Llama4 요약본' (Llama4)
    -           형태의 최종 프롬프트를 구성하여 GPT-OSS에 전달합니다.
    """
    try:
        import json, os
        from pathlib import Path

        image_name = payload.get("image_name")
        guide_text = payload.get("prompt_text", "") # (Llama4의 요약본)
        
        # [NEW] 프론트엔드에서 SDIFF 객체 전체를 받음
        structured_diff = payload.get("structured_diff") 

        if not image_name or not guide_text:
            return JSONResponse({"ok": False, "error": "image_name and prompt_text(Llama4 output) required"}, status_code=200)

        stem = Path(_normalize_name(image_name)).stem

        # --- [REMOVED] 'Sanitizer' 로직 제거 ---
        # (Llama4의 요약본을 그대로 사용할 것이므로 Sanitizer는 더 이상 필요 없습니다.)
        
        # --- [NEW] 1. Qwen 원본(raw_text) 추출 ---
        qwen_raw_text = ""
        if isinstance(structured_diff, dict):
            qwen_raw_text = (structured_diff.get("notes") or {}).get("raw_text") or ""
        
        if not qwen_raw_text:
            log.warning("[gptoss_generate] SDIFF.notes.raw_text not found in payload. Using Llama4 summary only.")
            qwen_raw_text = "(Qwen raw_text not available)"


        # --- 2. `meta_summary` (클래스 정보) 재조회 (Failsafe용) ---
        meta_sum = {}
        if image_name:
            try:
                mask_name = _mask_name_from_merge(_normalize_name(image_name))
                if mask_name:
                    mask_path = (Path(IMAGE_DIR).parent/"mask"/mask_name).resolve()
                    if mask_path.exists():
                         meta_sum = _meta_summary(mask_path)
            except Exception as e:
                log.warning(f"Failed to re-fetch meta for GPT-OSS: {e}")

        # --- 3. Top-1 Few-shot '전체 코드' 로드 (스타일 참조용) ---
        sels = []
        fewshot_texts = []
        try:
            sels = _select_topk_fewshots_for(image_name, k=1)
            if sels:
                best_dir = Path(sels[0]["_dir"])
                code_path = best_dir / "code.py"
                if code_path.exists():
                    full_code_str = code_path.read_text(encoding="utf-8", errors="ignore")
                    preamble = (
                        "아래는 가장 유사한 측정 코드 예시입니다. "
                        "이 스타일과 구조를 참고하여 현재 요청에 맞는 코드를 작성하세요: "
                        "(단, 이 예제에 `load_sdiff`나 `json.load`가 있더라도 절대 따라하지 마세요.)" 
                    )
                    fewshot_texts = [f"{preamble}\n\n```python\n{full_code_str}\n```"]
        except Exception as e:
            log.warning(f"[gptoss_generate] fewshot load failed: {e}")

        structured_hint_body = ""
        if isinstance(structured_diff, dict):
            try:
                structured_hint_body = _structured_diff_semantic_summary(structured_diff).strip()
            except Exception as e:
                log.warning(f"[gptoss_generate] structured diff summary failed: {e}")
                structured_hint_body = ""

        # --- [CRITICAL FIX] 4. Qwen 원본 + SDIFF 힌트 + Llama4 요약본 + Meta 결합 ---
        full_prompt_list = []

        # (A) [NEW] Qwen 원본(raw_text)을 '0) IMAGE SUMMARY'로 주입
        full_prompt_list.append("## 0) IMAGE SUMMARY (Qwen's Original Analysis)")
        full_prompt_list.append(qwen_raw_text)

        # (B) [NEW] STRUCTURED_DIFF 힌트 주입 (있을 때만)
        if structured_hint_body:
            full_prompt_list.append("")
            full_prompt_list.append("## [STRUCTURED_DIFF HINT]")
            full_prompt_list.append(structured_hint_body)

        # (C) Llama4의 요약본 주입
        full_prompt_list.append("")
        full_prompt_list.append("## [Llama4's Summary & Instructions]")
        full_prompt_list.append(guide_text) # (Llama4의 요약본)

        # (D) MASK METADATA 주입 (Failsafe 힌트)
        if meta_sum:
            try:
                meta_json_str = json.dumps(meta_sum, ensure_ascii=False, indent=2)
                full_prompt_list.append("\n\n### MASK METADATA (Data/Hints for Algorithm)\n")
                full_prompt_list.append("IMPORTANT: Use the 'classes' list (e.g., [10, 30, 50]) from this section to select the correct `class_val` for your algorithm.")
                full_prompt_list.append(meta_json_str)
            except Exception as e:
                log.warning(f"Failed to serialize Meta Summary for prompt: {e}")

        full_prompt = "\n".join(full_prompt_list)

        # ---------------------------\
        # 5. GPT-OSS 입력 프롬프트 전체 로깅 (잘림 없음)\
        # ---------------------------\
        if os.getenv("DEBUG_GPTOSS_PROMPT", "1") == "1":
            try:
                debug_path = (RUN_DIR / stem / "gptoss_gen.debug.txt")
                debug_content = []
                debug_content.append(f"[gptoss_generate log @ {time.strftime('%Y-%m-%d %H:%M:%S')}]\\n")

                debug_content.append("--- 1. Combined Prompt (Qwen + SDIFF Hint + Llama4 + Meta) ---\n")
                debug_content.append(full_prompt) # [CHANGED]

                if structured_hint_body:
                    debug_content.append("\n\n--- 1-b. Structured Diff Hint Section ---\n")
                    debug_content.append("## [STRUCTURED_DIFF HINT]\n" + structured_hint_body)

                debug_content.append("\n\n--- 2. SDIFF-JSON (Injected) ---\n")
                debug_content.append("(REMOVED: No longer injected directly, used for raw_text extraction)") # [CHANGED]

                debug_content.append("\n\n--- 3. Few-Shot Code (Injected) ---\n")
                if fewshot_texts:
                    debug_content.append(fewshot_texts[0])
                    if sels:
                        debug_content.append(f"\n[Fewshot Info] id={sels[0]['id']}, score={sels[0]['score']}\n")
                else:
                    debug_content.append("(No few-shot code injected)")

                debug_path.write_text("\n".join(debug_content), encoding="utf-8")
            except Exception as e:
                log.warning(f"[gptoss_generate] debug log failed: {e}")
                _dump_debug(stem, "gptoss_gen.debug.txt", f"\n[DEBUG_LOG_WRITE_FAILED] {e}\n")

        # ---------------------------\
        # 6. 1차 생성 (기존 방화벽이 설치된 _safe_invoke_gptoss_for_code 호출)\
        # ---------------------------\
        code = _safe_invoke_gptoss_for_code(full_prompt, fewshot_texts).strip()
        code = _strip_code_fence(code)
        
        # (이하 로깅, 린팅, 재생성 로직은 기존과 동일)
        if os.getenv("DEBUG_GPTOSS_PROMPT", "1") == "1":
            _dump_debug(
                stem,
                "gptoss_gen.debug.txt",
                "\n--- 4. Model Output (Head) ---\n" + (code[:4000] if code else "(empty)") + "\n"
            )

        if not code.strip():
            if os.getenv("DEBUG_GPTOSS_PROMPT", "1") == "1":
                _dump_debug(stem, "gptoss_gen.debug.txt", "\n[note] content empty after first attempt\n")
            return JSONResponse({"ok": False, "error": "LLM returned empty code."}, status_code=200)

        violation = _reject_merge_copying(code)
        if violation:
            feedback = (
                "You violated NO-COLOR-COPY rule: " + violation + "\n"
                "Rewrite the entire script STRICTLY following the Llama4 Prompt and MASK METADATA:\n"
                "- Use ONLY `mask_path` and implement the logic (PCA/contours).\n"
                "- Use the correct 'class_val' from MASK METADATA.\n"
            )
            full_prompt_2 = full_prompt + "\n\n[FEEDBACK]\n" + feedback + "\n"

            if os.getenv("DEBUG_GPTOSS_PROMPT", "1") == "1":
                _dump_debug(stem, "gptoss_gen.debug.txt", "\n--- 5. Feedback Prompt (Retry) ---\n" + feedback + "\n")

            code2 = _safe_invoke_gptoss_for_code(full_prompt_2, fewshot_texts).strip()
            code2 = _strip_code_fence(code2)

            if os.getenv("DEBUG_GPTOSS_PROMPT", "1") == "1":
                _dump_debug(stem, "gptoss_gen.debug.txt", "\n--- 6. Model Output (Retry Head) ---\n" + (code2[:4000] if code2 else "(empty)") + "\n")

            if code2.strip():
                violation2 = _reject_merge_copying(code2)
                if not violation2:
                    code = code2
                else:
                    if os.getenv("DEBUG_GPTOSS_PROMPT", "1") == "1":
                        _dump_debug(stem, "gptoss_gen.debug.txt", f"\n[note] violation persists: {violation2}\n")
            else:
                if os.getenv("DEBUG_GPTOSS_PROMPT", "1") == "1":
                    _dump_debug(stem, "gptoss_gen.debug.txt", f"\n[note] empty after reattempt due to violation: {violation}\n")

        return JSONResponse({"ok": True, "code": code})
    except Exception as e:
        log.exception("[gptoss/generate] failed")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=200)
    
@app.post("/gptoss/chat")
async def gptoss_chat(payload: dict = Body(...)):
    try:
        image_name = payload.get("image_name")
        message = payload.get("message","")
        current_code = payload.get("current_code","")
        if not image_name or not message or not current_code:
            return JSONResponse({"ok": False, "error":"image_name/message/current_code required"}, status_code=200)

        guide = (
            "다음은 현재 측정 코드입니다. 사용자가 말한 변경/수정 사항만 반영하고, "
            "다른 기능은 절대 깨지지 않게 전체 코드를 다시 작성하세요. "
            "마크다운 코드펜스 없이 순수 파이썬만 출력하세요.\n\n"
            f"=== 사용자 요청 ===\n{message}\n\n=== 현재 코드 ===\n{current_code}\n"
        )
        llm = _build_gptoss()
        resp = llm.invoke(guide)
        code = _extract_text_from_aimessage(resp).strip()
        code = _strip_code_fence(code)
        if not code.strip():
            return JSONResponse({"ok": False, "error": "LLM returned empty code."}, status_code=200)
        return JSONResponse({"ok": True, "code": code})
    except Exception as e:
        log.exception("[gptoss/chat] failed")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=200)

@app.post("/code/save")
async def code_save(payload: dict = Body(...)):
    image_name = payload.get("image_name")
    code_text  = payload.get("code","")
    add_to_fewshot = bool(payload.get("add_to_fewshot", True))
    include_ppt    = bool(payload.get("include_ppt", True))
    ppt_id = payload.get("ppt_id")

    if not image_name or not code_text:
        return JSONResponse({"error":"image_name and code are required"}, status_code=400)

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    save_name = _code_label_name(image_name)
    save_path = LABEL_DIR / save_name
    save_path.write_text(code_text, encoding="utf-8")

    fewshot_id=None
    if add_to_fewshot:
        fewshot_id = _register_fewshot_case(image_name, code_text, include_ppt, ppt_id)
    return JSONResponse({"status":"saved","path":str(save_path),"filename": save_name,"fewshot_id": fewshot_id})


@app.post("/code/undo")
async def code_undo(payload: dict = Body(...)):
    try:
        image_name = payload.get("image_name")
        if not image_name:
            return JSONResponse({"status": "error", "error": "image_name required"}, status_code=400)

        normalized = _normalize_name(image_name)
        py_name = _code_label_name(normalized)
        target = (LABEL_DIR / py_name).resolve()

        if LABEL_DIR not in target.parents:
            return JSONResponse({"status": "error", "error": "invalid label path"}, status_code=400)

        if target.exists():
            try:
                target.unlink()
            except Exception as e:
                return JSONResponse({"status": "error", "error": f"unlink failed: {e}"}, status_code=500)
            return JSONResponse({"status": "undone", "filename": py_name})
        else:
            # [CHANGED] 존재하지 않아도 정상 응답(사용자 피드백 반영)
            return JSONResponse({"status": "undone", "filename": py_name, "note": "file not found (already deleted)"})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

@app.post("/code/revert_last_run")
async def code_revert_last_run(payload: dict = Body(...)):
    try:
        image_name = payload.get("image_name")
        if not image_name: return JSONResponse({"ok": False, "error":"image_name required"}, status_code=200)
        stem = Path(_normalize_name(image_name)).stem
        run_dir = RUN_DIR / stem
        if not run_dir.exists(): return JSONResponse({"ok": False, "error":"run dir not found"}, status_code=200)

        logs = list(run_dir.rglob("run.log"))
        if not logs:
            p = run_dir/"measure.py"
            if p.exists(): return JSONResponse({"ok": True, "code": p.read_text(encoding="utf-8",errors="ignore")})
            return JSONResponse({"ok": False, "error":"no previous code"})
        latest = max(logs, key=lambda x: x.stat().st_mtime)
        pcode = latest.parent/"measure.py"
        if not pcode.exists(): return JSONResponse({"ok": False, "error":"measure.py not found near latest run.log"})
        return JSONResponse({"ok": True, "code": pcode.read_text(encoding="utf-8",errors="ignore")})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=200)
    
@app.get("/debug/prompt")
async def debug_prompt(name: str = Query(...)):
    stem = Path(_normalize_name(name)).stem
    p = RUN_DIR / stem / "gptoss_gen.debug.txt"
    if not p.exists():
        return JSONResponse({"ok": False, "error": "debug not found"})
    return JSONResponse({"ok": True, "text": p.read_text(encoding="utf-8", errors="ignore")})

# -----------------------------
# few-shot 등록(유지)
# -----------------------------
def _register_fewshot_case(image_name: str, code_text: str, include_ppt: bool, ppt_id: Optional[str]) -> str:
    _ensure_dir(FEWSHOT_DIR)
    normalized = _normalize_name(image_name)
    merge_path = (IMAGE_DIR / normalized).resolve()
    if not merge_path.exists(): raise FileNotFoundError(merge_path)

    mask_name = _mask_name_from_merge(normalized)
    mask_path = (Path(IMAGE_DIR).parent / "mask" / mask_name).resolve() if mask_name else None
    run_dir   = _run_dir_for(image_name)
    overlay_path = (run_dir / "overlay.png").resolve()

    ts  = time.strftime("%Y%m%d-%H%M%S")
    uid = uuid.uuid4().hex[:6]
    stem= Path(normalized).stem
    item_dir = FEWSHOT_DIR / f"{ts}__{stem}__{uid}"
    _ensure_dir(item_dir)

    _save_thumb(merge_path, item_dir / "merge.jpg")
    if mask_path and mask_path.exists(): _save_thumb(mask_path, item_dir / "mask.jpg")
    if overlay_path.exists(): _save_thumb(overlay_path, item_dir / "overlay.jpg")

    (item_dir/"code.py").write_text(code_text, encoding="utf-8")
    (item_dir/"code_summary.json").write_text(json.dumps(_code_summary_text(code_text), ensure_ascii=False, indent=2), encoding="utf-8")
    meta_sum = _meta_summary(mask_path)
    (item_dir/"meta_summary.json").write_text(json.dumps(meta_sum, ensure_ascii=False, indent=2), encoding="utf-8")

    has_ppt=False
    if include_ppt and ppt_id:
        ppt_path = (BASE_DIR/"ppt_uploads"/ppt_id).resolve()
        if ppt_path.exists():
            _save_thumb(ppt_path, item_dir/"ppt.jpg"); has_ppt=True

    hashes = {
        "merge_phash":   _phash_hex(_phash64_from_path(item_dir/"merge.jpg"))   if (item_dir/"merge.jpg").exists() else None,
        "overlay_phash": _phash_hex(_phash64_from_path(item_dir/"overlay.jpg")) if (item_dir/"overlay.jpg").exists() else None
    }
    item = {
        "id": item_dir.name,
        "source_merge": normalized,
        "name_tokens": _name_tokens(normalized),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "has_ppt": has_ppt,
        "meta": {"um_per_px_x": meta_sum.get("um_per_px_x"), "um_per_px_y": meta_sum.get("um_per_px_y"), "classes": meta_sum.get("classes")},
        "tags": [],
        "hashes": hashes
    }
    (item_dir/"item.json").write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
    return item["id"]

# -----------------------------
# 메인
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = 8000
    if "--port" in sys.argv:
        i = sys.argv.index("--port")+1
        if i < len(sys.argv):
            try: port = int(sys.argv[i])
            except ValueError: pass
    log.info(f"Start 0.0.0.0:{port}")
    
    # --- [NEW] Log the GPU setting on startup ---
    try:
        log.info(_log_msg_gpu)
    except NameError:
        log.warning("Could not log GPU setting (variable not found).")
    # --- End 
    
    log.info(f"IMAGE_DIR={IMAGE_DIR}")
    log.info(f"LABEL_DIR={LABEL_DIR}")
    log.info(f"RUN_DIR={RUN_DIR}")
    log.info(f"META_ROOT={META_ROOT}")
    log.info(f"FEWSHOT_DIR={FEWSHOT_DIR}")
    log.info(f"SCRIBBLE_DIR={SCRIBBLE_DIR}")
    log.info(f"GEMMA_BASE_URL set? {bool(GEMMA_BASE_URL)}  MODEL={GEMMA_MODEL}")
    log.info(f"LLAMA4_BASE_URL set? {bool(LLAMA4_BASE_URL)} MODEL={LLAMA4_MODEL}")
    log.info(f"GPTOSS_BASE_URL set? {bool(GPTOSS_BASE_URL)} MODEL={GPTOSS_MODEL}")
    log.info(f"QWEN_ENABLE={QWEN_ENABLE} MODEL_ID={QWEN_MODEL_ID} DEVICE={QWEN_DEVICE} DTYPE={QWEN_DTYPE}")
    uvicorn.run(app, host="0.0.0.0", port=port)


    
