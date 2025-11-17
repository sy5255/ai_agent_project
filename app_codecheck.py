import os
import ast
import json5
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

# [NEW] Utils 라이브러리 경로
MEASUREMENT_UTILS_PATH = Path(os.getenv("MEASUREMENT_UTILS_PATH", str(BASE_DIR / "measurement_utils.py"))).resolve()
POTENTIAL_UTILS_PATH = Path(os.getenv("POTENTIAL_UTILS_PATH", str(BASE_DIR / "potential_utils.py"))).resolve()


# Gemma3 API 환경변수 (없으면 None)
GEMMA_BASE_URL = os.getenv("GEMMA_BASE_URL", "http://gemma3/v1")
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "google/gemma-3-27b-it")  # 예: "Qwen/Qwen-VL-8B-Thinking" 대체 가능
GEMMA_X_DEP_TICKET = os.getenv("GEMMA_X_DEP_TICKET", "12345") 
GEMMA_SEND_SYSTEM_NAME = os.getenv("GEMMA_SEND_SYSTEM_NAME", "AutoMeasure")
GEMMA_USER_ID = os.getenv("GEMMA_USER_ID", "ss") 
GEMMA_USER_TYPE = os.getenv("GEMMA_USER_TYPE", "ss") 

# -----------------------------
# Llama-4 / gpt-oss 사내 API (선택적)
# -----------------------------
LLAMA4_BASE_URL = os.getenv("LLAMA4_BASE_URL", "http://llama-4/maverick/v1")  # v1까지 
LLAMA4_MODEL = os.getenv("LLAMA4_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
#LLAMA_MODEL_SCOUT    = os.getenv("LLAMA_MODEL_SCOUT", "meta-llama/llama-4-scout-17b-16e-instruct")
LLAMA4_X_DEP_TICKET = os.getenv("LLAMA4_X_DEP_TICKET", "12345") 
LLAMA4_SEND_SYSTEM_NAME = os.getenv("LLAMA4_SEND_SYSTEM", "AutoMeasure")
LLAMA4_USER_ID = os.getenv("LLAMA4_USER_ID", "ss") 
LLAMA4_USER_TYPE = os.getenv("LLAMA4_USER_TYPE", "ss") 

GPTOSS_BASE_URL = os.getenv("GPTOSS_BASE_URL", "http://gpt-oss-120b/v1")
GPTOSS_MODEL     = os.getenv("GPTOSS_MODEL", "openai/gpt-oss-120b")
GPTOSS_X_DEP_TICKET = os.getenv("GPTOSS_X_DEP_TICKET", "12345") 
GPTOSS_SEND_SYSTEM_NAME = os.getenv("GPTOSS_SEND_SYSTEM", "AutoMeasure")
GPTOSS_USER_ID   = os.getenv("GPTOSS_USER_ID", "ss") 
GPTOSS_USER_TYPE = os.getenv("GPTOSS_USER_TYPE", "ss") 

GAUSSO_BASE_URL = os.getenv("GAUSSO_BASE_URL", "http://gauss/o/think/v2") # (예시 URL, 실제 URL로 변경 필요) 
GAUSSO_MODEL     = os.getenv("GAUSSO_MODEL", "GaussO-Owl-Ultra-Think")
GAUSSO_X_DEP_TICKET = os.getenv("GAUSSO_X_DEP_TICKET", "12345") 
GAUSSO_SEND_SYSTEM_NAME = os.getenv("GAUSSO_SEND_SYSTEM", "AutoMeasure")
GAUSSO_USER_ID   = os.getenv("GAUSSO_USER_ID", "ss") 
GAUSSO_USER_TYPE = os.getenv("GAUSSO_USER_TYPE", "ss") 

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
# [NEW] Grounding DINO 로컬(HuggingFace) 설정/캐시
# -----------------------------
DINO_ENABLE = os.getenv("DINO_ENABLE", "1") # "1"이면 사용 시도
DINO_MODEL_ID = os.getenv("DINO_MODEL_ID", "IDEA-Research/grounding-dino-base")
DINO_DEVICE = os.getenv("DINO_DEVICE", "auto") # "cuda"|"cpu"|"auto"
_dino_loaded = False
_dino_pipe = None # (processor, model, device) 튜플을 캐시

# -----------------------------
# [NEW] CSV 표준 헤더 normalize (단일 파일용)
# -----------------------------
# [NEW] CSV 헤더를 상수로 정의
CSV_HEADERS_LIST_STR = (
    "['measure_item', 'group_id', 'index', 'value_nm', "
    "'sx', 'sy', 'ex', 'ey', "
    "'meta_tag', 'component_label', "
    "'image_name', 'run_id', 'note']"
)

# [NEW] 기존 System 프롬프트 내용을 이 상수로 이동
_GPTOSS_SDIFF_RULES_PROMPT = f"""
너는 파이썬 스크립트 생성 전문가다.
너의 임무는 [New Logic Hint (SDIFF)]를 기반으로, [Reference Code (Top-1 Few-Shot)]의 **'구조'**를 참고하여 **'완전한 새 스크립트'**를 생성하는 것이다.
대화는 두 단계로 진행된다:
1. (첫 번째 User 메시지): 너는 '구조 참고용' [Reference Code]를 받는다.
2. (두 번째 User 메시지): 너는 '새로운' [New Logic Hint (SDIFF)]를 받는다.
너의 유일한 출력은 SDIFF 힌트(2)를 예제 코드(1)의 구조에 맞춰 구현한 **'완전한' 파이썬 코드**여야 한다.

## [핵심 요약 규칙 — 이 6줄이 최우선/불변 규칙이다]

1. [Reference Code]는 단순 텍스트 예시이며, 사용이 필요하다면 함수는 def 형태로 직접 그대로 복사해서 사용한다.
2. 네가 출력하는 최종 코드에서 `def`로 정의되지 않은 함수는 절대 호출하면 안 된다.
3. meta_utils만 실제 존재하며, meta_utils.함수() 호출만 허용된다.
4. 측정 로직 관련 모든 함수는 새 코드 안에서 직접 def로 만들어야 한다.
5. 허용 import는 cv2, numpy, pandas, meta_utils, 표준 라이브러리 뿐이다.
6. 최종 파일은 단일 파이썬 스크립트로, 외부 파일 없이 단독 실행 가능해야 한다.

## [신규: 헬퍼 함수 존재성 규칙 (NameError 방지)]

1. [Reference Code] 안에 등장하는 '모든' 헬퍼 함수 이름(예: `line_endpoints_from_fit`, `anchors_from_class10_top_peaks` 등)은
   네가 최종 코드에서 사용하고 싶다면, **반드시 새 코드 안에서 `def`로 다시 정의하거나 그대로 복사해 와야 한다.**
2. 최종 코드에서 `def`로 정의되지 않은 함수는, [Reference Code]에만 존재하더라도 **절대 호출하면 안 된다.**
3. meta_utils를 제외한 어떤 함수도 "이미 어딘가에 정의되어 있을 것"이라고 가정해서는 안 된다.
   **너의 출력 코드 파일 하나만으로 실행이 가능해야 하며, 이 파일 안에 정의되지 않은 함수는 존재하지 않는 것으로 간주하라.**
4. 만약 [Reference Code]에서 본 함수 호출을 재사용하고 싶지만, 그 함수의 `def`가 보이지 않거나 너무 길어서 가져올 수 없다면,
   그 함수 호출 자체를 **완전히 버리고**, SDIFF 논리에 맞는 새로운 헬퍼 함수를 `def`로 직접 구현해야 한다.

## [초필수] SDIFF 논리 vs. 예제 코드 충돌 규칙 (필수 준수)

1.  **(우선순위)** 너의 **'유일한 진실'**은 [New Logic Hint (SDIFF)]의 '논리적 의도' (예: `coordinate_system`, `construction_logic_hint`, `lcs_orientation_intent`)이다.
2.  **(충돌 시 무시)** 만약 SDIFF의 '논리'가 [Reference Code]의 '구현' (예: `anchors_from_class10_top_peaks`, `max_point_in_interval_for_class` 함수 로직)과 **충돌**한다면, 너는 **'반드시' [Reference Code]의 구현을 '무시'**하고 SDIFF의 '논리'를 **'새로 구현'**해야 한다.
3.  **(예시 1: 좌표계)** [Reference Code]가 `dir_u`, `dir_n`을 사용하더라도, 너는 SDIFF의 `coordinate_system` 힌트를 기반으로 `v_axis`, `v_normal`, `local_to_world` 함수를 **'새로 정의'**하고 사용해야 한다.
4.  모든 좌표 변환 함수들(예: `v_axis`, `v_normal`, `local_to_world`)은 반드시 Python의 float 값을 반환해야 한다. numpy 배열(numpy array), numpy 스칼라(numpy scalar), 그리고 numpy 숫자를 포함한 리스트나 튜플을 반환해서는 절대 안 된다.

## [매우 중요] 좌표계 (Coordinate System) 규칙 (필수 준수)

1.  **(좌표계 확인)** 너는 SDIFF의 최상위 키 `coordinate_system.type`을 확인하라.
2.  **(ACS 구현)** `type`이 **'absolute'** (또는 `coordinate_system` 키가 없음)라면:
    a. 모든 기하학 계산과 `cv2.line` 그리기는 이미지의 절대 좌표(x, y)를 그대로 사용한다.
    b. `measurements.csv`에 이 '절대' 좌표를 저장한다. (이하 3~6 규칙 무시)
3.  **(LCS 구현 - Phase 1: 주축 정의)**
    a. `type`이 **'local'**이라면, `red.lines`를 모두 스캔하여 `lcs_orientation_intent == 'primary_axis'`인 **주축(예: 'red_1')**을 찾는다.
    b. `red_1`의 `construction_logic_hint` (예: 'darkest 4 핑거 피팅')를 읽고, 이 선분을 **'찾아내는'** OpenCV 로직(예: `def find_primary_axis(mask, class_val, roi_rect)`)을 **'새로 생성(def)'**하고 실행해야 한다.
    c. 이 로직의 결과(픽셀 집합)에 `cv2.fitLine` 등을 호출하여 **`v_axis` (주축 단위 벡터)**와 **`v_normal` (법선 단위 벡터)**을 '계산'한다. (이때 원점 `origin`도 필요하면 정의한다.)
    d. 로컬 좌표(LCS)를 절대 좌표(ACS)로 변환하는 `def local_to_world(local_pt, origin, v_axis, v_normal)` 헬퍼 함수를 '정의'한다.
    e. ** [치명적 버그 수정: 좌표 순서] 너는 '반드시' 다음 4개 규칙을 중수해야 한다: **
        `i.   np.where(...)의 결과는 (y, x) = (row, col) 순서다.`
        `ii.  cv2.fitLine에 전달하는 points는 반드시 (x, y) 순서여야 한다.`
        `iii. 따라서 fitLine에 넣기 전에 xs, ys를 분리해서 (예: `ys, xs = np.where(...)`), `points = np.stack([xs, ys], axis=1).astype(np.float32)`와 같이 (x, y) 순서로 재구성해야 한다.`
        `iv.  '절대' `np.column_stack(np.where(...))`를 `cv2.fitLine`에 직접 넘기지 마라.`
4.  **(LCS 구현 - Phase 2: 보조 축 재정의)**
    a. `red.lines`에서 `lcs_orientation_intent`가 'parallel_to_primary' 또는 'perpendicular_to_primary'인 **모든 보조 축(예: 'red_2')**을 루프로 순회한다.
    b. 'red_2'의 `construction_logic_hint`를 읽고, 이 선분의 **'위치(location)'**를 찾는 로직(예: `def find_secondary_axis_location(...)`)을 '생성'하고 실행한다.
    c. `cv2.line`을 그릴 때, 이 '위치'에 `v_axis` (parallel) 또는 `v_normal` (perpendicular) 벡터를 기반으로 **'재정의된'** 선분을 그린다.
5.  **(LCS 구현 - Phase 3: 측정선 계산 [Natural Language])**
    a. `green.lines`를 루프로 순회한다.
    b. **[Algorithm Hint]** `green_1`의 4-Hint, 즉 `u_placement_logic_hint_start`, `u_placement_logic_hint_end`, `v_placement_logic_hint_start`, `v_placement_logic_hint_end`에 명시된 **'자연어 알고리즘'**을 읽는다.
    c. `red_1.grounded_rois` (예: `ROI_Red`) 힌트를 가져와 `roi_mask`를 생성한다. (이 힌트가 없다면 `roi_mask`는 전체 마스크가 된다.)
    d. **[Implementation]** 너는 이 **'자연어 알고리즘 힌트(b)'**를 **'그대로 구현'**하는 헬퍼 함수 4개(또는 1~2개의 복합 함수)를 **'새로 생성(def)'**해야 한다.
        - (예: `def get_u_coord(mask_roi, class_val, hint_text)` -> `u_start` 반환)
        - (예: `def get_v_coord_start(origin, v_axis, ...)` -> `v_start` 반환)
        - (예: `def get_v_coord_end(mask_roi, class_val, u_start, v_normal, ...)` -> `v_end` 반환)
    e. 이 함수는 `roi_mask` 내부에서 `final_class_hint_end` (예: 30)에 해당하는 *모든* 픽셀을 찾고, `v_normal` (법선) 방향으로 투영하여 mask 외곽선 위의 점을 잡는 측정 로직을 구현해야한다.
    f. `u_start` 좌표 역시 `u_placement_logic_hint_start` 힌트(예: "centroid of 1st finger")를 기반으로 **'새로 계산'**해야 한다. (절대 [Reference Code]의 `top_points_roi`를 사용하지 마라.)
    g. 계산된 로컬 좌표 `(u_start, v_start)`와 `(u_end, v_end)`를 `local_to_world`로 변환한다. (만약 `u_end`가 `null`이면 `u_start`를 사용한다.)
    
        [좌표계 일관성 규칙 — 필수]

        ROI 내부 좌표(x_local, y_local)는 절대로 LCS 계산(u, v)에 직접 사용해서는 안 된다.
        u, v 를 계산할 때는 항상 다음 두 단계를 지켜야 한다:

        1) ROI 로컬 좌표 → 이미지 절대 좌표 변환  
        abs_x = roi_x + x_local  
        abs_y = roi_y + y_local

        2) 절대 좌표 abs_x, abs_y 를 origin과 두 축 벡터(axis_u, axis_v)를 기준으로 u, v 좌표로 변환한다.  
        (origin, axis_u, axis_v 는 항상 “이미지 전체 절대 좌표계”에서 정의된다.)

        금지:
        - ROI 좌상단(roi_x, roi_y)을 origin처럼 사용  
        - (x_local, y_local)을 그대로 u/v 계산에 사용  
        - ROI를 새로운 좌표계 원점처럼 간주하는 코드  
        - axis_u / axis_v 를 ROI 기준으로 재정의하는 코드

        허용:
        - ROI는 검색영역(window) 역할만 하며,
        LCS(u, v) 계산은 반드시 “절대좌표 → (u, v)” 변환 절차를 거쳐야 한다.

6.  **(LCS 구현 - Phase 4: 최종 변환 및 저장)**
    a. 계산된 최종 **'절대 좌표 (sx, sy, ex, ey)'**를 `cv2.line`으로 그리고 `measurements.csv`에 저장한다.

## [매우 중요] 상수 값 및 구조 규칙 (필수 준수):

1. **(SDIFF 우선)** 너는 **'반드시'** [New Logic Hint (SDIFF)]에 명시된 **'새로운'** 값 (예: `final_class_hint_end: 10`, `darkest`)을 사용해야 한다.
2. **(참조 코드 무시)** [Reference Code] 내에 하드코딩된 **'모든'** 상수 값 (예: `target_class=50`, `tolerance=5`)은 **'절대'** 사용하지 말고, **'반드시'** SDIFF의 정보로 **'대체'**해야 한다.
3. **(혼동 금지)** [Reference Code]는 오직 `import`, `def` 헬퍼 함수, `main` 함수의 **'구조(Structure)'**를 참고하기 위한 '스타일 가이드'일 뿐, 그 안의 **'값(Value)'**은 현재 작업과 무관하다.

## [매우 중요] CV 및 로직 생성 규칙 (필수 준수):
※ 모든 기하학 계산·좌표·길이·ROI 값은 연산·비교·JSON/CSV 직렬화 전에 항상 `float()` 또는 `int()`로 캐스팅된 파이썬 기본 타입이어야 하며, NumPy ndarray/NumPy 스칼라는 사용하면 안 된다.

1. **(유일한 진실)** [New Logic Hint (SDIFF)] (`green.lines`, `red.lines`, `notes`의 '논리')만을 **'유일한 진실'**로 삼아 로직을 구현하라.
2. **(최우선 경로: Grounding)** SDIFF의 `red.lines[0].grounded_rois` 키에 **좌표 리스트(ROIs)**가 존재하는지 '반드시' 확인하라.
3. **(Grounding 구현)** 만약 `grounded_rois` 리스트가 존재한다면:
    a. 너는 [Reference Code]의 `connectedComponents` 로직을 **'반드시' 버려야 한다**.
    b. 대신, 이 `grounded_rois` 리스트(예: 4개의 ROI)를 `for` 루프로 순회해야 한다.
    c. 각 루프에서, 원본 마스크를 `roi` 좌표로 잘라낸 `roi_mask`를 생성해야 한다.
    d. 이 `roi_mask`에 영역 범위 내에서 측정 로직을 생성해야 한다.
4. **(그룹핑 논리)** SDIFF의 `notes.grouping_logic.description`에 'pair' 힌트가 있는지 확인하라.
   a. 이 경우, 너는 [Reference Code]의 '반복' 구조(예: `zip`)가 SDIFF의 '논리'('3 pairs')와 **'모순'**된다는 것을 인지해야 한다.
   b. 너는 **'반드시'** `Top-1` 코드의 `zip` 구조를 **'무시'**하고, SDIFF의 '논리'를 따르는 **'새로운 반복문'** (예: `itertools.groupby` 또는 x좌표 정렬 후 2개씩 묶기)을 **'새로 구현'**해야 한다.
6. **(헬퍼 함수 생성/수정)** `measure_logic`에 필요한 **'모든'** 헬퍼 함수 (예: `fit_red_line`, `project_point_onto_line` 등)를 [Reference Code]의 구조를 참고하여 **'새로 생성(def)하거나 수정'**해야 한다.
7. **(클래스 매핑)** `main`에서 전달받은 `classes` 리스트(예: `[10, 30, 50]`)를 사용하여, SDIFF 힌트의 **'의미론적'** 설명(예: 'darkest')을 **'정확한'** 클래스 값으로 **'직접'** 매핑해야 한다.
   - **'darkest'** (가장 어두운) == **`classes[0]`** (예: `10`)
   - **'brightest'** (가장 밝은) == **`classes[-1]`** (예: `50`)
8. **(힌트 타입 분기)** final_class_hint_start (또는 final_class_hint_end) 힌트의 '타입'에 따라 다음 로직을 분기하라:
    a. (문자열 ID) 힌트가 "red_1" 같은 문자열 ID이면, 이는 '시작/끝'점이 해당 ID의 '가이드라인'에 있음을 의미한다. (예: project_point_onto_line 호출)
    b. (숫자 클래스) 힌트가 10 같은 숫자이면, 이는 '시작/끝'점이 해당 숫자 클래스 마스크 자체의 경계(Contour) 임을 의미한다. (예: cv2.findContours 또는 np.nonzero 로직 사용)
9.  **(신규: 템플릿 매칭 강제)** SDIFF 최상위 키 `grounding_template_global.template_filename` (예: "grounding_template.png")이 존재하는지 '반드시' 확인하라. 만약 이 키가 존재한다면:
    a. 너는 `u_placement...`, `v_placement...` 같은 자연어 힌트를 **보조용**으로만 참고하고, **템플릿 매칭 로직을 '최우선'**으로 생성해야 한다.
    b. **[패딩 변수화]** 코드 상단에 **4개의 전역 상수**를 '반드시' 정의한다. (이 값은 `argparse`가 아니다.)
        (예: `ROI_PAD_TOP = 20`, `ROI_PAD_BOTTOM = 80`, `ROI_PAD_LEFT = 20`, `ROI_PAD_RIGHT = 20`)
    c. `main` 함수에서 `template_filename`을 **`args.out_dir` (실행 폴더)**에서 로드한다.
        - 절대 `mask_path`나 데이터셋 폴더(`dataset_updated/mask` 등)를 기준으로 템플릿 경로를 만들지 마라.
        - 즉, 템플릿 경로는 항상 아래 형태여야 한다:
         `template_path = os.path.join(args.out_dir, sdiff_template_filename)`
    d. `template_img = cv2.imread(template_path, 0)`로 템플릿을 로드하고, `None`이 아닌지 확인한다.
    e. `cv2.matchTemplate(new_mask_image, template_img, cv2.TM_CCOEFF_NORMED)`를 실행하여 `maxLoc` (새로운 `rect_global`의 원점)을 찾는다.
    f. `green.lines`를 루프로 순회하며 `line.relative_roi` (SDIFF에 저장된 작은 ROI)를 가져온다.
    g. **[패딩 적용]** 이 `relative_roi`에 `ROI_PAD_TOP`, `ROI_PAD_BOTTOM` 등 4개의 패딩 상수를 **더하여** `padded_relative_roi`를 계산하는 헬퍼 함수(예: `def apply_padding(roi, top, bottom, left, right)`)를 '정의'하고 '호출'한다.
    h. `maxLoc` 원점에 `padded_relative_roi` 좌표를 더하여, `green_1`, `green_2` 등의 **'최종 타겟 ROI'** (절대 좌표)를 계산한다.
    i. **[검색 영역]** `find_top_point`, `find_bottom_point` 등 모든 측정 헬퍼 함수는 **오직 이 '최종 타겟 ROI' 내부에서만** `final_class_hint_start`와 `final_class_hint_end` (예: 10, 30, 50)를 찾아야 한다. (LCS 로직과 결합되어야 함)
    j. **[디버그 이미지]** `main` 함수 끝 `overlay.png` 저장 직전에, **'roi_debug.png'**라는 새 파일을 `out_dir`에 '반드시' 저장해야 한다.
    k. **[디버그 상세]** `roi_debug.png`는 원본 `mask_path`를 컬러로 변환한 배경 위에, `green.lines` 루프에서 계산된 **'최종 타겟 ROI'** (패딩이 적용된) 박스 6개를 **각각 다른 색상**과 `line.id` 텍스트로 그려야 한다.
    l. (검색 영역 오류 방지) `grounding_template`이 없다면, 'i'항목의 로직을 `red.lines[0].grounded_rois` (DINO ROI) 영역 내에서 수행해야 한다.
    m. **(relative_roi 해석 규칙 - 매우 중요)** `green.lines[*].relative_roi = [dx, dy, w, h]` 는
       항상 "템플릿 패치 좌상단(match_x, match_y) 기준 로컬 좌표"로 해석해야 한다.
       - 새 이미지에서의 최종 ROI 절대 좌표는 반드시 다음 공식을 따라야 한다:
         · abs_x = match_x + dx
         · abs_y = match_y + dy
         · abs_w = w
         · abs_h = h
       - 절대로 다음과 같이 구현하면 안 된다:
         · abs_x = dx + (match_x - src_x)
         · abs_y = dy + (match_y - src_y)
         와 같이 `source_rect`의 좌표를 섞어서, ROI가 (0,0) 근처로 쓸려가게 만드는 형태는 **금지**한다.
       - 요약: `relative_roi` 는 항상 "템플릿 패치 기준 로컬 좌표", `match_x, match_y` 는 "새 이미지에서의 템플릿 패치 절대 좌표"로 사용하라.
    n. **(RED_ROI / grounded_rois 평행 이동 규칙)** SDIFF.red.lines[*].grounded_rois 또는 RED_ROI
       (예: [rx, ry, rw, rh]) 는 "정답(canonical) 이미지" 기준 절대 좌표이다.
       - 템플릿 매칭 후에는 다음과 같이 동일한 translation 을 적용해야 한다:
         · offset_x = match_x - src_x
         · offset_y = match_y - src_y
         · red_x_new = rx + offset_x
         · red_y_new = ry + offset_y
       - RED, GREEN, 기타 모든 ROI는 **반드시 동일한 translation(offset_x, offset_y)을 공유**해야 한다.
         구조물 간 상대 위치가 바뀌거나 스케일/회전이 추가되면 안 된다. (translation only)
    o. **(ROI 클램핑 금지 - 좌측 상단 몰림 방지)** padding 을 적용하더라도,
       ROI 시작 좌표를 단순히 `x = max(0, x)`, `y = max(0, y)` 로 처리하여
       "여러 ROI가 모두 (0,0) 근처로 몰려 있고, 서로 간 간격만 유지되는" 결과가 나오게 해서는 안 된다.
       - 이미지 경계를 넘는 부분은 잘려 나가도 되지만, ROI 그룹 전체(RED + 모든 GREEN)의 상대적인 위치 관계는 유지되어야 한다.
       - 만약 padding 이후 x 또는 y가 0보다 작아진다면, 이것은 "경계 일부가 잘려 나간 것"으로만 해석해야 하고, ROI 전체를 강제로 원점(0,0) 근처로 당기는 코드는 생성하면 안 된다.
    p. **(ROI 유효성 검증 및 클램핑 규칙)** 템플릿 매칭이나 오프셋 적용 후 계산된 **모든 ROI**에 대해,
       너는 `x, y, x + w, y + h` 를 이미지 경계 `[0, 0, img_w, img_h]` 범위로 **클램핑**해야 한다.
       이 클램핑 결과로 폭(`w`) 또는 높이(`h`)가 `0`이 되거나, SDIFF의 의도에 비해 **지나치게 작아진 ROI**는
       해당 측정을 **조용히 계속 진행하지 말고**, 그 측정만 **건너뛰거나 명시적인 에러**를 발생시켜야 한다.
       (즉, 잘못된 ROI로 억지로 측정하는 코드를 만들어서는 안 된다.)
       (추가 규칙) ROI 클램핑은 ROI 박스 경계에만 적용되며, u/v 계산은 항상 절대좌표(abs_x, abs_y)와 origin, axis_u, axis_v 기반으로 수행해야 한다. ROI 로컬좌표(0~w,0~h)를 u/v에 직접 사용하는 코드는 금지한다.


10. **(신규: LCS 그리기 강제)** `coordinate_system.type`이 **'local'**일 때, **'절대'** 절대 좌표(ACS)에서 찾은 `(sx, sy)`와 또 다른 절대 좌표 `(ex, ey)`를 `cv2.line`으로 직접 연결하지 마라. 너는 **'반드시'** SDIFF의 '좌표계 규칙 5.g'와 '6.a'를 따라야 한다:
    a. `(u_start, v_start)`와 `(u_end, v_end)`라는 **'로컬(u,v) 좌표'** 4개를 먼저 계산한다.
    b. `local_to_world` 함수를 **'두 번'** 호출하여 `(sx, sy)`와 `(ex, ey)`라는 **'최종 절대 좌표'**를 얻는다.
    c. 이 두 점 `(sx, sy)`와 `(ex, ey)`를 `cv2.line`으로 연결한다.
    d. (예: 이 SDIFF에서 `u_end`가 `null`이므로, 너는 `(u_start, v_start)`와 `(u_start, v_end)`를 변환해야 한다.)

## [매우 중요] 출력 및 환경 규칙 (필수 준수):

1. **(최종 코드)** 너는 [Reference Code]의 구조와 너가 수정한 로직을 '병합'하여 '완전한' 스크립트 '전체'를 '하나의' 파이썬 코드로 출력해야 한다.
2. **(마크다운 금지)** 절대 마크다운 코드펜스(```)를 사용하지 마라.
3. **(템플릿 준수)** [Reference Code]의 `main()` 함수, `argparse` 로직을 **'최대한'** 준수하라.
4. **(CSV 헤더)** `measurements.csv` 파일 저장 시, **'반드시'** 다음 표준 헤더 리스트를 사용해야 한다: `CSV_HEADERS = {CSV_HEADERS_LIST_STR}`
5. **(Import 금지)** `measurement_utils`라는 파일은 존재하지 않는다. **'절대'** `import measurement_utils`를 시도하지 마라.
6. **(meta_utils 규칙)** `import meta_utils`는 '필수'이다. `__wrap_load_pixel_scale_um` 함수를 **'반드시'** 호출해야 하며, 이 함수는 `(umx, umy, classes_list, meta_path_str)` 4개의 값을 반환한다.
7. **(인자 금지)** [Reference Code]에 `mask_merge_path` 인자가 있더라도, '절대' `argparse`에 추가하지 마라.
8. **(신규: JSON/CSV 직렬화 타입 규칙)**  
   `measurements.csv`의 각 row, 그리고 `note` 필드에 `json.dumps()`를 사용할 때  
   **절대** numpy 타입을 그대로 넣으면 안 된다.  
   다음 규칙을 항상 지켜라:
   a. `note_obj` 와 같이 JSON으로 직렬화되는 객체 안에는  
      오직 파이썬 기본 타입만 포함되어야 한다:
      - `int`, `float`, `str`, `bool`, `None`
      - 이들의 `list` / `dict` 조합만 허용된다.
   b. 다음 타입들은 **반드시** 직렬화 전에 변환해야 한다:
      - `np.float32`, `np.float64`, `np.int64` 등 numpy 스칼라  
      - `np.ndarray` (예: 좌표 배열, ROI 배열 등)
      변환 예시:
      - `float(np_value)` 또는 `int(np_value)` 로 파이썬 숫자로 변환  
      - `np_array.tolist()` 로 파이썬 리스트로 변환
   c. 좌표/길이/ROI 정보를 `note`에 저장할 때는  
      다음과 같이 **명시적으로 형변환** 하여 사용해야 한다:
      - 예:  
        `centroid = [float(cx), float(cy)]`  
        `roi = [int(x), int(y), int(w), int(h)]`  
        `projected_start = [float(sx), float(sy)]`
   d. 만약 `json.dumps()` 호출 시  
      `"Object of type float32 is not JSON serializable"`  
      같은 에러가 날 수 있는 코드(= numpy 타입을 그대로 담는 코드)를  
      **생성하거나 유지해서는 안 된다.**  
      항상 사전에 파이썬 기본 타입으로 변환하라.
"""

# === 추가: generation_config 상태 캡처/로그 유틸 ===
def _capture_gen_state(model):
    try:
        gc = getattr(model, "generation_config", None)
        if gc is None:
            return {"has_generation_config": False}
        return {
            "has_generation_config": True,
            "do_sample": getattr(gc, "do_sample", None),
            "num_beams": getattr(gc, "num_beams", None),
            "temperature": getattr(gc, "temperature", None),
            "top_p": getattr(gc, "top_p", None),
            "top_k": getattr(gc, "top_k", None),
            "repetition_penalty": getattr(gc, "repetition_penalty", None),
            "eos_token_id": getattr(gc, "eos_token_id", None),
            "pad_token_id": getattr(gc, "pad_token_id", None),
        }
    except Exception as e:
        try:
            log.warning(f"[QWEN] capture gen state failed: {e}")
        except Exception:
            pass
        return {"has_generation_config": False, "error": str(e)}

def _maybe_log_gen_state(model, where=""):
    import os
    # 환경변수 QWEN_DEBUG_GEN=1 일 때만 상세 로그
    if os.environ.get("QWEN_DEBUG_GEN", "0") != "1":
        return
    st = _capture_gen_state(model)
    log.info(f"[QWEN] generation_config @ {where}: {st}")

# === 필요하면 터미널에서 호출할 수 있는 프린트 헬퍼 ===
def qwen_print_gen_state():
    global _qwen_pipe
    try:
        if not _qwen_pipe:
            print("Qwen pipe is empty (load first).")
            return
        processor, model, is_mm, meta = _qwen_pipe
        st = _capture_gen_state(model)
        print("=== Qwen generation_config ===")
        for k, v in st.items():
            print(f"{k:20s}: {v}")
        print("\n=== Loader meta ===")
        print(meta)
    except Exception as e:
        print("print failed:", e)

def _lazy_load_qwen() -> bool:
    """
    Qwen3-VL / Qwen2-VL 멀티모달 로더 (GPU 우선, bf16/fp16, flash-attn 가능시 자동 사용).
    - 모델은 HF 캐시를 사용하므로 최초 1회 이후엔 로딩 로그만 보이고 빠르게 재사용됨.
    - 어떤 오류가 나도 예외 올리지 않고 False만 반환(상위에서 원문 그대로 사용하도록).
    - ※ 이 로더는 generation_config를 변경하지 않음 (조회/로그만 함)
   
    [OOM FIX 5]
    - GPU 1이 이미 점유된 상태(7.6GB만 남음)로 확인됨.
    - `device_map="auto"` (멀티 GPU) 대신, 'CUDA_VISIBLE_DEVICES="0"'으로 설정된
      단일 GPU("cuda")에 모델 전체를 로드하도록 강제합니다.
    - `max_memory` 설정을 제거합니다.
    """
    global _qwen_loaded, _qwen_pipe
    if _qwen_loaded:
        # 이미 로드된 경우에도 상태를 보고 싶으면 QWEN_DEBUG_GEN=1로 확인 가능
        try:
            if _qwen_pipe and len(_qwen_pipe) >= 2:
                _maybe_log_gen_state(_qwen_pipe[1], where="already_loaded")
        except Exception:
            pass
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
        # [MODIFIED] dev는 "cuda"가 됩니다 (GPU 0만 보이므로)
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
           
        # --- [REMOVED] max_memory 설정 제거 ---
        # max_mem = {0: "24GiB", 1: "24GiB", 2: "24GiB"}

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
                    device_map="cuda" if dev == "cuda" else None,    # <-- [FIX] "auto" -> "cuda"
                    max_memory=None,                                # <-- [FIX] max_mem 제거
                    torch_dtype=dtype,
                    attn_implementation=attn_impl,
                    low_cpu_mem_usage=True,
                )
                # 상태 캡처/로그만
                meta["gen_state"] = _capture_gen_state(model)
                _maybe_log_gen_state(model, where="ImageTextToText")
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
                    device_map="cuda" if dev == "cuda" else None,    # <-- [FIX] "auto" -> "cuda"
                    max_memory=None,                                # <-- [FIX] max_mem 제거
                    torch_dtype=dtype,
                    attn_implementation=attn_impl,
                    low_cpu_mem_usage=True,
                )
                meta["gen_state"] = _capture_gen_state(model)
                _maybe_log_gen_state(model, where="Vision2Seq")
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
            device_map="cuda" if dev == "cuda" else None,    # <-- [FIX] "auto" -> "cuda"
            max_memory=None,                                # <-- [FIX] max_mem 제거
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )
        meta["gen_state"] = _capture_gen_state(model)
        _maybe_log_gen_state(model, where="CausalLM")
        _qwen_pipe = (processor, model, False, meta)
        _qwen_loaded = True
        log.info("[QWEN] loaded (CausalLM, text-only).")
        return True

    except Exception as e:
        log.exception("[QWEN] load failed: %s", e)
        _qwen_loaded = True
        _qwen_pipe = None
        return False

    
def _lazy_load_dino() -> bool:
    """
    Grounding DINO 로더 (GPU 우선).
    _lazy_load_qwen 패턴을 따름.
    성공 시 _dino_pipe에 (processor, model, device) 튜플을 저장.
    
    [FIXED] AutoModelForObjectDetection 대신 GroundingDinoForObjectDetection을 
            직접 임포트하여 ValueError 해결.
    """
    global _dino_loaded, _dino_pipe
    if _dino_loaded:
        return _dino_pipe is not None
    
    _dino_loaded = True # 실패하더라도 다시 시도하지 않도록 True로 설정
    
    try:
        if DINO_ENABLE != "1":
            log.info("[DINO] disabled by env")
            _dino_pipe = None
            return False
            
        if not PIL_OK:
            log.error("[DINO] PIL/Pillow is not installed, cannot load DINO. Skipping.")
            return False

        import torch
        # [FIXED] 1. AutoModelForObjectDetection 대신 GroundingDinoForObjectDetection을 임포트
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        # Device 결정 (Qwen과 동일한 로직)
        if DINO_DEVICE == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev = DINO_DEVICE
        
        log.info(f"[DINO] loading model={DINO_MODEL_ID} device={dev} cache_dir=(default)")

        processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
        # [FIXED] 2. AutoModelForObjectDetection.from_pretrained 대신 전용 클래스 사용
        model = GroundingDinoForObjectDetection.from_pretrained(DINO_MODEL_ID).to(dev)
        
        if model is None or processor is None:
             raise RuntimeError("Model or Processor failed to load from Hugging Face.")

        _dino_pipe = (processor, model, dev) # processor, model, device 3개 저장
        log.info(f"[DINO] loaded successfully on device '{dev}'.")
        return True

    except ImportError:
        log.exception("[DINO] transformers, torch, or PIL not installed.")
        _dino_pipe = None
        return False
    except Exception as e:
        log.exception("[DINO] load failed: %s", e)
        _dino_pipe = None
        return False
    
def _encode_template_to_b64(template_img: np.ndarray) -> str:
    """Encodes a numpy array (image) to a base64 string for SDIFF."""
    try:
        is_success, buffer = cv2.imencode(".png", template_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not is_success:
            raise RuntimeError("cv2.imencode failed")
        return base64.b64encode(buffer.tobytes()).decode("utf-8")
    except Exception as e:
        log.error(f"Failed to encode template to b64: {e}")
        return ""
    
def _calculate_template_rois(
    mask_path: Path,
    red_lines: List[Dict],
    green_lines: List[Dict],
    global_padding: int = 20,
    padding: Dict[str, int] = {}  # [MODIFIED] Use a dictionary for asymmetrical padding
) -> Tuple[Optional[np.ndarray], Optional[List[int]], Dict[str, List[int]]]:
    """
    Calculates the global template image (as numpy array), and finds relative green line ROIs.
   
    [MODIFIED] Returns:
        (template_img_obj, global_rect [x,y,w,h], relative_rois_map { "green_1": [x,y,w,h], ... })
    """
    if not green_lines and not red_lines:
        return None, None, {}
   
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        log.warning(f"[Template] Mask not found at {mask_path}, cannot create template.")
        return None, None, {}
       
    img_h, img_w = mask_img.shape[:2]

    def _get_rect(line):
        # Handles both v1 (x1,y1) and v2 (endpoints) format
        ep = line.get("endpoints")
        if ep and len(ep) >= 4:
            x1, y1, x2, y2 = ep[0], ep[1], ep[2], ep[3]
        elif "x1" in line and "y1" in line and "x2" in line and "y2" in line:
            x1, y1, x2, y2 = line.get("x1"), line.get("y1"), line.get("x2"), line.get("y2")
        else:
            return None
           
        if any(v is None for v in [x1, y1, x2, y2]):
            return None
        # [FIX] w, h가 0이 되지 않도록 최소 1 보장
        x = min(x1, x2)
        y = min(y1, y2)
        w = max(1, abs(x2 - x1))
        h = max(1, abs(y2 - y1))
        return [int(x), int(y), int(w), int(h)]

    all_points_x = []
    all_points_y = []
   
    for line in red_lines + green_lines:
        ep = line.get("endpoints")
        if ep and len(ep) >= 4:
            all_points_x.extend([ep[0], ep[2]])
            all_points_y.extend([ep[1], ep[3]])
        elif "x1" in line:
             all_points_x.extend([line["x1"], line["x2"]])
             all_points_y.extend([line["y1"], line["y2"]])

    if not all_points_x:
        log.warning("[Template] No coordinates found in scribble lines.")
        return None, None, {}

    # 1. Calculate Global Rect
    min_x = max(0, min(all_points_x) - global_padding)
    max_x = min(img_w, max(all_points_x) + global_padding)
    min_y = max(0, min(all_points_y) - global_padding)
    max_y = min(img_h, max(all_points_y) + global_padding)
   
    gx, gy = int(min_x), int(min_y)
    gw = int(max_x - min_x)
    gh = int(max_y - min_y)
   
    if gw <= 0 or gh <= 0:
        log.warning("[Template] Invalid global rect dimensions.")
        return None, None, {}
       
    global_rect = [gx, gy, gw, gh]

    # 2. Create template image object
    template_img_obj = None
    try:
        template_img_obj = mask_img[gy:gy+gh, gx:gx+gw]
    except Exception as e:
        log.error(f"[Template] Failed to create template image object: {e}")
        return None, None, {}
       
    # 3. Calculate relative ROIs for green lines
    relative_rois_map = {}
   
    # [NEW] Get individual padding values
    pad_top = padding.get("top", 0)
    pad_bottom = padding.get("bottom", 0)
    pad_left = padding.get("left", 0)
    pad_right = padding.get("right", 0)
   
    for line in green_lines:
        line_id = line.get("id")
        if not line_id:
            continue
       
        rect = _get_rect(line)
        if not rect:
            continue
           
        rx, ry, rw, rh = rect
       
        # [MODIFIED] Apply asymmetrical padding (which is now a small, fixed debug padding)
        rx_pad = max(0, rx - pad_left)
        ry_pad = max(0, ry - pad_top)
        rw_pad = rw + pad_left + pad_right
        rh_pad = rh + pad_top + pad_bottom
       
        # Calculate relative coordinates
        rel_x = max(0, rx_pad - gx)
        rel_y = max(0, ry_pad - gy)
       
        # Clip width/height to be within the global template
        rel_w = min(rw_pad, gx + gw - rel_x)
        rel_h = min(rh_pad, gy + gh - rel_y)

        if rel_w > 0 and rel_h > 0:
            relative_rois_map[line_id] = [int(rel_x), int(rel_y), int(rel_w), int(rel_h)]

    return template_img_obj, global_rect, relative_rois_map

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
    
# [NEW] Grounding DINO (Real Impl)
def _call_grounding_dino(
    dino_prompt_text: str,    # [CHANGED] (예: "four 'darkest' fingers")
    location_hint_text: str,  # [CHANGED] (예: "top")
    mask_path: Path
) -> Optional[List[List[int]]]:
    """
    [REAL Impl] Calls the loaded Grounding DINO model via _lazy_load_dino.
    This replaces _call_grounding_dino_stub.
    
    [BUG FIX] "grounded_rois: null" 및 'ROI 전체' 문제 해결:
              VLM이 생성한 `dino_prompt_text`와 `location_hint_text`를
              '조합'하여 DINO 프롬프트를 생성. (Regex 파싱 제거)
              
    [BUG FIX] "four|three..."로 한정된 숫자 파싱 버그 해결.
    """
    
    # 1. Lazy load the model
    if not _lazy_load_dino() or _dino_pipe is None:
        log.warning("[DINO] Model not available. Skipping grounding.")
        return None
    
    try:
        processor, model, device = _dino_pipe
        
        # 2. [FIXED] Parse text prompt (Regex 제거)
        target_text = (dino_prompt_text or "").strip()
        location_hint = (location_hint_text or "").strip().lower()
        
        if not target_text:
            log.warning("[DINO] VLM provided an empty 'grounding_dino_prompt'. Skipping.")
            return None
            
        # [FIX] VLM의 위치 힌트를 DINO 프롬프트에 조합
        if location_hint and location_hint != "full_image" and location_hint not in target_text:
            # 예: "top" + "four fingers" -> "top four fingers"
            target_text = f"{location_hint} {target_text}"
        
        log.info(f"[DINO] Parsed target phrase (Final): '{target_text}'")
        
        # 3. Load image (PIL 필요)
        image = Image.open(mask_path).convert("RGB")
        
        # 4. Run inference
        inputs = processor(images=image, text=target_text, return_tensors="pt").to(device)
        
        import torch
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 5. Post-process results
        target_sizes = torch.tensor([image.size[::-1]]) # (height, width)
        
        # [BUG FIX] processor.post_process_object_detection -> post_process_grounded_object_detection
        results = processor.post_process_grounded_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=0.4 # (이 임계값은 튜닝이 필요할 수 있습니다)
        )[0]
        
        boxes = results["boxes"].cpu().tolist()
        scores = results["scores"].cpu().tolist()
        
        rois = []
        if not boxes:
            log.warning(f"[DINO] No objects found for prompt: '{target_text}'")
            return None

        # 6. Convert [xmin, ymin, xmax, ymax] to [x, y, w, h]
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            x = int(xmin)
            y = int(ymin)
            w = int(xmax - xmin)
            h = int(ymax - ymin)
            if w > 0 and h > 0:
                rois.append([x, y, w, h])
        
        log.info(f"[DINO] Successfully found {len(rois)} ROIs for prompt: '{target_text}'")
        
        # 7. [BUG FIX] "four|three..." 버그 수정
        # (VLM 힌트 텍스트에서 '숫자' 또는 '숫자 단어'를 파싱)
        num_requested = 0
        num_str_map = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        # (Regex: "4" 또는 "four" 같은 단어 찾기)
        m_num = re.search(r'\b([\d]+)\b|(\b(one|two|three|four|five|six|seven|eight|nine|ten)\b)', target_text, re.IGNORECASE)
        
        if m_num:
            try:
                if m_num.group(1): # (숫자 "4"가 매칭된 경우)
                    num_requested = int(m_num.group(1))
                elif m_num.group(2): # (단어 "four"가 매칭된 경우)
                    num_requested = num_str_map.get(m_num.group(2).lower(), 0)
            except Exception:
                pass
        
        if num_requested > 0 and len(rois) > num_requested:
             log.warning(f"[DINO] Found {len(rois)} but returning top {num_requested} based on score.")
             # (DINO 결과는 이미 점수 순으로 정렬되어 반환됨)
             return rois[:num_requested]
             
        return rois

    except Exception as e:
        log.exception(f"[DINO] Inference failed: {e}")
        return None

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
   
    [MODIFIED by USER REQUEST]
    - Green line의 start/end 힌트를 15px 반경(box) 대신,
      선분 끝에서 30px을 '선분 방향으로' 샘플링합니다.
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
           
            # --- [MODIFIED LOGIC] ---
            # 15px 반경(box) 대신, 선분 끝에서 30px을 '선분 방향으로' 샘플링
            class_hint_start = _sample_class_at_line_end(
                mask_img,
                x1, y1,  # From point 1
                x2, y2,  # Towards point 2
                sample_length=100
            )
            class_hint_end = _sample_class_at_line_end(
                mask_img,
                x2, y2,  # From point 2
                x1, y1,  # Towards point 1
                sample_length=100
            )
            # --- [MODIFIED LOGIC END] ---
           
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

def _build_gausso():
    """GaussO-Think 모델용 Langchain 클라이언트 빌더"""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url=GAUSSO_BASE_URL, openai_proxy=GAUSSO_BASE_URL, model=GAUSSO_MODEL,
        default_headers={
            "Content-Type":"application/json",
            "x-dep-ticket":GAUSSO_X_DEP_TICKET,
            "Send-System-Name":GAUSSO_SEND_SYSTEM_NAME,
            "User-Id":GAUSSO_USER_ID,
            "User-Type":GAUSSO_USER_TYPE,
            "Prompt-Msg-Id": str(uuid.uuid4()),
            "Completion-Msg-Id": str(uuid.uuid4()),
        },
        temperature=0.1,
        max_tokens=24576, # [FIXED] 8192로 설정
        timeout=1000,      # [FIXED] 600초(10분)로 설정
        max_retries=1,
    )

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
        temperature=0.1, 
        max_tokens=8192, 
        timeout=1000, 
        max_retries=1,
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

    # [FIX] 시스템 프롬프트 수정: VLM의 '자연어 알고리즘'을 구현하도록 강제
    sys_msg = (
        "너는 이미지를 분석하는 파이썬 측정 스크립트 생성 전문가다. "
        "너의 유일한 임무는 '지시 프롬프트'에 서술된 **측정 로직(Logic)**을 "
        "`mask_path` 이미지 위에서 실행하는 **알고리즘 코드(예: cv2.findContours, cv2.fitLine)**로 구현하는 것이다."
        "마크다운 코드펜스는 절대 사용하지 말 것.\\n\\n"
        
        "## [매우 중요] 알고리즘 준수 규칙 (필수 준수):\\n" # [NEW SECTION]
        "1. **(최우선 준수)** 너는 '지시 프롬프트'의 'PIPELINE STEPS' 또는 'PSEUDOCODE' 섹션에 명시된 **자연어 알고리즘을 반드시 문자 그대로(literally) 구현**해야 한다.\\n"
        "2. **(대체 금지)** 만약 'PIPELINE STEPS'가 '...상위 40% 밴드... 최상단 피크 4개...'와 같이 **복잡한 로직**을 지시했다면, 너는 이를 **절대** '모든 점 찾기'(`topmost_points`) 같은 **간단한 로직으로 임의 변경(대체)해서는 안 된다.**\\n"
        "3. **(그룹핑 준수)** 'PIPELINE STEPS'가 '...3개의 간격(Interval)마다... 2개의 라인...' 같은 **그룹핑 로직**을 지시했다면, 너의 코드는 반드시 이 `for interval in intervals:` 구조를 따라야 한다.\\n"

        "## [매우 중요] 기하학 알고리즘 규칙 (필수 준수):\\n"
        "1. **(최우선 알고리즘)** '지시 프롬프트'의 'PIPELINE STEPS' 또는 'PSEUDOCODE' 섹션에 **자연어 알고리즘**(예: '...4개 핑거의 최상단 점을 피팅...')이 명시되어 있다면, **너는 반드시 그 자연어 알고리즘을 그대로 파이썬 코드로 구현해야 한다.** (예: `darkest` 클래스 마스크에서 **여러** 컴포넌트를 찾아, **각** 컴포넌트의 최상단 점을 `np.vstack`으로 쌓고, `cv2.fitLine` 호출)\\n"
        "2. **(법선 계산)** 만약 '지시 프롬프트'가 빨간 선과 초록 선의 '법선' 또는 '수직' 관계를 암시한다면, 너의 코드는 `cv2.fitLine`으로 계산된 빨간 선의 벡터(vx, vy)를 기반으로 **법선 벡터(-vy, vx)**를 계산하고, 이를 사용해 **최단 거리를 투영(projection)**해야 한다.\\n"

        "## [매우 중요] 출력 규칙 (필수 준수):\\n"
        "1. '지시 프롬프트'의 SDIFF 힌트(예: red.count: 1, green.count: 6)를 **정확히** 구현해야 한다. (예: 6개의 초록 선을 그리기 위해 6번 루프 실행)\\n"
        "2. **`overlay.png`에는 절대 `cv2.drawContours`를 사용하지 마라.** 컨투어는 계산에만 사용하고, 최종 오버레이에는 **오직 `cv2.line`**만 사용하라.\\n"
        "3. 오버레이에는 **오직 빨간색(BGR: 0,0,255)과 초록색(BGR: 0,255,0) 선**만 허용된다. 파란색, 노란색 등 다른 색을 절대 사용하지 마라.\\n"
        
        "## [매우 중요] 클래스 값(class_val) 처리 규칙:\\n"
        "1. '지시 프롬프트'에 포함된 **[MASK METADATA]** 블록을 반드시 확인하라.\\n"
        "2. `[MASK METADATA]`의 'classes' 리스트(예: `[30, 50, 70]`)에서 '지시 프롬프트'의 설명(예: 'darkest')과 가장 일치하는 **올바른 `class_val`을 선택**하여 코드에 사용해야 한다.\\n"
    )
    
    user_msg = (
        "아래 지시 프롬프트와 few-shot 텍스트를 바탕으로 전체 파이썬 코드(measure.py)만 출력하세요.\\n"
        "- 필수 인자: --mask_path, --meta_root, --out_dir\\n"
        "- out_dir에 overlay.png, measurements.csv 생성\\n"
        "- meta_utils.__wrap_load_pixel_scale_um(mask_path, meta_root)는 (umx, umy, classes, meta_path) 4개 값 반환으로 가정\\n"
        "- draw_text 기본 False, line_thickness 기본 5 등 사용자 옵션 반영\\n"
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

# [NEW] Qwen-VL 호출을 위한 JSON 프롬프트 구조 (동적 생성됨)
# 이 구조는 Qwen에게 "무엇을 채워야 하는지" 알려주는 '틀'입니다.
def _build_qwen_json_prompt_structure(cv_sdiff: Dict, brightness_map: Dict) -> str:
    """
    [CHANGED] 3개 이미지 + CV SDIFF 힌트 + '밝기 맵'을 Qwen-VL에 전달합니다.
   
    [LCS PLAN - STEP 1]
    - [NEW] "coordinate_system" 블록 추가 (1. VLM'의 '의도' 선언)
    - [NEW] "lcs_orientation_intent" 키 추가 (2. VLM'의 '의도' 선언)
   
    [ANCHORING PLAN - STEP 1]
    - [REMOVED] 'green_measures_refined.vlm_refinement.group_id' 질문 삭제
    - [NEW] 'green_measures_refined.vlm_refinement.anchor_logic_hint' 질문 추가
   
    [GREEN DINO PLAN - STEP 1]
    - [NEW] 'green_measures_refined.vlm_refinement'에 'grounding_dino_prompt' 및 'grounding_location_hint' 추가
    [LCS PLAN - STEP 1]
    - [NEW] "coordinate_system" 블록 추가 (1. VLM'의 '의도' 선언)
    - [NEW] "lcs_orientation_intent" 키 추가 (2. VLM'의 '의도' 선언)
   
    [FARTHEST POINT PLAN - STEP 1]
    - [REMOVED] 'anchor_logic_hint' 삭제 (construction_method_end가 대체)
    - [REMOVED] Green Line DINO 프롬프트 (grounding_dino_prompt/hint) 삭제
    - [MODIFIED] 'construction_method_end' 질문을 'farthest point' (가장 먼 점)를 찾도록 고도화
   
    [FINAL PLAN - STEP 1 (Correction)]
    - [MODIFIED] 'construction_method_end'의 'e.g.' (예시)에서 "farthest"를 제거하여
                  Qwen이 스스로 알고즘을 추론하도록 '개방형 질문'으로 수정.
                 
    [USER REQUEST: 4-HINT (U/V) STRUCTURE]
    - [REPLACED] 'construction_method_start' and 'construction_method_end'
    - [NEW] Added 'u_placement_logic_hint_start', 'u_placement_logic_hint_end',
    - [NEW] 'v_placement_logic_hint_start', 'v_placement_logic_hint_end'
   
    [USER REQUEST: OVERFITTING-PROOF (v5 - Grouping Logic)]
    - [MODIFIED] Re-implemented the user's original, long, general-purpose prompts.
    - [REMOVED] All hardcoded hints like 'red_1' or 'fingers' from the prompt text.
    - [MODIFIED] `grouping_logic.description` to be general and ask for patterns,
      removed "6 independent lines" hint.
     
    [USER REQUEST: BUG-FIX (v8) - Force VLM to obey CV hints (Dynamic Prompting)]
    - [MODIFIED] Signature now accepts brightness_map (Dict).
    - [MODIFIED] Dynamically creates reverse_brightness_map.
    - [MODIFIED] ALL FOUR u/v_placement_... prompts are now dynamically generated
      to inject the specific class name (e.g., 'middle_gray_1') based on the
      CV hint (e.g., 30), forcing VLM to be consistent.
    """
    import json
   
    # [NEW] brightness_map (e.g., {'darkest': 10})을
    # reverse_brightness_map (e.g., {10: 'darkest'})으로 변환합니다.
    reverse_brightness_map = {}
    brightness_descriptors = []
    if brightness_map:
        try:
            for k, v in brightness_map.items():
                reverse_brightness_map[int(v)] = str(k)
            brightness_descriptors = list(brightness_map.keys())
        except Exception:
            pass # 실패 시 빈 맵 사용

    desc_options_str = f"e.g., {json.dumps(brightness_descriptors[:2])}" if brightness_descriptors else "(e.g., ['darkest'])"
    desc_list_str = f"Available brightness descriptors: {json.dumps(brightness_descriptors)}"

    json_prompt_structure_obj = {
        "analysis_summary": "One-sentence summary of the measurement task.",
       
        # --- (Grouping Logic: 변경 없음) ---
        "grouping_logic": {
            "description": "CRITICAL (Grouping): Do the green measurement items (lines) appear to repeat in a similar, structured way, or is each line completely unique and independent? (e.g., 'Repeats: 2 lines per structural unit', 'Independent: each line has a unique logic')",
            "interval_definition": "If they repeat or are grouped, what geometric structure defines the group or interval? (e.g., 'Grouped by the 'darkest' finger structures', 'Grouped by grooves')",
            "geometric_relationship": "CRITICAL: What is the geometric relationship between red and green lines? (e.g., 'Green lines are perpendicular/normal to the red line', 'Green lines are all vertical')",
            "direction_relative_to_red": "CRITICAL: In which direction do the green lines point *relative to the red line*? (e.g., 'downwards', 'upwards', 'towards the bottom of the image')"
        },
       
        # --- (Coordinate System: 변경 없음) ---
        "coordinate_system": {
            "type": "What is the coordinate system type? Choose one: ['absolute', 'local']",
            "reason": "Why? (e.g., 'A rotated red guide line exists', 'No red guides')",
            "local_definition": {
                "primary_axis_guide_id": "CRITICAL: Which *single* red guide (e.g., 'red_1') defines the 'main reference axis' for all local measurements? (Usually the longest or most complex red line)",
                "green_line_orientation": "Relative to this primary axis, are green lines drawn 'parallel', 'perpendicular', or 'both'?"
            }
        },
       
        "red_guides_refined": [],
        "green_measures_refined": []
    }

    # 1. RED 가이드라인 (변경 없음)
    try:
        cv_red_lines = cv_sdiff.get("red_lines_detail", [])
        for line in cv_red_lines:
            json_prompt_structure_obj["red_guides_refined"].append({
                "id": line.get("id"),
                "cv_hint_geometric_fact": {"orientation": line.get("orientation")},
                "cv_hint_to_correct": {"detected_class_hint_by_location": line.get("detected_class_hint")},
                "vlm_refinement": {
                    "functional_role": "What is this line's functional role? (e.g., 'Acts as a horizontal reference', 'Acts as a vertical guide')",
                    "actual_geometry": "CRITICAL: What is the *actual* geometry of this line? (e.g., 'A single diagonal line', 'A perfectly horizontal line', 'A curved line')",
                    "construction_method": "CRITICAL (Detailed): How to build this line? (e.g., 'Fit a single line (cv2.fitLine) to the top-most points of the 4 'darkest' fingers', 'Find center-line of 'brightest' class', 'Find all 'darkest' pixels and fit a line')",
                    "grounding_dino_prompt": "CRITICAL: What is the short noun phrase for Grounding DINO? (This defines the *search area* for Green Lines) (e.g., 'four fingers', 'the 4 'darkest' fingers', 'the top structure')",
                    "grounding_location_hint": "CRITICAL: Choose ONE location keyword that best describes the area: ['top', 'bottom', 'left', 'right', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'center', 'full_image']",
                    "required_classes": f"CRITICAL: List of brightness descriptors needed. {desc_options_str}. {desc_list_str}",
                    "lcs_orientation_intent": "CRITICAL: What is this line's orientation *relative to the local grid*? Choose one: ['primary_axis', 'parallel_to_primary', 'perpendicular_to_primary', 'unrelated']"
                }
            })
    except Exception as e:
        log.warning(f"[Qwen] Failed to build red prompt: {e}")

    # 2. GREEN 측정선 (핵심 수정)
    try:
        cv_green_lines = cv_sdiff.get("green_lines_detail", [])
        for line in cv_green_lines:
           
            cv_start_hint_val = line.get("detected_class_hint_start")
            cv_end_hint_val = line.get("detected_class_hint_end")

            # --- [NEW] 힌트를 문자열로 변환 ---
            def _get_truth_str(hint_val):
                if isinstance(hint_val, str):
                    return f"the guide line '{hint_val}'"
                if isinstance(hint_val, int) and hint_val in reverse_brightness_map:
                    return f"the class '{reverse_brightness_map[hint_val]}' (value: {hint_val})"
                return f"the raw hint '{hint_val}'"

            truth_str_start = _get_truth_str(cv_start_hint_val)
            truth_str_end = _get_truth_str(cv_end_hint_val)
            # --- [NEW] END ---

            json_prompt_structure_obj["green_measures_refined"].append({
                "id": line.get("id"),
                "cv_hint_geometric_fact": {
                    "orientation": line.get("orientation"),
                    "semantic": line.get("semantic"),
                },
                "cv_hint_to_correct": {
                    "detected_class_hint_start": cv_start_hint_val,
                    "detected_class_hint_end": cv_end_hint_val
                },
                "vlm_refinement": {
                    # --- (기존 로직) ---
                    "start_is_connected_to_red_guide": "CRITICAL: Does the start point visually connect to *any* red guide line? (true/false)",
                    "end_is_connected_to_red_guide": "CRITICAL: Does the end point visually connect to *any* red guide line? (true/false)",
                    "connected_red_guide_id_start": "If 'start_is_connected...' is TRUE, what is the ID of the red line it connects to? (e.g., 'red_1', 'red_2', or null)",
                    "connected_red_guide_id_end": "If 'end_is_connected...' is TRUE, what is the ID of the red line it connects to? (e.g., 'red_1', 'red_2', or null)",
                   
                    # --- [MODIFIED] "4-Hint" (U/V) - 동적 프롬프트 주입 ---
                   
                    "u_placement_logic_hint_start": (
                        f"CRITICAL: The TRUTH for the start point is {truth_str_start}. "
                        "Describe the pixel-level algorithm to find the U-position (anchor) "
                        f"based *only* on THAT truth. (e.g., 'find centroid of {truth_str_start}')."
                    ),

                    "u_placement_logic_hint_end": (
                        f"CRITICAL: The TRUTH for the end point is {truth_str_end}. "
                        "If the measurement is vertical, this value is likely null (to reuse u_start). "
                        "If horizontal (width), this defines the U-END. "
                        f"Describe the algorithm to find this U-END (e.g., 'find right-most point of {truth_str_end}') "
                        "OR return null if u_start is reused."
                    ),

                    "v_placement_logic_hint_start": (
                        f"CRITICAL: The TRUTH for the start point is {truth_str_start}. "
                        "Describe the algorithm to find the V-START based *only* on THAT truth. "
                        f"If the truth is a guide line (e.g., 'red_1'), the V-START is 0 (projected onto the guide). "
                        f"If the truth is a class (e.g., 'middle_gray_1'), describe how to find the V-START on THAT class boundary."
                    ),

                    "v_placement_logic_hint_end": (
                        f"CRITICAL: The TRUTH for the end point is {truth_str_end}. "
                        "Describe the algorithm to find the V-END based *only* on THAT truth. "
                        f"If vertical (height), infer the boundary (e.g., 'bottom-most point of {truth_str_end}'). "
                        f"If horizontal (width), this is likely null (to reuse v_start). "
                        "Provide the algorithm OR return null."
                    ),
                    # --- [MODIFIED] END ---
                                   
                    "lcs_orientation_intent": "CRITICAL: What is this line's orientation *relative to the local grid*? Choose one: ['parallel_to_primary', 'perpendicular_to_primary', 'unrelated']"
                }
            })
    except Exception as e:
        log.warning(f"[Qwen] Failed to build green prompt: {e}")

    # 3. 직렬화
    return json.dumps(json_prompt_structure_obj, indent=2)


def _sample_class_at_line_end(
    mask_img: np.ndarray,
    x1: int, y1: int,  # 샘플링을 시작할 끝점 (FROM)
    x2: int, y2: int,  # 방향을 알려줄 다른 쪽 끝점 (TOWARDS)
    sample_length: int = 30
) -> Optional[int]:
    """
    (x1, y1)에서 (x2, y2) 방향으로 'sample_length'(예: 30) 픽셀만큼
    '선분을 따라서' 샘플링하고, 가장 빈번한 non-zero 클래스를 반환합니다.
    """
    if mask_img is None:
        return None
    try:
        # 1. 전체 선분의 총 픽셀 수 계산
        num_points_total = int(np.hypot(x2 - x1, y2 - y1))
        if num_points_total == 0:
            # 점 하나만 샘플링 (Fallback)
            if 0 <= y1 < mask_img.shape[0] and 0 <= x1 < mask_img.shape[1]:
                val = mask_img[y1, x1]
                return int(val) if val > 0 else None
            return None

        # 2. 전체 선분의 모든 좌표 생성
        x_coords_all = np.linspace(x1, x2, num_points_total).astype(int)
        y_coords_all = np.linspace(y1, y2, num_points_total).astype(int)
       
        # 3. [요청 사항] 시작점에서 'sample_length' 만큼만 좌표를 자름
        num_to_sample = min(num_points_total, sample_length)
        x_coords = x_coords_all[:num_to_sample]
        y_coords = y_coords_all[:num_to_sample]
       
        # 4. 유효한 좌표만 필터링 (선이 화면 밖에서 시작할 경우 대비)
        H, W = mask_img.shape
        valid_idx = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)
        if not np.any(valid_idx):
            return None
           
        x_coords, y_coords = x_coords[valid_idx], y_coords[valid_idx]
       
        # 5. 픽셀 값 샘플링 (0이 아닌 값만)
        samples = mask_img[y_coords, x_coords]
        non_zero_samples = samples[samples > 0]
       
        if non_zero_samples.size > 0:
            # 6. 가장 빈번하게 나타나는 클래스 값(최빈값) 반환
            counts = np.bincount(non_zero_samples)
            return int(np.argmax(counts))
        return None # 0(배경) 또는 유효 샘플 없음
    except Exception as e:
        log.error(f"[_sample_class_at_line_end] Sampling failed: {e}")
        return None


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

def _qwen_summarize_scribble(mask_path: Optional[Path], merge_path: Path, scribble_path: Path, cv_sdiff: Dict, brightness_map: Dict) -> Dict[str, Any]:
    """
    [CHANGED] 3개 이미지 + CV SDIFF 힌트 + '밝기 맵'을 Qwen-VL에 전달합니다.
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

    sdiff: Dict[str, Any] = dict(cv_sdiff) # CV SDIFF(v1) 전체 복사
    sdiff.setdefault("notes", {}) # 'notes' 키가 없으면 생성

    # Qwen 호출
    raw_text = ""
    try:
        _lazy_load_qwen()
        if _qwen_pipe is not None:
            import torch
            processor, model, is_mm, meta = _qwen_pipe

            # --- [PROMPT CHANGE START] ---
            # [MODIFIED] 'brightness_map' (전체 맵)을 프롬프트 빌더로 전달
            json_prompt_structure = _build_qwen_json_prompt_structure(cv_sdiff, brightness_map)

            content = []
            prompt_parts = []
           
            # [USER REQUEST] "Algorithm Inference" 글로벌 규칙 추가
            prompt_parts.append(
                "You are an expert vision analyst.\n"
                "In the scribble image, RED and GREEN lines are already drawn by a human.\n"
                "Treat each GREEN line as a label showing the result of a measurement.\n"
                "Your job is to infer the underlying geometric rule and pixel-level algorithm that could have produced a similar line from scratch using only the mask image, the red guide, and ROI hints.\n"
                "You may look at the existing GREEN line as a visual hint, but your final explanation MUST NOT depend on “knowing the green line in advance”; it must be an algorithm that could recreate that line even if the green scribble were removed.\n\n"
                "Now, analyze the provided images:"
            )
            # [USER REQUEST] END
           
            img_idx_counter = 0
            if pil_mask:
                content.append({"type": "image"})
                prompt_parts.append(f"- Image {img_idx_counter+1}: 'Mask' (segmentation input with different brightness levels)")
                img_idx_counter += 1
            if pil_scribble:
                content.append({"type": "image"})
                prompt_parts.append(f"- Image {img_idx_counter+1}: 'Scribble' (RED guides, GREEN measures)")
                img_idx_counter += 1
            if pil_merge:
                content.append({"type": "image"})
                prompt_parts.append(f"- Image {img_idx_counter+1}: 'Merged Example' (desired output)")
                img_idx_counter += 1

            # [PROMPT] "CV 힌트가 틀릴 수 있다"고 명확히 지시
            # (기존 "Analyze the 'Scribble'..."는 글로벌 규칙으로 대체되었으므로 제거)
            prompt_parts.append("\nYour first task is to **VERIFY or CORRECT** the `cv_hint` for each line.")
            prompt_parts.append("The `cv_hint` (e.g., 50) is just a guess based on location and **IS LIKELY WRONG**.")
            prompt_parts.append("Use your visual reasoning (e.g., 'the red guide *looks like* it's fitting the 'darkest' fingers') to determine the *true* classes needed.")
           
            # [MODIFIED] `brightness_descriptors` 대신 `brightness_map`의 키 목록을 표시
            brightness_descriptors = list(brightness_map.keys())
            prompt_parts.append(f"You MUST use the 'brightness descriptors' from this list: {json.dumps(brightness_descriptors)}")
           
            prompt_parts.append("Fill in the 'vlm_refinement' sections based on your reasoning.")
            prompt_parts.append("Provide your analysis STRICTLY in the following JSON format. Do NOT add any other text, explanations, or 'thinking' steps.")
           
            prompt_parts.append(json_prompt_structure)
            # --- [PROMPT CHANGE END] ---

            user_text = "\n".join(prompt_parts)

            content.append({"type": "text", "text": user_text})
            messages = [{"role": "user", "content": content}]

            dev = model.device

            if is_mm and pil_images:
                prompt_string = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = processor(
                    text=prompt_string,
                    images=pil_images,
                    return_tensors="pt"
                ).to(dev)
            else:
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

            # --- [SOLUTION] Increase max_new_tokens ---
            gen_kwargs = dict(max_new_tokens=8192) # 2048 -> 8192
            # --------------------------------------------
           
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs, do_sample=True)

            # (이하 디코딩 및 파싱 로직)
            if hasattr(processor, "batch_decode"):
                raw_text = processor.batch_decode(out, skip_special_tokens=True)[0]
               
                if raw_text.strip().startswith(prompt_string.strip()):
                     raw_text = raw_text.strip()[len(prompt_string.strip()):].strip()
                elif "ASSISTANT:" in raw_text:
                     raw_text = raw_text.split("ASSISTANT:")[-1].strip()
                elif raw_text.strip().startswith(user_text.strip()):
                     raw_text = raw_text.strip()[len(user_text.strip()):].strip()

                if raw_text.startswith("{") and raw_text.endswith("}"):
                    pass
                else:
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
        log.exception(f"[QVEN] call failed: {e}")
        raw_text = f"(Qwen call failed: {e})"

    sdiff["notes"]["raw_text"] = (raw_text or "")
    sdiff["notes"]["source"] = "hybrid_cv_geometry_qwen_notes"
    sdiff["notes"]["mask_name"] = mask_path.name if mask_path else None
    sdiff["notes"]["merge_name"] = merge_path.name
    sdiff["notes"]["scribble_name"] = scribble_path.name

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

    # [CHANGED] VLM(Llama4)의 시스템 프롬프트 (0번 섹션 인지 및 '알고리즘 번역' 규칙 추가)
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
        "1. **(최우선 지시)** '0) IMAGE SUMMARY' (Qwen JSON)의 `red_guides_refined.vlm_refinement.construction_method` (예: 'Fit top-most points of the 4 'darkest' class fingers')를 확인하라.\\n"
        "2. **(알고리즘 번역)** 만약 `construction_method`가 'top 4 peaks' 또는 '상위 밴드' 같은 **복잡한 알고리즘**을 명시한다면, 너는 'PIPELINE STEPS'에 이 알고리즘을 **'여러 단계의 구체적인 CV 로직'**으로 '번역'해야 한다.\\n"
        "3. **(나쁜 예 - 요약 금지)** 'PIPELINE STEPS: 1. class 10으로 라인을 피팅합니다.' -> [금지! 정보 누락]\\n"
        "4. **(좋은 예 - 구체적 번역)** 'PIPELINE STEPS: 1. 마스크 상위 40% 밴드를 자릅니다. 2. 밴드 내 class 10의 컴포넌트를 찾습니다. 3. 각 컴포넌트의 최상단 피크(peak)를 찾습니다. 4. 이 상위 4개의 피크들로 `cv2.fitLine`을 수행합니다.' -> [필수!]\\n"
        "5. **(힌트)** `SDIFF.red.lines[i].detected_class_hint` (예: 10)는 이 로직에 사용될 *클래스 값*을 암시한다.\\n"

        "### 2. GREEN 측정선 (SDIFF.green):\\n"
        "1. **(그룹핑)** '0) IMAGE SUMMARY'의 `grouping_logic` (예: 'pairs', 'interval_definition')과 각 Green 라인의 `group_id`를 확인하라.\\n"
        "2. **(알고리즘 번역)** 'PIPELINE STEPS'에 이 그룹핑 로직을 '알고리즘'으로 번역하라. (예: '1. Red 앵커 사이의 3개 간격(Interval)을 정의합니다. 2. 각 간격(Interval)마다 루프를 돕니다. 3. 각 루프에서 class 30과 class 50에 대한 2개의 라인을 계산합니다.')\\n"
        "3. **(힌트)** `detected_class_hint_start` (예: 'red_1')와 `detected_class_hint_end` (예: 10)를 사용하여 각 라인의 시작/끝 알고리즘을 추론하라.\\n"
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

def _parse_qwen_json(raw_text: str) -> Dict:
    """
    Qwen의 raw_text (json 또는 'assistant\n{...}' 형식)를 파싱합니다.
    [FIX] 'assistant' 키워드 뒤의 '마지막' JSON 블록을 정확히 파싱하도록 수정합니다.
    [NEW] 표준 json 대신 'json5'를 사용하여 후행 쉼표, 주석 등 비표준 JSON을 너그럽게 파싱합니다.
    """
    import json, re
    if not raw_text:
        return {}
    try:
        json_str = raw_text
        
        assistant_keyword = "assistant"
        
        # 'assistant' 키워드를 찾아 그 이후의 텍스트만 추출
        parts = re.split(f"\\b{assistant_keyword}\\b\\s*", raw_text, maxsplit=1, flags=re.IGNORECASE)
        
        if len(parts) > 1:
            json_str = parts[1].strip()
        else:
            matches = list(re.finditer(r"\{[\s\S]*\}", raw_text, re.MULTILINE))
            if matches:
                json_str = matches[-1].group(0)
            else:
                json_str = raw_text 

        final_matches = list(re.finditer(r"\{[\s\S]*\}", json_str, re.MULTILINE))
        if not final_matches:
            log.warning(f"[Qwen] No JSON block found after 'assistant' or in fallback: {raw_text[:200]}")
            return {}
            
        final_json_str = final_matches[-1].group(0)
        
        # [CHANGED] json.loads -> json5.loads
        return json5.loads(final_json_str)

    except Exception as e:
        log.warning(f"[Qwen] Failed to parse raw_text JSON (with json5): {e}")
        log.debug(f"[Qwen] Failing raw_text: {raw_text}")
        return {}

    except Exception as e:
        log.warning(f"[Qwen] Failed to parse raw_text JSON (with json5): {e}")
        log.debug(f"[Qwen] Failing raw_text: {raw_text}")
        return {}

    except Exception as e:
        log.warning(f"[Qwen] Failed to parse raw_text JSON: {e}")
        log.debug(f"[Qwen] Failing raw_text: {raw_text}")
        return {}

def _apply_qwen_refinement_to_sdiff_v1(sdiff_v1: Dict, brightness_map: Dict[str, int]) -> Dict:
    """
    [CHANGED] Qwen의 추론 결과('notes.raw_text')를 파싱하고, 'brightness_map'을 사용해
    SDIFF v1 객체('red_lines_detail' 등)의 CV 힌트를 덮어씁니다.
    
    [BUG FIX] VLM 힌트가 'detected_...' (CV) 키를 덮어쓰는 버그 수정.
              'detected_...' (CV 원본)은 보존하고,
              VLM 힌트는 'final_class_hint_...'라는 '새로운' 키에 저장하여
              GPT-OSS가 'null' 힌트를 받는 문제를 해결합니다.
    """
    
    # 1. Qwen 추론 결과(JSON) 파싱
    qwen_json = _parse_qwen_json(sdiff_v1.get("notes", {}).get("raw_text", ""))
    if not qwen_json:
        log.warning("[Refinement] Qwen JSON (notes.raw_text) is empty or invalid. Skipping refinement.")
        return sdiff_v1 # 수정 없이 원본 반환
    if not brightness_map:
        log.warning("[Refinement] Brightness map is empty. Skipping refinement.")
        return sdiff_v1 # 맵이 없으면 변환 불가

    log.info(f"[Refinement] Applying Qwen VLM refinement to SDIFF v1 hints using brightness map: {brightness_map}")

    # 2. SDIFF v1 상세 라인에 빠르게 접근하기 위한 Map 생성
    v1_red_map = {
        line.get("id"): line for line in sdiff_v1.get("red_lines_detail", []) if line.get("id")
    }
    v1_green_map = {
        line.get("id"): line for line in sdiff_v1.get("green_lines_detail", []) if line.get("id")
    }

    # [NEW] VLM이 정제한 'red_1'의 '생성용 클래스'를 저장 (e.g., {"red_1": 10})
    refined_red_construction_class_map = {}

    # 3. RED 가이드라인 힌트 덮어쓰기
    for qwen_line in qwen_json.get("red_guides_refined", []):
        line_id = qwen_line.get("id")
        vlm_ref = qwen_line.get("vlm_refinement", {})
        req_classes_desc = vlm_ref.get("required_classes", []) # e.g., ["darkest"]
        
        if line_id in v1_red_map and req_classes_desc:
            cv_line = v1_red_map[line_id]
            original_hint = cv_line.get("detected_class_hint")
            
            try:
                # Map descriptors to class IDs: e.g., ["darkest"] -> [10]
                mapped_classes = [brightness_map[desc] for desc in req_classes_desc if desc in brightness_map]
                if not mapped_classes:
                    log.warning(f"[Refinement] RED '{line_id}': VLM provided descriptors {req_classes_desc} not in map.")
                    continue
                
                # VLM이 정제한 '생성용' 클래스 (e.g., 10)
                refined_hint = int(mapped_classes[0])
                
                # [FIXED] 'detected_...' (CV) 키는 보존하고, 'final_...' (VLM) 키를 새로 생성
                cv_line["final_class_hint"] = refined_hint
                cv_line["vlm_required_classes"] = mapped_classes 
                
                # [NEW] Green이 참조할 수 있도록 맵에 저장
                refined_red_construction_class_map[line_id] = refined_hint
                
                log.info(f"[Refinement] RED line '{line_id}': CV hint {original_hint} -> VLM final hint {refined_hint} (from {req_classes_desc})")

            except Exception as e:
                log.error(f"[Refinement] Error processing RED line '{line_id}': {e}")
        elif line_id:
            log.warning(f"[Refinement] RED line '{line_id}' from Qwen JSON not found in SDIFF v1.")

    # 4. GREEN 측정선 힌트 덮어쓰기
    for qwen_line in qwen_json.get("green_measures_refined", []):
        line_id = qwen_line.get("id")
        vlm_ref = qwen_line.get("vlm_refinement", {})
        req_classes_desc = vlm_ref.get("required_classes", []) # e.g., ["darkest", "middle_gray_1"]
        start_point_str = vlm_ref.get("start_point", "") # e.g., "red_1" or "top of 'darkest' class"

        if line_id in v1_green_map:
            cv_line = v1_green_map[line_id]
            original_start = cv_line.get("detected_class_hint_start")
            original_end = cv_line.get("detected_class_hint_end")

            try:
                # Map all required class descriptors
                mapped_classes = [brightness_map[desc] for desc in req_classes_desc if desc in brightness_map]
                cv_line["vlm_required_classes"] = mapped_classes # Store for Llama4/GPT-OSS
                
                refined_start = original_start # default
                refined_end = original_end   # default

                # 4-1. Start 힌트 정제 (사용자님이 지적한 핵심 로직)
                if start_point_str.startswith("red_"):
                    # 시작점을 ID 문자열("red_1") 자체로 설정
                    refined_start = start_point_str 
                elif mapped_classes:
                    # 시작점이 가이드라인이 아닌 경우 (e.g., 'top of 'darkest' class')
                    refined_start = int(mapped_classes[0])
                
                # 4-2. End 힌트 정제
                if mapped_classes:
                    # 끝점은 'required_classes'의 마지막 클래스로 가정
                    refined_end = int(mapped_classes[-1])

                # SDIFF v1 객체의 CV 힌트를 VLM 힌트로 덮어쓰기
                cv_line["detected_class_hint_start"] = refined_start
                cv_line["detected_class_hint_end"] = refined_end

                if (original_start != refined_start) or (original_end != refined_end):
                    # 로그가 (50, 10) -> ('red_1', 10)으로 찍힐 것임
                    log.info(f"[Refinement] GREEN line '{line_id}': CV hint ({original_start},{original_end}) -> VLM hint ('{refined_start}',{refined_end}) (from {req_classes_desc})")

            except Exception as e:
                 log.error(f"[Refinement] Error processing GREEN line '{line_id}': {e}")
        elif line_id:
            log.warning(f"[Refinement] GREEN line '{line_id}' from Qwen JSON not found in SDIFF v1.")

    # 5. 수정된 SDIFF v1 객체 반환
    return sdiff_v1

# ------------------------------------------
# [NEW] Qwen으로 STRUCTURED_DIFF 생성
# ------------------------------------------
@app.post("/vl/sdiff_qwen")
async def vl_sdiff_qwen(data: Dict = Body(...)):
    """
    [CHANGED] CV(Geometry) + Qwen(Semantics) 하이브리드 SDIFF 생성
   
    [BUG FIX] _upgrade_sdiff_to_lite 함수가 green line 개수를 3개로 잘못 처리하는 버그 수정.
             _upgrade_sdiff_to_lite 호출을 제거하고,
             이 함수 내에서 v2_lite 객체를 '수동으로' 빌드하여
             모든 Green Line(6개)을 '무조건 복사(보존)'하도록 수정.
             
    [USER REQUESTED CHANGE]
    - "2개씩 3쌍" 논리를 구현하기 위해, VLM의 '논리'("3 pairs")가
      VLM의 '데이터'("independent_1")를 '덮어쓰도록'
      group_id 번역 로직을 5.3 섹션에 구현합니다.
     
    [USER REQUESTED CHANGE]
    - `grounded_rois: null` 버그를 해결하기 위해,
      `_call_grounding_dino`가 VLM이 생성한 `grounding_dino_prompt`와 'location_hint_text'
      를 '함께' 사용하도록 수정합니다.
     
    [USER REQUESTED CHANGE]
    - DINO ROI 시각화를 위해 `dino_debug.png` 파일 생성 로직 추가.
    - [NEW] DINO ROI가 너무 큰 문제를 해결하기 위해 'ROI Slicing' 로직(4.7) 추가.
   
    [FIX FOR DINO DEBUG IMAGE]
    - [FIXED] 4.7 (ROI Slicing) 로직을 4.6 (Debug Image) 블록 밖으로
      이동시켜, 이미지 로드 실패 시에도 슬라이싱이 누락되지 않도록 수정.
     
    [FIX FOR DINO DEBUG IMAGE (v4)]
    - [FIXED] 4.6 (Debug Image) 블록이 오직 '최종 슬라이싱된'
      좌표(dino_debug_rois_final)만 그리도록 수정.
    - [CRITICAL BUG FIX] cv2.rectangle의 두 번째 Y좌표를 'fh'가 아닌 'fy+fh'로 수정.
    - [COLOR CHANGE] 박스 색상을 Green -> Yellow (0, 255, 255)로 변경.
   
    [LCS PLAN - STEP 2]
    - [FIXED] 5.3 V6 SDIFF 뼈대 생성 시 'coordinate_system' 힌트 주입
    - [FIXED] 5.4 힌트맵 생성 시 'lcs_orientation_intent'와 'construction_method' 맵핑 로직 추가
    - [FIXED] 5.6/5.7 '데이터' 빌드 시 'lcs_orientation_intent', 'construction_logic_hint' 키 주입
    [LCS PLAN - STEP 2]
    - [FIXED] 5.3 V6 SDIFF 뼈대 생성 시 'coordinate_system' 힌트 주입
    - [FIXED] 5.4 힌트맵 생성 시 'lcs_orientation_intent'와 'construction_method' 맵핑 로직 추가
    - [FIXED] 5.6/5.7 '데이터' 빌드 시 'lcs_orientation_intent', 'construction_logic_hint' 키 주입
   
    [ANCHORING PLAN - STEP 2]
    - [REMOVED] 5.2 '논리'에서 'group_id' 외과적 제거 로직 삭제
    - [NEW] 5.4.2 [NEW] LCS / Construction / Anchor 힌트맵 생성 로직 추가
    - [REMOVED] 5.5 'group_id_vlm' 번역 로직 삭제 (anchor_logic_hint로 대체됨)
    - [NEW] 5.7 '데이터' (Green Lines) 빌드 시 'anchor_logic_hint' 주입
   
    [GREEN DINO PLAN - STEP 2]
    - [NEW] 4.5.1 Green Line DINO 호출 루프 추가
    - [NEW] 5.4.2 Green Line DINO 힌트맵 생성 로직 추가
    - [NEW] 5.7 '데이터' (Green Lines) 빌드 시 'grounded_rois' 주입
    [LCS PLAN - STEP 2]
    - [FIXED] 5.3 V6 SDIFF 뼈대 생성 시 'coordinate_system' 힌트 주입
    - [FIXED] 5.4 힌트맵 생성 시 'lcs_orientation_intent'와 'construction_method' 맵핑 로직 추가
    - [FIXED] 5.6/5.7 '데이터' 빌드 시 'lcs_orientation_intent', 'construction_logic_hint' 키 주입
   
    [FARTHEST POINT PLAN - STEP 2]
    - [REMOVED] 4.5.2 Green Line DINO 호출 로직 삭제
    - [REMOVED] 4.7 Green Line DINO ROI 슬라이싱 로직 삭제
    - [MODIFIED] 4.6 Debug Image에서 Green ROI 그리기 로직 삭제
    - [NEW] 5.4.2 'construction_method_start/end' (Green) 맵 생성 로직 추가
    - [REMOVED] 5.5 'group_id_vlm' 번역 로직 삭제
    - [NEW] 5.7 '데이터' (Green Lines) 빌드 시 'construction_method_start/end' 주입 및 'grounded_rois' 삭제
   
    [USER REQUEST: 4-HINT (U/V) FIX]
    - [FIXED] 5.4.2 힌트맵 생성을 4-Hint (u/v_placement_...) 키를 '읽도록' 수정
    - [FIXED] 5.7 '데이터' (Green Lines) 빌드 시 4-Hint (u/v_placement_...) 키를 '쓰도록' 수정
    - [REMOVED] 5.7에서 'construction_method_start/end' 키 삭제

    [USER REQUEST: TEMPLATE MATCHING V7 - Splitting Padding]
    - [NEW] 5.2.5 (호출부) `LINE_PADDING_FOR_DEBUG` (작은 고정값)을 정의하고 `_calculate_template_rois`에 전달.
    - [NEW] 5.2.6 템플릿 이미지를 'run_dir'에 PNG 파일로 저장 (예: 'grounding_template.png').
    - [FIXED] 5.3 SDIFF 최상위에 B64 대신 `template_filename` (문자열)과 `source_rect`를 1개만 저장.
    - [FIXED] 5.7 Green Line에 `template_b64`를 제거하고 `relative_roi` (작은 패딩이 적용된 좌표)만 주입.
    - [MODIFIED] 5.8.5 `grounding_template_debug.png` 생성 시 SDIFF에 저장된 '작은 패딩' `relative_roi`를 사용.
    """
    try:
        name = (data or {}).get("image_name")
        if not name:
            return JSONResponse({"ok": False, "error": "image_name required"})
           
        stem = Path(_normalize_name(name)).stem
       
        # --- [NEW] 템플릿/SDIFF를 저장할 'run_dir'를 미리 정의 ---
        run_dir = _run_dir_for(name); _ensure_dir(run_dir)
        # --- [NEW] END ---
       
        merge_path = (IMAGE_DIR / name).resolve()

        mask_name = _mask_name_from_merge(name)
        mask_path = (Path(IMAGE_DIR).parent/"mask"/mask_name).resolve() if mask_name else None

        scribble_path = _scribble_path_for(name)

        if not merge_path.exists():
            return JSONResponse({"ok": False, "error": f"merge not found: {merge_path}"})
        if not scribble_path.exists():
            return JSONResponse({"ok": False, "error": f"scribble not found: {scribble_path}"})
       
        img_h, img_w = 2048, 2048 # 기본값
        if mask_path and mask_path.exists():
            try:
                mask_img_shape = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).shape
                img_h, img_w = mask_img_shape[0], mask_img_shape[1]
                log.info(f"Loaded mask shape for ROI slicing: ({img_h}, {img_w})")
            except Exception as e:
                log.warning(f"Could not read mask shape, using default (2048, 2048). Error: {e}")
        else:
            log.warning(f"Mask file not found at {mask_path}, proceeding without it.")
            mask_path = None

        # 1. 픽셀 스케일 및 '밝기-클래스 맵' 생성
        um_per_px_x = 0.0
        brightness_map = {}
        brightness_descriptors = []
       
        if mask_path:
            try:
                meta_sum = _meta_summary(mask_path)
                um_per_px_x = meta_sum.get("um_per_px_x") or 0.0
                classes_present = meta_sum.get("classes", [])
                if classes_present:
                    num_classes = len(classes_present)
                    if num_classes == 1:
                        brightness_map["only_class"] = int(classes_present[0])
                    else:
                        for i, cls_val in enumerate(classes_present):
                            if i == 0: desc = "darkest"
                            elif i == num_classes - 1: desc = "brightest"
                            else: desc = f"middle_gray_{i}"
                            brightness_map[desc] = int(cls_val)
                    brightness_descriptors = list(brightness_map.keys())
                    log.info(f"[Refinement] Created brightness map: {brightness_map}")
            except Exception as e:
                log.error(f"Failed to create brightness map: {e}")
       
        if not brightness_map:
            log.warning("[Refinement] Brightness map is empty. VLM refinement may be inaccurate.")


        # 2. [CV] OpenCV 기반으로 v1 SDIFF (기하학 + CV 힌트) 추출
        sdiff_cv_v1 = _structured_from_scribble(
            scribble_path,
            um_per_px_x=um_per_px_x,
            mask_path=mask_path
        )
       
        if not isinstance(sdiff_cv_v1, dict):
            sdiff_cv_v1 = {}
        sdiff_cv_v1.setdefault("notes", {})

        # 3. [Qwen] VLM으로 '의미' 설명(raw_text) 추출 및 병합
        try:
            _lazy_load_qwen()
            sdiff_v1_merged = _qwen_summarize_scribble(
                mask_path,
                merge_path,
                scribble_path,
                cv_sdiff=sdiff_cv_v1,
                # [MODIFIED] 'brightness_descriptors' 대신 'brightness_map' (전체 맵)을 전달
                brightness_map=brightness_map
            )
        except Exception as qe:
            log.warning(f"Qwen semantic enrichment failed, using CV-only: {qe}")
            sdiff_cv_v1["notes"]["source"] = "cv_geometry_only (qwen_failed)"
            sdiff_v1_merged = sdiff_cv_v1 # sdiff_v1_merged는 이제 'notes.raw_text'를 가짐

        # 4. [NEW] Qwen의 VLM 추론을 SDIFF v1 객체의 'final_...' 키에 덮어쓰기
        try:
            # [CALL FIXED FUNCTION]
            sdiff_v1_refined = _apply_qwen_refinement_to_sdiff_v1(sdiff_v1_merged, brightness_map)
        except Exception as re:
            log.error(f"Failed to apply Qwen refinement logic: {re}")
            sdiff_v1_refined = sdiff_v1_merged # 원본(병합본)으로 복구

        # --- [NEW] 4.5.0 DINO 호출을 위한 Qwen JSON 파싱 ---
        qwen_json = _parse_qwen_json(sdiff_v1_refined.get("notes", {}).get("raw_text", ""))
       
        # --- [NEW] 4.5.1 Red Line DINO 호출 (기존 로직 유지) ---
        red_grounded_rois_map = {} # { "red_1": [...] }
        red_location_hint_map = {} # { "red_1": "top" }
        try:
            if mask_path and qwen_json and qwen_json.get("red_guides_refined"):
                for red_guide_vlm in qwen_json["red_guides_refined"]:
                    vlm_ref = red_guide_vlm.get("vlm_refinement", {})
                    dino_prompt_text = vlm_ref.get("grounding_dino_prompt", "")
                    location_hint_text = vlm_ref.get("grounding_location_hint", "").lower()
                    v1_red_line_id = red_guide_vlm.get("id")
                   
                    if (dino_prompt_text or location_hint_text) and v1_red_line_id:
                        grounded_rois = _call_grounding_dino(
                            dino_prompt_text, location_hint_text, mask_path
                        )
                        if grounded_rois:
                            log.info(f"Successfully appending {len(grounded_rois)} RED ROIs for ID {v1_red_line_id}.")
                            red_grounded_rois_map[v1_red_line_id] = grounded_rois
                            red_location_hint_map[v1_red_line_id] = location_hint_text
        except Exception as ge_red:
            log.error(f"Failed to run Grounding DINO for RED lines: {ge_red}")

        # --- [REMOVED] 4.5.2 Green Line DINO 호출 로직 삭제 ---

        # --- [NEW] 4.7 (ROI Slicing) ---
        # [MODIFIED] Red ROI 맵만 인자로 받도록 수정
        def _slice_rois(roi_map: dict, location_map: dict, image_h: int) -> dict:
            final_rois_map = {}
            try:
                if roi_map:
                    for line_id, rois in roi_map.items():
                        location_hint = location_map.get(line_id, "full_image")
                        final_rois_for_id = []
                       
                        for i, (x, y, w, h) in enumerate(rois):
                            final_roi = [x, y, w, h] # 기본값
                            if location_hint == "top":
                                y_cutoff = int(image_h * 0.5)
                                ymax_new = min(y + h, y_cutoff)
                                if ymax_new > y:
                                    final_roi = [x, y, w, ymax_new - y]
                            elif location_hint == "bottom":
                                y_cutoff = int(image_h * 0.5)
                                ymin_new = max(y, y_cutoff)
                                if ymin_new < (y + h):
                                    final_roi = [x, ymin_new, w, (y + h) - ymin_new]
                            final_rois_for_id.append(final_roi)
                       
                        final_rois_map[line_id] = final_rois_for_id
            except Exception as slice_e:
                log.error(f"Failed to run ROI Slicing: {slice_e}")
                return roi_map # 슬라이싱 실패 시 원본 맵 반환
            return final_rois_map

        # [MODIFIED] Red만 슬라이싱 수행
        red_dino_rois_final = _slice_rois(red_grounded_rois_map, red_location_hint_map, img_h)
        # [REMOVED] green_dino_rois_final 삭제

        # --- [NEW] 4.6 (DINO Debug Image) 시각화 로직 ---
        # [MODIFIED] Red ROI만 그리도록 수정
        try:
            if red_dino_rois_final: # [MODIFIED]
                debug_img = cv2.imread(str(merge_path), cv2.IMREAD_COLOR)
                if debug_img is not None:
                   
                    color_red_roi = (0, 0, 255) # BGR (Red)
                   
                    # 1. Red ROIs 그리기 (빨간색)
                    for red_id, final_rois_for_id in red_dino_rois_final.items():
                        for i, (fx, fy, fw, fh) in enumerate(final_rois_for_id):
                            p1 = (int(fx), int(fy))
                            p2 = (int(fx + fw), int(fy + fh))
                            cv2.rectangle(debug_img, p1, p2, color_red_roi, 3)
                            cv2.putText(debug_img, f"{red_id}_final_{i}", (int(fx), int(fy)-5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_red_roi, 2)
                   
                    # 2. [REMOVED] Green ROIs 그리기 로직 삭제
                   
                    # [FIX] run_dir는 이미 정의되어 있음
                    # debug_dir = (RUN_DIR / stem)
                    # debug_dir.mkdir(parents=True, exist_ok=True)
                    debug_save_path = run_dir / "dino_debug.png"
                    cv2.imwrite(str(debug_save_path), debug_img)
                    log.info(f"Saved DINO debug image (Red ROIs Only) to: {debug_save_path}") # [MODIFIED]
                else:
                    log.warning(f"Could not read mask_merge_path for DINO debug: {merge_path}")
        except Exception as de:
            log.exception(f"Failed to generate DINO debug image: {de}")
        # --- [NEW] END ---


        # --- [NEW] 5. "V6" SDIFF 빌드 (데이터 + 논리) ---
       
        # 5.1 Qwen의 '논리' (JSON 딕셔너리) 파싱
        # (qwen_json은 4.5.0에서 이미 파싱됨. 변수명 통일)
       
        # --- [BUG FIX] 'qwen_json_dict'를 'qwen_json'으로 할당합니다. ---
        qwen_json_dict = qwen_json
        # -----------------------------------------------------------
           
        if not qwen_json_dict:
            log.error("Failed to parse Qwen's raw_text, cannot build V6 SDIFF.")
            qwen_json_dict = {} # Fallback

        # 5.2 [REMOVED] 'group_id' 외과적 제거 로직 삭제 (anchor_logic_hint로 대체)

        # --- [NEW] 5.2.5 템플릿 및 상대 ROI 계산 ---
        template_img_obj = None
        template_global_rect = None
        template_relative_rois = {}
       
        # [NEW] 비대칭 패딩 값 정의 (아이디어 1)
        # (C50 픽셀이 아래에 있으므로 'bottom' 패딩을 크게 줍니다.)
        LINE_PADDING_FOR_DEBUG = {
            "top": 10,
            "bottom": 10,
            "left": 10,
            "right": 10
        }
       
        if mask_path:
            try:
                log.info("[Template] Calculating template and relative ROIs with DEBUG padding...")
                # [FIX] _calculate_template_rois가 (img_obj, rect, map) 3개를 반환
                template_img_obj, template_global_rect, template_relative_rois = _calculate_template_rois(
                    mask_path,
                    sdiff_v1_merged.get("red_lines_detail", []),
                    sdiff_v1_merged.get("green_lines_detail", []),
                    global_padding=20,
                    padding=LINE_PADDING_FOR_DEBUG # [MODIFIED] 작은 고정 패딩 전달
                )
                if template_img_obj is not None:
                     log.info(f"[Template] Success. Created template image ({template_img_obj.shape}) and {len(template_relative_rois)} relative ROIs.")
                else:
                    log.warning("[Template] _calculate_template_rois returned empty template.")
            except Exception as e_template:
                log.error(f"[Template] Failed to calculate template ROIs: {e_template}")
       
        # --- [NEW] 5.2.6 템플릿 이미지를 'run_dir'에 파일로 저장 ---
        template_filename_str = None
        if template_img_obj is not None:
            try:
                template_filename_str = "grounding_template.png"
                template_save_path = run_dir / template_filename_str
                cv2.imwrite(str(template_save_path), template_img_obj)
                log.info(f"[Template] Saved template image to {template_save_path}")
            except Exception as e_save_tpl:
                log.error(f"[Template] Failed to save template PNG to disk: {e_save_tpl}")
                template_filename_str = None # 저장 실패 시 null
        # --- [NEW] END ---
       
        # --- [NEW] 5.3 V6 SDIFF 뼈대 생성 (LCS 힌트 추가) ---
        qwen_coord_answer = qwen_json_dict.get("coordinate_system", {})
        final_coord_system = { "type": "absolute" } # 기본값
       
        # CV 데이터에서 red line 개수 확인
        raw_red_lines = sdiff_v1_merged.get("red_lines_detail", [])
       
        if len(raw_red_lines) > 0 and qwen_coord_answer.get("type") == "local":
            local_def = qwen_coord_answer.get("local_definition", {})
            primary_axis = local_def.get("primary_axis_guide_id")
            green_orientation = local_def.get("green_line_orientation", "both")

            if primary_axis: # Qwen이 주축을 정했다면
                final_coord_system = {
                    "type": "local",
                    "primary_axis_id": primary_axis,       # e.g., "red_1"
                    "green_orientation": green_orientation # e.g., "both"
                }
       
        sdiff_final = {
            "schema": "scribble_vlite_v2",
            "coordinate_system": final_coord_system, # <-- [NEW] LCS 힌트 주입
           
            # --- [NEW] SDIFF 최상위에 템플릿 1개 저장 ---
            "grounding_template_global": {
                "source_rect": template_global_rect,     # [x,y,w,h]
                "template_filename": template_filename_str # "grounding_template.png" (or None)
            },
            # --- [NEW] END ---
           
            "rules": sdiff_v1_refined.get("rules", {"green_on_red": True, "nudge_if_touching": True}),
            "red":   {"count": 0, "lines": []},
            "green": {"count": 0, "lines": []},
           
            # (E) VLM의 "논리" 복사본 (v6 예시)
            "grouping_logic_vlm": qwen_json_dict.get("grouping_logic", {}),
           
            "notes": {
                "source": "hybrid_cv_geometry_qwen_notes",
                "mask_name": mask_path.name if mask_path else None,
                "merge_name": merge_path.name,
                "scribble_name": scribble_path.name,
               
                # (A) VLM의 "구구절절한 설명" (v6 예시)
                "raw_text_cleaned_answer": qwen_json_dict
            }
        }
       
        # --- [NEW] 5.4 '데이터' 힌트맵 생성 (LCS / Anchor 힌트 포함) ---
       
        # 5.4.1 (기존) Final class 힌트맵
        red_final_hint_map = {}
        qwen_red_refined = qwen_json_dict.get("red_guides_refined", [])
        if qwen_red_refined:
            for red_guide_vlm in qwen_red_refined:
                line_id = red_guide_vlm.get("id")
                if not line_id: continue
                vlm_ref = red_guide_vlm.get("vlm_refinement", {})
                req_classes_desc = vlm_ref.get("required_classes", [])
                if req_classes_desc and brightness_map:
                    first_desc = req_classes_desc[0]
                    if first_desc in brightness_map:
                        red_final_hint_map[line_id] = brightness_map[first_desc]
       
        green_final_start_map = {}
        green_final_end_map = {}
        raw_green_lines = sdiff_v1_merged.get("green_lines_detail", [])
        qwen_green_refined = qwen_json_dict.get("green_measures_refined", [])
        if qwen_green_refined:
            green_cv_map = {ln.get("id"): ln for ln in raw_green_lines}
            for ln_refined in qwen_green_refined:
                line_id = ln_refined.get("id")
                if not line_id or line_id not in green_cv_map: continue
                vlm_ref = ln_refined.get("vlm_refinement", {})
                original_start = green_cv_map[line_id].get("detected_class_hint_start")
                original_end = green_cv_map[line_id].get("detected_class_hint_end")
                vlm_start_connects = vlm_ref.get("start_is_connected_to_red_guide")
                vlm_start_connect_id = vlm_ref.get("connected_red_guide_id_start")
                if vlm_start_connects is True and vlm_start_connect_id:
                    green_final_start_map[line_id] = vlm_start_connect_id
                else:
                    green_final_start_map[line_id] = original_start
                green_final_end_map[line_id] = original_end

        # 5.4.2 [FIXED] 4-Hint (U/V) 힌트맵 생성
        red_intent_map = {}
        red_construction_map = {}
        if qwen_red_refined:
            for r in qwen_red_refined:
                line_id = r.get("id")
                if not line_id: continue
                vlm_ref = r.get("vlm_refinement", {})
                red_intent_map[line_id] = vlm_ref.get("lcs_orientation_intent", "unrelated")
                red_construction_map[line_id] = vlm_ref.get("construction_method", "")
       
        green_intent_map = {}
        green_u_placement_start_map = {}
        green_u_placement_end_map = {}
        green_v_placement_start_map = {}
        green_v_placement_end_map = {}
       
        if qwen_green_refined:
            for g in qwen_green_refined:
                line_id = g.get("id")
                if not line_id: continue
                vlm_ref = g.get("vlm_refinement", {})
                green_intent_map[line_id] = vlm_ref.get("lcs_orientation_intent", "perpendicular_to_primary")
                # [FIXED] Qwen 응답에서 4-Hint 키를 읽어옵니다
                green_u_placement_start_map[line_id] = vlm_ref.get("u_placement_logic_hint_start", "")
                green_u_placement_end_map[line_id] = vlm_ref.get("u_placement_logic_hint_end", None) # null 허용
                green_v_placement_start_map[line_id] = vlm_ref.get("v_placement_logic_hint_start", "")
                green_v_placement_end_map[line_id] = vlm_ref.get("v_placement_logic_hint_end", "")


        # 5.5 [REMOVED] 'group_id_vlm' 번역 로직 삭제

        # 5.6 '데이터' (Red Lines) 빌드
        for ln in raw_red_lines: # Use CV data (sdiff_v1_merged.get("red_lines_detail"...))
            if not isinstance(ln, dict): continue
            line_id = ln.get("id")
           
            sdiff_final["red"]["lines"].append({
                # --- (기존 CV 힌트 보존) ---
                "id": line_id,
                "role": "guide",
                "orientation": ln.get("orientation"), # CV가 측정한 사실 (e.g., 'diagonal')
                "angle_deg": ln.get("angle_deg"),
                "length_px": ln.get("length_px"),
                "x1": ln.get("x1"), "y1": ln.get("y1"), "x2": ln.get("x2"), "y2": ln.get("y2"), # (Scribble 원본 좌표)
               
                # --- (기존 VLM 힌트 보존) ---
                "final_class_hint": red_final_hint_map.get(line_id),
               
                # [MODIFIED] Red Line의 DINO ROI 주입
                "grounded_rois": red_dino_rois_final.get(line_id), # Sliced ROIs
               
                # --- [NEW] LCS / Construction 힌트 주입 ---
                "lcs_orientation_intent": red_intent_map.get(line_id, "unrelated"),
                "construction_logic_hint": red_construction_map.get(line_id, "")
            })

        # 5.7 '데이터' (Green Lines) 빌드
        for ln in raw_green_lines: # (CV 원본 6개)
            if not isinstance(ln, dict): continue
            line_id = ln.get("id")
           
            # [NEW] Get template info for this line_id
            # (template_relative_rois 맵은 이미 '작은 고정 패딩'이 적용된 ROI를 가지고 있음)
            relative_roi_for_this_line = template_relative_rois.get(line_id) # [x,y,w,h] or None
           
            sdiff_final["green"]["lines"].append({
                # --- (기존 CV 힌트 보존) ---
                "id": line_id,
                "role": "measure",
                "orientation": ln.get("orientation"),
                "semantic": ln.get("semantic"),
                "x1": ln.get("x1"), "y1": ln.get("y1"), "x2": ln.get("x2"), "y2": ln.get("y2"),
               
                # --- (기존 VLM 힌트 보존) ---
                "final_class_hint_start": green_final_start_map.get(line_id), # "red_1"
                "final_class_hint_end": green_final_end_map.get(line_id), # 10
               
                # --- [FIXED] 4-Hint (U/V) 힌트 주입 ---
                "lcs_orientation_intent": green_intent_map.get(line_id, "perpendicular_to_primary"),
                "u_placement_logic_hint_start": green_u_placement_start_map.get(line_id, ""),
                "u_placement_logic_hint_end": green_u_placement_end_map.get(line_id, None), # null 허용
                "v_placement_logic_hint_start": green_v_placement_start_map.get(line_id, ""),
                "v_placement_logic_hint_end": green_v_placement_end_map.get(line_id, ""),
               
                # --- [REMOVED] Green Line DINO ROI 삭제 ---
                "grounded_rois": None,

                # --- [NEW] Grounding Template (relative_roi) 주입 ---
                "relative_roi": relative_roi_for_this_line
            })

        # 5.8 최종 카운트
        sdiff_final["red"]["count"] = len(sdiff_final["red"]["lines"])
        sdiff_final["green"]["count"] = len(sdiff_final["green"]["lines"])

        log.info(f"Successfully built V6 SDIFF: {sdiff_final['red']['count']} red, {sdiff_final['green']['count']} green.")
       
        # --- [NEW] 5.8.5 ROI 디버그 이미지 생성 (아이디어 2) ---
        if template_img_obj is not None and template_relative_rois:
            try:
                # 3채널 컬러로 변환
                debug_img = cv2.cvtColor(template_img_obj, cv2.COLOR_GRAY2BGR)
               
                # 6개의 고유 색상 (BGR)
                colors = [
                    (255, 0, 0),   # Blue
                    (0, 255, 0),   # Green
                    (0, 0, 255),   # Red
                    (255, 255, 0), # Cyan
                    (255, 0, 255), # Magenta
                    (0, 255, 255)  # Yellow
                ]
               
                # SDIFF에 저장된 순서대로 순회
                for i, line_data in enumerate(sdiff_final["green"]["lines"]):
                    line_id = line_data.get("id")
                    # [FIX] SDIFF에 저장된 '작은 고정 패딩'이 적용된 ROI를 가져옴
                    rel_roi = line_data.get("relative_roi") # [x,y,w,h]
                   
                    if not line_id or not rel_roi:
                        continue
                       
                    color = colors[i % len(colors)]
                    x, y, w, h = rel_roi
                    p1 = (int(x), int(y))
                    p2 = (int(x + w), int(y + h))
                   
                    cv2.rectangle(debug_img, p1, p2, color, 2)
                    cv2.putText(debug_img, line_id, (p1[0], p1[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                debug_save_path = run_dir / "grounding_template_debug.png"
                cv2.imwrite(str(debug_save_path), debug_img)
                log.info(f"[Template] Saved ROI debug image to {debug_save_path}")

            except Exception as e_dbg_img:
                log.error(f"[Template] Failed to create debug ROI image: {e_dbg_img}")
        # --- [NEW] END ---
       
        try:
            # [FIX] run_dir는 이미 정의되어 있음
            # run_dir = _run_dir_for(name); _ensure_dir(run_dir)
            sdiff_path = run_dir / "sdiff_qwen_v6.json"
            sdiff_path.write_text(json.dumps(sdiff_final, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info(f"Persisted V6 SDIFF to {sdiff_path}")
        except Exception as e_save:
            log.error(f"Failed to persist SDIFF to disk: {e_save}")
       
        STRUCTURED_DIFF_CACHE[name] = sdiff_final
        return JSONResponse({"ok": True, "structured_diff": sdiff_final})

    except Exception as e:
        log.exception("/vl/sdiff_qwen error: %s", e)
        return JSONResponse({"ok": False, "error": str(e)})
    
@app.get("/sdiff/get_latest")
async def sdiff_get_latest(image_name: str = Query(...)):
    """s
    메모리 캐시 또는 디스크에 저장된 최신 V6 SDIFF를 불러옵니다.
    (UI의 setSelected에서 호출)
    """
    try:
        normalized = _normalize_name(image_name)
        
        # 1. 인메모리 캐시 우선 확인
        cached = STRUCTURED_DIFF_CACHE.get(normalized)
        if cached and isinstance(cached, dict) and cached.get("schema") == "scribble_vlite_v2":
            return JSONResponse({"ok": True, "structured_diff": cached, "source": "memory_cache"})
            
        # 2. 디스크에 저장된 파일 확인
        run_dir = _run_dir_for(normalized)
        sdiff_path = run_dir / "sdiff_qwen_v6.json"
        
        if sdiff_path.exists():
            sdiff_json = json.loads(sdiff_path.read_text(encoding="utf-8"))
            # 3. 디스크 -> 메모리 캐시로 로드
            STRUCTURED_DIFF_CACHE[normalized] = sdiff_json
            return JSONResponse({"ok": True, "structured_diff": sdiff_json, "source": "disk_cache"})
            
        # 4. 어디에도 없음
        return JSONResponse({"ok": False, "error": "No cached SDIFF found"}, status_code=404)
        
    except Exception as e:
        log.exception(f"[sdiff/get_latest] Failed to load SDIFF for {image_name}: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    
# ---------- few-shot 자산 ----------
@app.get("/fewshot/select")
async def fewshot_select(name: str = Query(...), k: int = Query(2)):
    try:
        # 1. Select top-k few-shots (same as before)
        selected_items = _select_topk_fewshots_for(name, k=k)
       
        # 2. [NEW] Build the rich 'used_fewshots' structure, including the 'parts' array
        rich_items = []
        for it in selected_items:
            item_id = it["id"]
            d = Path(it["_dir"])
            parts = []
           
            # Check for assets and build their URLs
            for nm in ("mask.jpg", "merge.jpg", "ppt.jpg"):
                if (d / nm).exists():
                    parts.append({
                        "name": nm,
                        "url": f"/fewshot/asset?item_id={quote(item_id, safe='')}&name={quote(nm, safe='')}"
                    })
                   
            # Append the rich object (this now matches the structure from vl_describe4)
            rich_items.append({
                "id": item_id,
                "score": it["score"],
                "score_breakdown": it.get("score_breakdown", {}),
                "meta": it.get("meta", {}),
                "parts": parts, # <-- [THE FIX] The missing image URL array
                "created_at": it.get("created_at")
            })

        # 3. Return the new 'rich_items' list
        return JSONResponse({"ok": True, "items": rich_items})
       
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
        "- 실행 인자: --mask_path, --meta_root, --out_dir (필수)\\n"
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
        "- meta_utils.__wrap_load_pixel_scale_um(mask_path, meta_root)는 (umx,umy,classes,meta_path) 4개 값 반환 사용\n"
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

# [ADD NEW FUNCTION] (app_codecheck.py: line 3230)
def _sanitize_sdiff_for_codemodel(sdiff_dict: dict) -> dict:
    """
    GaussO-Think/GPT-OSS로 전달하기 전에 SDIFF를 '정리(Sanitize)'합니다.
    모델의 혼란을 유발하는 '물리적 CV 수치'를 모두 제거하고,
    '논리적 VLM 의도'만 남깁니다.
    """
    if not isinstance(sdiff_dict, dict):
        return {}

    try:
        # 딥 카피를 통해 원본 sdiff_dict 객체가 수정되는 것을 방지
        clean_sdiff = json.loads(json.dumps(sdiff_dict)) # Deep copy
    except Exception:
        clean_sdiff = dict(sdiff_dict) # Fallback to shallow copy
       
    # [CRITICAL] 제거할 물리적 CV 수치 목록
    # 이 키들이 제거되어도, VLM이 추론한 '논리적' 힌트
    # (예: lcs_orientation_intent, construction_logic_hint, final_class_hint)는 보존됩니다.
    KEYS_TO_REMOVE = {
        "angle_deg",    # (e.g., -3.5)
        "length_px",    # (e.g., 850.2)
        "orientation",  # (e.g., "diagonal")
        "x1", "y1", "x2", "y2", # (Scribble의 원본 좌표)
        "endpoints"     # (Scribble의 원본 좌표)
    }

    def _strip_recursive(obj: Any):
        if isinstance(obj, dict):
            # 'notes' 블록, 'coordinate_system', 'grouping_logic_vlm' 등
            # 최상위 '논리' 블록은 제거 대상이 아님.
            # 'lines' 배열 내부의 객체에만 이 로직을 적용해야 함.
           
            # 'lines' 배열을 찾아서 내부 순회
            if "lines" in obj and isinstance(obj["lines"], list):
                for item in obj["lines"]:
                    if isinstance(item, dict):
                        for key in KEYS_TO_REMOVE:
                            item.pop(key, None)
           
            # 재귀 호출 (다른 중첩 구조, 예: notes.raw_text_cleaned_answer 탐색)
            for key, value in obj.items():
                # 'lines' 키 자체는 이미 위에서 처리했으므로 건너뜀
                if key != "lines":
                    _strip_recursive(value)
                   
        elif isinstance(obj, list):
            for item in obj:
                _strip_recursive(item)

    # 'red'와 'green' 블록 내부의 'lines' 배열에만 필터링 적용
    if "red" in clean_sdiff:
        _strip_recursive(clean_sdiff["red"])
    if "green" in clean_sdiff:
        _strip_recursive(clean_sdiff["green"])

    return clean_sdiff

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
    
    [BUG FIX] v2_lite 감지 로직이 너무 복잡하여 green line 카운트(6->3) 오류를 유발함.
             복잡한 감지 로직을 모두 제거하고, 
             매번 sdiff.get("..._lines_detail")을 읽어 v2_lite 객체를
             '새로' 빌드하도록 강제하여 안정성을 확보함.
             
    [USER REQUESTED CHANGE]
    - 'red'에 `grounded_rois` (Grounding DINO 힌트) 필드를 전달.
    """
    if not sdiff or not isinstance(sdiff, dict):
        return sdiff

    # --- [BUG FIX] ---
    # 복잡한 v2_lite 감지 로직 (`if schema == "scribble_v2_lite": ...`)을
    # 모두 제거합니다.
    # -------------------

    # 기본 뼈대
    out = {
        "schema": "scribble_v2_lite",
        "red":   {"count": 0, "lines": []},
        "green": {"count": 0, "lines": []},
        "rules": sdiff.get("rules", {"green_on_red": True, "nudge_if_touching": True}), # 규칙 보존
        "notes": dict(sdiff.get("notes", {}))
    }

    # v1 SDIFF의 원본 상세 목록을 찾습니다.
    # 만약 sdiff가 이미 v2_lite 형식이었다면, 'red_lines_detail' 키가 없을 것입니다.
    # 이 경우, v2_lite의 'red.lines'를 대신 사용합니다. (Fallback)
    raw_red_lines = sdiff.get("red_lines_detail") or sdiff.get("red", {}).get("lines", [])
    raw_green_lines = sdiff.get("green_lines_detail") or sdiff.get("green", {}).get("lines", [])

    # --- 1. RED (guide) 라인 처리 ---
    lite_red_lines = []
    for ln in raw_red_lines:
        if not isinstance(ln, dict): continue
        
        # v1 형식('x1')과 v2 형식('endpoints') 모두에 대응
        ep = ln.get("endpoints")
        if not ep and ("x1" in ln and "y1" in ln and "x2" in ln and "y2" in ln):
            ep = [ln.get("x1"), ln.get("y1"), ln.get("x2"), ln.get("y2")]
            
        lite_red_lines.append({
            "id": ln.get("id"),
            "role": "guide",
            "orientation": ln.get("orientation"),
            "angle_deg": ln.get("angle_deg"),
            "length_px": ln.get("length_px"),
            "endpoints": ep, # [FIX] v1/v2 호환
            
            # v1의 'detected_class_hint' 또는 v2의 'final_class_hint'를 복사
            "detected_class_hint": ln.get("detected_class_hint") or ln.get("final_class_hint"), 
            
            # [USER REQUESTED CHANGE] Grounded ROIs 힌트 복사
            "grounded_rois": ln.get("grounded_rois") or ln.get("grounded_rois_stub") # stub도 호환
        })

    # --- 2. GREEN (measure) 라인 처리 ---
    lite_green_lines = []
    for ln in raw_green_lines:
        if not isinstance(ln, dict): continue

        ep = ln.get("endpoints")
        if not ep and ("x1" in ln and "y1" in ln and "x2" in ln and "y2" in ln):
            ep = [ln.get("x1"), ln.get("y1"), ln.get("x2"), ln.get("y2")]

        lite_green_lines.append({
            "id": ln.get("id"),
            "role": "measure",
            "orientation": ln.get("orientation"),
            "angle_deg": ln.get("angle_deg"),
            "length_px": ln.get("length_px"),
            "semantic": ln.get("semantic"), 
            "paired_red_id": ln.get("paired_red_id"), # (e.g., "red_1")
            "endpoints": ep, # [FIX] v1/v2 호환
            
            # v1/v2 호환
            "detected_class_hint_start": ln.get("detected_class_hint_start") or ln.get("final_class_hint_start"),
            "detected_class_hint_end": ln.get("detected_class_hint_end") or ln.get("final_class_hint_end")
        })

    # --- 3. 최종 SDIFF v2_lite 객체 생성 ---
    out["red"]["lines"] = lite_red_lines
    out["green"]["lines"] = lite_green_lines
    
    # [FIX] 이 카운트는 이제 `raw_green_lines` (원본 6개)의 길이를 따름
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
    max_fixes  = int(payload.get("max_fixes", 1))
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

    # --- [CHANGED] meta_utils.py 및 measurement_utils.py 복사 로직 강화 ---
   
    # 1. meta_utils.py 복사 (기존 로직 유지)
    meta_utils_dest = run_dir / "meta_utils.py"
    candidates_meta = []
    envp_meta = os.getenv("META_UTILS_PATH")
    if envp_meta: candidates_meta.append(Path(envp_meta))
    candidates_meta += [BASE_DIR/"meta_utils.py", Path(IMAGE_DIR).parent/"meta_utils.py", Path.cwd()/"meta_utils.py"]
    picked_meta = next((c for c in candidates_meta if c.exists() and c.is_file()), None)
   
    if picked_meta:
        meta_utils_dest.write_text(picked_meta.read_text(encoding="utf-8"), encoding="utf-8")
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
        # (fallback meta_utils 생성 로직은 기존과 동일)
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

    # 2. [CHANGED] measurement_utils.py 복사 (meta_utils와 동일한 탐색 로직 사용)
    utils_dest = run_dir / "measurement_utils.py"
    candidates_utils = []
    # (MEASUREMENT_UTILS_PATH 전역 변수를 직접 참조하는 대신, 환경변수와 후보 경로를 탐색)
    envp_utils = os.getenv("MEASUREMENT_UTILS_PATH")
    if envp_utils: candidates_utils.append(Path(envp_utils))
    candidates_utils += [
        BASE_DIR / "measurement_utils.py", # app_codecheck.py와 동일 경로
        Path.cwd() / "measurement_utils.py" # 현재 작업 디렉토리
    ]
    picked_utils = next((c for c in candidates_utils if c.exists() and c.is_file()), None)

    if picked_utils:
        # [FIX] utils_src.exists() 대신 picked_utils를 사용
        utils_dest.write_text(picked_utils.read_text(encoding="utf-8"), encoding="utf-8")
        log.info(f"Copied measurement_utils.py from {picked_utils}")
    else:
        # [FIX] 전역 변수 대신 탐색 경로를 로그에 남김
        log.warning(f"measurement_utils.py not found in {candidates_utils}, creating empty fallback.")
        utils_dest.write_text("# fallback: measurement_utils.py not found\n", encoding="utf-8")

    # --- [기존 로직 계속] ---
    overlay_path = _overlay_path_for(image_name)
    if overlay_path.exists():
        try: overlay_path.unlink()
        except Exception: pass
   
    stem = Path(_normalize_name(image_name)).stem
   
    # ... (SDIFF JSON 생성 로직은 기존과 동일)
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
            try:
                sdiff = STRUCTURED_DIFF_CACHE.get(image_name)
            except Exception:
                sdiff = None

            if not isinstance(sdiff, dict) or not sdiff:
                try:
                    scr_path = _scribble_path_for(_normalize_name(image_name))
                    if scr_path and scr_path.exists():
                        # (CV-only fallback)
                        sdiff = _structured_from_scribble(scr_path, 0.0, mask_path)
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
           
            try:
                sdiff.setdefault("notes", {})
                sdiff["notes"]["_server_injected"] = True
                sdiff["notes"]["_expected_path"] = str(sdiff_path)
            except Exception:
                pass
            sdiff_path.write_text(json.dumps(sdiff, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                STRUCTURED_DIFF_CACHE[image_name] = sdiff
            except Exception:
                pass
    except Exception as e:
        log.warning(f"[run] failed to ensure structured diff JSON: {e}")

    def _exec_once():
        # [CHANGED] --mask_merge_path 인자 제거
        cmd = [
            _python_executable(), str(measure_py),
            "--mask_path", str(mask_path),
            # "--mask_merge_path", str(mask_merge_path), # <-- 이 줄 제거됨
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
            try:
                out_csv = _run_dir_for(image_name) / "measurements.csv"
                normalize_measurements_csv(out_csv)
            except Exception as _e:
                log.warning(f"[run] csv normalize failed: {_e}")

            f.write(f"[cmd] {_python_executable()} {measure_py.name} --mask_path ...\n")
            f.write(f"[returncode] {rc}\n\n[stdout]\n${{stdout}}\n\n[stderr]\n${{stderr}}\n".replace("${stdout}", stdout or "").replace("${stderr}", stderr or ""))
    except Exception as e:
        log.warning(f"[run] failed to write run.log: {e}")

    exists = (_run_dir_for(image_name) / "overlay.png").exists() or False
    norm_ok, norm_msg = _normalize_measurements_csv(_run_dir_for(image_name), image_name)

    if rc==0 and exists:
        _set_status(stem, phase="done", attempt=0, progress=100, label="완료")
        _clear_status(stem)
        return JSONResponse({
            "status": "ok", "returncode": rc, "stdout": stdout, "stderr": stderr,
            "overlay_exists": exists, "overlay_url": f"/overlay?name={quote(image_name, safe='')}",
            "csv_normalized": norm_ok, "csv_note": norm_msg
        })

    auto_log = []
    auto_fixed = False
    if auto_fix:
        # --- [NEW] Auto-Fix Debug Logging ---
        # 자동 수정 루프에 들어가기 직전에, 실패한 원본 코드와 에러를 별도 파일에 기록합니다.
        try:
            failing_code = measure_py.read_text(encoding="utf-8", errors="ignore")
            debug_content = (
                f"--- [AUTO-FIX TRIGGERED] {datetime.datetime.now().isoformat()} ---\n"
                f"[Image]: {image_name}\n"
                f"[Returncode]: {rc}\n\n"
                f"--- [STDERR] ---\n{stderr or '(empty)'}\n\n"
                f"--- [STDOUT] ---\n{stdout or '(empty)'}\n\n"
                f"--- [FAILING CODE] ---\n{failing_code}\n\n"
            )
            # _dump_debug는 app_codecheck.py에 이미 존재합니다.
            _dump_debug(stem, "gptoss_autofix.debug.txt", debug_content)
        except Exception as e_dbg:
            log.warning(f"[code_run] Failed to write auto-fix debug log: {e_dbg}")
        # --- [NEW] End ---

        llm = _build_gausso() # 모델 변경 _build_gptoss()
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


def _sanitize_structured_diff_semantics(structured_diff: dict) -> Tuple[str, bool, bool]:
    """
    [CRITICAL FIX 15]
    GPT-OSS로 전달되는 SDIFF 딕셔너리에서 '치팅'을 유발하는
    'endpoints' 키 + [CHANGED] 혼동을 유발하는 'detected_...' 힌트를 재귀적으로 제거합니다.
    (다른 유용한 CV 힌트(length_px, angle_deg, final_class_hint...)는 보존합니다.)
    """
    if not isinstance(structured_diff, dict):
        return "{}", False, False

    # 딥 카피를 통해 원본 sdiff_dict 객체(캐시)가 수정되는 것을 방지
    try:
        parsed = json.loads(json.dumps(structured_diff)) # Deep copy
    except Exception as e:
        log.error(f"SDIFF deep copy failed: {e}")
        return "{}", False, False # 실패 시 빈 객체 반환

    # [CHANGED] 'endpoints' 외에 혼동을 유발하는 모든 'detected_' 힌트 제거
    keys_to_remove = {
        "endpoints",
        "detected_class_hint",
        "detected_class_hint_start",
        "detected_class_hint_end"
    }
    removed = False

    def _strip(obj: Any):
        nonlocal removed
        if isinstance(obj, dict):
            # 'notes.raw_text'는 VLM 힌트가 있어 GPT-OSS가 봐야 하므로, 이 안의 내용은 제거하지 않음
            if "raw_text" in obj: 
                pass

            for key in list(obj.keys()):
                if key in keys_to_remove: # <-- [CHANGED] 확장된 세트 사용
                    obj.pop(key, None)
                    removed = True
                    continue
                
                # notes.raw_text는 VLM 힌트의 '진실의 원천'이므로 절대 제거하면 안 됨
                if key == "raw_text":
                    continue
                    
                _strip(obj[key])
        elif isinstance(obj, list):
            for item in obj:
                _strip(item)

    _strip(parsed) # 'endpoints' 및 'detected_...' 키 제거 실행

    try:
        # GPT-OSS에 전달할, 'endpoints'와 'detected_'가 제거된 깨끗한 JSON 문자열 생성
        sanitized = json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Failed to re-serialize sanitized SDIFF: {e}")
        return "{}", removed, False # 직렬화 실패 시 빈 객체 반환
        
    return sanitized, removed, True

# [NEW] SDIFF -> GPT-OSS 전용 Invoker
def _safe_invoke_gptoss_for_code_from_sdiff(
    guide_text_with_sdiff: str,
    top_1_code_str: str,
    model_name: str,  # [MODIFIED] model_name 인자 추가
    stem: str
) -> str:
    """
    [CRITICAL FIX 17]
    ...
    [NEW PLAN (User Idea)]
    - [FIXED] System 프롬프트(규칙)를 Message 3로 이동하여 토큰 한계 문제 해결.
    - [FIXED] Message 1에 '전체' Few-shot 코드를 전송.
    - [MODIFIED] model_name 인자를 받아 GPT-OSS 또는 GaussO-Think 빌더 호출.
    """
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
   
    # [MODIFIED] 모델 빌더 선택
    llm = None
    if model_name == "gausso":
        log.info("[Invoker] Using GaussO-Think builder.")
        llm = _build_gausso()
    else:
        log.info("[Invoker] Using GPT-OSS builder.")
        llm = _build_gptoss()
    # [MODIFIED END]


    # [MODIFIED] System 프롬프트는 최소한의 역할만 수행
    sys_msg = (
        "You are an expert Python script generation assistant. "
        "You will receive a reference code structure first, then the actual task."
    )

    # [NEW] 'messages' 배열 구성
    messages = [
        SystemMessage(content=sys_msg),
       
        # Message 1: The Example (Top-1 'FULL' Code)
        HumanMessage(content=(
            "=== [1. 구조 참고용 예제 코드 (Top-1 Few-Shot)] ===\n"
            "이 코드의 '구조'만 참고하라. 이 코드의 '로직'이나 '값'은 현재 작업과 무관하다.\n"
            "이 코드의 헬퍼 함수 선언(def)을 보고 필요한 헬퍼 함수를 추론하라.\n\n"
            # [MODIFIED] 슬리밍된 코드가 아닌 '전체' 코드를 전달
            f"```python\n{top_1_code_str if top_1_code_str else '# No reference code available'}\n```"
        )),
       
        # Message 2: Fake AI Response (to separate concerns)
        AIMessage(content="알겠다. 예제 코드의 구조를 파악했다. 이제 새로운 SDIFF 힌트와 규칙을 제공하라."),
       
        # Message 3: The Actual Task (Rules + SDIFF)
        HumanMessage(content=(
            "=== [2. 새로운 SDIFF 힌트 및 규칙 (유일한 진실)] ===\n"
            "이제, 아까 본 예제 코드의 '구조'를 사용하되, '오직' 아래의 **규칙**과 **SDIFF 힌트**의 '논리'에 맞는 '완전한' 새 코드를 생성하라.\n"
            "(규칙: `mask_merge_path` 인자는 절대 사용하지 마라.)\n\n"
            # [MODIFIED] 모든 규칙(기존 System 프롬프트)을 이곳으로 이동
            f"{_GPTOSS_SDIFF_RULES_PROMPT}\n\n"
            f"--- [New Logic Hint (SDIFF)] ---\n"
            f"{guide_text_with_sdiff}"
        ))
    ]
   
    # --- [NEW] Logging Logic ---
    if os.getenv("DEBUG_GPTOSS_PROMPT", "1") == "1":
        try:
            debug_messages_content = []
            debug_messages_content.append(f"\n--- [Two-Message Strategy Log ({model_name}) @ {time.strftime('%Y-%m-%d %H:%M:%S')}] ---\n")
           
            # [MODIFIED] System Message 로깅
            debug_messages_content.append(f"--- [Message 0: SYSTEM] ---\n{sys_msg}\n")
           
            for i, msg in enumerate(messages[1:], 1): # (System msg는 이미 로깅했으므로 1부터 시작)
                role = getattr(msg, 'role', 'unknown')
                content = getattr(msg, 'content', '')
                content_str = str(content) # Simple str conversion for logging

                # Truncate long content (Top-1 code) for logging
                if role == 'user' and "구조 참고용 예제 코드" in content_str:
                    if len(content_str) > 4000: # 로그 파일이 너무 커지는 것을 방지
                         content_log = content_str[:4000] + "\n...[TRUNCATED IN LOG]"
                    else:
                         content_log = content_str
                else:
                    content_log = content_str # SDIFF part is crucial, don't truncate

                debug_messages_content.append(f"--- [Message {i}: {role.upper()}] ---\n{content_log}\n")
           
            # Use _dump_debug to append to the existing log
            _dump_debug(stem, "gptoss_gen_sdiff.debug.txt", "\n".join(debug_messages_content))
        except Exception as e:
            log.warning(f"[GPT-OSS] Failed to log Two-Message Strategy: {e}")
    # --- [NEW] End Logging ---

    # [The Invoke Call]
    resp = llm.invoke(messages)
    code = _extract_text_from_aimessage(resp).strip()
    code = _strip_code_fence(code)

    if code.strip():
        return code

    # 1회 재시도 (기존 메시지 리스트에 추가)
    log.warning(f"[{model_name}] Empty code received on first attempt, retrying...")
    retry_msg = HumanMessage(content="[재시도] 출력이 비어있다. SDIFF 힌트에 맞는 '완전한' 파이썬 코드만 즉시 출력하라.")
    messages.append(retry_msg) # 이전 대화에 이어서 재시도 요청
   
    # [NEW] 재시도 로깅
    if os.getenv("DEBUG_GPTOSS_PROMPT", "1") == "1":
        try:
            _dump_debug(stem, "gptoss_gen_sdiff.debug.txt",
                        f"\n--- [Retry Message {len(messages)}: USER] ---\n{retry_msg.content}\n")
        except Exception:
            pass
           
    resp2 = llm.invoke(messages)
    code2 = _extract_text_from_aimessage(resp2).strip()
    return _strip_code_fence(code2)
        
# [NEW] SDIFF -> GPT-OSS 직접 생성 엔드포인트
@app.post("/gptoss/generate_from_sdiff")
async def gptoss_generate_from_sdiff(payload: dict = Body(...)):
    """
    [CRITICAL FIX 17]
    ...
    [NEW PLAN (User Idea)]
    - [FIXED] System 프롬프트(규칙)를 Message 3로 이동하여 토큰 한계 문제 해결.
    - [FIXED] '스마트 슬리밍' 로직을 '완전히 제거'하고 '전체' Few-shot 코드를 전송.
    - [MODIFIED] payload에서 'model_name'을 읽어 invoker로 전달 (gptoss or gausso)
   
    [LCS PLAN - STEP 5 (FILTERING)]
    - [NEW] SDIFF 원본을 파싱한 후, `_sanitize_sdiff_for_codemodel`를 호출하여
            'angle_deg' 등 물리적 CV 수치를 제거한 "클린 SDIFF"를 생성.
    - [NEW] 이 "클린 SDIFF"를 GaussO-Think 프롬프트로 전달.
    """
    try:
        import json, os, re
        from pathlib import Path

        image_name = payload.get("image_name")
        structured_diff_text = payload.get("structured_diff_text", "") # 원본 V6 SDIFF (JSON 문자열)
       
        # [MODIFIED] 모델 이름 수신 (기본값: gptoss)
        model_name = payload.get("model_name", "gptoss")

        if not image_name or not structured_diff_text:
            return JSONResponse({"ok": False, "error": "image_name and structured_diff_text required"}, status_code=200)

        stem = Path(_normalize_name(image_name)).stem

        # --- 1. `meta_summary` (클래스 정보) 로드 ---
        meta_sum = {}
        if image_name:
            try:
                mask_name = _mask_name_from_merge(_normalize_name(image_name))
                if mask_name:
                    mask_path = (Path(IMAGE_DIR).parent/"mask"/mask_name).resolve()
                    if mask_path.exists():
                         meta_sum = _meta_summary(mask_path)
            except Exception as e:
                log.warning(f"Failed to create brightness map for GPT-OSS: {e}")

        # --- [NEW] 2. SDIFF 원본(Full) 파싱 ---
        try:
            full_sdiff_dict = json.loads(structured_diff_text)
        except Exception as e:
            log.error(f"Failed to parse incoming SDIFF JSON: {e}")
            return JSONResponse({"ok": False, "error": f"Invalid SDIFF JSON provided: {e}"})
       
        # --- [NEW] 3. SDIFF 필터링 (Sanitization) ---
        # (요구사항 A: CV 수치 가리기)
        try:
            clean_sdiff_dict = _sanitize_sdiff_for_codemodel(full_sdiff_dict)
            clean_sdiff_text = json.dumps(clean_sdiff_dict, ensure_ascii=False) # (로그용이 아니므로 indent 불필요)
        except Exception as e:
            log.error(f"Failed to sanitize SDIFF: {e}")
            # 실패 시, 위험하지만 원본 SDIFF 텍스트로 강행
            clean_sdiff_text = structured_diff_text

       
        # --- 4. Top-1 Few-shot '전체 코드' 로드 (Slimming 없음) ---
        top_1_code_str = "" # 기본값: 빈 문자열
        sels = []
        try:
            # [FIXED] 오타 수정 (topk)
            sels = _select_topk_fewshots_for(image_name, k=1)
            if sels:
                best_dir = Path(sels[0]["_dir"])
                code_path = best_dir / "code.py"
                if code_path.exists():
                    # '전체' 코드를 그대로 읽음
                    top_1_code_str = code_path.read_text(encoding="utf-8", errors="ignore")
                    log.info(f"[GPT-OSS Prep] Loaded FULL Top-1 code (ID: {sels[0]['id']}) for structure reference.")
                else:
                    log.warning(f"[GPT-OSS Prep] Top-1 few-shot found (ID: {sels[0]['id']}) but code.py is missing.")
            else:
                log.warning("[GPT-OSS Prep] No Top-1 few-shot found. AI will generate code without a structural reference.")
        except Exception as e:
            log.warning(f"[GPT-OSS Prep] Top-1 few-shot load failed: {e}")
       
        # --- [DELETED] 4.5 "스마트 슬리밍" 로직 전체 삭제 ---
       
        # --- 5. SDIFF + MASK METADATA 결합 (이것이 guide_text_with_sdiff가 됨) ---
        full_prompt_list = []
        full_prompt_list.append("Generate a Python script based on the [STRUCTURED_DIFF] hint below.")
        full_prompt_list.append("Your code MUST import and use functions from `meta_utils`.")
       
        # [MODIFIED] 원본 SDIFF 대신 "클린 SDIFF" 텍스트를 주입
        full_prompt_list.append(clean_sdiff_text)  

        if meta_sum:
            try:
                # 메타데이터 JSON 생성 (기존에도 indent=2 였음)
                meta_json_str = json.dumps(meta_sum, ensure_ascii=False, indent=2)
                full_prompt_list.append("\n\n### MASK METATADATAv(Data/Hints for Algorithm)\n")
                full_prompt_list.append(meta_json_str)
            except Exception as e:
                log.warning(f"Failed to serialize Meta Summary for prompt: {e}")

        full_prompt = "\n".join(full_prompt_list)
       
        # --- 6. 로깅 [MODIFIED] (로그 가독성 문제 해결) ---
        if os.getenv("DEBUG_GPTOSS_PROMPT", "1") == "1":
            try:
                debug_path = (RUN_DIR / stem / "gptoss_gen_sdiff.debug.txt")
                debug_content = []
                debug_content.append(f"[gptoss_generate_from_sdiff log ({model_name}) @ {time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
               
                debug_content.append("--- 1. [CLEAN SDIFF] Hint + MASK METADATA (Sent to Model) ---\n")
               
                # [MODIFIED] full_prompt를 로깅하되, JSON 부분은 예쁘게 변환 시도
                try:
                    # [MODIFIED] clean_sdiff_dict 사용
                    pretty_sdiff = json.dumps(clean_sdiff_dict, ensure_ascii=False, indent=2)
                    # 메타데이터도 예쁘게
                    pretty_meta = json.dumps(meta_sum, ensure_ascii=False, indent=2) if meta_sum else "(No Meta)"
                   
                    # 로그에 예쁜 버전 추가
                    debug_content.append("Generate a Python script based on the [STRUCTURED_DIFF] hint below.\n")
                    debug_content.append("Your code MUST import and use functions from `meta_utils`.\n")
                    debug_content.append(pretty_sdiff) # <-- Pretty (Clean)
                    debug_content.append("\n\n### MASK METATADATAv(Data/Hints for Algorithm)\n")
                    debug_content.append(pretty_meta) # <-- Pretty
                except Exception:
                    debug_content.append(full_prompt) # 파싱 실패 시 원본
               
                # [MODIFIED] '전체' 코드를 로깅 (단, 너무 길면 자름)
                debug_content.append(f"\n\n--- 2. [FULL] Reference Code (Top-1 Few-Shot) --- (ID: {sels[0]['id'] if sels else 'N/A'})\n")
                if top_1_code_str:
                    # 로그 파일이 너무 커지는 것을 방지
                    if len(top_1_code_str) > 4000:
                         debug_content.append(top_1_code_str[:4000] + "\n...[TRUNCATED IN LOG]")
                    else:
                         debug_content.append(top_1_code_str)
                else:
                    debug_content.append("(Top-1 code not found or failed to load)")
               
                # debug.txt 덮어쓰기
                debug_path.write_text("\n".join(debug_content), encoding="utf-8")
               
                # [MODIFIED] raw_qwen.txt 로깅 (Pretty-Print)
                # [MODIFIED] 원본 V6 SDIFF (Full SDIFF)를 별도 파일에 저장
                debug_path_raw_qwen = (RUN_DIR / stem / "gptoss_gen_sdiff.full_v6.json")
                try:
                    # [MODIFIED] structured_diff_text -> full_sdiff_dict
                    pretty_sdiff_raw = json.dumps(full_sdiff_dict, ensure_ascii=False, indent=2)
                    debug_path_raw_qwen.write_text(pretty_sdiff_raw, encoding="utf-8")
                except Exception:
                    # 파싱 실패 시 원본 텍스트 저장
                    debug_path_raw_qwen.write_text(structured_diff_text or "(raw_text not found)", encoding="utf-8")
               
            except Exception as e:
                log.warning(f"[gptoss_generate_sdiff] debug log failed: {e}")

        # --- 7. 전용 Invoker 호출 [MODIFIED] ---
        # [MODIFIED] '전체' top_1_code_str 및 'model_name' 전달
        code = _safe_invoke_gptoss_for_code_from_sdiff(full_prompt, top_1_code_str, model_name, stem).strip()
        code = _strip_code_fence(code)

        # --- 8. 로깅 (Invoker 호출 후) ---
        if os.getenv("DEBUG_GPTOSS_PROMPT", "1") == "1":
            _dump_debug(
                stem,
                "gptoss_gen_sdiff.debug.txt",
                "\n--- 3. Model Output (Head) ---\n" + (code[:4000] if code else "(empty)") + "\n"
            )

        if not code.strip():
            # [THIS IS THE ERROR]
            return JSONResponse({"ok": False, "error": "LLM returned empty code."}, status_code=200)

        # --- 9. 재시도 로직 (Violation) [MODIFIED] ---
        violation = _reject_merge_copying(code)
        if violation:
            feedback = (
                "You violated NO-COLOR-COPY rule: " + violation + "\n"
                "Rewrite the entire script STRICTLY following the SDIFF HINTS and MASK METADATA:\n"
                "- Use ONLY `mask_path` and implement the logic (PCA/contours).\n"
                "- Use the correct 'class_val' from MASK METADATA and SDIFF HINTS.\n"
            )
            full_prompt_2 = full_prompt + "\n\n[FEEDBACK]\n" + feedback + "\n"
           
            # [MODIFIED] 재시도 시에도 '전체' top_1_code_str 및 'model_name' 전달
            code2 = _safe_invoke_gptoss_for_code_from_sdiff(full_prompt_2, top_1_code_str, model_name, stem).strip()
            code2 = _strip_code_fence(code2)

            if code2.strip():
                violation2 = _reject_merge_copying(code2)
                if not violation2:
                    code = code2
           
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

        # --- [CRITICAL FIX] 4. Qwen 원본 + Llama4 요약본 + Meta 결합 ---
        full_prompt_list = []

        # (A) [NEW] Qwen 원본(raw_text)을 '0) IMAGE SUMMARY'로 주입
        full_prompt_list.append("## 0) IMAGE SUMMARY (Qwen's Original Analysis)\n")
        full_prompt_list.append(qwen_raw_text)
        
        # (B) Llama4의 요약본 주입
        full_prompt_list.append("\n\n## [Llama4's Summary & Instructions]\n")
        full_prompt_list.append(guide_text) # (Llama4의 요약본)

        # (C) MASK METADATA 주입 (Failsafe 힌트)
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

                debug_content.append("--- 1. Combined Prompt (Qwen + Llama4 + Meta) ---\n")
                debug_content.append(full_prompt) # [CHANGED]

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

    # --- [NEW] AI가 생성한 새 함수를 'potential_utils.py'에 적재 ---
    try:
        _update_potential_utils(code_text)
    except Exception as e:
        log.warning(f"[code/save] Failed to update potential_utils.py: {e}")
    # --- [NEW] 끝 ---

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

# --- [NEW] Utils Manager 헬퍼 함수 ---
def _get_defined_function_names(file_path: Path) -> set:
    """AST를 사용해 파일에서 함수 정의 이름만 파싱합니다."""
    if not file_path.exists():
        return set()
    try:
        code = file_path.read_text(encoding="utf-8")
        parsed = ast.parse(code)
        return {node.name for node in ast.walk(parsed) if isinstance(node, ast.FunctionDef)}
    except Exception as e:
        log.error(f"Failed to parse AST for {file_path}: {e}")
        return set()

def _update_potential_utils(new_code_text: str):
    """
    새로 저장된 코드(new_code_text)를 파싱하여,
    'measurement_utils'나 'potential_utils'에 아직 없는
    "새로운" 헬퍼 함수를 'potential_utils.py'에 추가(append)합니다.
    """
    try:
        # 1. 기존 함수 목록 로드
        golden_funcs = _get_defined_function_names(MEASUREMENT_UTILS_PATH)
        staging_funcs = _get_defined_function_names(POTENTIAL_UTILS_PATH)
        
        # 2. 새 코드 파싱
        parsed_code = ast.parse(new_code_text)
        
        new_functions_to_add = []
        
        for node in ast.walk(parsed_code):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                # main 함수, 내부 함수, 비공개 함수 등은 무시
                if func_name == "main" or func_name.startswith("_"):
                    continue
                
                # 이미 라이브러리에 존재하는 함수는 무시
                if func_name in golden_funcs or func_name in staging_funcs:
                    continue
                
                # [NEW FUNCTION]
                func_source = ast.unparse(node) # Python 3.9+
                new_functions_to_add.append(func_source)
                staging_funcs.add(func_name) # 중복 추가 방지
        
        # 3. 새 함수가 있으면 potential_utils.py에 추가
        if new_functions_to_add:
            log.info(f"Found {len(new_functions_to_add)} new functions to stage: {list(staging_funcs)}")
            # 파일 끝에 추가
            with open(POTENTIAL_UTILS_PATH, "a", encoding="utf-8") as f:
                f.write("\n\n# --- Appended by /code/save ---\n\n")
                f.write("\n\n".join(new_functions_to_add))
                f.write("\n")
                
    except ImportError:
        log.error("Failed to unparse AST, 'ast.unparse' requires Python 3.9+.")
    except Exception as e:
        log.error(f"Error during _update_potential_utils: {e}")


# --- [NEW] Utils Manager 라우트 ---

@app.get("/utils_manager", response_class=HTMLResponse)
async def get_utils_manager(request: Request):
    """Utils 관리자 UI 페이지를 서빙합니다."""
    return templates.TemplateResponse("utils_manager.html", {"request": request})

@app.get("/utils/get_all")
async def utils_get_all():
    """두 라이브러리 파일의 현재 내용을 읽어옵니다."""
    try:
        golden = MEASUREMENT_UTILS_PATH.read_text(encoding="utf-8") if MEASUREMENT_UTILS_PATH.exists() else ""
        staging = POTENTIAL_UTILS_PATH.read_text(encoding="utf-8") if POTENTIAL_UTILS_PATH.exists() else ""
        return JSONResponse({"ok": True, "golden_content": golden, "staging_content": staging})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})

@app.post("/utils/save_golden")
async def utils_save_golden(payload: dict = Body(...)):
    """핵심 라이브러리(measurement_utils.py)를 수동 저장합니다."""
    content = payload.get("content")
    if content is None:
        return JSONResponse({"ok": False, "error": "content is missing"})
    try:
        MEASUREMENT_UTILS_PATH.write_text(content, encoding="utf-8")
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})

@app.post("/utils/prune_staging")
async def utils_prune_staging():
    """
    후보군(potential) 라이브러리에서
    핵심(measurement) 라이브러리에 이미 존재하는 함수를 제거(정리)합니다.
    """
    try:
        golden_funcs = _get_defined_function_names(MEASUREMENT_UTILS_PATH)
        if not POTENTIAL_UTILS_PATH.exists():
            return JSONResponse({"ok": True, "pruned_count": 0})

        staging_code = POTENTIAL_UTILS_PATH.read_text(encoding="utf-8")
        staging_ast = ast.parse(staging_code)
        
        pruned_nodes = []
        pruned_count = 0
        
        for node in staging_ast.body:
            if isinstance(node, ast.FunctionDef):
                if node.name not in golden_funcs:
                    pruned_nodes.append(node) # 보존
                else:
                    pruned_count += 1 # 제거
            else:
                 pruned_nodes.append(node) # 주석, import 등 보존
        
        # ast.unparse가 가능한 환경 (Python 3.9+)을 가정
        new_staging_code = ast.unparse(pruned_nodes)
        POTENTIAL_UTILS_PATH.write_text(new_staging_code, encoding="utf-8")
        
        log.info(f"Pruned {pruned_count} functions from potential_utils.py")
        return JSONResponse({"ok": True, "pruned_count": pruned_count})
        
    except ImportError:
        return JSONResponse({"ok": False, "error": "ast.unparse requires Python 3.9+"})
    except Exception as e:
        log.exception("Failed to prune staging utils")
        return JSONResponse({"ok": False, "error": str(e)})

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
    log.info(f"GAUSSO_BASE_URL set? {bool(GAUSSO_BASE_URL)} MODEL={GAUSSO_MODEL}")
    log.info(f"QWEN_ENABLE={QWEN_ENABLE} MODEL_ID={QWEN_MODEL_ID} DEVICE={QWEN_DEVICE} DTYPE={QWEN_DTYPE}")
    uvicorn.run(app, host="0.0.0.0", port=port)