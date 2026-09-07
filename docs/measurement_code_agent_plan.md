# 스크리블 기반 측정 코드 생성 Agent — 시스템 개발 계획서

> **문서 버전** v1.0
> **대상 저장소** `sy5255/ai_agent_project` (as-is 분석) → 신규 시스템(to-be) 설계
> **한 줄 정의** 사용자가 이미지 위에 **그림판처럼 스크리블(선)을 그리면**, 시스템이 그 **측정 의도(intent)를 복원·보정·확정**하고, 그 의도를 **다른 이미지에도 그대로 적용 가능한 파이썬 측정 스크립트**로 자동 생성·검증·배포하는 Agent.

---

## 목차

- [Part 0. 문서의 목적과 읽는 법](#part-0-문서의-목적과-읽는-법)
- [Part 1. 기존 시스템(As-Is) 완전 분석](#part-1-기존-시스템as-is-완전-분석)
- [Part 2. 신규 시스템(To-Be) 컨셉과 아키텍처](#part-2-신규-시스템to-be-컨셉과-아키텍처)
- [Part 3. 전체 기능 명세 (F-01 ~ F-72)](#part-3-전체-기능-명세)
- [Part 4. 핵심 데이터 모델](#part-4-핵심-데이터-모델)
- [Part 5. ★ 모델 API 통합 설계 (Gemma4 / GaussO4.1)](#part-5--모델-api-통합-설계-gemma4--gausso41)
- [Part 6. 구현 계획 (모듈 · 기술스택 · 마일스톤)](#part-6-구현-계획)
- [Part 7. 평가 체계와 완료 기준](#part-7-평가-체계와-완료-기준)
- [Part 8. 리스크와 대응](#part-8-리스크와-대응)
- [부록 A. API 목록 / 디렉터리 구조 / 마이그레이션](#부록-a-api-목록--디렉터리-구조--마이그레이션)

---

# Part 0. 문서의 목적과 읽는 법

이 문서는 두 가지를 한 번에 담는다.

1. **기존 저장소가 무엇을 만들려고 했는지**를 코드의 세부 구현이 아니라 **설계 의도와 시스템 구조 수준**에서 정리한다. (Part 1)
2. 그 컨셉을 계승하되, **"사용자가 그림판처럼 스크리블을 그리면 측정 코드를 만들어주는 Agent"** 라는 목표에 맞춰 **새로 개발할 시스템의 전체 기능과 구현 계획**을 정의한다. (Part 2~8)

Part 3의 기능들은 모두 `F-xx` 번호를 가진다. (F-01~F-58은 제품 기능, F-59~F-72는 모델 API 통합 기능으로 Part 5에서 상세히 다룬다.) Part 6의 마일스톤과 Part 7의 KPI가 이 번호를 참조하므로, 개발 착수 시 이 번호를 그대로 이슈 트래커의 Epic/Story 키로 사용하면 된다.

**우선순위 표기**
- `P0` — MVP 필수. 이것이 없으면 시스템이 성립하지 않음.
- `P1` — 실사용 품질 확보에 필요. MVP 직후 착수.
- `P2` — 고도화/차별화. 여유가 생기면 착수.

---

# Part 1. 기존 시스템(As-Is) 완전 분석

## 1.1 시스템의 목적

반도체 단면(SEM/TEM 계열) 이미지에 대한 **계측(metrology) 자동화**다. 사람이 이미지마다 손으로 재던 치수를, **"한 장에 한 번 시연(demonstration)하면 나머지는 코드가 알아서 재도록"** 만드는 것이 목표다.

구체적으로:

- 입력: 세그멘테이션 **마스크 이미지**(클래스별 gray value, 예: 10/30/50), 원본과 마스크를 겹친 **merge 이미지**, 그리고 픽셀 스케일·클래스 목록을 담은 **meta JSON**.
- 사용자 시연: 검은 배경 위에 **빨간 선(가이드/기준선)** 과 **초록 선(실제 측정선)** 만 그린 **스크리블 이미지**.
- 출력: `--mask_path / --out_dir / --meta_root` 만 받아 **단독 실행 가능한 `measure.py`**, 그리고 그 실행 결과인 `measurements.csv` + `overlay.png`.

즉 이 시스템의 진짜 산출물은 "측정값"이 아니라 **"측정 프로그램"** 이다. 값은 그 프로그램을 돌리면 나온다. 이 구분이 전체 설계를 지배한다.

## 1.2 전체 파이프라인

```
[mask.tif] [merge.png] [scribble.png] [meta.json]
        │        │            │            │
        └────────┴──────┬─────┴────────────┘
                        ▼
   ① CV 파싱 : 스크리블에서 빨강/초록 선분 벡터화
      - HSV inRange → 형태학 연산 → findContours → cv2.fitLine
      - 선분별 길이/각도/방향(h·v·diagonal) 산출
      - 선분의 끝점 주변 마스크를 샘플링 → "이 끝점은 클래스 10에 닿아 있다" 힌트 생성
                        ▼
   ② VLM 의미 추론 (Qwen3-VL, 로컬 HF)
      - 이미지 3장(mask/merge/scribble) + ①의 CV 사실 + 밝기맵(darkest/middle/brightest→클래스값)
      - 정해진 JSON 스키마의 빈칸을 채우게 함:
        · coordinate_system(absolute/local, 주축이 될 red 선)
        · grouping_logic(반복 구조인가? 몇 개씩 묶이나? red 대비 green 방향은?)
        · red 선별 construction_method(이 기준선을 어떻게 "찾아낼" 것인가)
        · green 선별 u/v placement hint 4종(시작/끝의 U좌표·V좌표를 찾는 자연어 알고리즘)
                        ▼
   ③ Grounding DINO (open-vocab detection)
      - VLM이 만든 명사구("four darkest fingers")+위치힌트("top")로 ROI 박스 검출
      - 위치 힌트로 ROI 상/하 절반 슬라이싱, 개수 힌트로 top-N 절단
                        ▼
   ④ 템플릿 추출
      - 모든 스크리블 선을 감싸는 전역 bounding box를 마스크에서 잘라 grounding_template.png 저장
      - 각 green 선의 ROI를 "템플릿 좌상단 기준 상대좌표"로 저장 (relative_roi)
        → 새 이미지에서는 템플릿 매칭으로 원점을 다시 찾고 상대좌표를 더해 ROI 재현
                        ▼
   ⑤ SDIFF v6 (Structured Diff) 조립
      - CV 기하 사실 + VLM 논리 + DINO ROI + 템플릿 상대좌표를 하나의 JSON으로 병합
      - runs/<stem>/sdiff_qwen_v6.json 으로 영속화
                        ▼
   ⑥ 코드 생성 (GPT-OSS-120B 또는 GaussO-Think)
      - SDIFF를 "정제(sanitize)": angle_deg/length_px/x1..y2/endpoints 등 물리 수치를 제거
        → 모델이 좌표를 베끼지 못하게 하고 "논리"만 보고 알고리즘을 짜게 강제
      - Few-shot Top-1 코드 전문을 "구조 참고용"으로 함께 제공
      - 초대형 시스템 프롬프트(수백 줄)로 규칙 강제
                        ▼
   ⑦ 실행 & 자동 수정
      - subprocess로 measure.py 실행 (timeout 300s)
      - 실패 시 stderr를 LLM에 던져 코드 재작성 → 재실행 (최대 N회)
      - overlay.png 생성 + measurements.csv 표준 헤더 정규화
                        ▼
   ⑧ 자산화
      - 승인된 코드를 label 폴더에 .py로 저장
      - few-shot 저장소에 (썸네일, 코드, meta 요약, pHash) 케이스 등록
      - AST로 새 헬퍼 함수를 추출해 potential_utils.py(스테이징)에 적재 → 사람이 검토 후 measurement_utils.py(골든)로 승격
                        ▼
   ⑨ 전이 신뢰도 평가 (오프라인 스크립트)
      - shift/rotate/scale/shear + 랜덤 50종 augmentation 생성
      - 각각에 measure.py 실행 → 성공/실패/에러단계/CSV행수를 transfer_summary.csv로 집계
```

## 1.3 구성요소별 상세

### (a) 서버 / UI
- `app_codecheck.py` — FastAPI 단일 파일(약 5,400줄). 라우트 약 30개.
- `index_codecheck.html` — 파일 목록 · 4장 이미지 뷰 · SDIFF 뷰어 · few-shot 카드 · 코드 에디터(textarea) · 실행 로그 · overlay 뷰어 · 어시스턴트 채팅.
- `utils_manager.html` — 골든/스테이징 유틸 라이브러리 편집기.

### (b) 데이터 규약 (파일명 기반)
| 항목 | 규칙 |
|---|---|
| merge | `<base>_gray_merge.png` (IMAGE_DIR) |
| mask | `<base>_gray.tif` (IMAGE_DIR의 형제 폴더 `mask/`) |
| scribble | `<base>.png` (SCRIBBLE_DIR) |
| meta | `<base>.json` (META_ROOT) — `pixel_scale_um_x/y`, `mask_info.classes_present` |
| 산출물 | `runs/<stem>/` 아래 measure.py, measurements.csv, overlay.png, sdiff_qwen_v6.json, 각종 debug png/txt |
| 라벨 | `<base>.py` (LABEL_DIR) |

### (c) 표준 CSV 스키마
```
measure_item, group_id, index, value_nm, sx, sy, ex, ey,
meta_tag, component_label, image_name, run_id, note
```

### (d) 모델 구성
| 역할 | 모델 | 호출 방식 |
|---|---|---|
| 스크리블 의미 해석 | Qwen3-VL-8B-Instruct | 로컬 HuggingFace (CUDA 0,1,2) |
| ROI 그라운딩 | grounding-dino-base | 로컬 HuggingFace |
| 코드 생성 | gpt-oss-120b / GaussO-Think | 사내 OpenAI 호환 게이트웨이 |
| 보조 설명 | Gemma3-27B, Llama-4-Maverick | 사내 게이트웨이 |

> **신규 시스템에서는 위 5종 모델을 전부 폐기하고 Gemma4 · GaussO4.1 두 개만 사용한다.** 로컬 HuggingFace 추론(Qwen3-VL, Grounding DINO)도 제거되어 GPU 상주 부담이 사라진다. 상세 설계는 [Part 5](#part-5--모델-api-통합-설계-gemma4--gausso41).

### (e) 안전장치 (이 시스템이 실제로 싸우고 있던 문제들)
기존 코드가 방어하려 한 실패 모드는 신규 설계의 요구사항 그 자체다.

1. **정답 베끼기(cheating)** — 생성 코드가 merge/scribble 이미지에서 초록·빨강 픽셀을 직접 읽어 "측정한 척"하는 것. → `_reject_merge_copying()` 정규식 차단 + SDIFF에서 좌표 제거.
2. **환각 함수 호출(NameError)** — few-shot 코드에만 있던 헬퍼를 정의 없이 호출. → 프롬프트에 "def로 정의되지 않은 함수 호출 금지" 규칙을 반복 명시.
3. **상수 하드코딩** — few-shot의 `target_class=50` 같은 값을 그대로 가져옴. → "SDIFF 값이 유일한 진실" 규칙.
4. **좌표계 혼동** — `np.where`의 (y,x)를 `cv2.fitLine`의 (x,y)로 잘못 전달, ROI 로컬 좌표를 LCS(u,v)에 직접 사용, ROI 클램핑으로 모든 박스가 (0,0)으로 몰림. → 프롬프트에 케이스별 금지 규칙 나열.
5. **numpy 직렬화 오류** — `float32 is not JSON serializable`. → 모든 값 `float()/int()` 캐스팅 강제.
6. **일반화 실패** — 시연 이미지에서만 동작. → 템플릿 매칭 기반 재정위 + 전이 신뢰도 평가 스크립트.

## 1.4 설계 의도 요약 — 이 시스템의 5가지 핵심 아이디어

이 저장소가 진짜로 만들려던 것은 다음 다섯 문장으로 요약된다. **이 다섯 가지는 신규 시스템에서도 그대로 계승한다.**

1. **시연 학습(Programming by Demonstration).** 사용자는 프로그래밍하지 않는다. 한 장에 선을 그어 보여줄 뿐이고, 시스템이 그것을 프로그램으로 번역한다.
2. **좌표가 아니라 의도를 옮긴다.** 스크리블의 픽셀 좌표는 그 이미지에만 유효하다. 그래서 좌표를 지우고 "가장 어두운 클래스의 최상단에서 아래쪽 경계까지"라는 **알고리즘적 의미**로 바꾼 뒤 코드로 만든다.
3. **역할 분담: CV는 사실, VLM은 의미, LLM은 코드.** 기하는 OpenCV가 정확히 재고, "왜 그렇게 그었는가"는 VLM이 추론하고, 그 둘을 합친 명세로 코드 모델이 구현한다.
4. **결과물은 검증 가능한 아티팩트다.** 실행되어야 하고, 표준 CSV를 뱉어야 하고, overlay로 눈으로 확인되어야 하고, 흔들어도(augmentation) 살아남아야 한다.
5. **자산은 축적된다.** 승인된 코드는 few-shot이 되고, 그 안의 헬퍼 함수는 공용 라이브러리로 승격된다. 시스템은 쓸수록 좋아진다.

## 1.5 진단 — 신규 개발이 필요한 이유

| # | 문제 | 근거 | 신규 설계에서의 해결 |
|---|---|---|---|
| 1 | **스크리블 입력 수단이 없다** | UI는 `/get_scribble`로 미리 만들어둔 PNG를 **보여주기만** 한다. 실제 드로잉은 외부 그림판에서 해야 한다. | 브라우저 드로잉 캔버스를 1급 기능으로 (F-01~F-14) |
| 2 | **의도를 래스터에서 "다시 추측"한다** | 사용자는 그릴 때 이미 "이 경계에 붙이려고" 했는데, 저장은 PNG로 되고, 시스템은 HSV+contour+VLM으로 그 의도를 되찾으려 애쓴다. 정보 손실이 크다. | **스냅 = 의미 캡처**. 그리는 순간의 스냅 대상을 구조화 기록 (F-06, F-15) |
| 3 | **명세(SDIFF)가 비정형** | 스키마 버전이 v1/v2_lite/v6로 섞여 있고, 키가 코드 곳곳에서 즉석 조립된다. 검증기가 없다. | Pydantic으로 정의된 **MIR(Measurement Intent Representation)** + JSON Schema 검증 (F-15) |
| 4 | **프롬프트가 방어 로직을 대신한다** | 수백 줄의 "절대 ~하지 마라" 규칙으로 NameError·좌표계 버그를 막는다. 확률적이라 새 케이스마다 규칙이 늘어난다. | **타입 있는 DSL + 검증된 유틸 라이브러리 위로 생성**. 자유 코드 생성 → 계획(plan) 생성 + 결정론적 렌더링 (F-30, F-31) |
| 5 | **자기 검증이 없다** | "실행이 되고 overlay가 나오면 성공"이다. **사용자가 그린 선을 재현했는지**는 확인하지 않는다. | **시연 재현 검증(Demo Replay)** 을 필수 게이트로 (F-36) |
| 6 | **일반화 검증이 파이프라인 밖** | `transfer_reliability_eval.py`는 경로가 하드코딩된 수동 스크립트다. | 전이 신뢰도 테스트를 Agent 루프 안의 게이트로 편입 (F-37) |
| 7 | **상태가 프로세스 메모리** | `SESSIONS`, `CODE_BACKUPS`, `STRUCTURED_DIFF_CACHE`가 dict. 재시작하면 사라진다. | SQLite/Postgres 영속화 + 이력 관리 (F-48) |
| 8 | **환경 이식성** | Windows 절대경로 하드코딩(`D:\aip-expert\...`), API 키 리터럴(`os.environ['OPENAI_API_KEY']='api_key'`). | 설정 주입 + 시크릿 관리 + Docker (F-50, F-52) |
| 9 | **동기 블로킹 · 폴링** | VLM 추론과 subprocess 실행이 요청 스레드를 막고, 진행률은 `/run/status` 폴링. | 작업 큐 + SSE 스트리밍 (F-49) |
| 10 | **단일 파일 5,400줄** | 테스트 불가, 변경 시 회귀 위험. | 레이어드 패키지 구조 + 단위 테스트 (Part 6) |
| 11 | **모델이 5종으로 흩어짐** | Qwen3-VL(로컬 GPU) + DINO(로컬 GPU) + GPT-OSS + GaussO + Gemma3/Llama4. 각각 호출 방식이 제각각이고 GPU 3장을 상주 점유한다. | **Gemma4 / GaussO4.1 2종으로 통일.** 로컬 GPU 추론 제거, 전송 계층 단일화 (Part 5) |
| 12 | **이미지 페이로드 관리 부재** | `MAX_IMAGES_PER_PROMPT = 5` 상수만 있고 실제 강제·크기 검사·좌표 역변환 관리가 없다. | 슬롯 예산 플래너 + 크기 파이프라인 + 변환 메타 (F-63~F-67) |

---

# Part 2. 신규 시스템(To-Be) 컨셉과 아키텍처

## 2.1 제품 정의

> **ScribbleMetro** (가칭) — 계측 엔지니어가 이미지 위에 **대충 그은 선**을, 시스템이 **정확한 측정 의도로 다듬어 확정**하고, **다른 모든 이미지에 적용 가능한 파이썬 측정 스크립트**로 자동 생성·검증·운영하는 Agent 플랫폼.

**핵심 사용자 경험 (목표 5분 시나리오)**

1. 이미지를 연다. (마스크가 반투명 오버레이로 겹쳐 보인다)
2. 빨간 펜으로 기준선을 **대충** 긋는다 → 시스템이 *"4개 핑거의 최상단을 지나는 직선으로 피팅할까요?"* 라고 **고스트 프리뷰**를 띄운다 → `Enter`.
3. 초록 펜으로 측정선을 **대충** 긋는다 → 끝점이 **경계에 자석처럼 붙고**, 실시간으로 `43.2 nm` 가 표시된다.
4. 시스템이 *"이 구조가 4번 반복됩니다. 나머지 3곳에도 같은 측정을 넣을까요?"* → `Yes`.
5. 시스템이 의도를 자연어로 **되읽어준다**: *"가장 어두운 클래스 핑거 4개 각각에 대해, 최상단 점에서 주축의 법선 방향으로 중간계조 경계까지의 거리를 잽니다."* → 확인.
6. **[코드 생성]** → 생성 → 정적 검증 → 샌드박스 실행 → **시연 재현 검증**(내가 그은 선과 코드가 그린 선의 오차 1.4px) → **전이 테스트**(augmentation 50종 중 49종 통과) → 초록불.
7. 배치 탭에서 폴더 전체에 적용. 분포/이상치 대시보드 확인. 레시피 저장.

## 2.2 설계 원칙

| 원칙 | 의미 |
|---|---|
| **Intent-first** | 시스템의 1급 데이터는 이미지도 코드도 아닌 **MIR(측정 의도 표현)** 이다. 스크리블도 코드도 MIR의 투영(projection)이다. |
| **Snap is Semantics** | 스냅은 예쁘게 보이려는 기능이 아니라 **의도를 무손실로 캡처하는 수단**이다. 스냅될 때마다 "무엇에 붙었는지"가 구조화 기록된다. |
| **Recover, don't guess** | 그리는 순간에 잡을 수 있는 정보는 그때 잡는다. 나중에 픽셀에서 되추측(VLM)하는 것은 **보조 수단**으로 격하한다. |
| **Generate plans, not prose** | LLM은 자유 파이썬을 쓰지 않고, **검증된 연산자 조합(plan)** 을 낸다. 실행 코드는 결정론적 렌더러가 만든다. |
| **Nothing ships unverified** | 실행 성공 · 시연 재현 · 전이 안정성 — 3개 게이트를 통과해야 "완료"다. |
| **Human in the right loop** | 사람은 코드를 고치지 않는다. **의도를 확인·수정**한다. 수정은 항상 MIR 레벨에서 일어난다. |
| **Everything is versioned** | 이미지, MIR, 코드, 모델, 프롬프트, 실행 결과가 모두 해시로 묶여 재현 가능하다. |
| **Models accelerate, never gate** | 모델 호출은 품질을 높이는 가속기다. 모델이 죽어도 드로잉·스냅·패턴검출·측정 실행은 전부 동작해야 한다. (F-71) |

## 2.3 아키텍처

```
┌──────────────────────────────────────────────────────────────────────┐
│ Frontend (TypeScript, Canvas 2D/WebGL)                               │
│  ┌────────────┐ ┌──────────────┐ ┌────────────┐ ┌─────────────────┐  │
│  │ Draw Canvas│ │ Intent Panel │ │ Code View  │ │ Batch Dashboard │  │
│  │ (스크리블) │ │ (MIR 편집)   │ │ (diff/log) │ │ (분포/이상치)   │  │
│  └─────┬──────┘ └──────┬───────┘ └─────┬──────┘ └────────┬────────┘  │
└────────┼───────────────┼───────────────┼─────────────────┼───────────┘
         │ stroke+snap   │ MIR patch     │ generate/run    │ batch job
┌────────▼───────────────▼───────────────▼─────────────────▼───────────┐
│ API Gateway (FastAPI) + Job Queue (async worker)                     │
├──────────────────────────────────────────────────────────────────────┤
│ ① Perception Service      ② Intent Service     ③ Synthesis Service   │
│  · 마스크 전처리          · 스트로크→기본도형   · 검색(few-shot/MIR) │
│  · 앵커 프리미티브        · 스냅 의미 해석      · Planner(LLM)       │
│  · 반복/대칭 패턴 검출    · MIR 빌드/검증       · Renderer(결정론)   │
│  · 템플릿/특징 재정위     · 모호성 질문 생성    · Static Validator   │
│  · (선택) VLM 의미 보강   · 자연어 되읽기       · Repair Loop        │
├──────────────────────────────────────────────────────────────────────┤
│ ④ Execution Service       ⑤ Verification Svc   ⑥ Knowledge Service   │
│  · 샌드박스 실행          · Demo Replay 검증   · 레시피 레지스트리   │
│  · 배치 러너              · 전이(augment) 검증 · few-shot 코퍼스     │
│  · 아티팩트 수집          · QC 룰/이상치       · utils 라이브러리    │
│                           · 민감도 분석        · 평가 벤치마크       │
├──────────────────────────────────────────────────────────────────────┤
│ Storage: SQLite/Postgres(메타·이력) + 오브젝트 스토리지(이미지·산출물)│
│ Model Router: Gemma4(멀티모달 인식) / GaussO4.1(추론·계획)  ← Part 5   │
└──────────────────────────────────────────────────────────────────────┘
```

**데이터 흐름 한 줄 요약**
`Stroke + Snap` → `Intent(MIR)` → `Plan` → `Code` → `Artifacts` → `Verification` → `Recipe`

---

# Part 3. 전체 기능 명세

각 기능은 **무엇을 / 왜 / 어떻게 동작하고 / 어떻게 구현하는가**로 기술한다.

---

## A. 드로잉 캔버스 — "그림판" 계층

### F-01. 멀티레이어 드로잉 캔버스 `P0`
- **무엇** 브라우저에서 이미지 위에 직접 그리는 캔버스. 레이어: `배경(merge/원본)` / `마스크 오버레이(클래스별 색상, 투명도 조절)` / `가이드(빨강)` / `측정(초록)` / `보조(노랑)` / `주석(텍스트)`.
- **왜** 기존 시스템의 가장 큰 공백. 외부 그림판 왕복이 사라지고, 무엇보다 **그리는 순간의 컨텍스트(스냅 대상)** 를 잡을 수 있게 된다.
- **동작** 마우스/펜/터치. 레이어 표시·잠금·투명도. 이미지 좌표계 고정(줌해도 좌표 정확도 유지).
- **구현** `<canvas>` 2D 컨텍스트 2장(정적 이미지 / 동적 스트로크) + 오프스크린 캔버스. 좌표는 항상 **원본 이미지 픽셀 좌표**로 정규화해 저장. 마스크 오버레이는 클래스값→팔레트 LUT를 GPU(WebGL) 또는 오프스크린에서 1회 계산 후 캐시.

### F-02. 정밀 뷰포트: 무한 줌/팬 · 미니맵 · 픽셀 그리드 `P0`
- **왜** 2048×2048 이상 마스크에서 1~2px 단위 판단이 필요하다.
- **동작** 휠 줌(커서 기준), 스페이스+드래그 팬, 16배 이상 줌 시 픽셀 격자 및 클래스값 툴팁 표시, 미니맵으로 현재 위치 표시.
- **구현** 뷰 변환 행렬 하나(`scale, tx, ty`)로 모든 렌더/히트테스트 통일. 타일링 렌더로 대형 이미지 대응.

### F-03. 도구 팔레트 `P0`
- 자유곡선 펜 / **직선** / 폴리라인 / 원호 / 사각 ROI / 자유형 ROI / 점(앵커 마커) / 지우개 / 선택·이동 / 텍스트 주석.
- 각 도구는 `layer`(guide/measure/helper)와 결합. 예: "초록 직선"은 `measure` 레이어의 segment 도구.
- **구현** 도구는 `Tool` 인터페이스(`onDown/onMove/onUp/preview/commit`) 구현체. 신규 도구 추가가 플러그인처럼 되도록.

### F-04. 실시간 치수 표시(Live Readout) `P0`
- **무엇** 초록 선을 그리는 동안 길이가 **nm 단위로 실시간 표시**되고, 시작/끝점이 놓인 **클래스 이름**이 함께 뜬다. (`43.2 nm  [poly → SiN]`)
- **왜** 잘못 그은 것을 **그 자리에서** 알 수 있다. 코드 생성까지 가서야 발견하는 낭비를 없앤다.
- **구현** meta의 `pixel_scale_um_x/y`로 즉시 환산. 끝점 클래스는 마스크 배열에서 반경 샘플링(최빈값).

### F-05. Undo/Redo · 자동저장 · 스크리블 버전 `P0`
- 커맨드 패턴 기반 무제한 undo. 5초 주기 자동저장. 스크리블은 세션마다 버전으로 남고 diff 비교 가능.

### F-06. ★ 스마트 스냅 (Semantic Snapping) `P0`
> **이 기능이 신규 시스템의 심장이다.**

- **무엇** 커서/끝점이 의미 있는 기하 대상에 자석처럼 붙고, **무엇에 붙었는지가 구조화되어 기록**된다.
- **스냅 대상 (우선순위 순)**
  1. **클래스 경계(edge)** — 특정 클래스 영역의 컨투어. 서브픽셀 보정 포함.
  2. **특징점** — 코너, 영역 내 최상단/최하단/최좌/최우 점, 무게중심, 볼록껍질 꼭짓점.
  3. **기존 선** — 다른 스트로크의 끝점 / 선 위의 최근접점 / 두 선의 교점.
  4. **투영점** — 기준선(빨강)에 내린 수선의 발.
  5. **각도** — 0°/45°/90°, 또는 **주축(primary axis) 기준 평행/수직**.
  6. **대칭축 / 반복 격자** — 패턴 검출 결과(F-11)의 격자점.
- **왜** 사용자가 "여기 붙이려고 했다"는 사실을 **추측이 아니라 기록**으로 남긴다. `snap.target = {kind:"class_boundary", class:10, feature:"topmost", roi:"finger[0]"}` — 이 한 줄이 기존 시스템이 VLM 두 번 돌려 얻으려 했던 정보다.
- **구현**
  - 마스크 로드시 클래스별 컨투어/거리변환(distance transform)/코너맵을 **사전 계산해 인덱싱**(KD-Tree).
  - 커서 반경 R(줌 보정) 내 후보를 점수화: `score = w1·(1-d/R) + w2·priority + w3·(사용자 최근 선택 이력)`.
  - 스냅 대상은 화면에 **하이라이트 + 라벨**로 표시. `Alt`로 일시 해제, `Tab`으로 후보 순환.

### F-07. ★ 스트로크 의도 인식 및 정형화 (Beautify / Intent Recognition) `P0`
> **사용자가 직접 요청한 기능. "대충 그어도 알아서 다듬어준다."**

- **무엇** 손으로 흔들리게 그린 획을 분석해 **어떤 도형을 그리려 했는지 후보를 제안**하고, 고스트 프리뷰로 보여준 뒤 사용자가 1키로 확정한다.
- **후보 유형** 직선분 / 수평선 / 수직선 / 특정 각도(45°) 선 / **주축 평행·수직선** / 원호 / 폴리라인(꺾인 선) / 닫힌 영역 / 점(짧은 획).
- **판정 로직**
  1. 획을 리샘플링 후 **Douglas-Peucker**로 단순화 → 정점 수로 1차 분류(직선/폴리라인).
  2. 각 후보 원형(primitive)에 **Total Least Squares** 피팅 → 잔차(RMS) 계산.
  3. **모델 선택**: `cost = RMS + λ·(파라미터 수)` (MDL/BIC 방식). 잔차 차이가 작으면 **더 단순한 도형이 이긴다**.
  4. 각도가 0/90/45/주축 기준 ±τ(기본 5°) 이내면 **각도 스냅 후보**를 추가 가점.
  5. 끝점이 스냅 후보(F-06)에 근접하면 그 스냅을 반영한 버전을 상위 후보로.
- **UX** 획을 떼는 순간 원래 획은 흐리게 남고, 1순위 후보가 **점선 고스트**로 겹쳐 표시된다.
  - `Enter` 수락 · `Tab` 다음 후보 · `Esc` 원본 유지 · 우측 하단에 후보 칩(예: `직선(수직) 98%` `원호 61%` `원본`).
- **왜** 계측에서 손떨림 3px는 곧 측정 오차다. 그리고 "수직선을 그으려 했다"는 정보는 곧 **코드의 방향 제약**이 된다. 미화가 아니라 **의도 획득**이다.
- **구현** `intent/stroke_fit.py` — 순수 함수(입력: 점 배열 + 컨텍스트, 출력: 랭킹된 후보 리스트). **프런트에서 즉시 1차 판정**(반응성), 서버에서 마스크 컨텍스트를 이용한 정밀 재판정(F-08)으로 갱신.

### F-08. 컨텍스트 인식 보정 (Mask-aware Refinement) `P1`
- **무엇** 스트로크 정형화 시 **마스크 구조를 근거로** 보정한다.
  - 기준선을 그었는데 근처에 "가장 어두운 클래스 영역 4개의 최상단 점들"이 거의 일직선이면 → *"이 4점에 피팅한 직선"* 을 제안. (기존 시스템이 VLM에게 자연어로 물어보던 `construction_method`를 **CV로 직접 제안**)
  - 측정선을 그었는데 끝점이 경계에서 2px 떠 있으면 → 경계로 당기고 *"경계에 스냅함"* 배지 표시.
  - 측정선이 클래스 A와 B의 경계를 지나면 → *"A→B 두께"* 라는 의미 라벨 자동 부여.
- **구현** 스트로크 주변 밴드(±20px) 내 클래스별 픽셀 통계 + 컨투어 교차 분석 → 후보 해석 생성.

### F-09. 제약 조건 인스펙터 (Constraint Inspector) `P0`
- **무엇** 선 하나를 선택하면 우측 패널에 **그 선의 의미가 편집 가능한 폼**으로 뜬다.

  | 필드 | 예시 값 | 출처 |
  |---|---|---|
  | 이름 | `Gate_CD` | 사용자/자동제안 |
  | 시작 앵커 | `클래스 poly 영역의 최상단 점` | 스냅 기록 |
  | 끝 앵커 | `클래스 SiN 경계 (법선 방향 최초 교차)` | 스냅 기록 |
  | 방향 | `red_1에 수직` | 각도 스냅 |
  | 탐색 ROI | `finger[i]` (반복 패턴 인스턴스) | 패턴 검출 |
  | 강건화 | `상위 5행 중앙값` | 기본값/권고 |
  | 기대 범위 | `35 ~ 55 nm` | 사용자 |
- **왜** "추천/제안/수정"의 최종 착지점. 시스템 추론이 틀렸을 때 사용자가 **코드가 아니라 의미를 고친다**.
- **구현** MIR의 `measurements[i]` 노드를 그대로 폼 바인딩. 각 필드에 **provenance 배지**(스냅/CV/VLM/사용자)와 신뢰도 표시.

### F-10. 측정 항목 명명 · 그룹핑 · 페어링 UI `P0`
- 측정선에 이름(`measure_item`) 부여, 그룹(`group_id`)으로 묶기, 쌍(pair) 지정. 드래그로 그룹 편집.
- 기존 시스템이 VLM의 `grouping_logic` 자연어에 의존하던 부분을 **명시적 UI**로 대체(+VLM 제안은 초기값으로만 사용).

### F-11. ★ 반복 구조 검출 및 자동 전파 (Pattern Propagation) `P0`
- **무엇** 사용자가 **한 곳에만** 측정선을 그리면, 시스템이 동일 구조가 몇 번 반복되는지 찾아 **나머지에 같은 측정을 제안**한다.
- **동작** *"이 구조가 가로 방향으로 6회 반복됩니다(주기 187px). 나머지 5곳에 동일 측정을 추가할까요?"* → 미리보기 오버레이 → 개별 체크박스로 취사선택.
- **구현**
  1. 사용자 스트로크 주변 패치를 템플릿으로 삼아 **정규화 상호상관(NCC) 매칭** → 피크 검출.
  2. 병행 검증: 클래스 마스크의 **행/열 프로파일 자기상관(autocorrelation)** 으로 주기 추정.
  3. 연결요소(connected components) 기반 인스턴스 분할로 반복 단위 확정.
  4. 결과를 MIR의 `patterns[]`에 `periodic_1d{axis, period, count, instances[]}`로 기록 → 각 측정은 `roi: finger[i]`로 인스턴스를 참조.
- **왜** 시연 비용이 N배 줄고, 무엇보다 **"반복 구조를 순회하라"는 논리가 코드에 정확히 반영**된다. (기존 시스템이 프롬프트로 "zip 구조를 버리고 3쌍으로 묶어라"고 설득하던 문제의 근본 해결)

### F-12. 대칭·배열 보조 `P1`
- 미러(수직/수평/임의축), 선형 배열, 그리드 배열. 대칭축은 마스크의 대칭성 분석으로 자동 제안.

### F-13. 온보딩 보조: 이전 레시피 고스트 오버레이 `P1`
- 유사한 이전 레시피(F-43 검색)의 측정선을 **연한 고스트**로 겹쳐 표시 → 사용자는 새로 긋지 않고 **끌어다 맞추기만** 하면 된다. 가장 빠른 입력 경로.

### F-14. 레거시 스크리블 PNG 임포트 `P1`
- 기존 자산(SCRIBBLE_DIR의 PNG 수백 장) 호환. HSV 추출 → 컨투어 → fitLine으로 **벡터 스트로크로 역변환** 후 F-06/F-07 파이프라인에 투입해 MIR 생성.
- **구현** 기존 `_extract_color_lines_componentwise` 로직을 `perception/legacy_scribble.py`로 이관·정리(면적 30px·길이 10px 필터 유지).

---

## B. 의도 표현 계층 (Intent / MIR)

### F-15. ★ MIR — Measurement Intent Representation `P0`
- **무엇** 시스템의 단일 진실원(single source of truth). 버전 있는 JSON 스키마.
- **구성** `image_context` / `classes` / `frames`(좌표계) / `patterns` / `rois` / `anchors` / `measurements` / `groups` / `constraints` / `provenance` / `confidence`. (전체 스키마는 Part 4)
- **왜** SDIFF가 하려던 일을 **검증 가능하고, 편집 가능하고, 왕복 가능한** 형태로 정식화.
- **구현** Pydantic v2 모델 → JSON Schema 자동 생성 → 프런트/백엔드 공용 검증. 스키마 버전 필드 + 마이그레이션 함수 체인.

### F-16. Provenance & Confidence 트래킹 `P0`
- MIR의 **모든 필드**에 `{source: snap|cv|vlm|user|default, confidence: 0~1}` 를 병기.
- **왜** UI가 "이건 AI 추측입니다(노란 배지)" / "이건 당신이 직접 지정했습니다(파란 배지)"를 구분해 보여줄 수 있고, 저신뢰 필드만 골라 사용자에게 확인 요청(F-18)할 수 있다.

### F-17. 자연어 되읽기 (Intent Read-back) `P0`
- **무엇** MIR을 사람 문장으로 번역해 확인받는다.
  > *"가장 어두운 클래스(값 10)로 이루어진 핑거 4개 각각에 대해, 핑거 최상단 점에서 시작해 주축 red_1의 법선 방향으로 진행하여 중간계조 클래스(값 30)의 첫 경계까지의 거리를 측정합니다. 총 4개 측정, 이름 Gate_CD, 예상 범위 35~55 nm."*
- **왜** 코드 리뷰보다 훨씬 빠른 검수 수단. 오해를 코드 생성 **이전에** 잡는다.
- **구현** 1차는 **템플릿 기반 결정론적 문장 생성**(모델 불필요, 항상 정확). 2차로 LLM이 자연스럽게 다듬기(선택).

### F-18. ★ 모호성 감지 및 표적 질문 (Clarification) `P0`
- **무엇** 신뢰도가 낮거나 해석이 갈리는 지점만 골라 **객관식 + 시각 미리보기**로 묻는다.
  > *"초록선 3번의 끝점이 애매합니다. 어느 쪽인가요?"*
  > `(A) SiN 층의 상단 경계` `(B) poly 핑거의 하단 경계` `(C) 기준선 red_2 위의 투영점`
  > — 각 선택지에 해당 위치가 캔버스에 하이라이트된다.
- **왜** 기존 시스템은 애매하면 **조용히 추측**했고, 그 결과 잘못된 코드가 나와 사람이 뒤늦게 발견했다.
- **구현** 규칙 기반 감지기 세트:
  - 상위 2개 스냅 후보 점수차 < 0.1
  - 끝점이 어느 클래스 경계에서도 5px 이상 떨어짐
  - 측정선 방향이 어떤 기준선과도 평행/수직이 아님(±10° 밖)
  - 반복 패턴 인스턴스 수와 그린 선 개수가 불일치
  - 실측값이 기대 범위 밖 또는 그룹 내 편차 > 20%
- **정책** 질문은 **최대 3개**까지만. 나머지는 기본값으로 진행하되 provenance에 `default`로 표기.

### F-19. 대화형 의도 수정 (Conversational Edit) `P1`
- *"3번 선은 아래쪽이 아니라 위쪽 경계까지야"* → LLM이 **MIR 패치(JSON Patch)** 를 생성 → 검증 후 적용 → 캔버스와 코드가 동시에 갱신.
- **왜** 코드를 직접 고치게 하면(기존 시스템의 채팅 어시스턴트) 의도와 코드가 어긋난다. 항상 MIR을 고치고 코드는 재생성한다.
- **구현** LLM 출력은 자유 텍스트가 아니라 **JSON Patch(RFC 6902)** 로 제한. 적용 전 스키마 검증 + 사용자 미리보기(diff).

### F-20. MIR ↔ 캔버스 양방향 동기화 `P0`
- MIR을 수정하면 캔버스 선이 다시 그려지고, 캔버스를 수정하면 MIR이 갱신된다. **왕복 손실 없음(round-trip lossless)** 이 인수 조건.

### F-21. 클래스 의미 사전 (Material Dictionary) `P1`
- `{10: "poly", 30: "SiN", 50: "oxide"}` 를 **공정 단계별로 1회 등록**해 재사용. `darkest/brightest` 같은 상대 표현 대신 실제 물질명을 쓰면 VLM·LLM 프롬프트 품질과 사람 가독성이 동시에 올라간다.
- 기존 시스템의 `brightness_map`(darkest/middle_gray_i/brightest)은 사전이 없을 때의 **폴백**으로 유지.

---

## C. 인식(Perception) 계층

### F-22. 마스크 전처리 & 인덱싱 서비스 `P0`
- 클래스별 이진 마스크, 컨투어, 거리변환, 코너맵, 행/열 프로파일을 **1회 계산해 캐시**(이미지 해시 키). 스냅·패턴검출·측정이 모두 이 캐시를 공유.

### F-23. ★ 앵커 프리미티브 라이브러리 `P0`
> 생성 코드가 "무엇이든 할 수 있는" 대신 **"검증된 것만 조합"** 하도록 만드는 기반.

결정론적이고 단위 테스트된 연산자 집합:

| 카테고리 | 연산자 |
|---|---|
| 영역 선택 | `class_mask(c)`, `roi(box)`, `component(i)`, `pattern_instance(p, i)`, `largest_component()` |
| 점 추출 | `extreme_point(dir)`, `centroid()`, `corner()`, `contour_points()`, `intersection(l1,l2)`, `projection(pt, line)` |
| 선 추출 | `fit_line(points)`, `pca_axis()`, `center_line()`, `boundary_between(cA,cB)` |
| 경로 탐색 | `first_crossing(from, dir, target_class)`, `farthest_point(from, within)`, `ray_cast(origin, dir)` |
| 강건화 | `median_of_k(op, k)`, `trimmed_mean`, `ransac_fit`, `subpixel_edge(profile)` |
| 측정 | `distance(a,b)`, `signed_distance(a,b,axis)`, `angle(l1,l2)`, `area(region)`, `radius(arc)` |
| 좌표 | `local_frame(origin, u_axis)`, `to_world(uv, frame)`, `to_local(xy, frame)` |

- **구현** `metro_ops/` 패키지. 모든 연산자는 순수 함수 + 타입 힌트 + docstring + pytest. **numpy 스칼라를 절대 반환하지 않는다**(항상 `float`/`int`) — 기존 시스템의 직렬화 버그를 타입 레벨에서 봉쇄.

### F-24. 서브픽셀 엣지 측정 `P1`
- 측정 방향의 강도 프로파일을 보간해 **50% 임계 교차점**을 서브픽셀로 산출. CD 계측에서 ±0.5px는 유의미한 오차다.
- 마스크(이산 클래스)뿐 아니라 원본 그레이 이미지를 함께 참조하는 옵션 제공.

### F-25. ★ 새 이미지 재정위 (Relocalization) `P0`
- **무엇** 시연 이미지에서 정의한 ROI/앵커를 **새 이미지의 대응 위치로 옮기는** 문제. 기존 시스템의 `grounding_template.png` 매칭을 일반화·강건화한다.
- **폴백 체인**
  1. **구조 기반**(1순위): 반복 패턴/클래스 배치로 직접 인스턴스를 재검출. **좌표 이동에 의존하지 않으므로 가장 강건**.
  2. **템플릿 매칭**: 다중 스케일 NCC + 매칭 점수 임계값.
  3. **특징 정합**: ORB/SIFT + RANSAC → 유사변환(translation+scale+rotation) 추정.
  4. **전역 정렬**: 클래스별 무게중심/주축 정렬(위상 상관).
- 각 단계는 **신뢰도 점수**를 내고, 임계 미달이면 다음 단계로. 전부 실패하면 해당 이미지는 `NEEDS_REVIEW`로 분류(조용한 오측정 금지).
- **왜** 기존 시스템의 가장 잦은 실패 원인이 "ROI가 (0,0) 근처로 몰림"이었다. 이는 단일 수단(템플릿 매칭)에 전량 의존한 결과다.

### F-26. 반복/대칭 구조 검출기 `P0`
- F-11의 백엔드. 자기상관 주기 추정 + NCC 피크 + 연결요소 분할의 앙상블. 결과는 인스턴스 리스트(각각 ROI + 인덱스 + 매칭 점수).

### F-27. 멀티모달 의미 보강 (Gemma4, 선택적 보조) `P1`
- **역할 재정의**: 기존 시스템에서 VLM(Qwen3-VL)은 **필수 경로**였다. 신규 시스템에서 멀티모달 호출은 **보조**다. 사용 모델은 **Gemma4** (`intent_enrich` 역할, Part 5.2.2).
  - 사용 시점 ①: 스냅 정보가 부족한 레거시 임포트(F-14).
  - 사용 시점 ②: F-18 모호성 질문의 **선택지 문구와 후보 생성**.
  - 사용 시점 ③: 반복 구조 검출 결과의 육안 확인(F-11), 측정 항목 자동 명명(F-57), 이상치 1차 분류(F-43).
- **제약 준수** 이미지는 **최대 4장**, 슬롯 배정은 F-64 예산 플래너를 반드시 경유. 대상이 4개를 넘으면 콘택트 시트(F-65)로 압축.
- **구현** 출력은 항상 **닫힌 선택지(enum)** 또는 스키마 고정 JSON. 자유 서술 금지, **좌표 응답 금지**(F-67). 실패 시 CV 기본값으로 폴백하며 파이프라인은 계속 진행(F-71).

### F-28. ~~Open-Vocabulary Detection~~ → 제거 `-`
- 기존 시스템의 **Grounding DINO는 신규 시스템에서 완전히 제거**한다. 사용 가능한 모델이 Gemma4 / GaussO4.1 두 개로 확정되었고, ROI 지정은 (a) 사용자의 직접 드로잉, (b) 반복 패턴 검출(F-26), (c) 재정위(F-25)로 전부 대체되기 때문이다.
- "사람이 말로 영역을 지정"하는 편의는 **Gemma4에게 콘택트 시트의 셀 인덱스를 고르게 하는 방식**으로 대체한다 (박스 좌표를 모델에게 받지 않는다).

### F-29. 이미지 품질/적합성 사전 점검 `P1`
- 배치 실행 전 각 이미지에 대해: 클래스 존재 여부, 대비, 마스크 결손, 스케일 메타 유무, 회전량 추정 → 부적합 이미지를 사전 배제하고 사유를 리포트.

---

## D. 합성(Synthesis) 계층 — 코드 생성

### F-30. ★ 2단 합성: Plan → Code `P0`
> 기존 시스템의 "수백 줄 프롬프트로 자유 파이썬 생성"을 대체하는 핵심 전환.

- **1단계 (Planner, LLM)**: MIR을 입력받아 **연산자 그래프(plan)** 를 JSON으로 출력한다.
  ```json
  {"steps":[
    {"id":"axis","op":"fit_line","args":{"points":{"op":"extreme_points","args":{"class":"poly","instances":"finger[*]","dir":"top"}}}},
    {"id":"frame","op":"local_frame","args":{"origin":"axis.p0","u_axis":"axis.dir"}},
    {"id":"m1","op":"for_each","args":{"over":"finger[*]","body":[
        {"id":"a","op":"extreme_point","args":{"class":"poly","roi":"$item","dir":"top","robust":"median_of_k:5"}},
        {"id":"b","op":"first_crossing","args":{"from":"a","dir":"frame.normal","target_class":"SiN"}},
        {"id":"d","op":"distance","args":{"a":"a","b":"b"},"emit":{"measure_item":"Gate_CD","group":"$index"}}
     ]}}
  ]}
  ```
- **2단계 (Renderer, 결정론적)**: plan을 검증(연산자 존재·인자 타입·참조 무결성)한 뒤 **Jinja 템플릿으로 단독 실행 가능한 `measure.py` 렌더링**. 필요한 유틸 함수는 라이브러리에서 **그대로 인라인 삽입**(단일 파일 요건 충족).
- **왜 결정적으로 나은가**
  - `NameError` 구조적으로 불가능 (미정의 연산자는 검증 단계에서 거부).
  - 좌표계 버그(`(y,x)` vs `(x,y)`), numpy 직렬화 버그를 **라이브러리가 한 번만 올바르게** 해결한다. 프롬프트로 매번 설득할 필요가 없다.
  - plan은 짧다 → 토큰·지연·비용 대폭 절감, 그리고 **모델 간 이식성**이 생긴다.
  - plan은 사람이 읽고 고칠 수 있고, diff가 의미 있다.
- **탈출구(Escape hatch)** 기존 연산자로 표현 불가능한 요구는 `custom_step`으로 자유 코드 생성을 허용하되, 반드시 F-33 정적 검증 + F-36 재현 검증을 통과해야 하고, 승인 시 F-41 라이브러리 승격 후보가 된다.

### F-31. 결정론적 코드 렌더러 `P0`
- 산출 스크립트의 고정 골격: `argparse(--mask_path --out_dir --meta_root)` → `meta_utils`로 스케일/클래스 로드 → 재정위(F-25) → 측정 루프 → `measurements.csv`(표준 헤더) + `overlay.png` + `roi_debug.png` + `run.json`.
- 코드 스타일 고정(black), 난수 시드 고정, 모든 수치 `float()/int()` 캐스팅, 예외 시 종료 코드/사유 표준화.

### F-32. Few-shot / 유사 레시피 검색 `P1`
- **구조 유사도 기반 검색**: MIR을 특징 벡터로 임베딩(측정 타입 분포, 클래스 조합, 반복 구조, 좌표계 유형) + 이미지 임베딩(pHash + 경량 CNN/CLIP) 결합.
- 기존 pHash+파일명 토큰 Jaccard 방식보다 정확. **실패 사례도 함께 검색**해 "이렇게 하면 실패한다"는 네거티브 예시로 제공.
- MMR로 다양성 확보(기존 로직 계승).

### F-33. ★ 정적 검증 게이트 `P0`
실행 **이전에** AST로 검사하고, 위반 시 실행하지 않는다.

| 검사 | 내용 |
|---|---|
| 미정의 참조 | 호출되는 모든 이름이 정의/임포트되었는가 |
| 임포트 허용목록 | `cv2, numpy, pandas, meta_utils, math, json, os, argparse, csv` 등만 |
| **부정행위 차단** | merge/scribble 이미지 읽기, 색상 `inRange`, 초록/빨강 BGR 비교 (기존 `_reject_merge_copying` 규칙 계승·확장) |
| **좌표 하드코딩 차단** | 시연 이미지의 좌표값과 일치하는 매직 넘버 탐지 (일반화 실패의 주범) |
| I/O 계약 | `--mask_path/--out_dir/--meta_root` 존재, CSV 표준 헤더, overlay 저장 |
| 타입 안전 | numpy 스칼라가 CSV/JSON 경로에 도달하지 않는가 |
| 리소스 | 무한 루프 위험, 전체 이미지 반복문 남용 |

### F-34. 샌드박스 실행 `P0`
- 별도 프로세스 + 리소스 제한(CPU 시간·메모리·wall clock) + **네트워크 차단** + 임시 작업 디렉터리 + 읽기 전용 입력 마운트.
- 산출물만 회수. 컨테이너(Docker) 또는 최소 `resource.setrlimit` + 격리 사용자.

### F-35. ★ 구조화된 자기 수정 루프 `P0`
- 실패를 **분류**한 뒤 분류별 전용 처방을 적용한다. 기존 시스템처럼 stderr 전문을 통째로 LLM에 던지지 않는다.

| 오류 유형 | 처방 |
|---|---|
| Plan 검증 실패 | 위반 연산자/인자만 지목해 plan 재생성 (코드 재생성 아님) |
| 런타임 예외 | 실패 step id + 입력 요약 + 스택 3줄만 제공 → 해당 step만 수정 |
| 측정 결과 0건 | ROI 확대 / 임계 완화 / 재정위 폴백 단계 하향 → **결정론적 자동 재시도**(LLM 불필요) |
| 결과가 시연과 불일치 | F-36의 기하 diff를 피드백으로 제공 |
| 전이 실패 | 실패한 augmentation 유형(회전/스케일)을 알려주고 강건 연산자 사용 유도 |
| 스키마/타입 오류 | 렌더러 버그로 간주 → 사람에게 에스컬레이션 (LLM 재시도 금지) |

- 재시도 예산: 유형별 최대 2회, 전체 최대 5회. 초과 시 **부분 성공 상태로 사람에게 인계**(무한 루프 금지).

### F-36. ★★ 시연 재현 검증 (Demo Replay Verification) `P0`
> **기존 시스템에 없던, 가장 가치가 큰 신규 게이트.**

- **무엇** 생성된 코드를 **사용자가 그림을 그린 바로 그 이미지**에 돌린 뒤, 코드가 그린 측정선과 **사용자가 그린 초록선을 기하학적으로 비교**한다.
- **지표**
  - 측정선 개수 일치 여부
  - 끝점 거리 (평균/최대, px)
  - 길이 상대 오차 (%)
  - 각도 차이 (°)
  - 그룹/이름 매칭 정확도
- **판정** 기본 임계: 끝점 평균 ≤ 3px, 길이 오차 ≤ 3%, 개수 100% 일치. 미달 시 F-35로 diff를 피드백해 자동 재시도.
- **왜** *"실행이 성공했다"* 와 *"내가 의도한 것을 쟀다"* 는 완전히 다른 문제다. 후자를 자동 판정할 수 있게 되면 사람의 검수 부담이 급감한다.
- **UI** 사용자 선(초록)과 코드 선(파랑)을 겹쳐 보여주고 오차를 숫자로 표기.

### F-37. ★ 전이 신뢰도 게이트 (Transfer Reliability) `P0`
- 기존 `validation/transfer_reliability_eval.py`를 **파이프라인 내부 서비스로 승격**.
- **결정론적 augmentation**: shift(±200px), rotate(±1~2°), scale(0.9~1.03), shear(±0.02) + **랜덤 조합 N종**(기본 30, 시드 고정) + 노이즈/밝기 변화 옵션.
- **홀드아웃 실이미지**: 같은 공정의 다른 이미지 5~20장에도 실행.
- **산출 지표**
  - 실행 성공률
  - 측정 산출률 (기대 개수 대비)
  - **값 안정성**: 동일 대상에 대한 변동계수 CV% (기하 변환은 값을 바꾸지 않아야 함)
  - 실패 단계 분류 (재정위 실패 / ROI 무효 / 측정 0건 / 예외)
- **판정** 성공률 ≥ 90% & CV ≤ 2% → 초록. 미달 시 리포트와 함께 재생성 또는 사람 개입.

### F-38. 민감도 분석 & 강건성 어드바이저 `P1`
- **무엇** 앵커 정의를 미세 변형(예: 최상단 1픽셀 → 상위 3/5/9행 중앙값, 임계 ±1)했을 때 측정값이 얼마나 흔들리는지 측정하고, **더 안정적인 정의를 추천**한다.
  > *"현재 정의(단일 최상단 픽셀)는 변동 3.1nm입니다. 상위 5행 중앙값으로 바꾸면 0.8nm로 줄어듭니다. 적용할까요?"*
- **왜** 계측 시스템의 신뢰도는 평균이 아니라 **분산**이 결정한다. 이 기능은 사람 전문가가 하던 판단을 자동화한다.

### F-39. Best-of-N 생성 및 자동 채점 `P1`
- plan을 N개(기본 3) 생성 → 각각 렌더·실행·재현검증·전이검증 → **점수 최고안 자동 채택**.
- 점수: `0.5·재현정확도 + 0.3·전이성공률 + 0.1·(1-CV) + 0.1·단순성(step 수)`.
- 병렬 실행이므로 지연은 거의 늘지 않는다.

### F-40. 레시피 카드 자동 문서화 `P1`
- 생성 코드 옆에 사람이 읽는 문서를 자동 생성: 무엇을 재는가 / 어디서 어디까지 / 전제 조건 / 알려진 실패 모드 / 검증 결과 요약 / 재현 정보(모델·프롬프트·코드 해시).
- 공정 이관·감사 대응에 그대로 사용.

### F-41. 유틸 라이브러리 승격 파이프라인 `P1`
- 기존 `potential_utils.py` ↔ `measurement_utils.py` 2단 구조 계승 + 강화:
  - `custom_step`에서 나온 함수를 AST로 추출 → **정규화(이름·시그니처)** → 기존 함수와 **AST 유사도 비교로 중복 제거**.
  - 승격 조건: 서로 다른 3개 이상 레시피에서 사용 + 단위 테스트 통과 + 리뷰 승인.
  - 승격 시 F-23 연산자 레지스트리에 등록되어 **Planner가 즉시 사용 가능**해진다. → 시스템이 자기 능력을 확장하는 루프.

---

## E. 실행 · 운영 계층

### F-42. 배치 러너 & 진행 대시보드 `P0`
- 폴더/필터/리스트로 대상 지정 → 병렬 실행(프로세스 풀) → 실시간 진행률, 성공/실패/검토필요 카운트.
- 결과: 통합 CSV/Parquet, 이미지별 overlay 썸네일 그리드, 실패 사유별 그룹핑.

### F-43. 결과 QC & 이상치 탐지 `P0`
- 규칙: 기대 범위 이탈 / 측정 개수 부족 / ROI 경계 클리핑 / 재정위 점수 저조 / 그룹 내 이상치(로버스트 z-score) / 인접 이미지 대비 급변.
- 위반 건은 `NEEDS_REVIEW` 큐로. **조용한 통과를 절대 허용하지 않는다.**

### F-44. 통계 리포팅 `P1`
- 항목별 히스토그램·박스플롯, 평균/σ/공정능력지수 유사 지표, 시간/로트별 추세(메타데이터가 있을 때).

### F-45. ★ 리뷰 피드백 루프 `P1`
- 리뷰어가 overlay에서 **잘못된 측정선의 끝점을 드래그로 수정** → 그 수정이
  1. 해당 이미지의 정답(ground truth)으로 기록되고,
  2. MIR의 제약으로 환류되며(예: "이 경우 상단이 아니라 하단"),
  3. 재현/전이 평가 벤치마크에 케이스로 추가된다.
- **왜** 운영 중 축적되는 사람의 수정이 시스템 개선으로 자동 환류되는 유일한 통로.

### F-46. 아티팩트 & 재현성 매니페스트 `P0`
- 실행마다 `run.json`: 입력 이미지 해시, MIR 해시, plan 해시, 코드 해시, 모델 ID/버전, 프롬프트 버전, 라이브러리 버전, 난수 시드, 소요 시간, 검증 결과.
- **동일 매니페스트 → 동일 결과**가 보장되어야 한다(결정론성 인수 조건).

### F-47. 레시피 레지스트리 `P1`
- 레시피 = (MIR + plan + 코드 + 검증 리포트 + 소유자 + 공정 단계 + 태그 + 버전).
- 버전 diff, 롤백, 승인 상태(draft/verified/production), 사용처 추적.

---

## F. 플랫폼 · 인프라

### F-48. 영속 저장소 `P0`
- SQLite(단일 노드) → Postgres(확장) 전환 가능한 SQLAlchemy 레이어. 테이블: `images, recipes, mir_versions, plans, runs, measurements, reviews, fewshots, ops_registry, jobs`.
- 이미지/산출물은 파일시스템 또는 S3 호환 스토리지, DB에는 경로+해시만.

### F-49. 작업 큐 & 실시간 스트리밍 `P0`
- 장시간 작업(VLM 추론, 배치 실행, 전이 검증)은 큐 워커로. 진행 상황은 **SSE/WebSocket 푸시**(기존 폴링 방식 대체).
- 작업 취소/재시도/우선순위 지원.

### F-50. 설정 · 시크릿 관리 `P0`
- 모든 경로·엔드포인트·모델 ID를 `settings.yaml` + 환경변수로 주입. **코드 내 절대경로·API 키 리터럴 금지**(기존 코드의 `os.environ['OPENAI_API_KEY']='api_key'` 제거).
- 프로필: `local-gpu`, `local-cpu`, `enterprise-gateway`.

### F-51. 모델 라우터 `P0`
- **Gemma4 / GaussO4.1** 2개 백엔드와 역할(role) 레지스트리를 설정으로 배선. 장애 시 폴백 체인(F-71), 토큰·비용·지연 계측(F-69), 응답 캐시(F-68).
- 상세 설계는 **Part 5**. 구현 항목은 F-59 ~ F-72.

### F-52. 패키징 & 배포 `P1`
- Docker 이미지 2종(app / sandbox-runner), docker-compose, GPU·CPU 프로파일, 헬스체크, 마이그레이션 스크립트.

### F-53. 관측성 `P1`
- 구조화 로깅, 단계별 지연/성공률 메트릭, 모델 호출 비용 집계, 실패 케이스 자동 아카이빙(재현 번들).

### F-54. 권한 · 감사 `P2`
- 역할(작성자/리뷰어/관리자), 레시피 승인 워크플로, 변경 감사 로그.

---

## G. 지능형 어시스트 (차별화 기능 모음)

### F-55. 드로잉 오류 실시간 진단 `P1`
그리는 도중/직후에 인라인 경고 + 원클릭 수정 제안:

| 감지 | 메시지 | 제안 |
|---|---|---|
| 측정선이 어떤 구조에도 닿지 않음 | *"이 선의 양 끝이 모두 배경입니다"* | 가까운 경계로 스냅 |
| 두 선이 거의 중복 | *"측정 2와 5가 96% 겹칩니다"* | 하나 삭제 |
| 기준선이 너무 짧아 축 피팅 불안정 | *"길이 42px로는 각도 오차가 ±3°입니다"* | 연장 또는 다중점 피팅 |
| 반복 개수 불일치 | *"구조는 6회 반복인데 측정선은 4개입니다"* | 나머지 2개 자동 추가 |
| 실측값이 비상식적 | *"이 값은 4.2 µm입니다. 스케일 메타가 맞나요?"* | meta 확인 |
| 그룹 내 편차 과다 | *"같은 그룹인데 값이 2배 차이납니다"* | 그룹 재지정 / 앵커 재확인 |

### F-56. 시연 자동 완성 (Demonstration Autocomplete) `P1`
- 6개 중 2개만 그리면 나머지 4개를 **예측해서 제안**(F-11의 일반화). 사용자는 확인만 한다.
- 더 나아가: 유사 레시피 이력이 있으면 **선을 하나도 긋기 전에** 초안을 제시(F-13과 결합).

### F-57. 측정 항목 자동 명명 `P2`
- 클래스 사전 + 방향 + 구조 유형으로 이름을 제안(`poly_top_to_SiN_vertical` → 사용자가 `Gate_CD`로 확정). 명명 규칙은 프로젝트별 설정 가능.

### F-58. 키보드 우선 워크플로 & 매크로 `P2`
- 모든 주요 동작에 단축키. 자주 쓰는 시퀀스(예: "수직 측정선 + 상단 스냅 + 반복 전파")를 매크로로 저장.

---

## H. 모델 API 통합 (F-59 ~ F-72)

Gemma4 / GaussO4.1 전용 전송 계층, reasoning 정책, 구조화 JSON 복구, 그리고 **이미지 4장 · 장당 200MB 제약**을 다루는 기능군이다. 제약이 설계에 직접 영향을 주므로 별도 장으로 분리했다 — **[Part 5](#part-5--모델-api-통합-설계-gemma4--gausso41)** 참조.

| # | 기능 | 우선순위 |
|---|---|---|
| F-59 | 모델 백엔드 · 역할 레지스트리 | P0 |
| F-60 | 게이트웨이 전송 계층 (헤더 · 풀 · 타임아웃 · 429) | P0 |
| F-61 | Reasoning 제어 정책 (Gemma4 omit / GaussO4.1 effort) | P0 |
| F-62 | 구조화 JSON 3단 복구 + 스키마 검증 + 부분 재질의 | P0 |
| F-63 | **이미지 페이로드 파이프라인 (200MB 제약)** | P0 |
| F-64 | **이미지 슬롯 예산 플래너 (4장 제약)** | P0 |
| F-65 | **콘택트 시트 합성** (4장 → 4장×N셀 확장) | P0 |
| F-66 | 다중 호출 맵리듀스 | P1 |
| F-67 | 이미지 변환 메타 & 좌표 역변환 | P0 |
| F-68 | 모델 응답 캐시 | P1 |
| F-69 | 모델 호출 텔레메트리 · 예산 가드 | P1 |
| F-70 | 프롬프트 · 모델 회귀 스위트 | P1 |
| F-71 | 폴백 및 열화 정책 (모델 없이도 동작) | P0 |
| F-72 | 페이로드 보안 · 민감정보 처리 | P2 |

---

# Part 4. 핵심 데이터 모델

## 4.1 MIR (Measurement Intent Representation) — 요약 스키마

```jsonc
{
  "schema": "mir/1.0",
  "recipe_id": "rcp_7f3a...",
  "created_by": "user@corp",

  "image_context": {
    "reference_image": "C2024_x_gray.tif",
    "size": [2048, 2048],
    "pixel_scale_nm": { "x": 1.953, "y": 1.953 },
    "meta_source": "meta/C2024_x.json"
  },

  "classes": [
    { "value": 10, "rank": "darkest",  "name": "poly" },
    { "value": 30, "rank": "middle_1", "name": "SiN" },
    { "value": 50, "rank": "brightest","name": "oxide" }
  ],

  "patterns": [
    { "id": "finger", "type": "periodic_1d", "axis": "u", "period_px": 187.4,
      "count": 4, "detector": "autocorr+ncc", "confidence": 0.94,
      "instances": [ {"index":0,"roi":[120,300,160,420]}, /* ... */ ] }
  ],

  "frames": [
    { "id": "world", "type": "absolute" },
    { "id": "lcs_1", "type": "local",
      "origin": { "ref": "anchor:axis_origin" },
      "u_axis": { "ref": "anchor:primary_axis" },
      "handedness": "image" }
  ],

  "anchors": [
    { "id": "primary_axis", "kind": "line",
      "op": "fit_line",
      "args": { "points": { "op": "extreme_points",
                            "args": { "class": "poly", "over": "pattern:finger[*]", "dir": "top" } } },
      "robust": { "method": "ransac", "residual_px": 2.0 },
      "provenance": { "source": "snap+cv", "confidence": 0.91 } },

    { "id": "m1_start", "kind": "point",
      "op": "extreme_point",
      "args": { "class": "poly", "roi": "pattern:finger[$i]", "dir": "top" },
      "robust": { "method": "median_of_k", "k": 5 },
      "provenance": { "source": "snap", "confidence": 0.97 } },

    { "id": "m1_end", "kind": "point",
      "op": "first_crossing",
      "args": { "from": "anchor:m1_start", "direction": "frame:lcs_1.normal",
                "target_class": "SiN", "max_distance_px": 300 },
      "provenance": { "source": "snap", "confidence": 0.88 } }
  ],

  "measurements": [
    { "id": "m1", "name": "Gate_CD", "type": "distance",
      "from": "anchor:m1_start", "to": "anchor:m1_end",
      "iterate_over": "pattern:finger[*]",
      "group_by": "instance_index",
      "unit": "nm",
      "expect": { "min": 35, "max": 55 },
      "provenance": { "source": "user", "confidence": 1.0 } }
  ],

  "clarifications": [
    { "field": "anchors.m1_end.args.target_class",
      "question": "끝점이 닿는 층은 어디입니까?",
      "options": ["SiN 상단 경계", "poly 하단 경계", "red_2 투영점"],
      "answered": "SiN 상단 경계", "answered_by": "user" }
  ],

  "demo_strokes": [
    { "id": "s3", "layer": "measure", "measurement": "m1", "instance": 0,
      "raw_points": [[512,300],[513,340],[514,381]],
      "fit": { "primitive": "segment", "p0": [512.0,300.0], "p1": [514.0,381.0], "rms_px": 0.7 },
      "snaps": [
        { "end": "start", "target": {"kind":"class_extreme","class":"poly","dir":"top","roi":"finger[0]"},
          "dist_px": 1.8, "score": 0.95 },
        { "end": "end",   "target": {"kind":"class_boundary","class":"SiN"},
          "dist_px": 2.4, "score": 0.89 }
      ] }
  ]
}
```

**핵심 포인트**
- `demo_strokes`가 **원본 시연을 보존**한다 → F-36 재현 검증의 정답지이자, 캔버스 왕복의 근거.
- `anchors`는 **좌표가 아니라 연산자 트리**다 → 다른 이미지에 그대로 적용 가능.
- 모든 노드에 `provenance`가 붙는다.

## 4.2 측정 결과 CSV (기존 스키마 계승 + 확장)

```
measure_item, group_id, index, value_nm,
sx, sy, ex, ey,
meta_tag, component_label, image_name, run_id, note
```
- **하위 호환 유지**(기존 자산·다운스트림 도구 보호).
- 확장 컬럼(선택): `relocalize_score`, `anchor_confidence`, `qc_flag`, `mir_version`.

## 4.3 실행 매니페스트 `run.json`

```jsonc
{ "run_id": "...", "recipe_id": "...", "mir_hash": "...", "plan_hash": "...", "code_hash": "...",
  "image": {"path": "...", "sha256": "..."},
  "models": {
    "planner":       {"backend": "gausso4_1", "model": "GaussO4.1-...", "reasoning": "medium", "thinking_budget": 1024},
    "intent_enrich": {"backend": "gemma4",    "model": "Gemma4-...",    "reasoning": null}
  },
  "llm_usage": {"calls": 3, "image_calls": 1, "images_sent": 3,
                "image_bytes_total": 4128332, "prompt_tokens": 18244, "completion_tokens": 3120,
                "retries": 0, "structured_json": {"primary": 3, "fallback": 0, "repair": 0}},
  "prompt_version": "p/2.3", "ops_version": "1.4.0", "seed": 1234,
  "verification": {"demo_replay": {"endpoint_px": 1.4, "length_err_pct": 0.9, "pass": true},
                   "transfer": {"n": 50, "success_rate": 0.98, "cv_pct": 0.7, "pass": true}},
  "timing_ms": {"plan": 4210, "render": 38, "exec": 2650, "verify": 18400} }
```

---

# Part 5. ★ 모델 API 통합 설계 (Gemma4 / GaussO4.1)

> **전제 (확정 제약)**
> 1. 사용 가능한 모델은 **Gemma4** 와 **GaussO4.1** **둘 뿐**이다. 기존 저장소가 쓰던 Qwen3-VL(로컬 HF), Grounding DINO, GPT-OSS, Gemma3, Llama-4는 **모두 제거**한다.
> 2. 두 모델 모두 **멀티모달(이미지 입력) 지원**.
> 3. **1회 호출당 이미지 최대 4장.**
> 4. **이미지 1장당 200MB 초과 시 호출 불가.**
>
> 호출 방식은 `sy5255/report-search` 저장소의 `app/model_client.py` · `app/config.py` · `app/llm_roles.py` 패턴을 **그대로 계승**한다. 이미 사내 게이트웨이의 특성(헤더 규약, reasoning 파라미터, response_format 미지원, 429 처리, 커넥션 재사용)이 검증되어 있으므로 재발명하지 않는다.

---

## 5.1 참조 구현에서 확정된 사실 (report-search 분석 결과)

`report-search`는 텍스트 전용 RAG이지만, **모델 전송 계층은 그대로 재사용 가능**하다. 분석에서 확인된 계약은 다음과 같다.

### 5.1.1 전송 방식 — OpenAI 호환 SDK

```python
from openai import OpenAI, DefaultHttpxClient, Timeout

client = OpenAI(
    base_url=profile.base_url,     # 예: http://api/gausso4-1/v1
    api_key=profile.api_key,       # 사내 게이트웨이는 "EMPTY" 사용
    default_headers=_connection_headers(user_id, profile.ticket),
    timeout=Timeout(connect=10, write=30, pool=10, read=120),
    http_client=DefaultHttpxClient(timeout=..., limits=..., event_hooks=...),
)
client.chat.completions.create(model=..., messages=[...], ...)
```

### 5.1.2 게이트웨이 헤더 규약

| 헤더 | 성격 | 값 |
|---|---|---|
| `Send-System-Name` | 커넥션 고정 | 시스템 식별자 (예: `ScribbleMetro`) |
| `User-Id` | 커넥션 고정 | 호출 사용자 |
| `User-Type` | 커넥션 고정 | 사용자 유형 |
| `x-dep-ticket` | 커넥션 고정 | 부서 티켓 |
| `Prompt-Msg-Id` | **호출마다 새로 발급** | `uuid4()` |
| `Completion-Msg-Id` | **호출마다 새로 발급** | `uuid4()` |

> **중요** 커넥션 풀을 캐시하면 트레이스 ID 2종이 프로세스 수명 내내 고정되는 버그가 생긴다. `report-search`는 이를 `_per_request_headers()`를 `extra_headers`로 매 호출 주입해 해결했다. **동일하게 구현한다.**

### 5.1.3 프로바이더 정책 (Provider Policy) — 두 모델의 차이

| 항목 | **Gemma4** | **GaussO4.1** |
|---|---|---|
| provider key | `gemma4` | `gausso4_1` |
| 모델명 판별 | `gemma4 / gemma-4 / gemma_4` 포함 | `gausso4.1 / gausso4-1 / gausso4_1` 포함 |
| `reasoning_control` | **omit** (reasoning 파라미터를 **보내면 안 됨**) | **explicit** |
| `reasoning_style` | none | **effort** |
| `supports_response_format` | **False** | **False** |
| `token_param` | `max_tokens` | `max_tokens` |
| `send_temperature` | True | True |
| `include_reasoning` | — | False (추론 텍스트를 되돌려받지 않음) |

**GaussO4.1 reasoning 전송 규약** (`extra_body`에 실어 보낸다)

```python
# mode = none  → 이것만 보낸다. thinking budget을 함께 보내면 안 된다.
extra_body["reasoning_effort"] = "none"

# mode = minimal | medium | high
extra_body["reasoning_effort"]      = {"minimal": "low", "medium": "medium", "high": "high"}[mode]
extra_body["thinking_token_budget"] = budget      # 기본 256 / 1024 / 2048
extra_body["include_reasoning"]     = False
```

> **주의 (계승할 설계 판단)** `"none"` 은 "적게 생각하기"가 아니라 **추론 패스 자체를 끄는 것**이다. 이때 thinking budget을 함께 보내면 껐다는 의미가 무너진다. Gemma4에는 이 세 파라미터를 **한 개도 보내지 않는다**.

### 5.1.4 `response_format` 미지원 → 구조화 JSON 3단 복구

두 프로바이더 모두 `supports_response_format=False`다. 즉 **JSON Schema 강제(strict)가 불가능**하다. `report-search`의 `create_json_completion()`이 쓰는 3단 복구를 그대로 채택한다.

```
1차 (primary)   : 역할의 reasoning 모드로 호출 → 본문에서 { ... } 추출 시도
   ↓ 실패(파싱 오류 또는 빈 본문)
2차 (fallback)  : reasoning="minimal", response_format 제거,
                  finish_reason=="length" 였다면 max_tokens 상향 후 재호출
   ↓ 실패
3차 (repair)    : json_repair 역할(경량 모델)에게 "깨진 JSON만 고쳐라" 지시
   ↓ 실패 → StructuredJSONError 발생 (조용한 실패 없음)
```

- JSON 추출은 `text.find("{") ~ text.rfind("}")` 구간을 `json.loads(strict=False)` 로 파싱.
- 각 단계는 `structured_json` 텔레메트리 이벤트로 성공/실패가 기록된다.

### 5.1.5 응답 본문 추출 — reasoning 콘텐츠 분리

응답의 `message.content` 는 문자열일 수도, **content parts 배열**일 수도 있다. 배열인 경우 `type` 에 `reason` 또는 `think` 가 포함된 파트는 **본문에서 제외**해야 한다. 본문이 비면 `model_extra` 의 `output_text / final_text / final / text` 순으로 폴백한다. (`extract_message_text()` 그대로 이식)

### 5.1.6 재시도 · 타임아웃 · 커넥션 재사용

- **429만 재시도**한다. `Retry-After` 헤더가 있으면 그 값, 없으면 `min(30, 2**attempt)` 초 대기.
- 그 외 예외는 즉시 전파(조용한 삼킴 금지). 모든 실패는 `request` 이벤트로 기록.
- 타임아웃은 4분할: `connect=10s / write=30s / pool=10s / read=120s`. **모델이 생각하는 구간(read)만 길게**.
- 커넥션 풀은 `(base_url, api_key, ticket, user_id)` 키로 캐시. 한 턴에 십수 회 호출하면서 매번 TLS 핸드셰이크를 하는 낭비를 없앤다.
- `keepalive_expiry`(기본 5초)를 짧게 잡아 게이트웨이가 먼저 끊은 유휴 커넥션에 써 넣고 read 타임아웃 전체를 날리는 stall을 방지한다.

### 5.1.7 백엔드 / 역할 레지스트리

`report-search`는 **backend(엔드포인트+모델) ↔ role(용도별 토큰·추론 설정)** 을 분리했다. 이 2단 구조를 그대로 쓴다.

```python
LLM_BACKENDS = {"gemma4": BackendSpec(...), "gausso4_1": BackendSpec(...)}
LLM_ROLES    = {"planner": RoleSpec(backend="gausso4_1", max_tokens=..., reasoning_mode="medium"), ...}
```

- 역할별 백엔드는 `<ROLE>_BACKEND` 환경변수로 재정의 가능 → **코드 변경 없이 모델을 바꿔 A/B**할 수 있다.
- 기동 시 `validate_llm_role_registry()` 로 모든 역할이 실재하는 백엔드를 가리키는지 검증하고, 미설정이면 **즉시 실패**한다(런타임에 조용히 기본값으로 흐르지 않는다).

---

## 5.2 ScribbleMetro의 모델 역할 배치

### 5.2.1 역할 분담 원칙

| | **Gemma4** | **GaussO4.1** |
|---|---|---|
| 성격 | 멀티모달 인식 · 빠른 판정 · 경량 구조화 | 추론(reasoning) · 계획 수립 · 코드 논리 |
| 주 사용처 | **이미지를 보는 모든 작업** | **plan 합성과 수정** |
| 추론 예산 | 없음(omit) | none/minimal/medium/high 조절 |

> 이 배치는 `report-search`의 판단(런타임 결정은 Gemma4에 집중, 최종 산출은 GaussO4.1)과 동일한 철학이다.

### 5.2.2 역할 레지스트리 (초기값)

| role | backend | reasoning | max_tokens | 이미지 | 설명 |
|---|---|---|---|---|---|
| `intent_enrich` | gemma4 | — | 2,000 | ≤4 | 스냅 정보가 부족할 때 의미 보강 (F-27) |
| `clarify_options` | gemma4 | — | 1,500 | ≤4 | 모호성 선택지 생성 (F-18) |
| `pattern_verify` | gemma4 | — | 1,000 | ≤4 | 반복 구조 검출 결과 육안 확인 (F-11) |
| `legacy_import` | gemma4 | — | 3,000 | ≤4 | 레거시 스크리블 PNG 해석 (F-14) |
| `naming` | gemma4 | — | 500 | 0~1 | 측정 항목 명명 제안 (F-57) |
| `qc_triage` | gemma4 | — | 1,500 | ≤4 | 이상치 overlay 1차 분류 (F-43) |
| `planner` | **gausso4_1** | medium | 8,000 | 0~2 | MIR → plan 합성 (F-30) |
| `plan_repair` | **gausso4_1** | medium | 8,000 | 0~2 | 검증 실패 step 수정 (F-35) |
| `custom_step` | **gausso4_1** | high | 6,000 | 0 | 연산자로 표현 불가한 로직의 자유 코드 (F-30 탈출구) |
| `judge` | **gausso4_1** | minimal | 2,000 | ≤4 | Best-of-N 채점 보조 (F-39) |
| `json_repair` | gemma4 | — | 3,000 | 0 | 깨진 JSON 복구 (5.1.4 3단계) |
| `readback_polish` | gemma4 | — | 1,000 | 0 | 되읽기 문장 다듬기 (F-17, 선택) |

**의도된 비대칭**: 이미지를 쓰는 역할은 거의 전부 Gemma4다. GaussO4.1은 **MIR(텍스트)만 보고** plan을 짠다. 이렇게 하면
- 4장 제약이 걸리는 지점이 인식 단계에 국한되고,
- plan 합성은 이미지 없이 **결정론적이고 캐시 가능**해지며,
- 가장 비싼 추론 호출에 대용량 페이로드가 실리지 않는다.

---

## 5.3 신규 기능 (F-59 ~ F-72)

### F-59. 모델 백엔드 · 역할 레지스트리 `P0`
- **무엇** `BackendSpec`(엔드포인트/모델/키/티켓/api_profile) + `RoleSpec`(백엔드/토큰/추론 모드/추론 예산) 2단 레지스트리와 `resolve_role()`.
- **구현** `platform/llm/backends.py`, `platform/llm/roles.py`. `report-search/app/llm_roles.py` 구조를 그대로 이식하되 역할 목록을 5.2.2로 교체.
- 기동 시 전체 역할 검증. `sanitized_llm_routing()`(키·티켓 마스킹된 라우팅 덤프)을 `/api/system/llm-routing` 으로 노출해 운영자가 현재 배선을 확인할 수 있게 한다.

### F-60. 게이트웨이 전송 계층 `P0`
- **무엇** 헤더 규약(5.1.2), 커넥션 풀 캐시, 4분할 타임아웃, 429 전용 재시도, HTTP 이벤트 훅 기반 느린 요청 로깅.
- **구현** `platform/llm/transport.py` — `report-search/app/model_client.py` 의 `_transport / _connection_headers / _per_request_headers / request_timeout / _connection_limits / _is_rate_limit_error / _retry_after_seconds` 이식.
- **인수 조건** 20회 연속 호출 시 TCP 핸드셰이크가 1회만 발생(커넥션 재사용 테스트).

### F-61. Reasoning 제어 정책 `P0`
- **무엇** 프로바이더별 정책 테이블로 reasoning 파라미터 전송 여부/형태를 결정. Gemma4에는 절대 보내지 않고, GaussO4.1에는 5.1.3 규약대로 보낸다.
- **왜** 이 규약을 어기면 게이트웨이가 요청을 거부하거나(파라미터 미지원) 의도와 다르게 추론이 켜진다.
- **인수 조건** 프로바이더별 페이로드 스냅샷 테스트(요청 body를 캡처해 키 집합 검증).

### F-62. 구조화 JSON 3단 복구 + 스키마 검증 `P0`
- **무엇** 5.1.4의 3단 복구 위에 **Pydantic 검증**을 얹는다.
- **추가 4단계(신규)**: 파싱은 됐지만 **스키마 위반**인 경우 → 전체 재생성이 아니라 **위반 필드만 지목해 부분 재질의**한다.
  ```
  "다음 필드만 다시 답하라. steps[2].args.target_class 는
   ['poly','SiN','oxide'] 중 하나여야 하는데 'dark_layer' 가 왔다."
  ```
- **왜** plan/MIR은 스키마가 곧 실행 안전성이다. `response_format` 강제가 불가능한 환경에서 **검증을 우리 쪽에서 두껍게** 가져가야 한다.
- **인수 조건** 고의로 깨뜨린 응답 10종에 대해 복구 성공률 ≥ 90%, 실패 시 예외가 반드시 전파(무음 실패 0건).

### F-63. ★ 이미지 페이로드 파이프라인 (200MB 제약) `P0`

- **무엇** 어떤 이미지든 **호출 가능한 형태로 안전하게 변환**하고, 변환 정보를 좌표 역변환용으로 보존한다.
- **처리 순서**
  1. **디코드** — TIFF/PNG/JPEG. 멀티페이지 TIFF는 대상 페이지만.
  2. **채널 축소** — 마스크는 8bit 단일 채널로. 16bit는 윈도잉 후 8bit.
  3. **가시화 렌더** — 클래스 마스크는 그대로 보내면 사람도 모델도 못 읽는다. **팔레트 컬러맵 + 스크리블 오버레이**로 렌더링.
  4. **리사이즈** — 긴 변 기준 상한(기본 1,536px). **단, 판단용 크롭은 원해상도 유지**(5.4 슬롯 정책).
  5. **인코딩** — PNG(무손실, 마스크·라인 아트에 유리) 우선, 사진성 원본은 JPEG q85.
  6. **base64 인코딩**.
  7. **크기 검사 — 반드시 base64 인코딩 *후* 바이트 수로 검사한다.** base64는 원본 대비 약 **4/3배(≈33% 팽창)** 되므로, 파일 크기로 검사하면 통과했는데 전송에서 초과하는 사고가 난다.
  8. **초과 시 자동 열화 루프** — 긴 변 상한을 0.75배씩 낮추며 재인코딩(최대 4회) → 그래도 초과하면 **타일 분할**(F-66)로 전환 → 그래도 불가하면 해당 이미지를 슬롯에서 제외하고 provenance에 `image_dropped` 기록.
- **2단 상한 정책**

  | 상한 | 값 | 성격 |
  |---|---|---|
  | **하드 상한** | **200MB** (base64 후) | API 계약. 초과하면 호출 자체가 불가 |
  | **소프트 상한(운영 기본)** | **장당 6MB, 호출 합계 20MB** | 지연·게이트웨이 타임아웃 방지. 설정으로 조절 |

  > 실무적으로 2048×2048 8bit PNG는 1~3MB이므로 소프트 상한으로 충분하다. 하드 상한은 **대형 스티치 이미지·16bit 원본·멀티페이지 TIFF** 같은 예외를 막는 안전선이다. 200MB를 그대로 쓰면 요청 하나가 read 타임아웃을 넘겨 실패한다 — **크기 제약은 통과해도 시간 제약에서 죽는다**는 점을 설계에 반영한다.
- **구현** `platform/llm/image_payload.py`. 결과 객체는 `ImagePart{ data_url, bytes, width, height, transform }`.

### F-64. ★ 이미지 슬롯 예산 플래너 (4장 제약) `P0`

- **무엇** "이번 호출에 **어떤 4장**을 보낼 것인가"를 결정하는 **명시적 플래너**. 아무 데서나 이미지를 첨부하는 것을 금지하고, 모든 멀티모달 호출은 이 플래너를 통과한다.
- **표준 슬롯 배치 (의미 보강 호출 기준)**

  | 슬롯 | 내용 | 해상도 | 목적 |
  |---|---|---|---|
  | **1** | 전체 뷰: 마스크 컬러맵 + 스크리블 오버레이 + 격자/스케일바 | 축소(≤1,536px) | 전역 맥락 |
  | **2** | 관심 영역 **원해상도 크롭** (측정 대상 주변) | 원본 배율 | 세부 판단 |
  | **3** | **콘택트 시트** — 여러 후보/인스턴스를 격자로 합성 + 인덱스 라벨 (F-65) | 셀별 원해상도 | 다수 항목을 1장으로 |
  | **4** | 비교 뷰: 선택지 A / B 를 나란히 하이라이트 (모호성 질문용) 또는 원본 그레이 크롭 | 원본 배율 | 대안 대조 |

- **우선순위 규칙**
  1. 슬롯 1은 항상 포함(맥락 없는 판단 금지).
  2. 남은 3슬롯은 **신뢰도가 낮은 항목부터** 배정한다(F-16의 confidence 오름차순).
  3. 4장을 넘는 대상이 있으면 → **콘택트 시트로 압축**(F-65) → 그래도 넘치면 **다중 호출 맵리듀스**(F-66).
  4. 텍스트만으로 답할 수 있는 역할(`planner`, `custom_step`, `json_repair`)은 **이미지 0장**이 기본값이다.
- **강제 장치** 전송 계층은 `len(image_parts) > 4` 이면 **요청을 거부**한다(런타임 assert). 조용한 절단 금지 — 어떤 이미지가 빠졌는지 모른 채 모델이 답하는 상황이 가장 위험하다.
- **구현** `platform/llm/image_budget.py` — `plan_slots(request) -> list[ImagePart]` + 결정 근거를 `slot_plan` 텔레메트리로 기록.

### F-65. ★ 콘택트 시트 합성 (Contact Sheet) `P0`

- **무엇** N개의 ROI/후보를 **격자 한 장**으로 합성하고 각 셀에 `#1 ~ #N` 인덱스와 얇은 테두리를 그린다. 모델에게는 *"각 셀 번호에 대해 답하라"* 고 지시하고, 응답은 셀 인덱스 키를 가진 JSON으로 받는다.
- **왜** "4장" 제약을 실질적으로 **4장 × 셀 수**로 확장한다. 6개 핑거 검증을 6회 호출이 아니라 **1회 호출**로 끝낸다.
- **설계 규칙**
  - 셀당 최소 해상도 보장(기본 ≥256px). 미달하면 시트를 2장으로 분할.
  - 격자는 **행 우선 순서 고정**, 셀 라벨은 이미지에 직접 렌더(모델이 순서를 헷갈리지 않게).
  - 셀 ↔ 원본 좌표 매핑을 `transform` 에 보존(F-67).
  - 응답 스키마: `{"cells": {"1": {...}, "2": {...}}}` — 셀 수와 응답 키 수가 다르면 스키마 위반으로 재질의.
- **구현** `perception/contact_sheet.py` (OpenCV로 합성).

### F-66. 다중 호출 맵리듀스 `P1`
- **무엇** 대상이 4슬롯·콘택트 시트로도 안 들어가면, **결정론적으로 배치를 나눠 여러 번 호출**하고 결과를 병합한다.
- **규칙**
  - 분할은 **인덱스 순서 고정**(재현성). 배치 크기는 설정값.
  - 각 배치는 **독립적으로 유효한 프롬프트**여야 한다(맥락 슬롯 1은 매 배치에 재첨부).
  - 병합은 결정론적: 인덱스 키로 dict 병합, 충돌 시 신뢰도 높은 쪽 채택 후 충돌 로그 기록.
  - 배치 간 **일관성 검사**: 동일 대상이 두 배치에 겹쳐 들어가면 답이 같은지 확인(불일치 시 F-18 모호성 질문으로 승격).
- **구현** `platform/llm/map_reduce.py`.

### F-67. 이미지 변환 메타 & 좌표 역변환 `P0`
- **무엇** 크롭·리사이즈·시트 합성으로 좌표계가 바뀌므로, 모든 `ImagePart`에 변환 메타를 동봉한다.
  ```json
  { "source_image": "C2024_x_gray.tif",
    "crop": [512, 300, 640, 640], "scale": 0.75,
    "sheet_cell": {"index": 3, "origin": [256, 0]},
    "to_source": "affine 2x3 행렬" }
  ```
- **왜** 기존 시스템이 `_rescale_endpoints_to_original()` 로 뒤늦게 처리하던 문제를 **구조적으로 봉쇄**한다.
- **정책** 신규 시스템은 **모델에게 좌표를 묻지 않는 것을 원칙**으로 한다(좌표는 CV와 스냅이 담당). 그럼에도 모델이 좌표를 반환하는 경우(예: 셀 내 대략적 위치 지목)에는 **반드시 역변환을 통과**해야 MIR에 들어갈 수 있다.

### F-68. 모델 응답 캐시 `P1`
- **키** `(role, backend, model, prompt_hash, image_content_hashes, reasoning_mode, max_tokens)`.
- **왜** 같은 이미지·같은 질문의 재호출은 개발/디버깅 중 대량 발생한다. VLM 호출이 가장 비싸므로 캐시 효과가 크다.
- **정책** plan 생성 캐시는 **MIR 해시 기준**이므로, 사용자가 의도를 고치면 자동 무효화된다. 캐시 히트/미스는 텔레메트리에 기록하고, `?no_cache=1` 로 우회 가능.

### F-69. 모델 호출 텔레메트리 · 예산 `P1`
- **기록 항목** role, backend, provider, model, 요청 이미지 수/총 바이트, prompt/completion 토큰, `finish_reason`, `open_ms`(요청 수락~헤더 수신), 총 지연, 재시도 횟수, rate-limited 여부, structured-json 단계별 성공.
- **예산 가드** 레시피 1건 생성당 **호출 수 / 이미지 장수 / 총 토큰** 상한을 두고, 초과 시 작업을 중단하고 사람에게 보고한다(무한 재시도 루프 금지).
- **목표 예산 (레시피 1건, 정상 경로)**

  | 단계 | 호출 | 이미지 | 비고 |
  |---|---|---|---|
  | 의미 보강(선택) | 0~1 | ≤4 | 스냅이 충분하면 **0회** |
  | 모호성 선택지 | 0~1 | ≤4 | 모호할 때만 |
  | plan 합성 | 1 | 0 | 텍스트만 |
  | plan 수정 | 0~2 | 0 | 검증 실패 시 |
  | **합계** | **1~5회** | **0~8장** | |

  > 기존 시스템은 VLM 1회 + DINO N회 + 코드생성 1~2회 + 자동수정 최대 5회로 **최대 9회 이상**, 그중 상당수가 대용량 이미지 호출이었다. 신규 설계는 **정상 경로에서 이미지 호출 0~2회**를 목표로 한다.

### F-70. 프롬프트 · 모델 회귀 스위트 `P1`
- 프롬프트나 모델(Gemma4→차기 버전 등)을 바꿀 때 **B4 회귀 세트**(Part 7.1)를 자동 실행해 이전 대비 성능 변화를 리포트.
- 모든 프롬프트에 버전(`p/2.3`)을 부여하고 `run.json`에 기록(F-46) → 어떤 프롬프트가 어떤 결과를 냈는지 사후 추적 가능.

### F-71. 폴백 및 열화 정책 `P0`
- **모델 장애 시**
  1. Gemma4 실패 → 동일 역할을 GaussO4.1로 1회 재시도(멀티모달 지원되므로 대체 가능).
  2. GaussO4.1 실패 → `planner`는 **재시도 후 중단**. 잘못된 plan을 만드느니 멈춘다.
  3. 둘 다 불가 → **CV + 스냅만으로 MIR 확정**하고, plan은 **템플릿 기반 결정론 합성**(단순 거리 측정 등 표준 패턴만)으로 축약 생성. 사용자에게 "AI 보강 없이 생성됨"을 명시.
- **원칙** 모델 장애가 **드로잉과 스냅을 막지 않는다.** 캔버스·스냅·패턴검출·연산자 실행은 전부 로컬 CV이므로 오프라인에서도 동작해야 한다. 모델은 **가속기이지 필수 부품이 아니다.**

### F-72. 페이로드 보안 · 민감정보 처리 `P2`
- 로그에 **base64 이미지 본문을 절대 남기지 않는다**(해시와 바이트 수만).
- 이미지에 웨이퍼 ID/로트 번호 등이 렌더링되어 있으면 전송 전 마스킹하는 옵션.
- API 키·티켓은 환경변수. 라우팅 덤프는 항상 마스킹.
- 외부 전송 여부(사내 게이트웨이 vs 외부망)를 설정에 명시하고, 외부일 경우 이미지 전송을 기본 차단.

---

## 5.4 멀티모달 프롬프트 규약

### 5.4.1 메시지 형태

```python
messages = [
  {"role": "system", "content": SYSTEM_RULES},   # 역할별 고정, 짧게
  {"role": "user", "content": [
      {"type": "text",      "text": task_text},              # 지시 + 슬롯 설명
      {"type": "image_url", "image_url": {"url": slot1_data_url}},
      {"type": "image_url", "image_url": {"url": slot2_data_url}},
      {"type": "image_url", "image_url": {"url": slot3_data_url}},
      {"type": "image_url", "image_url": {"url": slot4_data_url}},
      {"type": "text",      "text": OUTPUT_SCHEMA_TEXT},     # 출력 스키마를 이미지 뒤에 재확인
  ]},
]
```

**규칙**
1. **텍스트로 슬롯을 반드시 명명한다.** *"이미지 1은 전체 뷰, 이미지 2는 #3 핑거의 원해상도 크롭, 이미지 3은 6개 후보 콘택트 시트다."* — 모델이 몇 번째 이미지인지 헷갈리는 것이 멀티모달 오류의 최대 원인이다.
2. **출력 스키마를 이미지 뒤에 한 번 더 붙인다.** 긴 이미지 블록 뒤에 지시가 희석되는 것을 막는다.
3. **닫힌 선택지를 쓴다.** 자유 서술 대신 enum·boolean·인덱스. `response_format` 강제가 불가능한 만큼 프롬프트로 좁힌다.
4. **좌표를 묻지 않는다.** 위치가 필요하면 콘택트 시트의 **셀 인덱스**나 후보 ID로 답하게 한다.
5. **프롬프트는 파일로 버전 관리**한다(`prompts/<role>/<version>.md`). 코드 안 f-string에 흩어두지 않는다 — 기존 시스템의 수백 줄 인라인 프롬프트가 유지보수 불가능해진 원인이다.

### 5.4.2 시스템 프롬프트 최소주의

기존 시스템은 코드 생성 규칙 수백 줄을 프롬프트에 넣었다. 신규 시스템은 **Plan→Code 구조(F-30)** 덕분에 그 규칙 대부분이 불필요하다.

| 기존 프롬프트 규칙 | 신규 시스템에서의 처리 |
|---|---|
| "def로 정의 안 된 함수 호출 금지" | plan 검증기가 미등록 연산자를 거부 |
| "`np.where`는 (y,x), `fitLine`은 (x,y)" | 연산자 라이브러리가 내부에서 한 번만 올바르게 처리 |
| "numpy 스칼라를 JSON에 넣지 마라" | 연산자 반환 타입이 `float`/`int`로 고정 |
| "CSV 표준 헤더를 써라" | 렌더러가 헤더를 생성 |
| "merge 이미지를 읽지 마라" | 정적 검증 게이트(F-33) |
| "SDIFF 값이 우선, 예제 상수 무시" | plan에 값이 명시되므로 충돌 자체가 없음 |

→ **planner 시스템 프롬프트 목표 길이: 60줄 이내.** 연산자 카탈로그(기계 생성)를 별도 메시지로 첨부한다.

---

## 5.5 구현 체크리스트

- [ ] `platform/llm/` 패키지 생성 (`backends / roles / transport / policy / structured / image_payload / image_budget / map_reduce / cache / telemetry`)
- [ ] `report-search`의 `model_client.py` 이식 — 텍스트 경로는 **동작 동등성 테스트**로 검증
- [ ] 멀티모달 확장: content parts 조립, 4장 강제, 200MB/소프트 상한 검사
- [ ] 프로바이더 정책 테이블에 `gemma4`, `gausso4_1` 만 등록 (`gpt_oss`/`openai_compat`는 테스트용으로만)
- [ ] 역할 레지스트리 5.2.2로 구성 + 기동 시 검증
- [ ] 프롬프트 디렉터리 + 버전 관리
- [ ] 테스트: 프로바이더 페이로드 스냅샷 / JSON 3단 복구 / 429 재시도 / 커넥션 재사용 / **이미지 4장 초과 거부** / **base64 후 크기 검사** / 콘택트 시트 좌표 역변환 / 맵리듀스 병합 결정론성
- [ ] `.env.example` 작성 (아래)

```bash
# --- 공통 게이트웨이 ---
SEND_SYSTEM_NAME=ScribbleMetro
USER_ID=
USER_TYPE=

# --- Gemma4 (멀티모달 인식) ---
GEMMA4_LLM_MODEL=
GEMMA4_LLM_BASE_URL=
GEMMA4_LLM_API_KEY=EMPTY
GEMMA4_LLM_TICKET=
GEMMA4_LLM_API_PROFILE=gemma4

# --- GaussO4.1 (추론/계획) ---
GAUSSO4_1_LLM_MODEL=
GAUSSO4_1_LLM_BASE_URL=
GAUSSO4_1_LLM_API_KEY=EMPTY
GAUSSO4_1_LLM_TICKET=
GAUSSO4_1_LLM_API_PROFILE=gausso4_1

# --- 역할 라우팅 (기본값 재정의용) ---
PLANNER_BACKEND=gausso4_1
INTENT_ENRICH_BACKEND=gemma4
JSON_REPAIR_BACKEND=gemma4

# --- 추론 예산 ---
LLM_DEFAULT_REASONING=minimal
LLM_THINKING_BUDGET_LOW=256
LLM_THINKING_BUDGET_MEDIUM=1024
LLM_THINKING_BUDGET_HIGH=2048

# --- 전송 ---
LLM_CONNECT_TIMEOUT=10
LLM_WRITE_TIMEOUT=30
LLM_POOL_TIMEOUT=10
LLM_REQUEST_TIMEOUT=120
LLM_KEEPALIVE_EXPIRY_SECONDS=5
LLM_TRANSPORT_CACHE=on

# --- 이미지 제약 ---
LLM_MAX_IMAGES_PER_CALL=4          # 하드 제약. 초과 시 요청 거부
LLM_IMAGE_HARD_LIMIT_MB=200        # API 계약 상한 (base64 인코딩 후 기준)
LLM_IMAGE_SOFT_LIMIT_MB=6          # 운영 권장 장당 상한
LLM_IMAGE_TOTAL_SOFT_LIMIT_MB=20   # 호출 합계 권장 상한
LLM_IMAGE_MAX_LONG_SIDE=1536       # 전체 뷰 리사이즈 상한 (크롭은 원해상도 유지)
LLM_CONTACT_SHEET_MIN_CELL_PX=256
```

---

# Part 6. 구현 계획

## 6.1 기술 스택

| 영역 | 선택 | 이유 |
|---|---|---|
| 백엔드 | Python 3.11 + FastAPI | 기존 자산 계승, CV/ML 생태계 |
| 스키마 | Pydantic v2 → JSON Schema | MIR 검증을 프런트/백엔드가 공유 |
| CV | OpenCV, numpy, scikit-image | 기존 로직 이관 |
| DB | SQLite → Postgres (SQLAlchemy 2.x + Alembic) | 단일 노드로 시작해 확장 |
| 큐 | asyncio 워커 (→ 필요 시 RQ/Celery) | 초기 복잡도 최소화 |
| 프런트 | TypeScript + Vite, Canvas 2D(+WebGL 오버레이) | 프레임워크는 얇게, 캔버스 엔진은 직접 제어 |
| 코드 검증 | `ast`, `libcst`, `ruff`, `black` | 정적 게이트 |
| 테스트 | pytest, playwright(E2E) | 연산자 단위 테스트가 시스템 신뢰의 기반 |
| 실행 격리 | Docker(sandbox-runner) 또는 rlimit+격리유저 | 생성 코드 실행 안전 |
| 모델 | **Gemma4** (멀티모달 인식) / **GaussO4.1** (추론·계획) — 사내 OpenAI 호환 게이트웨이 | 사용 가능한 2종으로 확정. 상세 Part 5 |
| 모델 SDK | `openai` (OpenAI 호환) + `httpx` 커넥션 풀 | `report-search/app/model_client.py` 패턴 이식 |
| 이미지 페이로드 | OpenCV + Pillow → PNG/JPEG → base64 data URL | 4장·200MB 제약 준수 (F-63/F-64) |

## 6.2 패키지 구조

```
scribblemetro/
├── apps/
│   ├── api/                 # FastAPI 엔트리, 라우터, DI
│   └── web/                 # TS 프런트엔드
│       ├── canvas/          # 렌더러, 뷰포트, 도구, 히트테스트
│       ├── intent/          # 스트로크 피팅(즉시 판정), 스냅 클라이언트
│       └── panels/          # 제약 인스펙터, 되읽기, 배치 대시보드
├── packages/
│   ├── mir/                 # Pydantic 모델, 스키마, 마이그레이션, 검증
│   ├── perception/          # 마스크 인덱싱, 스냅 후보, 패턴검출, 재정위, 레거시 임포트
│   ├── metro_ops/           # ★ 앵커 프리미티브 연산자 라이브러리 (+pytest)
│   ├── synthesis/           # planner(LLM), plan 검증, 렌더러(Jinja), 정적검증, repair
│   ├── execution/           # 샌드박스, 배치 러너, 아티팩트 수집
│   ├── verification/        # demo replay, transfer, 민감도, QC 룰
│   ├── knowledge/           # 레시피/few-shot 저장소, 검색, ops 승격
│   └── platform/            # 설정, DB, 큐, 로깅
│       └── llm/             # ★ 모델 API 계층 (Part 5)
│           ├── backends.py      # BackendSpec: gemma4 / gausso4_1
│           ├── roles.py         # RoleSpec + resolve_role + 기동 검증
│           ├── policy.py        # ProviderPolicy (reasoning/response_format/token param)
│           ├── transport.py     # OpenAI 클라이언트, 헤더, 풀 캐시, 타임아웃, 429 재시도
│           ├── structured.py    # JSON 3단 복구 + Pydantic 스키마 검증 + 부분 재질의
│           ├── image_payload.py # 리사이즈/인코딩/base64/200MB 검사/변환 메타
│           ├── image_budget.py  # 4장 슬롯 예산 플래너
│           ├── contact_sheet.py # 격자 합성 (perception과 공용)
│           ├── map_reduce.py    # 4장 초과 시 배치 분할·병합
│           ├── cache.py         # 응답 캐시
│           └── telemetry.py     # 호출 계측 · 예산 가드
├── templates/               # measure.py Jinja 템플릿
├── benchmarks/              # 평가 데이터셋 + 회귀 스위트
└── deploy/                  # Docker, compose, migrations
```

## 6.3 마일스톤

> 인원 가정: 백엔드 1.5 · 프런트 1 · CV/ML 1 (총 3~4인). 기간은 목표치.

### M0 — 기반 정비 (2주)
- 리포지토리 재구성, 설정/시크릿 외부화, DB 스키마, 작업 큐, 로깅.
- **기존 코드 이관**: 스크리블 CV 추출, meta 로더, CSV 정규화, few-shot 저장소, 전이 평가 스크립트를 각 패키지로 분해 이식.
- **★ 모델 API 계층 구축 (Part 5)**: `report-search/app/model_client.py` 패턴 이식 → Gemma4/GaussO4.1 백엔드·역할 레지스트리, 전송 계층, reasoning 정책, JSON 3단 복구, **이미지 페이로드 파이프라인(200MB) + 4장 슬롯 플래너**. 이 시점에 두 모델 엔드포인트로 **왕복 스모크 테스트**를 통과시켜 둔다(뒤 마일스톤이 전부 여기에 의존한다).
- 산출: 빈 화면이지만 이미지 목록/뷰가 뜨고 배치 잡이 큐를 통해 돌며, 두 모델에 이미지 첨부 호출이 성공한다.
- 관련: F-48, F-49, F-50, F-14, **F-59~F-64, F-67, F-71**

### M1 — 드로잉 캔버스 MVP (3주)
- F-01, F-02, F-03, F-04, F-05 + F-06(스냅 1차: 경계·특징점·기존 선).
- 스트로크는 `demo_strokes`로 저장되고 스냅 기록이 남는다.
- **인수 조건**: 사용자가 브라우저에서만 스크리블을 완성할 수 있다. 저장/재로드 시 왕복 손실 없음.

### M2 — 의도 계층 (3주)
- F-15(MIR), F-16(provenance), F-07(스트로크 정형화), F-09(제약 인스펙터), F-17(되읽기), F-20(양방향 동기화).
- F-11/F-26(반복 검출·전파) 1차.
- **인수 조건**: 스크리블 → 검증 통과한 MIR이 자동 생성되고, 사람이 폼에서 고칠 수 있으며, 문장으로 되읽어 준다.

### M3 — 합성 & 실행 (4주)
- F-23(연산자 라이브러리 v1: 25~35개), F-30(Planner), F-31(렌더러), F-33(정적검증), F-34(샌드박스), F-35(수정 루프).
- **인수 조건**: MIR → 실행되는 `measure.py` → `measurements.csv` + `overlay.png`. 첫 실행 성공률 ≥ 90%.

### M4 — 검증 게이트 (3주)
- F-36(시연 재현), F-37(전이 신뢰도), F-25(재정위 폴백 체인), F-43(QC), F-46(매니페스트).
- **인수 조건**: 검증 3게이트가 자동으로 초록/빨강을 판정하고, 빨강이면 자동 재시도가 돈다.

### M5 — 운영 & 지능형 어시스트 (3주)
- F-42(배치 대시보드), F-44(리포팅), F-45(리뷰 루프), F-47(레시피 레지스트리).
- F-18(모호성 질문), F-55(드로잉 진단), F-56(자동 완성), F-13(고스트 오버레이).
- **인수 조건**: 폴더 단위 운영이 가능하고, 사람이 검토할 항목만 큐에 올라온다.

### M6 — 품질 고도화 (지속)
- F-38(민감도), F-39(Best-of-N), F-24(서브픽셀), F-32(구조 유사도 검색), F-41(ops 승격), F-19(대화형 편집), F-21(클래스 사전), F-40(레시피 카드), F-52~F-54.
- 모델 계층 고도화: F-65(콘택트 시트 최적화), F-66(맵리듀스), F-68(캐시), F-69(예산 가드), F-70(프롬프트 회귀 스위트), F-72(페이로드 보안).
- 벤치마크 확장 및 프롬프트/모델 회귀 테스트 정착.

**총 소요: 약 18주 (M0~M5) + 지속 개선** — M0가 2주에서 3주로 늘어날 수 있다(모델 API 계층 포함). 다만 이 투자는 M2 이후 모든 마일스톤이 재사용하므로 회수된다.

## 6.4 병렬화 및 의존성

```
M0 ──┬── M1 (프런트) ────┬── M2 ── M3 ── M4 ── M5
     └── metro_ops (CV)  ┘         ▲
                                   └─ perception(재정위/패턴)은 M2와 병행 착수 가능
```
- `metro_ops`(F-23)는 프런트와 무관하므로 **M0 직후 즉시 병행 착수**한다. 이것이 M3의 임계 경로다.
- `platform/llm`(F-59~F-64)도 프런트와 무관하다. **M0에서 끝내 두는 것이 원칙**이며, 늦어도 M2 시작 전에는 완료되어야 F-18(모호성 질문)과 F-27(의미 보강)이 막히지 않는다.
- 연산자 라이브러리가 두꺼울수록 Planner가 쉬워진다 → **초기 투자 대비 회수율이 가장 높은 항목**.

---

# Part 7. 평가 체계와 완료 기준

## 7.1 벤치마크 구성
- **B1 시연 세트**: 대표 공정 20종 × 이미지 1장 + 사람이 그린 정답 스크리블 + 정답 측정값.
- **B2 전이 세트**: 각 공정당 실이미지 10~30장(시연에 쓰지 않은 것) + 정답 측정값.
- **B3 스트레스 세트**: augmentation 자동 생성(시드 고정) + 저품질/결손 마스크 포함.
- **B4 회귀 세트**: 과거 실패 사례 아카이브(모델·프롬프트 변경 시 반드시 통과).
- **B5 모델 계약 세트**: 프로바이더별 요청 페이로드 스냅샷, 깨진 JSON 응답 10종, 429/타임아웃 주입, 초대형 이미지(>200MB 원본) 및 5장 첨부 시도 — **모델을 실제로 호출하지 않고** 전송 계층만 검증하는 오프라인 스위트.

## 7.2 KPI

| # | 지표 | 목표 | 측정 |
|---|---|---|---|
| K1 | 스트로크 의도 추천 수락률 | ≥ 80% | `Enter`로 1순위 수락 비율 |
| K2 | 스냅 정확도 | ≥ 95% | 사용자가 스냅을 되돌린 비율의 여집합 |
| K3 | 코드 첫 실행 성공률 | ≥ 95% | 정적검증 통과분 기준 |
| K4 | **시연 재현 정확도** | 끝점 평균 ≤ 3px, 길이 오차 ≤ 3% | F-36 |
| K5 | **전이 성공률** | ≥ 90% (B2), ≥ 95% (B3) | F-37 |
| K6 | 값 안정성 CV | ≤ 2% (기하 변환 하에서) | F-37 |
| K7 | 사람 개입 시간 | 신규 레시피 ≤ 5분 | 세션 텔레메트리 |
| K8 | 자동 수정 루프 평균 반복 | ≤ 1.2회 | F-35 로그 |
| K9 | 조용한 오측정률 | 0% (모든 이상은 반드시 플래그) | F-43 |
| K10 | 재현성 | 동일 매니페스트 → 100% 동일 결과 | F-46 |
| K11 | **레시피 1건당 모델 호출 수** | ≤ 5회 (이미지 호출 ≤ 2회) | F-69 예산 계측 |
| K12 | **이미지 제약 위반** | 0건 (4장 초과 / 200MB 초과 요청이 전송되지 않음) | 전송 계층 assert + 테스트 |
| K13 | **구조화 JSON 최종 성공률** | ≥ 99% (3단 복구 + 부분 재질의 포함) | F-62 `structured_json` 이벤트 |
| K14 | **모델 장애 시 열화 동작** | 드로잉·스냅·측정 실행 100% 정상 | F-71 장애 주입 테스트 |
| K15 | 커넥션 재사용률 | 연속 20회 호출 시 TLS 핸드셰이크 1회 | F-60 |

## 7.3 릴리스 게이트
1. **Draft** — MIR 검증 통과.
2. **Generated** — 정적검증 + 실행 성공.
3. **Verified** — 시연 재현 통과 + 전이 통과. *여기서부터 배치 실행 허용.*
4. **Production** — 리뷰어 승인 + 홀드아웃 실이미지 검증 통과 + 레시피 카드 작성 완료.

---

# Part 8. 리스크와 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| 연산자 라이브러리가 표현력 부족 → 특수 측정 불가 | 높음 | `custom_step` 탈출구(F-30) + 승격 파이프라인(F-41). 초기 20개 실사례로 연산자 커버리지를 먼저 검증 |
| 스냅/패턴검출이 특정 공정에서 오작동 | 높음 | 항상 수동 모드 존재(`Alt` 해제, 인스펙터 직접 편집). 스냅 실패가 파이프라인을 막지 않게 설계 |
| LLM plan 품질 편차 | 중간 | plan 스키마 강제 + Best-of-N(F-39) + 검증 게이트. 최악의 경우에도 잘못된 코드가 Verified로 승격되지 않음 |
| 재정위 실패(공정 변화, 큰 회전) | 높음 | 4단 폴백(F-25) + 실패 시 `NEEDS_REVIEW`. 조용한 실패 금지 |
| 캔버스 성능(대형 이미지 + 다수 스트로크) | 중간 | 타일 렌더, 오프스크린 캐시, 스냅 후보 사전 인덱싱, 정형화 판정은 프런트 즉시/서버 정밀 2단 |
| GPU 자원 경합(VLM) | 중간 | VLM을 크리티컬 패스에서 제외(F-27). 큐 우선순위 분리 |
| 기존 자산(수백 개 스크리블/코드) 사장 | 중간 | F-14 임포터 + 기존 CSV 스키마 유지 + few-shot 코퍼스 이관 |
| 사람의 신뢰 확보 실패("AI가 잰 값을 못 믿겠다") | 높음 | 되읽기(F-17) + 재현 검증 수치(F-36) + 레시피 카드(F-40) + provenance 배지. **설명 가능성을 UI의 기본값으로** |
| **이미지 4장 제약으로 판단 정보가 부족** | 높음 | 콘택트 시트(F-65)로 4장×N셀 확장, 맵리듀스(F-66)로 배치 분할. 근본적으로는 **스냅이 의도를 잡으므로 모델이 볼 것이 적다**(F-06) |
| **200MB / 게이트웨이 타임아웃** | 중간 | base64 **인코딩 후** 크기 검사, 소프트 상한(장당 6MB)으로 지연 억제, 초과 시 단계적 열화 후 타일 분할 (F-63) |
| **`response_format` 미지원 → JSON 파싱 실패** | 높음 | 3단 복구 + Pydantic 검증 + **위반 필드 부분 재질의**(F-62). 최종 실패는 예외로 전파하고 절대 조용히 넘기지 않음 |
| **Gemma4/GaussO4.1 파라미터 규약 변경** | 중간 | 규약을 `ProviderPolicy` 테이블 1곳에 집중(F-61). 페이로드 스냅샷 테스트(B5)가 회귀를 즉시 잡음 |
| **모델 엔드포인트 장애·정원 초과** | 중간 | 429 전용 재시도 + Retry-After 준수, 역할별 폴백(Gemma4↔GaussO4.1), 최종적으로 CV-only 열화 경로(F-71) |
| **모델 교체/버전업 시 품질 회귀** | 중간 | 역할↔백엔드가 환경변수로 분리되어 A/B 가능(F-59) + B4/B5 회귀 스위트 자동 실행(F-70) |

---

# 부록 A. API 목록 / 디렉터리 구조 / 마이그레이션

## A.1 주요 API (신규)

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/images` | 이미지 목록/필터/상태 |
| GET | `/api/images/{id}/context` | 마스크 인덱스, 클래스, 스케일, 스냅 후보 캐시 키 |
| POST | `/api/strokes/fit` | 스트로크 → 정형화 후보 랭킹 (F-07/F-08) |
| POST | `/api/snap/candidates` | 커서 위치 → 스냅 후보 (F-06) |
| POST | `/api/patterns/detect` | 반복/대칭 구조 검출 (F-11/F-26) |
| POST | `/api/mir/build` | 스트로크+스냅 → MIR 생성 (F-15) |
| POST | `/api/mir/validate` | 스키마·의미 검증, 모호성 목록 반환 (F-18) |
| PATCH | `/api/mir/{id}` | JSON Patch로 부분 수정 (F-19) |
| GET | `/api/mir/{id}/readback` | 자연어 되읽기 (F-17) |
| POST | `/api/synthesis/plan` | MIR → plan (F-30) |
| POST | `/api/synthesis/render` | plan → measure.py (F-31) |
| POST | `/api/runs` | 실행(단일/배치) 잡 생성 (F-34/F-42) |
| GET | `/api/runs/{id}/events` | SSE 진행 스트림 (F-49) |
| POST | `/api/verify/replay` | 시연 재현 검증 (F-36) |
| POST | `/api/verify/transfer` | 전이 신뢰도 검증 (F-37) |
| GET/POST | `/api/recipes` | 레시피 레지스트리 (F-47) |
| POST | `/api/reviews` | 리뷰 수정 환류 (F-45) |
| GET/POST | `/api/ops` | 연산자 레지스트리 / 승격 (F-41) |
| GET | `/api/system/llm-routing` | 현재 백엔드·역할 배선 확인 (키/티켓 마스킹, F-59) |
| GET | `/api/system/llm-budget` | 레시피별 모델 호출·이미지·토큰 사용량 (F-69) |
| POST | `/api/vlm/enrich` | Gemma4 의미 보강 (내부용, 슬롯 플래너 경유, F-27) |

## A.2 기존 → 신규 매핑

| 기존 | 신규 |
|---|---|
| `/vl/sdiff_qwen` | `/api/mir/build` (+ `/api/patterns/detect`, 선택적 `/api/vlm/enrich`) |
| `/gptoss/generate_from_sdiff` | `/api/synthesis/plan` + `/api/synthesis/render` |
| `/code/run` (+auto_fix) | `/api/runs` + F-35 구조화 수정 루프 |
| `/sdiff/get_latest` | `/api/mir/{id}` |
| `/fewshot/*` | `/api/recipes/search` (구조 유사도 기반, F-32) |
| `/utils/*` | `/api/ops` (레지스트리 + 승격 워크플로) |
| `validation/transfer_reliability_eval.py` | `/api/verify/transfer` (F-37) |
| `_reject_merge_copying` | F-33 정적 검증 게이트의 한 규칙 |
| `_normalize_measurements_csv` | 렌더러가 스키마를 보장 + 실행 후 검증 |
| `_lazy_load_qwen()` (로컬 Qwen3-VL) | **삭제.** Gemma4 게이트웨이 호출로 대체 (`intent_enrich` 역할) |
| `_lazy_load_dino()` / `_call_grounding_dino()` | **삭제.** 반복 패턴 검출(F-26) + 재정위(F-25) + 사용자 드로잉으로 대체 |
| `_build_gptoss()` / `_build_gausso()` / `_build_llm_for()` | `platform/llm` 의 `make_role_client(user_id, role)` 로 통합 |
| `_safe_invoke_gptoss_for_code*()` | `synthesis.planner` (GaussO4.1) + 결정론적 렌더러 |
| `_extract_text_from_aimessage()` | `extract_message_text()` (reasoning 파트 분리 포함) |
| `MAX_IMAGES_PER_PROMPT = 5` (상수만 존재) | **F-64 슬롯 예산 플래너 + 전송 계층 강제(4장 초과 요청 거부)** |
| `_prep_for_vision_llm()` (max_side 1280, JPEG q80) | F-63 이미지 페이로드 파이프라인 (변환 메타 보존 + base64 후 크기 검사) |
| `_rescale_endpoints_to_original()` | F-67 변환 메타 기반 역변환 (원칙적으로 모델에게 좌표를 묻지 않음) |
| 인라인 수백 줄 시스템 프롬프트 | `prompts/<role>/<version>.md` 파일 + Plan→Code 구조로 규칙 대부분 불필요 (Part 5.4.2) |

## A.3 마이그레이션 전략

1. **데이터 우선 이관** — 기존 `fewshot_repo`, `measure_label/*.py`, `scribble/*.png`를 신규 스토리지로 적재. 스크리블은 F-14로 MIR 초안 자동 생성(사람 확인 후 확정).
2. **CSV 스키마 동결** — 다운스트림 도구 보호를 위해 13개 표준 컬럼을 그대로 유지, 확장은 추가 컬럼으로만.
3. **병행 운영 기간** — 신규 시스템이 생성한 코드와 기존 시스템 코드를 같은 이미지에 돌려 값 차이를 비교(회귀 리포트). 차이가 임계 이내일 때 전환.
4. **연산자 커버리지 선검증** — 착수 직후, 기존 `measure_label/`의 실제 코드 20~30개를 읽어 **어떤 연산이 실제로 쓰였는지 목록화**하고, 그것을 `metro_ops` v1의 스펙으로 삼는다. (설계 리스크를 가장 크게 줄이는 첫 작업)
5. **모델 전환 검증** — `report-search`의 전송 계층을 이식한 직후, Gemma4·GaussO4.1 두 엔드포인트에 대해 (a) 텍스트 왕복, (b) 이미지 1장 왕복, (c) 이미지 4장 왕복, (d) 5장 시도 시 거부, (e) 200MB 초과 이미지 자동 열화를 **스모크 테스트로 확인**한다. 이 5개가 통과하기 전에는 상위 기능 개발을 시작하지 않는다.
6. **GPU 반납** — 로컬 Qwen3-VL / Grounding DINO 제거로 `CUDA_VISIBLE_DEVICES=0,1,2` 상주 점유가 사라진다. 남는 GPU 자원은 배치 실행(F-42)과 전이 검증(F-37) 병렬화에 재배치한다.

---

## 마무리 — 이 계획의 한 문장

기존 시스템은 **"그려진 그림에서 의도를 되찾으려" 애썼다.** 신규 시스템은 **"그리는 순간에 의도를 함께 붙잡고", 그것을 검증 가능한 명세로 확정한 뒤, 검증된 연산자만으로 코드를 조립하고, 사람이 그린 것을 실제로 재현하는지 스스로 확인한다.** 나머지 기능은 모두 이 한 줄을 빠르고, 정확하고, 믿을 수 있게 만들기 위한 장치다.

그리고 모델 측면에서 이 전환은 **의존을 줄이는 방향**이다. 사용 가능한 모델이 Gemma4·GaussO4.1 둘뿐이고 한 번에 이미지 4장까지만 보낼 수 있다는 제약은, 역설적으로 이 설계와 잘 맞는다. 의도를 그리는 순간에 붙잡아 두면 **모델에게 보여줄 것이 애초에 적기 때문**이다. 기존 시스템은 이미지 3장을 매번 VLM에 밀어 넣고 그 안에서 의미를 찾아내야 했지만, 신규 시스템의 정상 경로에서는 이미지 호출이 **0~2회**에 그친다.
