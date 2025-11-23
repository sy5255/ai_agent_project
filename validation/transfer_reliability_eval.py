# transfer_reliability_eval.py
# ----------------------------
# 폴더 안의 모든 mask 이미지를 대상으로
#   - 여러 augmentation (shift/rot/scale/shear + 랜덤 5개)를 적용하고
#   - 각 케이스마다 measure.py를 실행한 뒤
#   - 성공/실패/에러 단계/CSV 행수 등을 transfer_summary.csv로 정리하는 스크립트
#
# meta_utils.py의 규칙:
#   filename = mask_path.split('_gray.tif')[0] + '.json'
#   → 그래서 augmentation 이미지 파일명 안에 반드시 "_gray.tif" 패턴을 그대로 포함시킨다.
#
# 예)
#   원본: C2024_..._gray.tif
#   aug : C2024_..._gray.tif__rot+1.0.png
#         C2024_..._gray.tif__shift_x+100.png
#   → meta_utils.find_meta_path(mask_path, meta_root) 에서
#      "_gray.tif" 기준으로 앞부분만 자르고, 같은 meta JSON을 찾을 수 있음.

import os
import shutil
import subprocess
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd


# ====================== 사용자 설정 ==========================

# 1) 측정 대상 이미지들이 들어 있는 폴더
#    (예: *_gray.tif 파일들)
INPUT_IMAGE_DIR = r"D:\aip-expert\testset\mask_imageset"  # <-- 수정해서 사용

# 2) meta_root 경로 (meta_utils.DEFAULT_META_ROOT와 동일하게 맞추거나, measure.py에 넘길 값)
META_ROOT = r"D:\aip-expert\testset\meta"  # <-- 네 환경에 맞게 수정

# 3) measure.py 경로
MEASURE_SCRIPT_PATH = r"D:\aip-expert\testset\variance_test\measure.py"

# 4) grounding_template.png 위치
GROUNDING_TEMPLATE_SRC = str(Path(MEASURE_SCRIPT_PATH).with_name("grounding_template.png"))

# 5) 실험 결과를 쌓을 루트 디렉토리
EXPERIMENT_ROOT = r"D:\aip-expert\testset\variance_test\experiments_transfer"

# 6) augmentation 설정
SHIFT_PX = 200          # 좌/우 shift 크기 (px) - 고정
ROT_DEG = 1          # 회전 각도 (deg) - 고정
SCALE_FACTORS = [1.03, 0.97]   # scale factor - 고정
SHEAR_FACTORS = [0.02, -0.02]  # x 방향 shear - 고정

N_RANDOM_AUG = 50       # 이미지당 랜덤 augmentation 개수

# 랜덤 augmentation 범위 설정 (너가 원하면 나중에 조정 가능)
RAND_SHIFT_MAX = SHIFT_PX      # [-100, 100] 범위
RAND_ROT_MAX = ROT_DEG * 2   # [-1.5, 1.5] 도
RAND_SCALE_MIN = 0.9
RAND_SCALE_MAX = 1.03
RAND_SHEAR_MAX = 0.02          # [-0.03, 0.03]

# 7) measure.py가 생성하는 파일 이름
OVERLAY_NAME = "overlay.png"
CSV_NAME = "measurements.csv"

# 8) python 실행 커맨드 템플릿
MEASURE_CMD_TEMPLATE = (
    r'python "{script}" '
    r'--mask_path "{image_path}" '
    r'--meta_root "{meta_root}" '
    r'--out_dir "{run_dir}"'
)

# 9) 처리할 이미지 확장자 (원본 mask)
VALID_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# 10) 재현성 있는 랜덤을 위해 seed 고정 (원하면 바꿔도 됨)
RANDOM_SEED = 1234

# ============================================================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_images(folder: str) -> List[str]:
    paths: List[str] = []
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in VALID_EXTS:
            paths.append(p)
    return sorted(paths)


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def save_gray(path: str, img: np.ndarray):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)


# ------------------ Augmentation 함수들 ----------------------

def aug_shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE
    )
    return shifted


def aug_rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def aug_scale_center(img: np.ndarray, scale: float) -> np.ndarray:
    """중심 기준 scale, 출력 크기는 원래 이미지와 동일하게 유지"""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
    scaled = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE
    )
    return scaled


def aug_shear_x(img: np.ndarray, shear: float) -> np.ndarray:
    """x 방향 shear, 중심 기준, 출력 크기 유지"""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    # 중심 기준 shear 변환: (x', y') = (x + shear * (y - cy), y)
    M = np.array([[1, shear, -shear * cy],
                  [0, 1, 0]], dtype=np.float32)
    sheared = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE
    )
    return sheared


def generate_deterministic_variants(img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    원본 + deterministic augmentation들을 생성.
    key: aug_name (나중에 파일명/summary에 사용)
    """
    variants: Dict[str, np.ndarray] = {}
    variants["orig"] = img

    # shift (좌우)
    variants[f"shift_x+{SHIFT_PX}"] = aug_shift(img, SHIFT_PX, 0)
    variants[f"shift_x-{SHIFT_PX}"] = aug_shift(img, -SHIFT_PX, 0)

    # [NEW] shift (상하)
    variants[f"shift_y+{SHIFT_PX}"] = aug_shift(img, 0, SHIFT_PX)
    variants[f"shift_y-{SHIFT_PX}"] = aug_shift(img, 0, -SHIFT_PX)

    # rotation
    variants[f"rot+{ROT_DEG}"] = aug_rotate(img, ROT_DEG)
    variants[f"rot-{ROT_DEG}"] = aug_rotate(img, -ROT_DEG)

    # scale
    for s in SCALE_FACTORS:
        variants[f"scale_{s:.3f}"] = aug_scale_center(img, s)

    # shear
    for sh in SHEAR_FACTORS:
        variants[f"shear_x_{sh:.3f}"] = aug_shear_x(img, sh)

    return variants


def generate_random_variants(img: np.ndarray, n_random: int) -> Dict[str, np.ndarray]:
    """
    이미지당 n_random개의 랜덤 augmentation 생성.
    shift / rot / scale / shear를 작은 범위 내에서 랜덤 조합.
    """
    variants: Dict[str, np.ndarray] = {}
    h, w = img.shape[:2]

    for i in range(n_random):
        dx = float(np.random.uniform(-RAND_SHIFT_MAX, RAND_SHIFT_MAX))
        dy = float(np.random.uniform(-RAND_SHIFT_MAX, RAND_SHIFT_MAX))  # 필요 없으면 0으로 줄여도 됨
        rot = float(np.random.uniform(-RAND_ROT_MAX, RAND_ROT_MAX))
        scale = float(np.random.uniform(RAND_SCALE_MIN, RAND_SCALE_MAX))
        shear = float(np.random.uniform(-RAND_SHEAR_MAX, RAND_SHEAR_MAX))

        aug_name = (
            f"rand{i+1}_dx{int(dx)}_dy{int(dy)}"
            f"_rot{rot:.2f}_scale{scale:.3f}_shear{shear:.3f}"
        )

        # 조합 변환: scale → rotate → shear → shift (대략적인 순서)
        aug_img = img.copy()
        if abs(scale - 1.0) > 1e-6:
            aug_img = aug_scale_center(aug_img, scale)
        if abs(rot) > 1e-6:
            aug_img = aug_rotate(aug_img, rot)
        if abs(shear) > 1e-6:
            aug_img = aug_shear_x(aug_img, shear)
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            aug_img = aug_shift(aug_img, dx, dy)

        variants[aug_name] = aug_img

    return variants

def collect_all_overlays(experiment_dir: str, overlay_name: str = "overlay.png"):
    """
    experiment_dir 아래 runs/**/overlay.png 를 전부 찾아서
    experiment_dir/all_overlays/ 폴더에 한 번에 모아준다.

    파일 이름 형식:
      runs/<image_name>/<aug>/overlay.png
      → all_overlays/<image_name>__<aug>.png
    """
    from pathlib import Path
    import shutil

    exp_path = Path(experiment_dir)
    runs_root = exp_path / "runs"
    dest_dir = exp_path / "all_overlays"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not runs_root.exists():
        print(f"[WARN] runs 디렉토리가 없습니다: {runs_root}")
        return

    count = 0
    for p in runs_root.rglob(overlay_name):
        # p: runs/<image_name>/<aug>/overlay.png
        try:
            aug_name = p.parent.name              # 마지막 폴더: aug 이름
            image_name = p.parent.parent.name     # 그 위 폴더: 원본 이미지 파일명
        except Exception:
            # 예상치 못한 구조면 건너뛰기
            continue

        dest_name = f"{image_name}__{aug_name}.png"
        dest_path = dest_dir / dest_name

        shutil.copy2(p, dest_path)
        count += 1

    print(f"[COLLECT] overlay 이미지 {count}개를 모았습니다 → {dest_dir}")


# ------------------ measure.py 실행 / 에러 분류 ----------------------

def classify_error_stage(stdout: str, stderr: str) -> str:
    """
    measure.py의 stdout/stderr를 대략적으로 파싱해서
    에러 단계(stage)를 문자열로 분류.
    """
    text = (stdout or "") + "\n" + (stderr or "")
    text_lower = text.lower()

    if "meta json not found" in text_lower:
        return "meta_not_found"
    if "pixel_scale_um_x" in text_lower and "not found" in text_lower:
        return "meta_field_missing"
    if "template not found" in text_lower or "failed to load template image" in text_lower:
        return "template_load_fail"
    if "matchtemplate" in text_lower and "error" in text_lower:
        return "template_match_fail"
    if "error fitting red line" in text_lower:
        return "red_line_fit_fail"
    if "insufficient points for line fitting" in text_lower:
        return "red_line_points_insufficient"
    if "roi has zero area" in text_lower:
        return "invalid_roi"
    if "unable to read image" in text_lower:
        return "input_image_read_fail"
    if "centroid" in text_lower and "none" in text_lower:
        return "centroid_missing"
    if "bottom-most" in text_lower and "none" in text_lower:
        return "bottom_point_missing"

    return "unknown"


def run_measurement(
    image_path: str,
    run_dir: str,
) -> Tuple[str, str, int, int, str]:
    """
    한 장의 (augmented) 이미지에 대해 measure.py 실행 후 상태를 반환.

    Returns:
        status: "ok" | "error" | "no_output" | "empty_csv"
        error_stage: string label
        n_rows: CSV 행 수 (성공 시)
        returncode: measure.py의 반환 코드
        log_path: stdout/stderr를 저장한 로그 파일 경로
    """
    ensure_dir(run_dir)

    # grounding_template를 run_dir로 복사
    if not os.path.exists(GROUNDING_TEMPLATE_SRC):
        raise FileNotFoundError(f"GROUNDING_TEMPLATE_SRC 없음: {GROUNDING_TEMPLATE_SRC}")
    dst_template = Path(run_dir) / "grounding_template.png"
    if not dst_template.exists():
        shutil.copy2(GROUNDING_TEMPLATE_SRC, dst_template)

    # 커맨드 구성
    cmd = MEASURE_CMD_TEMPLATE.format(
        script=MEASURE_SCRIPT_PATH,
        image_path=image_path,
        meta_root=META_ROOT,
        run_dir=run_dir,
    )

    print(f"[RUN] {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=str(Path(MEASURE_SCRIPT_PATH).parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    returncode = result.returncode

    # 로그 저장
    log_path = os.path.join(run_dir, "measure_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== CMD ===\n")
        f.write(cmd + "\n\n")
        f.write("=== STDOUT ===\n")
        f.write(stdout + "\n\n")
        f.write("=== STDERR ===\n")
        f.write(stderr + "\n")

    overlay_path = os.path.join(run_dir, OVERLAY_NAME)
    csv_path = os.path.join(run_dir, CSV_NAME)

    if returncode != 0:
        error_stage = classify_error_stage(stdout, stderr)
        print(f"[ERROR] returncode={returncode}, stage={error_stage}")
        return "error", error_stage, 0, returncode, log_path

    # returncode == 0 인데 output 파일이 없는 경우
    if not os.path.exists(overlay_path) or not os.path.exists(csv_path):
        error_stage = "no_output_files"
        print(f"[ERROR] overlay 또는 CSV가 없음: {overlay_path}, {csv_path}")
        return "no_output", error_stage, 0, returncode, log_path

    # CSV 열어서 행 수 확인
    try:
        df = pd.read_csv(csv_path)
        n_rows = len(df)
    except Exception as e:
        error_stage = f"csv_read_fail({e.__class__.__name__})"
        print(f"[ERROR] CSV 읽기 실패: {csv_path}, {e}")
        return "error", error_stage, 0, returncode, log_path

    if n_rows == 0:
        error_stage = "empty_csv"
        print(f"[WARN] CSV에 행이 없음: {csv_path}")
        return "empty_csv", error_stage, 0, returncode, log_path

    # 여기까지 왔으면 정상 동작으로 판단
    return "ok", "none", n_rows, returncode, log_path


# ------------------ 메인 워크플로우 ----------------------

def main():
    # 랜덤 seed 고정
    np.random.seed(RANDOM_SEED)

    # 실험 폴더 생성
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(EXPERIMENT_ROOT, f"transfer_eval_{ts}")
    images_out_dir = os.path.join(experiment_dir, "augmented_images")
    runs_root_dir = os.path.join(experiment_dir, "runs")
    ensure_dir(images_out_dir)
    ensure_dir(runs_root_dir)

    print(f"[INFO] 실험 폴더: {experiment_dir}")

    # 입력 이미지 목록
    image_paths = list_images(INPUT_IMAGE_DIR)
    if not image_paths:
        print(f"[WARN] 입력 폴더에 이미지가 없습니다: {INPUT_IMAGE_DIR}")
        return

    summary_rows: List[Dict[str, object]] = []

    for img_path in image_paths:
        img_name = os.path.basename(img_path)   # 예: C2024_..._gray.tif
        print(f"\n[IMAGE] {img_name}")

        try:
            gray = load_gray(img_path)
        except Exception as e:
            print(f"[ERROR] 원본 이미지 읽기 실패: {img_path}, {e}")
            summary_rows.append({
                "image_name": img_name,
                "image_path": img_path,
                "aug": "orig",
                "aug_image_path": "",
                "status": "input_read_error",
                "error_stage": "input_read_error",
                "n_rows": 0,
                "returncode": -1,
                "run_dir": "",
                "log_path": "",
            })
            continue

        # deterministic + random augmentation 생성
        variants = generate_deterministic_variants(gray)
        rand_variants = generate_random_variants(gray, N_RANDOM_AUG)
        variants.update(rand_variants)

        # 각 augmentation에 대해 실행
        for aug_name, aug_img in variants.items():
            # 파일명은 "원본파일이름 + '__' + aug_name + '.png'"
            # 예: C2024_..._gray.tif__rot+1.0.png
            #  → meta_utils.find_meta_path에서 "_gray.tif" 기준으로 잘 잘려서
            #    원본과 같은 meta JSON을 사용하게 됨.
            aug_filename = f"{img_name}__{aug_name}.png"
            aug_img_path = os.path.join(images_out_dir, aug_filename)

            print(f"[CASE] {aug_filename}")
            save_gray(aug_img_path, aug_img)

            # run_dir 설정: runs/<원본파일이름>/<aug_name>/
            stem = Path(img_name).name  # 전체 파일명 그대로 사용
            run_dir = os.path.join(runs_root_dir, stem, aug_name)

            status, error_stage, n_rows, rc, log_path = run_measurement(
                image_path=aug_img_path,
                run_dir=run_dir,
            )

            summary_rows.append({
                "image_name": img_name,
                "image_path": img_path,
                "aug": aug_name,
                "aug_image_path": aug_img_path,
                "status": status,             # ok / error / no_output / empty_csv / input_read_error
                "error_stage": error_stage,   # template_load_fail / red_line_fit_fail / ...
                "n_rows": n_rows,             # CSV 행 수 (성공 시)
                "returncode": rc,
                "run_dir": run_dir,
                "log_path": log_path,
            })

    # summary DataFrame 저장
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(experiment_dir, "transfer_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    # 간단한 집계 출력
    total_cases = len(summary_df)
    ok_cases = (summary_df["status"] == "ok").sum()
    print("\n================ SUMMARY ================")
    print(f"Total cases : {total_cases}")
    print(f"OK cases    : {ok_cases}")
    if total_cases > 0:
        print(f"Success rate: {ok_cases / total_cases * 100:.2f}%")
        print("\nStatus breakdown:")
        print(summary_df["status"].value_counts())
        print("\nError stage breakdown (에러/empty/no_output만):")
        mask_err = summary_df["status"].isin(["error", "no_output", "empty_csv"])
        if mask_err.any():
            print(summary_df.loc[mask_err, "error_stage"].value_counts())
        else:
            print("에러가 발생하지 않았습니다. 🎉")

    print(f"\n[RESULT] transfer_summary.csv 저장 위치: {summary_csv_path}")
    print("[DONE] 전이 성능(신뢰성) 평가 완료.")
    
    # 모든 overlay.png를 한 폴더로 모으기
    collect_all_overlays(experiment_dir)



if __name__ == "__main__":
    main()
