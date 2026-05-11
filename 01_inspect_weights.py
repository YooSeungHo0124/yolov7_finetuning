#!/usr/bin/env python3
"""
YOLOv7 weight 검사 도구
클래스 수, 클래스명, 아키텍처 정보 출력

지원 형식:
- .pt           : 표준 PyTorch 가중치 (전체 정보)
- _d / _detect  : 배포용 detect 레이어 zip (클래스 정보만)
- _t / _traced  : 배포용 traced 모델 zip (현재 미지원)
"""

import sys
import os
import argparse
import torch
import numpy
import zipfile
import tempfile
import getpass
import subprocess
import shutil

# numpy._core 호환성 패치 (numpy 1.23+ with older weight files)
sys.modules['numpy._core'] = numpy
sys.modules['numpy._core.multiarray'] = numpy.core.multiarray

YOLOV7_DIR = os.path.join(os.path.dirname(__file__), "yolov7")
sys.path.insert(0, YOLOV7_DIR)

from models.experimental import attempt_load


def detect_weight_type(path):
    """원본 파일명에서 weight 종류 판단

    Returns: 'pt' | 'detect' | 'traced'
    """
    base = os.path.basename(path)
    if base.endswith('_d') or base.endswith('_detect'):
        return 'detect'
    if base.endswith('_t') or base.endswith('_traced'):
        return 'traced'
    return 'pt'


def extract_zip(zip_path, password=None):
    """zip을 임시 디렉토리에 추출하고 내부 weight 파일 경로 반환

    시스템 `unzip` 명령 사용 (Python zipfile은 password-protected zip이
    매우 느림 — 큰 파일에서 사실상 멈춘 것처럼 보임).
    fallback으로 zipfile 모듈 사용.
    """
    if not zipfile.is_zipfile(zip_path):
        return None

    if password is None:
        password = getpass.getpass("Enter zip password: ")

    temp_dir = tempfile.mkdtemp()

    unzip_bin = shutil.which("unzip")
    if unzip_bin:
        try:
            cmd = [unzip_bin, "-o", "-q", "-P", password, zip_path, "-d", temp_dir]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print(f"✗ unzip failed (code {result.returncode}): {result.stderr.strip()}")
                return None
        except subprocess.TimeoutExpired:
            print(f"✗ unzip timeout (>120s)")
            return None
        except Exception as e:
            print(f"✗ unzip error: {e}")
            return None
    else:
        # fallback (느림)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir, pwd=password.encode() if password else None)
        except Exception as e:
            print(f"✗ Extraction failed: {e}")
            return None

    extracted_files = []
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            extracted_files.append(os.path.join(root, file))

    if not extracted_files:
        print(f"✗ No files extracted from zip")
        return None

    for f in extracted_files:
        if f.endswith('.pt'):
            print(f"  Inner file: {os.path.basename(f)}")
            return f

    print(f"  Inner files: {[os.path.basename(f) for f in extracted_files]}")
    print(f"  Using: {os.path.basename(extracted_files[0])}")
    return extracted_files[0]


def main():
    parser = argparse.ArgumentParser(description="Inspect YOLOv7 weight file")
    parser.add_argument("--weights", required=True, help="Weight file path (.pt / _d / _t)")
    parser.add_argument("--device", default="cpu", help="Device (cpu/0/1/...)")
    parser.add_argument("--password", default=None, help="Zip file password (if needed)")
    args = parser.parse_args()

    print(f"Loading weight: {args.weights}")

    # 원본 파일명으로 타입 판단 (추출 후에도 변하지 않음)
    weight_type = detect_weight_type(args.weights)
    print(f"Detected type: {weight_type}\n")

    # zip 추출이 필요한 타입
    weights_path = args.weights
    if weight_type in ('detect', 'traced'):
        if not zipfile.is_zipfile(args.weights):
            print(f"✗ Expected zip file but '{args.weights}' is not a valid zip")
            return 1
        print(f"Extracting zip...")
        extracted = extract_zip(args.weights, args.password)
        if extracted is None:
            return 1
        weights_path = extracted
        print(f"✓ Extracted to: {weights_path}\n")

    # detect 레이어 (클래스 정보만)
    if weight_type == 'detect':
        try:
            detect_layer = torch.load(weights_path, map_location=args.device, weights_only=False)
            print(f"✓ Detect layer loaded\n")

            if hasattr(detect_layer, 'names'):
                names = detect_layer.names
                nc = len(names)
                print(f"Classes (nc={nc}):")
                for i, name in enumerate(names):
                    print(f"  [{i}] {name}")
                print(f"\nNote: This is a detect-layer-only file (class info only)")
                print(f"For inference, use the corresponding _t/_traced file or .pt")
                return 0
            else:
                print(f"✗ No 'names' attribute in detect layer")
                return 1
        except Exception as e:
            print(f"✗ Failed to load detect layer: {e}")
            return 1

    # traced 모델 (현재 미지원)
    if weight_type == 'traced':
        print(f"✗ Traced model (_t/_traced) inspection is not supported.")
        print(f"  Use the corresponding _d/_detect file for class information,")
        print(f"  or use a standard .pt weight for full inspection.")
        return 1

    # 표준 .pt
    try:
        model = attempt_load(weights_path, map_location=args.device)
        print(f"✓ Weight loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load weight: {e}")
        return 1

    names = model.module.names if hasattr(model, "module") else model.names
    nc = len(names)

    print(f"Classes (nc={nc}):")
    for i, name in enumerate(names):
        print(f"  [{i}] {name}")

    n_modules = sum(1 for _ in model.modules())
    print(f"\nTotal modules: {n_modules}")

    yaml_file = getattr(model, "yaml_file", "unknown")
    if yaml_file != "unknown":
        print(f"YAML config: {yaml_file}")

    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    except Exception:
        print("\n(Parameter info not available for TorchScript model)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
