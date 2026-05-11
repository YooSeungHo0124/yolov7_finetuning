#!/usr/bin/env python3
"""
YOLOv7 Image Annotation Tool (이미지 폴더 기반)

사용법:
  source /data/venv/finetuning/bin/activate
  cd /data/yolov7_finetuning
  python 02_annotation_tool.py --images ./images --split train
  python 02_annotation_tool.py --split val --classes "fire,smoke"

단축키:
  q : -10 image
  w : -1 image
  e : +1 image
  r : +10 image
  s : save & next
  n : skip (라벨 없이 다음으로)
  d : 선택된 bbox 삭제
  c : 모든 bbox 삭제
  a : 모든 bbox 수락 (model bbox 그대로 save & next)
  1-9 : 클래스 선택
  ESC / x : 종료
"""

import os
import sys
import glob
import argparse
import cv2
import torch
import numpy as np
import zipfile
import tempfile
import getpass
import subprocess
import shutil
from pathlib import Path

# numpy._core 호환성 패치 (numpy 1.23+ with older weight files)
sys.modules['numpy._core'] = np
sys.modules['numpy._core.multiarray'] = np.core.multiarray

YOLOV7_DIR = os.path.join(os.path.dirname(__file__), "yolov7")
DEFAULT_WEIGHTS = "/data/yolo7_fire_model/runs/train_20260430/weights/yolov7-fire-v5.pt"

sys.path.insert(0, YOLOV7_DIR)
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox


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

    내부에 .pt가 있으면 그걸 우선,
    없으면 첫 번째 일반 파일 (확장자 없는 배포 포맷 대응).
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


def hsv_to_bgr(h, s=0.8, v=0.9):
    """HSV to BGR color conversion"""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(b * 255), int(g * 255), int(r * 255))


def load_model(weights_path, device_str="cpu", weight_type="pt"):
    """Load YOLOv7 model

    weight_type:
      'pt'     : 표준 PyTorch (attempt_load)
      'traced' : TorchScript traced (torch.jit.load)
      'detect' : detect 레이어만 → 추론 불가 (호출 안 됨)
    """
    device = torch.device(device_str if device_str != "cpu" else "cpu")
    if device_str != "cpu" and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_str}" if device_str.isdigit() else "cuda")

    if weight_type == 'traced':
        model = torch.jit.load(weights_path, map_location=device)
        model.eval()
        return model, device

    # 표준 .pt
    model = attempt_load(weights_path, map_location=device)
    model.eval()
    if device.type == "cuda":
        model.half()
    return model, device


def find_companion_detect(traced_orig_path):
    """_t/_traced 원본 경로에서 대응하는 _d/_detect 파일 경로 반환 (없으면 None)"""
    parent = os.path.dirname(traced_orig_path)
    base = os.path.basename(traced_orig_path)
    if base.endswith('_t'):
        d_base = base[:-1] + 'd'
    elif base.endswith('_traced'):
        d_base = base[:-6] + 'detect'
    else:
        return None
    d_path = os.path.join(parent, d_base)
    return d_path if os.path.exists(d_path) else None


def extract_classes_from_detect(detect_orig_path, password=None):
    """_d zip 또는 detect .pt에서 클래스 목록 추출"""
    path = detect_orig_path
    if zipfile.is_zipfile(detect_orig_path):
        path = extract_zip(detect_orig_path, password)
        if path is None:
            return None
    try:
        layer = torch.load(path, map_location="cpu", weights_only=False)
        if hasattr(layer, 'names'):
            return list(layer.names)
    except Exception as e:
        print(f"Warning: Failed to load detect layer: {e}")
    return None


def resolve_classes(weights_path, override_str=None, weight_type="pt"):
    """클래스 목록 결정 (traced/detect는 호출 전 처리됨)"""
    if override_str and override_str != "auto":
        return [c.strip() for c in override_str.split(",")]

    # 표준 .pt에서 추출
    try:
        model, _ = load_model(weights_path, device_str="cpu", weight_type="pt")
        if hasattr(model, 'names'):
            return list(model.names)
        elif hasattr(model, 'module') and hasattr(model.module, 'names'):
            return list(model.module.names)
    except Exception as e:
        print(f"Warning: Failed to extract class names: {e}")
    return None


def run_inference(model, device, img_bgr, conf_thres=0.1):
    """
    Run YOLOv7 inference
    Supports both standard and TorchScript models
    """
    h0, w0 = img_bgr.shape[:2]
    img = letterbox(img_bgr, 640, stride=32, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()
    img = torch.from_numpy(img).to(device)
    img = img.half() if device.type == "cuda" else img.float()
    img /= 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        # TorchScript 모델은 리스트 형태로 반환, 표준 모델은 텐서로 반환
        output = model(img)
        if isinstance(output, (list, tuple)):
            pred = output[0]
        else:
            pred = output

    pred = non_max_suppression(pred, conf_thres, 0.45)
    results = []
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (h0, w0)).round()
            for *xyxy, conf, cls in det:
                results.append({
                    "x1": int(xyxy[0]), "y1": int(xyxy[1]),
                    "x2": int(xyxy[2]), "y2": int(xyxy[3]),
                    "conf": float(conf), "cls": int(cls),
                })
    return results


def save_annotation(frame_bgr, boxes, output_dir, split, filename):
    """
    이미지와 라벨 저장 (YOLO 정규화 좌표)
    """
    img_dir = os.path.join(output_dir, "images", split)
    label_dir = os.path.join(output_dir, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # 이미지 저장
    img_path = os.path.join(img_dir, filename)
    cv2.imwrite(img_path, frame_bgr)

    # 라벨 저장 (YOLO format: class cx cy w h 정규화)
    label_name = os.path.splitext(filename)[0] + ".txt"
    label_path = os.path.join(label_dir, label_name)
    h, w = frame_bgr.shape[:2]

    with open(label_path, "w") as f:
        for b in boxes:
            bx = ((b["x1"] + b["x2"]) / 2) / w
            by = ((b["y1"] + b["y2"]) / 2) / h
            bw = (b["x2"] - b["x1"]) / w
            bh = (b["y2"] - b["y1"]) / h
            f.write(f"{b['cls']} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")

    return img_path, label_path


def find_box_at(boxes, x, y):
    """(x,y) 위치에 있는 bbox 인덱스 반환"""
    for i, b in enumerate(boxes):
        if b["x1"] <= x <= b["x2"] and b["y1"] <= y <= b["y2"]:
            return i
    return -1


def draw_frame(display, boxes, selected_idx, current_cls, class_names, colors_bgr,
               img_idx, total_imgs, filename, split):
    """프레임 위에 bbox, 상태 정보 그리기"""
    vis = display.copy()
    h, w = vis.shape[:2]

    # bbox 그리기
    for i, b in enumerate(boxes):
        cls_i = b["cls"] % len(colors_bgr)  # 클래스 수 초과 방지
        color = colors_bgr[cls_i]
        thick = 3 if i == selected_idx else 2
        if i == selected_idx:
            color = (0, 255, 255)  # 선택 = 노란색
        cv2.rectangle(vis, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, thick)
        label = class_names[b["cls"]] if b["cls"] < len(class_names) else f"cls{b['cls']}"
        if "conf" in b:
            label += f" {b['conf']:.2f}"
        cv2.putText(vis, label, (b["x1"], b["y1"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 상태 바
    bar_h = 40
    cv2.rectangle(vis, (0, 0), (w, bar_h), (40, 40, 40), -1)

    cur_cls_safe = current_cls % len(colors_bgr)
    cls_name = class_names[current_cls] if current_cls < len(class_names) else f"cls{current_cls}"
    cls_color = colors_bgr[cur_cls_safe]

    info = f"Image {img_idx+1}/{total_imgs} | {filename} | {len(boxes)} boxes | Class: {cls_name} | Split: {split}"
    cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # 클래스 색상 표시
    cv2.circle(vis, (w - 20, 20), 8, cls_color, -1)

    # 단축키 안내
    help_y = h - 10
    help_text = "q/w/e/r:nav | s:save | a:accept | n:skip | d:del | c:clear | 1-9:class | drag:draw | ESC:quit"
    cv2.putText(vis, help_text, (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    return vis


def main():
    parser = argparse.ArgumentParser(description="YOLOv7 Image Annotation Tool")
    parser.add_argument("--images", default="./images", help="Input images folder")
    parser.add_argument("--output", default="./dataset", help="Output dataset folder")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLOv7 weights (.pt or .zip)")
    parser.add_argument("--classes", default="auto", help='Classes (comma-separated or "auto")')
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Train/Val split")
    parser.add_argument("--conf", type=float, default=0.1, help="Detection confidence threshold")
    parser.add_argument("--width", type=int, default=1440, help="Display width")
    parser.add_argument("--device", default="cpu", help="Device (cpu/0/1/...)")
    parser.add_argument("--password", default=None, help="Zip file password (if needed)")
    parser.add_argument("--skip-labeled", action="store_true", default=True, help="Skip already labeled images")
    args = parser.parse_args()

    # 원본 파일명으로 weight 타입 판단 (추출 후에도 변하지 않음)
    weight_type = detect_weight_type(args.weights)
    print(f"Weight type: {weight_type}")

    # _d/_detect는 클래스 정보만 → 추론 불가 → 거부
    if weight_type == 'detect':
        print("ERROR: _d/_detect 파일은 클래스 정보만 포함하고 추론에는 사용할 수 없습니다.")
        print("       대신 _t/_traced 파일이나 .pt 파일을 사용하세요.")
        return 1

    # zip 추출이 필요한 타입 (_t/_traced)
    weights_path = args.weights
    if weight_type == 'traced':
        if not zipfile.is_zipfile(args.weights):
            print(f"✗ Expected zip file but '{args.weights}' is not a valid zip")
            return 1
        print(f"Extracting zip...")
        extracted = extract_zip(args.weights, args.password)
        if extracted is None:
            return 1
        weights_path = extracted
        print(f"✓ Extracted to: {weights_path}\n")

    # 클래스 목록 결정
    print("Loading class names...")
    if args.classes and args.classes != "auto":
        # 명시적 지정
        class_names = [c.strip() for c in args.classes.split(",")]
    elif weight_type == 'traced':
        # traced 모델: _d 파일에서 자동 추출 시도
        d_path = find_companion_detect(args.weights)
        if d_path:
            print(f"  Found companion detect file: {os.path.basename(d_path)}")
            class_names = extract_classes_from_detect(d_path, args.password)
        else:
            class_names = None

        if class_names is None:
            print(f"ERROR: traced 모델의 클래스를 자동 추출할 수 없습니다.")
            print(f"  1. --classes \"class1,class2\" 로 명시하거나")
            print(f"  2. 대응하는 _d 파일을 같은 경로에 두세요 (자동 탐색됨)")
            return 1
    else:
        # .pt 표준 모델
        class_names = resolve_classes(weights_path, args.classes, weight_type)
        if class_names is None:
            print(f"ERROR: 클래스 이름을 추출하지 못했습니다.")
            print(f"  --classes \"class1,class2\" 로 명시하세요.")
            return 1

    print(f"Classes ({len(class_names)}): {class_names}\n")

    # 색상 생성: 최소 32개 확보 (traced 모델처럼 nc 모를 때 IndexError 방지)
    n_colors = max(len(class_names), 32)
    colors_bgr = [hsv_to_bgr(i / n_colors) for i in range(n_colors)]

    # 이미지 파일 수집
    image_patterns = [
        os.path.join(args.images, "*.jpg"),
        os.path.join(args.images, "*.JPG"),
        os.path.join(args.images, "*.jpeg"),
        os.path.join(args.images, "*.JPEG"),
        os.path.join(args.images, "*.png"),
        os.path.join(args.images, "*.PNG"),
    ]
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(pattern))
    image_files = sorted(set(image_files))

    if not image_files:
        print(f"No images found in {args.images}")
        return 1

    print(f"Found {len(image_files)} images")

    # 라벨된 이미지 제외 (skip-labeled)
    if args.skip_labeled:
        labeled_files = set()
        label_dir = os.path.join(args.output, "labels", args.split)
        if os.path.exists(label_dir):
            for txt_file in glob.glob(os.path.join(label_dir, "*.txt")):
                labeled_files.add(os.path.splitext(os.path.basename(txt_file))[0])

        image_files = [f for f in image_files
                      if os.path.splitext(os.path.basename(f))[0] not in labeled_files]
        print(f"After excluding labeled: {len(image_files)} images\n")

    if not image_files:
        print("All images already labeled.")
        return 0

    # 모델 로드
    print(f"Loading model from {weights_path}...")
    model, device = load_model(weights_path, args.device, weight_type)
    print("Model loaded.\n")

    # UI 초기화
    window_name = "YOLOv7 Annotation Tool"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, int(args.width * 0.75))

    # 상태 변수
    img_idx = 0
    current_cls = 0
    boxes = []
    selected = -1
    drawing = False
    draw_start = None
    draw_end = None
    scale = 1.0  # 초기값

    def to_orig(b):
        """display 좌표 → 원본 좌표"""
        return {
            "x1": int(b["x1"] / scale), "y1": int(b["y1"] / scale),
            "x2": int(b["x2"] / scale), "y2": int(b["y2"] / scale),
            "cls": b["cls"]
        }

    def to_disp(b):
        """원본 좌표 → display 좌표"""
        r = {
            "x1": int(b["x1"] * scale), "y1": int(b["y1"] * scale),
            "x2": int(b["x2"] * scale), "y2": int(b["y2"] * scale),
            "cls": b["cls"]
        }
        if "conf" in b:
            r["conf"] = b["conf"]
        return r

    def load_image(idx):
        """이미지 로드 및 추론"""
        nonlocal img_idx, boxes, selected, scale, image_files
        img_idx = max(0, min(len(image_files) - 1, idx))
        img_path = image_files[img_idx]
        img_orig = cv2.imread(img_path)
        if img_orig is None:
            print(f"Failed to read {img_path}")
            return None, None

        orig_h, orig_w = img_orig.shape[:2]
        scale = args.width / orig_w
        display_h = int(orig_h * scale)
        img_disp = cv2.resize(img_orig, (args.width, display_h))

        # 추론
        det = run_inference(model, device, img_orig, args.conf)
        boxes = [to_disp(b) for b in det]
        selected = -1

        return img_orig, img_disp

    def mouse_cb(event, x, y, flags, param):
        nonlocal drawing, draw_start, draw_end, selected

        if event == cv2.EVENT_LBUTTONDOWN:
            idx = find_box_at(boxes, x, y)
            if idx >= 0:
                selected = idx
                drawing = False
            else:
                drawing = True
                draw_start = (x, y)
                draw_end = (x, y)
                selected = -1

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                draw_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if drawing and draw_start:
                x1 = min(draw_start[0], x)
                y1 = min(draw_start[1], y)
                x2 = max(draw_start[0], x)
                y2 = max(draw_start[1], y)
                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cls": current_cls})
                    selected = len(boxes) - 1
                drawing = False
                draw_start = None
                draw_end = None

    cv2.setMouseCallback(window_name, mouse_cb)

    # 첫 이미지 로드
    img_orig, img_disp = load_image(0)
    if img_orig is None:
        print("Failed to load first image")
        return 1

    print(f"Starting annotation on {len(image_files)} images...")
    print(f"Press 'h' for help or ESC to quit\n")

    # 메인 루프
    while True:
        # 그리는 중인 bbox
        draw_box = None
        if drawing and draw_start and draw_end:
            x1 = min(draw_start[0], draw_end[0])
            y1 = min(draw_start[1], draw_end[1])
            x2 = max(draw_start[0], draw_end[0])
            y2 = max(draw_start[1], draw_end[1])
            draw_box = (x1, y1, x2, y2)

        filename = os.path.basename(image_files[img_idx])
        vis = draw_frame(img_disp, boxes, selected, current_cls, class_names,
                        colors_bgr, img_idx, len(image_files), filename, args.split)

        # 그리는 중인 박스 시각화
        if draw_box:
            x1, y1, x2, y2 = draw_box
            color = colors_bgr[current_cls]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        cv2.imshow(window_name, vis)

        key = cv2.waitKey(30) & 0xFF

        if key == 27 or key == ord('x'):  # ESC or x
            break

        elif key == ord('q'):  # -10
            img_orig, img_disp = load_image(img_idx - 10)

        elif key == ord('w'):  # -1
            img_orig, img_disp = load_image(img_idx - 1)

        elif key == ord('e'):  # +1
            img_orig, img_disp = load_image(img_idx + 1)

        elif key == ord('r'):  # +10
            img_orig, img_disp = load_image(img_idx + 10)

        elif key == ord('s'):  # save & next
            if img_orig is not None:
                orig_boxes = [to_orig(b) for b in boxes]
                filename = os.path.basename(image_files[img_idx])
                img_path, label_path = save_annotation(img_orig, orig_boxes,
                                                       args.output, args.split, filename)
                print(f"✓ Saved {filename} ({len(orig_boxes)} boxes)")
                img_orig, img_disp = load_image(img_idx + 1)

        elif key == ord('n'):  # skip without saving
            img_orig, img_disp = load_image(img_idx + 1)

        elif key == ord('a'):  # accept all & next
            if img_orig is not None and boxes:
                orig_boxes = [to_orig(b) for b in boxes]
                filename = os.path.basename(image_files[img_idx])
                img_path, label_path = save_annotation(img_orig, orig_boxes,
                                                       args.output, args.split, filename)
                print(f"✓ Accepted {filename} ({len(orig_boxes)} boxes)")
            img_orig, img_disp = load_image(img_idx + 1)

        elif key == ord('d'):  # delete selected
            if 0 <= selected < len(boxes):
                del boxes[selected]
                selected = -1

        elif key == ord('c'):  # clear all
            boxes.clear()
            selected = -1

        # 클래스 선택 (1-9)
        elif ord('1') <= key <= ord('9'):
            cls_idx = key - ord('1')
            if cls_idx < len(class_names):
                current_cls = cls_idx

    cv2.destroyAllWindows()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
