#!/usr/bin/env python3
"""
YOLOv7 Fine-tuning Wrapper (4-level training intensity)

기본 동작:
- tmux 세션 자동 생성 (train + tensorboard 창)
- TensorBoard 자동 시작 (포트 6006)
- 단일 레벨 또는 4단계 순차 실행

사용법:
  # 단일 레벨
  python 03_finetune.py --level 2

  # 4단계 모두 순차 실행 (모두 동일 base weight에서 시작)
  python 03_finetune.py --all

  # tmux 없이 foreground 실행
  python 03_finetune.py --level 1 --no-tmux

  # epochs override
  python 03_finetune.py --level 1 --epochs 50

Training Levels (모두 epochs=200, freeze/lr0/hyp만 차이):
  1 (minimal)  : --freeze 75, lr0=0.0001  → 헤드만 미세조정
  2 (light)    : --freeze 50, lr0=0.0003  → backbone 일부 고정
  3 (moderate) : --freeze 0,  lr0=0.001   → 전체 학습, 보수적 LR
  4 (heavy)    : --freeze 0,  lr0=0.005   → 전체 학습, 공격적
"""

import os
import sys
import argparse
import subprocess
import shutil
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

# numpy._core 호환성 패치 (numpy 1.23+ with older weight files)
sys.modules['numpy._core'] = np
sys.modules['numpy._core.multiarray'] = np.core.multiarray

SCRIPT_DIR = Path(__file__).parent.resolve()
YOLOV7_DIR = SCRIPT_DIR / "yolov7"
DEFAULT_WEIGHTS = "/data/yolo7_fire_model/runs/train_20260430/weights/yolov7-fire-v5.pt"
VENV_PATH = "/data/venv/finetuning"
TB_PORT = 6006

# 4단계 학습 — epochs는 모두 200으로 통일, freeze/lr0/hyp만 차이
LEVEL_CONFIG = {
    1: {"freeze": 75, "hyp": "hyp.l1_minimal.yaml",  "epochs": 200, "lr0": 0.0001, "desc": "Minimal"},
    2: {"freeze": 50, "hyp": "hyp.l2_light.yaml",    "epochs": 200, "lr0": 0.0003, "desc": "Light"},
    3: {"freeze": 0,  "hyp": "hyp.l3_moderate.yaml", "epochs": 200, "lr0": 0.001,  "desc": "Moderate"},
    4: {"freeze": 0,  "hyp": "hyp.l4_heavy.yaml",    "epochs": 200, "lr0": 0.005,  "desc": "Heavy"},
}


def resolve_classes(weights_path):
    """weight에서 클래스 목록 추출"""
    sys.path.insert(0, str(YOLOV7_DIR))
    try:
        import torch
        from models.experimental import attempt_load
        model = attempt_load(weights_path, map_location="cpu")
        names = model.module.names if hasattr(model, "module") else model.names
        return list(names)
    except Exception as e:
        print(f"Warning: Could not extract classes from weight: {e}")
        return ["fire", "smoke"]


def create_data_yaml(dataset_dir, output_yaml, classes):
    """data.yaml 자동 생성"""
    dataset_dir = Path(dataset_dir).resolve()
    train_path = dataset_dir / "images" / "train"
    val_path = dataset_dir / "images" / "val"

    if not train_path.exists():
        print(f"Error: {train_path} does not exist")
        return False
    if not val_path.exists():
        print(f"Warning: {val_path} does not exist, will create empty val set")

    data = {
        "train": str(train_path),
        "val": str(val_path),
        "nc": len(classes),
        "names": classes,
    }
    with open(output_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"✓ Wrote {output_yaml} (nc={len(classes)}, names={classes})")
    return True


def run_single_level(level, args, run_name):
    """한 레벨의 학습 실행 (foreground subprocess)"""
    level_info = LEVEL_CONFIG[level]

    print(f"\n{'='*60}")
    print(f"YOLOv7 Fine-tuning - Level {level}: {level_info['desc']}")
    print(f"{'='*60}\n")

    # Sanity check: verify LEVEL_CONFIG lr0 matches hyp YAML
    hyp_path = SCRIPT_DIR / "hyp" / level_info["hyp"]
    if hyp_path.exists():
        try:
            with open(hyp_path, 'r') as f:
                hyp_data = yaml.safe_load(f)
            if hyp_data and 'lr0' in hyp_data:
                hyp_lr0 = hyp_data['lr0']
                if abs(hyp_lr0 - level_info['lr0']) > 1e-8:
                    print(f"⚠ Warning: LEVEL_CONFIG lr0={level_info['lr0']} doesn't match hyp YAML lr0={hyp_lr0}")
                    print(f"  Using LEVEL_CONFIG value. Please verify hyp/{level_info['hyp']} is correct.\n")
        except Exception as e:
            print(f"⚠ Warning: Could not verify hyp lr0 ({e})\n")

    epochs = args.epochs if args.epochs else level_info["epochs"]

    print(f"Configuration:")
    print(f"  Level:       {level} ({level_info['desc']})")
    print(f"  Freeze:      {level_info['freeze']}")
    print(f"  Hyp file:    {level_info['hyp']}")
    print(f"  Epochs:      {epochs}")
    print(f"  LR0:         {level_info['lr0']}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Img size:    {args.img_size}")
    print(f"  Device:      {args.device}")
    print(f"  Run name:    {run_name}\n")

    # 클래스
    if args.classes == "auto":
        classes = resolve_classes(args.weights)
    else:
        classes = [c.strip() for c in args.classes.split(",")]

    # data.yaml
    data_yaml = SCRIPT_DIR / "data.yaml"
    if not create_data_yaml(args.dataset, data_yaml, classes):
        return 1

    # hyp 파일
    hyp_path = SCRIPT_DIR / "hyp" / level_info["hyp"]
    if not hyp_path.exists():
        print(f"Error: {hyp_path} not found")
        return 1

    # project 디렉토리
    project_dir = Path(args.project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    # train.py 호출
    cmd = [
        "python", str(YOLOV7_DIR / "train.py"),
        "--weights", str(args.weights),
        "--data", str(data_yaml),
        "--hyp", str(hyp_path),
        "--epochs", str(epochs),
        "--batch-size", str(args.batch_size),
        "--img-size", str(args.img_size), str(args.img_size),
        "--freeze", str(level_info["freeze"]),
        "--device", str(args.device),
        "--project", str(project_dir),
        "--name", run_name,
    ]
    if args.cfg:
        cmd.extend(["--cfg", str(args.cfg)])

    print(f"Command: {' '.join(cmd)}\n")
    print(f"{'='*60}")
    print(f"Starting training (Level {level})...")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=SCRIPT_DIR)

    if result.returncode == 0:
        best_pt = project_dir / run_name / "weights" / "best.pt"
        print(f"\n✓ Level {level} training completed")
        if best_pt.exists():
            print(f"  Best weights: {best_pt}")
    else:
        print(f"\n✗ Level {level} training failed (exit code {result.returncode})")

    return result.returncode


def launch_tmux_session(args):
    """tmux 세션 생성 + tensorboard 창 + train 창 (self-call --no-tmux)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "all" if args.all else f"l{args.level}"
    session = f"yolov7_{suffix}_{timestamp}"

    project_dir = Path(args.project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    # 학습 self-call 명령 구성
    self_args = ["python", str(SCRIPT_DIR / "03_finetune.py"), "--no-tmux"]
    if args.all:
        self_args.append("--all")
    else:
        self_args.extend(["--level", str(args.level)])

    self_args.extend(["--weights", str(args.weights)])
    self_args.extend(["--dataset", str(args.dataset)])
    self_args.extend(["--classes", str(args.classes)])
    self_args.extend(["--img-size", str(args.img_size)])
    self_args.extend(["--batch-size", str(args.batch_size)])
    self_args.extend(["--device", str(args.device)])
    self_args.extend(["--project", str(args.project)])
    if args.epochs:
        self_args.extend(["--epochs", str(args.epochs)])
    if args.cfg:
        self_args.extend(["--cfg", args.cfg])
    if args.name:
        self_args.extend(["--name", args.name])

    train_inner = " ".join(self_args)

    activate = f"source {VENV_PATH}/bin/activate"
    train_cmd = (
        f"cd {SCRIPT_DIR} && {activate} && {train_inner}; "
        f"echo; echo '=== Training session ended (exit=$?). Press any key to close ==='; read"
    )
    tb_cmd = (
        f"cd {SCRIPT_DIR} && {activate} && "
        f"tensorboard --logdir {project_dir} --port {TB_PORT} --bind_all"
    )

    # tmux 세션 생성
    print(f"Creating tmux session: {session}")
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "-x", "220", "-y", "50"],
        check=True,
    )
    subprocess.run(
        ["tmux", "rename-window", "-t", f"{session}:0", "train"],
        check=True,
    )
    subprocess.run(
        ["tmux", "send-keys", "-t", f"{session}:train", train_cmd, "Enter"],
        check=True,
    )

    # tensorboard 창
    subprocess.run(
        ["tmux", "new-window", "-t", session, "-n", "tensorboard"],
        check=True,
    )
    subprocess.run(
        ["tmux", "send-keys", "-t", f"{session}:tensorboard", tb_cmd, "Enter"],
        check=True,
    )

    # train 창 활성화 (attach 시 train 창부터)
    subprocess.run(
        ["tmux", "select-window", "-t", f"{session}:train"],
        check=True,
    )

    print(f"\n{'='*60}")
    print(f"✓ tmux session started: {session}")
    print(f"{'='*60}\n")
    print(f"Attach to session:")
    print(f"  tmux attach -t {session}")
    print(f"\nDetach (inside tmux): Ctrl+b then d")
    print(f"Switch windows:       Ctrl+b then n / p")
    print(f"\nTensorBoard:          http://localhost:{TB_PORT}")
    print(f"Logdir:               {project_dir}")
    print(f"\nList sessions:        tmux ls")
    print(f"Kill this session:    tmux kill-session -t {session}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv7 Fine-tuning Wrapper (4-level + tmux + tensorboard)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 03_finetune.py --level 2                # 단일 레벨, tmux 자동
  python 03_finetune.py --all                    # 4단계 순차, tmux 자동
  python 03_finetune.py --level 1 --no-tmux      # foreground 단일
  python 03_finetune.py --all --no-tmux          # foreground 4단계
  python 03_finetune.py --level 1 --epochs 50    # epochs override
        """,
    )
    parser.add_argument("--level", type=int, choices=[1, 2, 3, 4],
                        help="Single training level (1=minimal ~ 4=heavy)")
    parser.add_argument("--all", action="store_true",
                        help="Run all 4 levels sequentially (each from same base weight)")
    parser.add_argument("--no-tmux", action="store_true",
                        help="Run in foreground without tmux session")

    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="Initial weights path (used for ALL levels)")
    parser.add_argument("--dataset", default="./dataset",
                        help="Dataset root folder (with images/{train,val})")
    parser.add_argument("--classes", default="auto",
                        help='Classes (comma-separated or "auto")')
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="0",
                        help="Device (0/1/... for GPU, cpu for CPU)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs (default: 200 per level)")
    parser.add_argument("--name", default=None,
                        help="Run name override (default: level{N}_{timestamp})")
    parser.add_argument("--project", default="./runs",
                        help="Project directory")
    parser.add_argument("--cfg", default="",
                        help="Model config YAML")

    args = parser.parse_args()

    # validation
    if not args.level and not args.all:
        parser.error("Specify either --level N or --all")
    if args.level and args.all:
        parser.error("Cannot combine --level and --all")

    # tmux 모드 (기본)
    if not args.no_tmux:
        if not shutil.which("tmux"):
            print("Warning: tmux not found in PATH, falling back to foreground mode")
            args.no_tmux = True
        else:
            return launch_tmux_session(args)

    # foreground 모드
    levels = [1, 2, 3, 4] if args.all else [args.level]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, level in enumerate(levels):
        run_name = args.name if args.name and len(levels) == 1 else f"level{level}_{timestamp}"
        print(f"\n{'#'*60}")
        print(f"# [{i+1}/{len(levels)}] Level {level}  →  run name: {run_name}")
        print(f"{'#'*60}")

        rc = run_single_level(level, args, run_name)
        if rc != 0:
            print(f"\n✗ Level {level} failed (exit {rc}). Stopping sequential run.")
            return rc

    print(f"\n{'='*60}")
    print(f"✓ All requested training completed: levels {levels}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
