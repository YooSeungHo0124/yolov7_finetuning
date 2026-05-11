# YOLOv7 Fine-tuning Workspace

오탐(FP)·미탐(FN) 이미지로 점진적으로 fine-tuning하는 자체 포함 워크스페이스.
YOLOv7 코드는 git clone으로 내부에 포함됨.

---

## 설치 (최초 1회)

```bash
python3 -m venv /data/venv/finetuning
source /data/venv/finetuning/bin/activate
pip install -r requirements.txt
```

---

## 빠른 시작

### Step 1. Weight 클래스 확인

```bash
cd /data/yolov7_finetuning
source /data/venv/finetuning/bin/activate

# 표준 .pt 파일
python 01_inspect_weights.py \
  --weights /data/yolo7_fire_model/runs/train_20260430/weights/yolov7-fire-v5.pt

# 배포용 detect zip (_d 파일, 확장자 없음)
python 01_inspect_weights.py \
  --weights /usr/local/share/opencv4/quality/tx/ws/hm_d \
  --password datonai1203
```

**지원 형식** (파일명 기반 자동 판별):

| 형식 | 지원 | 비고 |
|------|------|------|
| `.pt` | ✅ | 표준 PyTorch — 전체 정보 |
| `_d` / `_detect` zip | ✅ | 배포용 detect 레이어 — 클래스 정보만 |
| `_t` / `_traced` zip | ❌ | 이 스크립트에서는 미지원 (annotation_tool에서는 `--classes` 명시 시 사용 가능) |

**출력**: 클래스명(`fire`, `smoke`), 파라미터 수 등

---

### Step 2. 이미지 배치

라벨링할 오탐/미탐 이미지를 `./images/` 폴더에 복사:

```bash
cp /path/to/fp_fn_images/*.jpg ./images/
```

---

### Step 3. 라벨링 (train / val 각각)

```bash
# 기본 (.pt 파일 — 클래스 자동 인식)
python 02_annotation_tool.py --split train
python 02_annotation_tool.py --split val

# 배포용 가중치 (zip + 비밀번호)
python 02_annotation_tool.py --split train \
  --weights /usr/local/share/opencv4/quality/tx/ws/sh_t \
  --password datonai1203
```

**UI 단축키**

| 카테고리 | 키 | 동작 |
|----------|---|------|
| 네비게이션 | `q` / `w` | -10 / -1 image |
|           | `e` / `r` | +1 / +10 image |
| 그리기    | 마우스 drag | bbox 생성 |
| 클래스    | `1` ~ `9` | 클래스 선택 (`1`=fire, `2`=smoke ...) |
| 저장/스킵 | `s` | 저장 후 다음 |
|           | `a` | 모델 결과 그대로 수락 |
|           | `n` | 스킵 (라벨 없이) |
| 수정      | `d` | 선택 박스 삭제 |
|           | `c` | 전체 박스 삭제 |
| 종료      | `ESC` / `x` | 저장 후 종료 |

> **참고**: `.pt` 가중치는 클래스를 자동 인식합니다.
> TorchScript/detect/traced 모델은 `--classes "fire,smoke"`를 명시해야 합니다.

**출력 위치**:
- 이미지: `dataset/images/{train,val}/`
- 라벨: `dataset/labels/{train,val}/*.txt` (YOLO format)

---

### Step 4. 학습 (4단계 강도)

기본 동작: **tmux 세션 자동 생성** (train + tensorboard 창). TensorBoard는 `http://localhost:6006`에서 접근.

```bash
# 단일 레벨 (tmux 자동)
python 03_finetune.py --level 3

# 4단계 모두 순차 실행 (모두 동일 base weight에서 시작)
python 03_finetune.py --all

# tmux 없이 foreground 실행
python 03_finetune.py --level 1 --no-tmux

# epochs override (기본 모든 레벨 200)
python 03_finetune.py --level 2 --epochs 50
```

**레벨 설정** (epochs는 모두 200 통일, freeze/lr0/hyp만 차이):

| Level | freeze | lr0    | epochs | 특성 |
|-------|--------|--------|--------|------|
| 1     | 75     | 0.0001 | 200    | 헤드만 미세조정 (가장 보수적) |
| 2     | 50     | 0.0003 | 200    | backbone 일부 고정 |
| 3     | 0      | 0.001  | 200    | 전체 학습, 보수적 LR |
| 4     | 0      | 0.005  | 200    | 전체 학습, 공격적 |

**실행 후 출력 예시**:

```
✓ tmux session started: yolov7_l3_20260508_143022
Attach:  tmux attach -t yolov7_l3_20260508_143022
TensorBoard: http://localhost:6006
```

**tmux 단축키**: `Ctrl+b` → `n` (창 전환) · `Ctrl+b` → `d` (detach)

**결과 위치**: `runs/level{N}_{timestamp}/weights/best.pt`

> **`--all` 동작**: L1 → L2 → L3 → L4 순차 실행. 각 레벨은 모두 **동일 base weight**(`--weights`)에서 시작 (cascade 아님). 한 레벨 실패 시 즉시 중단.

---

## 주요 옵션

```bash
# 배치 크기 조정 (메모리 부족 시)
python 03_finetune.py --level 1 --batch-size 8

# 다른 weight에서 시작
python 03_finetune.py --level 2 \
  --weights /data/fire_validation/weights/yolov7-fire-v3.pt

# CPU에서 학습 (GPU 없을 시)
python 03_finetune.py --level 1 --device cpu

# 이미지 크기 변경
python 03_finetune.py --level 1 --img-size 480

# 학습된 모델 다시 검사
python 01_inspect_weights.py --weights ./runs/level1_*/weights/best.pt
```

---

## 디렉토리 구조

```
/data/yolov7_finetuning/
├── 01_inspect_weights.py       # weight 검사
├── 02_annotation_tool.py       # 라벨링 도구
├── 03_finetune.py              # 학습 wrapper
├── yolov7/                     # git clone (WongKinYiu/yolov7)
├── hyp/hyp.l{1-4}_*.yaml       # 4단계 하이퍼파라미터
├── images/                     # 입력 이미지 (사용자 배치)
├── dataset/                    # 라벨링 결과
│   ├── images/{train,val}/
│   └── labels/{train,val}/
├── runs/                       # 학습 결과
├── data.yaml                   # 자동 생성 (학습 데이터 정의)
└── requirements.txt
```

---

## 문제 해결

**GPU 인식 안 됨**

```bash
python -c "import torch; print(torch.cuda.is_available())"
# False면 CUDA 버전 맞게 torch 재설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**메모리 부족**

`--batch-size 4` 또는 `--img-size 480`으로 줄이기

**annotation_tool.py 창 안 뜸 (SSH)**

X11 forwarding 필요: `ssh -X ...`

**dataset이 비어있음**

`02_annotation_tool.py`를 실행했는지 확인
