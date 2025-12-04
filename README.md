# Surface Condition Classification System (Broken & Dirty Level Detection)

이 리포지토리는 제품/표면의 **파손(Broken)** 여부와 **오염도(Dirty Level)**를 7단계로 식별하는 딥러닝 프로젝트입니다. `EfficientNetV2B0`를 기반으로 전이 학습(Transfer Learning)을 수행하며, 단순 분류를 넘어 실제 **유지보수(교체/청소)가 필요한지**를 판단하는 로직을 포함합니다.

## 📂 Directory Structure

```bash
├── README.md
├── main.py              # [실행] 학습 파이프라인 (Freeze -> Fine-tuning)
├── test.py              # [실행] 모델 평가 및 현장 적용 지표 테스트
├── model.py             # [모델] EfficientNetV2B0 기반 Custom Model 정의
├── dataset.py           # [데이터] 데이터 로더(DataGenerator) 및 전처리
├── train.py             # [설정] 학습 Argument 및 컴파일 설정
├── utils.py             # [유틸] 경로 생성 등 헬퍼 함수
└── dataset/             # [데이터셋] (사용자 구성 필요)
    ├── datasets_broken/ # 파손된 이미지 (.jpg)
    └── datasets_dirty2/ # 오염 이미지 (.jpg) 및 라벨 (.txt)
```

## 🚀 Key Features
### 1. Model Architecture:
* Backbone: `EfficientNetV2B0` (ImageNet Pre-trained)
* Custom Head: Feature Extractor → Conv2D → Dense(128) → Output(7 Classes)
### 2. Two-Stage Training Strategy:
* Phase 1 (Freeze): 백본을 고정(Freeze)하고 분류기(Head)만 빠르게 학습합니다.
* Phase 2 (Fine-tuning): 전체 모델을 매우 낮은 학습률(1e-5)로 미세 조정하여 정확도를 극대화합니다.
### 3. Smart Metric (Need Changed):
* 단순 클래스 정확도(Accuracy)뿐만 아니라, **"실제 조치가 필요한가?"**에 대한 이진 분류 성능(Precision/Recall)을 별도로 측정합니다.

## 🛠️ Requirements
Bash
```
pip install tensorflow numpy opencv-python glob2 natsort scikit-learn
```
## 💾 Dataset Setup
데이터셋은 아래 폴더 구조를 준수해야 정상적으로 로드됩니다.
* Broken Data (`dataset/datasets_broken/`):
    * 이미지 파일(`.jpg`)만 존재하면 됩니다. (자동으로 Class 0 할당)
* Dirty Data (`dataset/datasets_dirty2/`):
    * 이미지 파일(`.jpg`)과 동일한 이름의 텍스트 파일(`.txt`)이 쌍으로 존재해야 합니다.
    * `.txt` 파일 내부에는 오염도 레벨(`0`~`5`) 정수 하나가 적혀 있어야 합니다.
## 💻 Usage
## 1. Training (학습)
Bash
```
python main.py --epochs 30 --finetune_epochs 10 --batch_size 32 --save_dir ./outputs/
```
* `--epochs`: 1단계(Freeze) 학습 에폭 수 (기본: 30)
* `--finetune_epochs`: 2단계(Fine-tuning) 학습 에폭 수 (기본: 10)
* `--save_dir`: 모델 저장 경로
## 2. Evaluation (테스트)
Bash
```
python test.py --model_path ./outputs/your_model_path/model.h5
```
테스트 결과는 두 가지 관점으로 출력됩니다:
1. Softmax Classifier Accuracy: 7개 클래스 전체에 대한 정밀도, 재현율, F1 Score.
2. Need Changed Accuracy: 유지보수 필요 여부(True/False)에 대한 성능.
## 📊 Class & Action Mapping
모델이 예측하는 7개 클래스와 실제 조치(Action) 기준은 다음과 같습니다.
| Class Index | Description | Dirty Level | Action Required (Need Change) |
| :--- | :--- | :--- | :--- |
| **0** | **Broken** | - | 🔴 **YES (True)** |
| **1** | Normal | Level 0 | 🟢 NO (False) |
| **2** | Normal | Level 1 | 🟢 NO (False) |
| **3** | Caution | Level 2 | 🔴 **YES (True)** |
| **4** | Dirty | Level 3 | 🔴 **YES (True)** |
| **5** | Very Dirty | Level 4 | 🔴 **YES (True)** |
| **6** | Extreme | Level 5 | 🔴 **YES (True)** |