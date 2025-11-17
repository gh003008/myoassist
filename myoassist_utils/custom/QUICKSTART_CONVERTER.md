# HDF5 → MyoAssist 변환 퀵스타트

## 🚀 빠른 시작 (5분)

### 1단계: 파일 준비
```
your_data/
└── S004.h5  # OpenSim HDF5 파일
```

### 2단계: 스크립트 수정
`myoassist_utils/custom/convert_hdf5_direct.py` 열기

**수정할 부분 (파일 상단):**
```python
CONFIG = {
    'input_hdf5': r'C:\your\path\S004.h5',     # ← 여기만 수정!
    'subject': 'S004',                         # ← 피험자 ID
    'speed': '08mps',                          # ← 속도 레벨
    'trial': 'trial_01',                       # ← 트라이얼 번호
    'output_dir': r'C:\workspace_home\myoassist\rl_train\reference_data',
    'output_name': 'S004_trial01_08mps_3D_HDF5_v7',  # ← 출력 파일명
    
    # 아래는 건드리지 마세요
    'height_offset_m': 0.0,
    'tilt_offset_deg': 0.0,
}
```

### 3단계: 실행
```bash
# Conda 환경 활성화
conda activate myoassist

# 변환 실행
cd C:\workspace_home\myoassist
python myoassist_utils/custom/convert_hdf5_direct.py
```

### 4단계: 확인
```
rl_train/reference_data/
└── S004_trial01_08mps_3D_HDF5_v7.npz  # ← 이 파일 생성됨
```

---

## ✅ 성공 확인

터미널에 다음과 같이 출력되면 성공:
```
✅ Computed velocities for 16 position channels

Saved: ...\S004_trial01_08mps_3D_HDF5_v7.npz
  Shape: (250, 16)
  DOF: 16
  Frames: 250
  Duration: 2.50 sec
```

---

## 🎯 RL 학습에서 사용

`rl_train/train_config_v7.json` 파일에서:
```json
{
    "reference_motion_file_path": "rl_train/reference_data/S004_trial01_08mps_3D_HDF5_v7.npz",
    ...
}
```

학습 실행:
```bash
python rl_train/run_train_ver1_0.py \
    --config_file_path rl_train/train_config_v7.json \
    --enable_wandb \
    --wandb_name my_training
```

---

## ❓ 문제 해결

### 에러: "KeyError: 'S004'"
→ HDF5 파일 구조 확인:
```bash
python myoassist_utils/custom/analyze_hdf5_structure.py
```

### 에러: "FileNotFoundError"
→ `input_hdf5` 경로 확인 (절대 경로 사용)

### 에러: "No module named 'h5py'"
→ 환경 확인:
```bash
conda activate myoassist
pip install h5py
```

---

## 📚 더 알아보기

자세한 설명: `README_HDF5_CONVERTER.md`
