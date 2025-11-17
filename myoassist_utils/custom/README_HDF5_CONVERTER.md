# HDF5 → MyoAssist NPZ Converter

OpenSim HDF5 모션 데이터를 MyoAssist RL 학습용 NPZ 포맷으로 변환하는 파이프라인입니다.

---

## 📋 파일 구조

```
myoassist_utils/custom/
├── convert_hdf5_direct.py          # 메인 변환 스크립트
├── render_hdf5_reference.py        # (옵션) NPZ 시각화/검증 (현재 rl_train/analyzer/custom/에 위치)
└── README_HDF5_CONVERTER.md        # 이 파일
```

---

## 🚀 사용 방법

### 1. HDF5 데이터 준비
OpenSim 모션 캡처 데이터가 HDF5 포맷으로 저장되어 있어야 합니다.

**HDF5 구조 예시:**
```
S004.h5
└── S004/
    └── level_08mps/
        └── trial_01/
            └── MoCap/
                └── ik_data/
                    ├── pelvis_tx       # 단위: meters
                    ├── pelvis_ty       # 단위: meters
                    ├── pelvis_tz       # 단위: meters
                    ├── pelvis_tilt     # 단위: degrees
                    ├── pelvis_list     # 단위: degrees
                    ├── pelvis_rotation # 단위: degrees
                    ├── hip_flexion_r   # 단위: degrees
                    ├── hip_adduction_r # 단위: degrees
                    ├── hip_rotation_r  # 단위: degrees
                    ├── hip_flexion_l   # 단위: degrees
                    ├── hip_adduction_l # 단위: degrees
                    ├── hip_rotation_l  # 단위: degrees
                    ├── knee_angle_r    # 단위: degrees
                    ├── knee_angle_l    # 단위: degrees
                    ├── ankle_angle_r   # 단위: degrees
                    ├── ankle_angle_l   # 단위: degrees
                    └── time            # 단위: seconds
```

### 2. 설정 수정

`convert_hdf5_direct.py` 파일 상단의 `CONFIG` 딕셔너리를 수정:

```python
CONFIG = {
    'input_hdf5': r'C:\path\to\your\data\S004.h5',  # HDF5 파일 경로
    'subject': 'S004',                               # 피험자 ID
    'speed': '08mps',                                # 속도 (level_08mps)
    'trial': 'trial_01',                             # 트라이얼 번호
    'output_dir': r'C:\workspace_home\myoassist\rl_train\reference_data',  # 출력 폴더
    'output_name': 'S004_trial01_08mps_3D_HDF5_v7',  # 출력 파일명 (확장자 제외)
    
    # Offsets (보통 0으로 유지)
    'height_offset_m': 0.0,   # 높이 오프셋 (사용 안 함)
    'tilt_offset_deg': 0.0,   # 골반 tilt 오프셋 (degrees)
}
```

### 3. 변환 실행

```bash
# Conda 환경 활성화
conda activate myoassist

# 변환 스크립트 실행
cd C:\workspace_home\myoassist
python myoassist_utils/custom/convert_hdf5_direct.py
```

### 4. 출력 확인

변환이 완료되면 `output_dir`에 NPZ 파일이 생성됩니다:

```
rl_train/reference_data/
└── S004_trial01_08mps_3D_HDF5_v7.npz
```

**출력 예시:**
```
================================================================================
HDF5 → MyoAssist Direct Converter
================================================================================
Loading HDF5: C:\workspace_home\opensim data\LD\S004.h5
  Loaded 17 datasets, 250 frames

Converting to MyoAssist format...
Using height offset: 0.000 m (body_height * 0.0)

Pelvis translation (converted to RELATIVE):
  TX (right):   [ -0.0234,   0.0189] m (mean subtracted: 0.0021)
  TY (up):      [ -0.0312,   0.0278] m (mean subtracted: 0.9956)
  TZ (forward): [ -0.0456,   0.0512] m (mean subtracted: 1.2345)

Pelvis rotation ranges (DIRECT mapping, no swaps):
  tilt:     [-12.3, +8.7] deg (offset=0.0deg)
  list:     [-3.2, +2.8] deg
  rotation: [-5.1, +4.9] deg

✅ Computed velocities for 16 position channels

Saved: C:\workspace_home\myoassist\rl_train\reference_data\S004_trial01_08mps_3D_HDF5_v7.npz
  Shape: (250, 16)
  DOF: 16
  Frames: 250
  Duration: 2.50 sec
  Metadata: {...}
```

---

## 📊 NPZ 파일 구조

생성된 NPZ 파일은 다음 구조를 가집니다:

```python
{
    'q_ref': ndarray(frames, 16),        # 관절 위치 (qpos order)
    'series_data': {                     # 시계열 데이터
        'q_pelvis_tx': ndarray(frames),  # 위치 (position)
        'dq_pelvis_tx': ndarray(frames), # 속도 (velocity)
        'q_hip_flexion_r': ndarray(frames),
        'dq_hip_flexion_r': ndarray(frames),
        ...
    },
    'metadata': dict,                    # 메타데이터
    'joint_names': list,                 # 관절 이름 리스트
    'num_dof': int,                      # DOF 수 (16)
    'sampling_rate': float,              # 샘플링 레이트 (100 Hz)
    'duration': float                    # 총 시간 (초)
}
```

---

## 🔧 주요 변환 로직

### 1. 단위 변환
- **Translation (pelvis_tx/ty/tz)**: Meters → Meters (유지)
- **Rotation (angles)**: Degrees → Radians

### 2. 좌표계 변환
OpenSim과 MuJoCo 모두 동일한 좌표계 사용:
- **X**: Right (오른쪽)
- **Y**: Up (위)
- **Z**: Forward (앞)

→ **좌표 변환 없음** (직접 매핑)

### 3. 상대 위치 변환
절대 위치 → 상대 위치 (평균 중심):
```python
pelvis_tx_relative = pelvis_tx - mean(pelvis_tx)
pelvis_ty_relative = pelvis_ty - mean(pelvis_ty)
pelvis_tz_relative = pelvis_tz - mean(pelvis_tz)
```

### 4. 속도 계산
위치 데이터에서 속도 자동 계산 (중앙 차분법):
```python
dq = np.gradient(q, dt)  # dt = 0.01 (100 Hz)
```

---

## 🎯 RL 학습에서 사용

생성된 NPZ 파일은 RL 학습 config에서 참조:

```json
{
    "reference_motion_file_path": "rl_train/reference_data/S004_trial01_08mps_3D_HDF5_v7.npz",
    ...
}
```

또는 학습 실행 시:

```bash
python rl_train/run_train_ver1_0.py \
    --config_file_path rl_train/train_config_v7.json \
    --enable_wandb \
    --wandb_name my_training_run
```

---

## 🐛 문제 해결

### 1. HDF5 구조가 다른 경우
`load_hdf5_data()` 함수에서 경로 수정:
```python
path = f[subject][f'level_{speed}'][trial]['MoCap']['ik_data']
```

### 2. 관절 이름이 다른 경우
`MYOASSIST_JOINTS` 리스트와 매핑 로직 수정

### 3. 샘플링 레이트가 다른 경우
`dt` 값 수정:
```python
dt = 1.0 / sampling_rate  # 예: 1.0/200 = 0.005 for 200Hz
```

### 4. NPZ 검증
변환 결과를 시각화하려면:
```bash
python rl_train/analyzer/custom/render_hdf5_reference.py
```

---

## 🎥 변환 결과 시각화 (비디오 생성)

변환이 제대로 되었는지 확인하기 위해 reference motion을 비디오로 렌더링할 수 있습니다.

### 사용 방법

```bash
# 기본 사용 (기본 설정으로 비디오 생성)
python rl_train/analyzer/custom/render_hdf5_reference.py \
    --data S004_trial01_08mps_3D_HDF5_v7

# 상세 옵션 지정
python rl_train/analyzer/custom/render_hdf5_reference.py \
    --data S004_trial01_08mps_3D_HDF5_v7 \
    --model models/26muscle_3D/myoLeg26_BASELINE.xml \
    --frames 300 \
    --output my_reference_video.mp4 \
    --height 0.95
```

### 옵션 설명

- `--data`: NPZ 파일 이름 또는 경로 (기본: `S004_trial01_08mps_3D_HDF5_v1`)
- `--model`: MuJoCo 모델 XML 경로 (기본: `myoLeg26_TUTORIAL.xml`)
- `--frames`: 렌더링할 프레임 수 (기본: 300)
- `--output`: 출력 비디오 파일명 (기본: `ref_{npz_name}.mp4`)
- `--height`: 모델을 들어올릴 높이 (기본: 0.95m)

### 출력 예시

```
Loading reference: rl_train/reference_data/S004_trial01_08mps_3D_HDF5_v7.npz
  Frames: 250
  DOF: 16
  Height offset: 0.950 m

Loading model: models/26muscle_3D/myoLeg26_BASELINE.xml

Camera settings:
  View angle: Diagonal (azimuth=135°, elevation=-20°)
  Distance: 5.0m
  Transparency: Enabled (can see through floor)

Rendering 300 frames...
  Frame 0/300...
  Frame 30/300...
  ...

Saving video: ref_S004_trial01_08mps_3D_HDF5_v7.mp4
  Video FPS: 5.0 (target duration: ~60 seconds)

Joint ranges:
  q_pelvis_tx         : [-0.023, +0.019] rad
  q_pelvis_ty         : [-0.031, +0.028] rad
  ...

✅ Done! Saved: ref_S004_trial01_08mps_3D_HDF5_v7.mp4
```

### 시각화 기능

- **카메라 앵글**: 대각선 뷰 (azimuth=135°, elevation=-20°)
- **투명도**: 바닥을 투명하게 처리하여 다리 움직임 명확히 확인
- **팔 숨김**: 팔 geom을 투명 처리하여 다리에 집중
- **높이 조정**: 모델을 바닥 위로 들어올려 자연스러운 걷기 시각화
- **비디오 길이**: 약 60초 (조정 가능)

### 검증 체크리스트

비디오를 보고 확인할 사항:
- [ ] 걷기 동작이 자연스러운가?
- [ ] 관절 각도가 정상 범위인가?
- [ ] 골반 회전/기울임이 합리적인가?
- [ ] 무릎/발목 각도가 이상하지 않은가?
- [ ] 모델이 바닥을 뚫고 들어가지 않는가?
- [ ] 발 접촉이 자연스러운가?

---

## 📝 추가 유틸리티

### 데이터 분석 스크립트들
- `analyze_hdf5_structure.py`: HDF5 파일 구조 확인
- `check_hdf5_units.py`: HDF5 데이터 단위 확인
- `inspect_npz.py`: NPZ 파일 내용 확인

모두 `myoassist_utils/custom/` 폴더에 있습니다.

### 시각화 스크립트
- `render_hdf5_reference.py`: NPZ reference motion을 비디오로 렌더링
- 위치: `rl_train/analyzer/custom/`

---

## 📚 참고

- **OpenSim 좌표계**: [OpenSim Documentation](https://simtk-confluence.stanford.edu/display/OpenSim/Coordinate+Systems)
- **MuJoCo 좌표계**: [MuJoCo Documentation](https://mujoco.readthedocs.io/en/stable/modeling.html#coordinate-frames)
- **MyoAssist 모델**: `models/26muscle_3D/myoLeg26_BASELINE.xml`

---

## ✅ 체크리스트

변환 전 확인사항:
- [ ] HDF5 파일 경로가 올바른가?
- [ ] HDF5 구조가 예상한 형태인가?
- [ ] CONFIG의 subject, speed, trial이 맞는가?
- [ ] 출력 폴더가 존재하는가?
- [ ] Conda 환경이 활성화되어 있는가?

변환 후 확인사항:
- [ ] NPZ 파일이 생성되었는가?
- [ ] 출력 로그에 경고/에러가 없는가?
- [ ] 관절 범위(range)가 합리적인가?
- [ ] 속도 데이터가 포함되었는가?

---

## 🔄 버전 히스토리

- **v7**: 직접 매핑 (좌표 변환 없음), 상대 위치, 속도 자동 계산
- **v6**: Arms 관절 추가
- **v5**: 단위 수정 (degrees → radians)
- **v1-v4**: 초기 버전 (deprecated)
