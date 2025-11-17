# Reference Motion 시각화 가이드

NPZ 파일로 변환된 reference motion이 제대로 되었는지 비디오로 확인하는 방법입니다.

---

## 🎥 기본 사용법

```bash
# NPZ 파일을 비디오로 렌더링
python rl_train/analyzer/custom/render_hdf5_reference.py \
    --data S004_trial01_08mps_3D_HDF5_v7
```

**출력:**
- `ref_S004_trial01_08mps_3D_HDF5_v7.mp4` 생성
- 약 60초 길이 비디오
- 대각선 시점, 투명 바닥

---

## 🔧 옵션

### 전체 경로 지정
```bash
python rl_train/analyzer/custom/render_hdf5_reference.py \
    --data rl_train/reference_data/S004_trial01_08mps_3D_HDF5_v7.npz
```

### 출력 파일명 지정
```bash
python rl_train/analyzer/custom/render_hdf5_reference.py \
    --data S004_trial01_08mps_3D_HDF5_v7 \
    --output my_walking_video.mp4
```

### 프레임 수 조정
```bash
python rl_train/analyzer/custom/render_hdf5_reference.py \
    --data S004_trial01_08mps_3D_HDF5_v7 \
    --frames 600  # 더 부드러운 영상
```

### 모델 변경
```bash
python rl_train/analyzer/custom/render_hdf5_reference.py \
    --data S004_trial01_08mps_3D_HDF5_v7 \
    --model models/26muscle_3D/myoLeg26_BASELINE.xml
```

### 높이 조정
```bash
python rl_train/analyzer/custom/render_hdf5_reference.py \
    --data S004_trial01_08mps_3D_HDF5_v7 \
    --height 1.0  # 더 높이 들어올림
```

---

## 📊 출력 정보

실행하면 다음 정보가 출력됩니다:

```
Loading reference: rl_train/reference_data/S004_trial01_08mps_3D_HDF5_v7.npz
  Frames: 250
  DOF: 16
  Joints: ['q_pelvis_tx', 'q_pelvis_ty', ...]
  Height offset: 0.950 m

Loading model: models/26muscle_3D/myoLeg26_BASELINE.xml

Camera settings:
  View angle: Diagonal (azimuth=135°, elevation=-20°)
  Distance: 5.0m
  Transparency: Enabled

Rendering 300 frames...
  Frame 0/300...
  Frame 30/300...
  Frame 60/300...
  ...

Saving video: ref_S004_trial01_08mps_3D_HDF5_v7.mp4
  Video FPS: 5.0 (target duration: ~60 seconds)

Joint ranges:
  q_pelvis_tx         : [-0.023, +0.019] rad
  q_pelvis_ty         : [-0.031, +0.028] rad
  q_pelvis_tz         : [-0.046, +0.051] rad
  q_pelvis_tilt       : [-0.215, +0.152] rad
  q_pelvis_list       : [-0.056, +0.049] rad
  q_pelvis_rotation   : [-0.089, +0.086] rad
  hip_flexion_r       : [-0.523, +0.698] rad
  hip_adduction_r     : [-0.234, +0.156] rad
  hip_rotation_r      : [-0.178, +0.134] rad
  hip_flexion_l       : [-0.512, +0.712] rad
  hip_adduction_l     : [-0.189, +0.201] rad
  hip_rotation_l      : [-0.145, +0.167] rad
  knee_angle_r        : [-1.234, -0.023] rad
  knee_angle_l        : [-1.198, -0.034] rad
  ankle_angle_r       : [-0.234, +0.123] rad
  ankle_angle_l       : [-0.212, +0.145] rad

✅ Done! Saved: ref_S004_trial01_08mps_3D_HDF5_v7.mp4
```

---

## ✅ 검증 체크리스트

비디오를 보고 확인할 사항:

### 기본 동작
- [ ] 걷기 동작이 자연스러운가?
- [ ] 좌우 다리가 교대로 움직이는가?
- [ ] 발이 바닥을 제대로 디디는가?

### 관절 각도
- [ ] 무릎이 자연스럽게 구부러지는가?
- [ ] 발목 각도가 정상 범위인가?
- [ ] 고관절 굴곡/신전이 합리적인가?

### 골반 움직임
- [ ] 골반 회전(rotation)이 과하지 않은가?
- [ ] 골반 기울임(tilt)이 자연스러운가?
- [ ] 골반 좌우 기울임(list)이 정상 범위인가?

### 물리적 타당성
- [ ] 모델이 바닥을 뚫고 들어가지 않는가?
- [ ] 관절이 비정상적으로 꺾이지 않는가?
- [ ] 전체적인 자세가 안정적인가?

### 데이터 품질
- [ ] 떨림(jitter)이 없는가?
- [ ] 부드럽게 연결되는가?
- [ ] 갑작스러운 점프가 없는가?

---

## 🎨 시각화 기능

### 카메라 설정
- **앵글**: 대각선 뷰 (azimuth=135°, elevation=-20°)
- **거리**: 5.0m (전체 움직임 확인 가능)
- **초점**: 골반 높이 (0.5m)

### 렌더링 옵션
- **투명도**: 바닥을 투명하게 처리하여 발 움직임 명확히 확인
- **팔 숨김**: 팔 geom을 투명 처리하여 다리에 집중
- **높이 조정**: 모델을 바닥 위로 들어올려 자연스러운 시각화

### 비디오 설정
- **해상도**: 1280x720
- **FPS**: 5.0 (60초 영상)
- **포맷**: MP4 (H.264)

---

## 🐛 문제 해결

### 에러: "No module named 'mujoco'"
```bash
pip install mujoco
```

### 에러: "No module named 'imageio'"
```bash
pip install imageio
```

### 비디오가 너무 빠름/느림
`--frames` 옵션으로 조정:
```bash
# 느리게: 더 많은 프레임
python render_hdf5_reference.py --data xxx --frames 600

# 빠르게: 더 적은 프레임
python render_hdf5_reference.py --data xxx --frames 150
```

### 모델이 바닥 아래로 떨어짐
`--height` 옵션으로 조정:
```bash
python render_hdf5_reference.py --data xxx --height 1.0
```

### 관절 각도가 이상함
→ NPZ 파일의 관절 순서/단위 확인:
```bash
python myoassist_utils/custom/inspect_npz.py
```

---

## 📁 파일 위치

```
rl_train/analyzer/custom/
└── render_hdf5_reference.py  # 시각화 스크립트

rl_train/reference_data/
└── S004_trial01_08mps_3D_HDF5_v7.npz  # 입력 NPZ

./  (루트)
└── ref_S004_trial01_08mps_3D_HDF5_v7.mp4  # 출력 비디오
```

---

## 🔄 Workflow

```
1. HDF5 변환
   ↓
2. NPZ 생성
   ↓
3. 시각화 (이 가이드) ← 여기!
   ↓
4. RL 학습
```

---

## 💡 팁

### 여러 파일 한 번에 시각화
```bash
for file in rl_train/reference_data/*.npz; do
    python rl_train/analyzer/custom/render_hdf5_reference.py \
        --data "$file" \
        --frames 300
done
```

### 특정 구간만 시각화
코드 수정 필요:
```python
# render_hdf5_reference.py 내부
start_frame = 50
end_frame = 150
q_ref = q_ref[start_frame:end_frame]
```

### 다양한 앵글로 렌더링
코드 수정 필요:
```python
# render_hdf5_reference.py 내부
camera.azimuth = 45   # 정면
camera.azimuth = 90   # 측면
camera.azimuth = 135  # 대각선 (기본)
camera.azimuth = 180  # 후면
```
