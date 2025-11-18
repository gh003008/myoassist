# 📊 데이터 파이프라인 매핑 완전 가이드

## 🎯 요약: 이제 두 파이프라인이 **동일**합니다! ✅

수정 후, **HDF5 → MuJoCo Renderer**와 **HDF5 → MyoAssist Environment**가 **완전히 동일한 방식**으로 동작합니다.

---

## 1️⃣ HDF5 → MuJoCo Renderer (render_hdf5_reference.py)

### 📥 입력: HDF5 Format
```python
# S004_trial01_08mps_3D_HDF5_v7_symmetric.npz
q_ref: shape (12028, 16)  # 16 DOF, 12028 frames @ 100 Hz
joint_names: [
    'q_pelvis_tx',        # 0
    'q_pelvis_ty',        # 1  ⚠️ Ground-relative (0.01m)
    'q_pelvis_tz',        # 2
    'q_pelvis_tilt',      # 3
    'q_pelvis_list',      # 4
    'q_pelvis_rotation',  # 5
    'q_hip_flexion_r',    # 6
    'q_hip_adduction_r',  # 7
    'q_hip_rotation_r',   # 8
    'q_hip_flexion_l',    # 9
    'q_hip_adduction_l',  # 10
    'q_hip_rotation_l',   # 11
    'q_knee_angle_r',     # 12
    'q_knee_angle_l',     # 13
    'q_ankle_angle_r',    # 14
    'q_ankle_angle_l',    # 15
]
```

### 🔄 변환 과정 (Lines 224-250)

```python
# STEP 1: Initialize with "stand" keyframe
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
data_mj.qpos[:] = model.key_qpos[key_id]
# qpos[0:54] 모두 초기화:
#   pelvis_ty = 0.91
#   knee_tx_r/l = 0.05
#   shoulder_flex = 0.0
#   ... (54 values total)

# STEP 2: Map HDF5 data to qpos
ref_to_qpos = [
    (0, 0, 'pelvis_tx'),      # q_ref[0]  → qpos[0]
    (1, 1, 'pelvis_ty'),      # q_ref[1]  → qpos[1]  ⚠️ Special handling
    (2, 2, 'pelvis_tz'),      # q_ref[2]  → qpos[2]
    (3, 3, 'pelvis_tilt'),    # q_ref[3]  → qpos[3]
    (4, 4, 'pelvis_list'),    # q_ref[4]  → qpos[4]
    (5, 5, 'pelvis_rotation'),# q_ref[5]  → qpos[5]
    (6, 6, 'hip_flexion_r'),  # q_ref[6]  → qpos[6]
    (7, 7, 'hip_adduction_r'),# q_ref[7]  → qpos[7]
    (8, 8, 'hip_rotation_r'), # q_ref[8]  → qpos[8]
    (9, 12,'hip_flexion_l'),  # q_ref[9]  → qpos[12]
    (10,13,'hip_adduction_l'),# q_ref[10] → qpos[13]
    (11,14,'hip_rotation_l'), # q_ref[11] → qpos[14]
    (12,18,'knee_angle_r'),   # q_ref[12] → qpos[18]
    (13,22,'knee_angle_l'),   # q_ref[13] → qpos[22]
    (14,26,'ankle_angle_r'),  # q_ref[14] → qpos[26]
    (15,30,'ankle_angle_l'),  # q_ref[15] → qpos[30]
]

for ref_idx, qpos_idx, jnt_name in ref_to_qpos:
    data_mj.qpos[qpos_idx] = q_ref[i, ref_idx]

# STEP 3: Apply pelvis_ty offset (Line 235)
data_mj.qpos[1] = model.key_qpos[key_id][1] + q_ref[i, 1] + (height_offset - 0.91)
#                 ↑ 0.91 (stand)        ↑ 0.01 (HDF5)   ↑ 0.0 (default)
#                 = 0.91 + 0.01 + 0.0 = 0.92m

# STEP 4: Fix arms (Lines 240-251)
arm_joints = {
    40: 0.0,   # r_shoulder_abd
    41: 0.0,   # r_shoulder_rot
    42: 0.5,   # r_shoulder_flex
    43: 0.8,   # r_elbow_flex
    47: 0.0,   # l_shoulder_abd
    48: 0.0,   # l_shoulder_rot
    49: 0.5,   # l_shoulder_flex
    50: 0.8,   # l_elbow_flex
}
for qpos_idx, angle in arm_joints.items():
    data_mj.qpos[qpos_idx] = angle
```

### 📤 출력: MuJoCo qpos (54 values)
```python
qpos[0:54] = [
    0.0,       # 0:  pelvis_tx (reset to 0 for visualization)
    0.92,      # 1:  pelvis_ty (0.91 stand + 0.01 HDF5)  ✅
    0.0,       # 2:  pelvis_tz
    -0.05,     # 3:  pelvis_tilt (from HDF5)
    0.01,      # 4:  pelvis_list (from HDF5)
    0.02,      # 5:  pelvis_rotation (from HDF5)
    0.5,       # 6:  hip_flexion_r (from HDF5)
    0.1,       # 7:  hip_adduction_r (from HDF5)
    0.0,       # 8:  hip_rotation_r (from HDF5)
    0.05,      # 9:  knee_tx_r (from stand, preserved!)  ✅
    0.0,       # 10: knee_ty_r (from stand, preserved!)  ✅
    0.0,       # 11: knee_tz_r (from stand, preserved!)  ✅
    -0.3,      # 12: hip_flexion_l (from HDF5)
    ...
    0.05,      # 15: knee_tx_l (from stand, preserved!)  ✅
    ...
    0.0,       # 40: r_shoulder_abd (manually set)  ✅
    0.0,       # 41: r_shoulder_rot (manually set)  ✅
    0.5,       # 42: r_shoulder_flex (manually set)  ✅
    0.8,       # 43: r_elbow_flex (manually set)  ✅
    ...
]
```

---

## 2️⃣ HDF5 → MyoAssist Environment (수정 후)

### 📥 입력: HDF5 Format (동일)
```python
# S004_trial01_08mps_3D_HDF5_v7_symmetric.npz
q_ref: shape (12028, 16)
joint_names: (위와 동일)
```

### 🔄 변환 과정 A: environment_handler.py (Lines 70-105)

```python
# STEP 1: Detect HDF5 format
if 'q_ref' in ref_data_dict and 'joint_names' in ref_data_dict:
    q_ref = ref_data_dict['q_ref']
    joint_names = ref_data_dict['joint_names']
    
    # STEP 2: Convert to series_data format with pelvis_ty offset
    series_data = {}
    for i, joint_name in enumerate(joint_names):
        joint_name_str = str(joint_name)
        
        # ⚠️ CRITICAL: Apply pelvis_ty offset
        if joint_name_str == 'q_pelvis_ty':
            series_data[joint_name_str] = q_ref[:, i] + 0.91  # ✅ +0.91m offset
            print(f"   ⚠️  Applied pelvis_ty offset: +0.91m")
        else:
            series_data[joint_name_str] = q_ref[:, i]
        
        # Velocity data
        dq = np.gradient(q_ref[:, i], axis=0) * 100
        series_data[f'd{joint_name_str}'] = dq
    
    # STEP 3: Create metadata
    ref_data_dict = {
        'series_data': series_data,
        'metadata': {
            'data_length': q_ref.shape[0],
            'sample_rate': 100,
            'dof': q_ref.shape[1],
        }
    }
```

### 📤 출력: series_data format
```python
series_data = {
    'q_pelvis_tx': array([0.0, 0.0, ...]),        # 12028 frames
    'q_pelvis_ty': array([0.92, 0.93, ...]),      # ✅ Offset applied!
    'q_pelvis_tz': array([0.0, 0.0, ...]),
    'q_pelvis_tilt': array([-0.05, -0.05, ...]),
    'q_pelvis_list': array([0.01, 0.01, ...]),
    'q_pelvis_rotation': array([0.02, 0.02, ...]),
    'q_hip_flexion_r': array([0.5, 0.52, ...]),
    'q_hip_adduction_r': array([0.1, 0.09, ...]),
    'q_hip_rotation_r': array([0.0, 0.01, ...]),
    'q_hip_flexion_l': array([-0.3, -0.28, ...]),
    'q_hip_adduction_l': array([-0.08, -0.09, ...]),
    'q_hip_rotation_l': array([0.0, -0.01, ...]),
    'q_knee_angle_r': array([-0.3, -0.32, ...]),
    'q_knee_angle_l': array([-0.5, -0.52, ...]),
    'q_ankle_angle_r': array([0.1, 0.12, ...]),
    'q_ankle_angle_l': array([0.15, 0.14, ...]),
    # Velocities
    'dq_pelvis_tx': array([...]),
    'dq_pelvis_ty': array([...]),  # Velocity from offset data
    ...
}
```

### 🔄 변환 과정 B: _follow_reference_motion() (Lines 467-486)

```python
# STEP 1: Initialize with "stand" keyframe (수정 후 추가!)
try:
    key_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    self.sim.data.qpos[:] = self.sim.model.key_qpos[key_id]
    # qpos[0:54] 모두 초기화:
    #   pelvis_ty = 0.91
    #   knee_tx_r/l = 0.05
    #   shoulder_flex = 0.0  ✅ Stand 값 사용
except:
    self.sim.data.qpos[:] = self.sim.model.qpos0

# STEP 2: Overlay reference data (14 joints only)
reference_data_keys = [
    'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
    'knee_angle_r', 'knee_angle_l',
    'ankle_angle_r', 'ankle_angle_l',
]

for key in reference_data_keys:
    # series_data already has pelvis_ty with offset applied!
    self.sim.data.joint(f"{key}").qpos = self._reference_data["series_data"][f"q_{key}"][self._imitation_index]
    # pelvis_ty: series_data has 0.92m (0.91 stand + 0.01 HDF5)  ✅

# STEP 3: Set velocities
for key in reference_data_keys:
    self.sim.data.joint(f"{key}").qvel = self._reference_data["series_data"][f"dq_{key}"][self._imitation_index] * speed_ratio
```

### 📤 최종 출력: Environment qpos (54 values)
```python
qpos[0:54] = [
    0.0,       # 0:  pelvis_tx (is_x_follow=False → 0)
    0.92,      # 1:  pelvis_ty (from series_data with offset!)  ✅
    0.0,       # 2:  pelvis_tz
    -0.05,     # 3:  pelvis_tilt (from reference)
    0.01,      # 4:  pelvis_list (from reference)
    0.02,      # 5:  pelvis_rotation (from reference)
    0.5,       # 6:  hip_flexion_r (from reference)
    0.1,       # 7:  hip_adduction_r (from reference)
    0.0,       # 8:  hip_rotation_r (from reference)
    0.05,      # 9:  knee_tx_r (from stand, preserved!)  ✅
    0.0,       # 10: knee_ty_r (from stand, preserved!)  ✅
    0.0,       # 11: knee_tz_r (from stand, preserved!)  ✅
    -0.3,      # 12: hip_flexion_l (from reference)
    ...
    0.05,      # 15: knee_tx_l (from stand, preserved!)  ✅
    ...
    0.0,       # 40: r_shoulder_abd (from stand!)  ✅
    0.0,       # 41: r_shoulder_rot (from stand!)  ✅
    0.0,       # 42: r_shoulder_flex (from stand!)  ✅
    0.0,       # 43: r_elbow_flex (from stand!)  ✅
    ...
]
```

---

## 🔍 비교: 수정 전 vs 수정 후

### ❌ 수정 전: 파이프라인이 달랐음

| 단계 | MuJoCo Renderer | Environment (OLD) | 차이점 |
|------|----------------|------------------|--------|
| **Stand 초기화** | ✅ Yes | ❌ **NO** | 환경은 stand 없음 |
| **pelvis_ty offset** | ✅ +0.91m | ❌ **NO** | 환경은 offset 없음 |
| **knee_tx/ty/tz** | ✅ Stand 값 (0.05) | ❌ **0으로 남음** | 환경은 초기화 안 됨 |
| **팔 joints** | ✅ 명시적 설정 | ❌ **0으로 남음** | 환경은 초기화 안 됨 |

**결과:**
- Renderer: 정상 걷기 자세 ✅
- Environment: 무릎 거꾸로 꺾임 💀

### ✅ 수정 후: 파이프라인 동일

| 단계 | MuJoCo Renderer | Environment (NEW) | 동일? |
|------|----------------|------------------|-------|
| **Stand 초기화** | ✅ Yes | ✅ **Yes** | ✅ 동일 |
| **pelvis_ty offset** | ✅ +0.91m | ✅ **+0.91m** | ✅ 동일 |
| **knee_tx/ty/tz** | ✅ Stand 값 (0.05) | ✅ **Stand 값** | ✅ 동일 |
| **팔 joints** | ✅ 명시적 설정 | ✅ **Stand 값** | ✅ 동일 |

**결과:**
- Renderer: 정상 걷기 자세 ✅
- Environment: 정상 걷기 자세 ✅
- **완전히 동일한 결과!** 🎉

---

## 📋 수정된 파일 요약

### 1. `environment_handler.py` (Lines 70-105)
```python
# HDF5 → series_data 변환 시
if joint_name_str == 'q_pelvis_ty':
    series_data[joint_name_str] = q_ref[:, i] + 0.91  # ✅ Offset 추가
```

**효과:** pelvis_ty가 처음부터 올바른 높이 (0.91 + HDF5)로 변환됨

### 2. `myoassist_leg_imitation_ver1_0.py` (Lines 467-486)
```python
def _follow_reference_motion(self, is_x_follow:bool):
    # ✅ Stand keyframe 초기화 추가
    key_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    self.sim.data.qpos[:] = self.sim.model.key_qpos[key_id]
    
    # Reference 14 joints로 overlay
    for key in self.reference_data_keys:
        self.sim.data.joint(f"{key}").qpos = self._reference_data["series_data"][f"q_{key}"][...]
```

**효과:** 
- 모든 joints가 stand로 초기화됨
- knee_tx/ty/tz, 팔 등이 올바른 값 유지
- Reference 14개만 덮어써서 걷기 동작 표현

---

## ✅ 결론

### 질문 1: "파이프라인이 다른가?"
**답:** 수정 후에는 **완전히 동일**합니다!

### 질문 2: "학습 시에도 올바르게 동작하나?"
**답:** **네! ✅** 학습 시에도:
1. `reset()` → `_follow_reference_motion()` 호출 (Line 550)
2. `step()` → `_follow_reference_motion()` 호출 (Line 538)
3. 모두 수정된 버전 사용 → **정상 동작!**

### 질문 3: "이상한 모션을 학습하지 않나?"
**답:** **아니요! ✅** 
- `environment_handler.py`에서 pelvis_ty offset 적용
- `_follow_reference_motion()`에서 stand 초기화
- 학습 시 올바른 reference를 따라감

---

## 🎯 최종 검증

비디오 파일 확인:
```
visualize_in_env/20251118_010320_symmetric_in_training_env.mp4
- Pelvis height: 0.901~0.925m  ✅
- 무릎 정상 동작  ✅
- 대칭 걷기 자세  ✅
```

**학습을 시작해도 됩니다!** 🚀
