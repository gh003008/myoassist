# 정강이가 힙으로 가는 문제 - 근본 원인 분석

**날짜:** 2024-11-18  
**문제:** 환경에서 reference motion 렌더링 시 kinematic chain 붕괴 (정강이가 힙에 붙어버림)  
**해결:** 환경 코드를 원본으로 복구하니 즉시 해결됨

---

## 🔍 근본 원인: `q_` prefix 불일치

### 문제의 핵심

**environment_handler.py**가 데이터를 로드할 때 **`q_` prefix를 제거**하는데,  
우리가 수정한 코드는 **`q_` prefix를 다시 추가**해서 찾으려 했음.

결과: **데이터를 못 찾음** → 초기화되지 않은 값 사용 → kinematic chain 붕괴

---

## 📊 데이터 플로우 분석

### 1. **데이터 저장 단계 (HDF5 → NPZ)**

```python
# convert_hdf5_to_npz.py (또는 symmetrize script)
q_ref = np.column_stack([
    q_pelvis_tx,    # "q_" prefix 포함
    q_pelvis_ty,
    ...
])
joint_names = ['q_pelvis_tx', 'q_pelvis_ty', ...]  # "q_" prefix 포함

np.savez(output_path,
    q_ref=q_ref,
    joint_names=joint_names  # ← "q_" prefix 있음
)
```

**저장 형식:** `S004_trial01_08mps_3D_HDF5_v7_symmetric.npz`
- `q_ref`: numpy array (12028, 16)
- `joint_names`: ['q_pelvis_tx', 'q_pelvis_ty', ..., 'q_ankle_angle_l']

---

### 2. **환경 로드 단계 (environment_handler.py)**

```python
# rl_train/envs/environment_handler.py (Lines 70-115)

elif 'q_ref' in ref_data_dict and 'joint_names' in ref_data_dict:
    # MuJoCo renderer format 감지
    q_ref = ref_data_dict['q_ref']
    joint_names = ref_data_dict['joint_names']
    
    series_data = {}
    for i, joint_name in enumerate(joint_names):
        joint_name_str = str(joint_name)
        
        # ✅ CRITICAL: "q_" prefix 제거!
        if joint_name_str.startswith('q_'):
            env_joint_name = joint_name_str[2:]  # "q_pelvis_tx" → "pelvis_tx"
        else:
            env_joint_name = joint_name_str
        
        # 환경에서 사용하는 키 이름 (prefix 없음)
        series_data[env_joint_name] = q_ref[:, i]  # ← "pelvis_tx"로 저장
```

**변환 결과:**
```python
self._reference_data = {
    "series_data": {
        "pelvis_tx": array([...]),      # ← "q_" prefix 제거됨!
        "pelvis_ty": array([...]),
        "hip_flexion_r": array([...]),
        ...
    },
    "metadata": {...}
}
```

---

### 3. **문제가 있던 코드 (3ff92eb)**

```python
# myoassist_leg_imitation_ver1_0.py (BROKEN)

def _follow_reference_motion(self, is_x_follow:bool):
    for key in self.reference_data_keys:  # ['pelvis_tx', 'pelvis_ty', ...]
        # ❌ 문제: "q_" prefix 다시 추가!
        self.sim.data.joint(f"{key}").qpos = \
            self._reference_data["series_data"][f"q_{key}"][self._imitation_index]
        #                                       ^^^^^^
        #                                       존재하지 않는 키!
```

**실행 시 에러:**
```
KeyError: 'q_pelvis_tx'  # series_data에는 'pelvis_tx'만 있음!
```

또는 더 나쁘게는:
- KeyError가 안 나고 **초기화되지 않은 값** 사용
- qpos가 0으로 남아있음
- **kinematic chain 붕괴** → 정강이가 힙으로 감

---

### 4. **정상 동작하는 코드 (032518e - 현재)**

```python
# myoassist_leg_imitation_ver1_0.py (WORKING)

def _follow_reference_motion(self, is_x_follow:bool):
    for key in self.reference_data_keys:  # ['pelvis_tx', 'pelvis_ty', ...]
        # ✅ 정상: prefix 없이 그대로 사용
        self.sim.data.joint(f"{key}").qpos = \
            self._reference_data["series_data"][f"{key}"][self._imitation_index]
        #                                       ^^^^^^
        #                                       올바른 키!
```

**결과:**
- `series_data['pelvis_tx']` → 올바른 데이터 접근 ✅
- qpos에 정상 값 설정 ✅
- kinematic chain 정상 유지 ✅

---

## 🎯 왜 stand keyframe도 문제였을까?

Stand keyframe 자체는 문제가 아니었습니다. 하지만 **추가적인 복잡성**을 만들었고,  
진짜 문제(`q_` prefix 불일치)를 가렸습니다.

### Stand keyframe 추가 시도:
```python
# 시도했던 코드
key_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
self.sim.data.qpos[:] = self.sim.model.key_qpos[key_id]  # 기본 자세 설정

# 그 위에 reference motion 덮어쓰기
for key in self.reference_data_keys:
    qpos_idx = self._joint_qpos_map[key]
    self.sim.data.qpos[qpos_idx] = self._reference_data["series_data"][f"{key}"][...]
```

**문제점:**
1. Stand keyframe은 **rendering에는 필요**하지만 **환경에는 불필요**
2. 환경은 `joint().qpos` API로 직접 설정하면 됨 (MuJoCo가 알아서 처리)
3. qpos index 직접 접근은 dm_control wrapper와 충돌
4. **근본 문제 (`q_` prefix)를 해결하지 못함**

---

## 🔬 타임라인 복기

### 문제 발생 과정:

1. **원본 코드 (동작함):**
   ```python
   self._reference_data["series_data"][f"{key}"]  # ✅
   ```

2. **어느 시점에 잘못 수정됨:**
   ```python
   self._reference_data["series_data"][f"q_{key}"]  # ❌
   ```
   
3. **에러 발생:**
   - 데이터를 못 찾음
   - qpos가 초기화되지 않음 (0으로 남음)
   - Kinematic chain 붕괴

4. **잘못된 해결 시도들:**
   - Stand keyframe 추가 → 근본 원인 해결 안 됨
   - qpos index 직접 접근 → dm_control 충돌
   - Joint name mapping 추가 → 복잡도만 증가
   - dm_control wrapper 처리 → 더 깊은 늪으로

5. **올바른 해결:**
   - 원본 코드로 복구
   - `f"{key}"` 사용 (prefix 없이)
   - 즉시 정상 동작 ✅

---

## 💡 핵심 교훈

### 1. **데이터 계약 (Data Contract) 준수**

```
environment_handler → series_data (NO q_ prefix)
                            ↓
                    environment uses it directly
```

Environment handler가 prefix를 제거했으면,  
environment는 **prefix 없이** 사용해야 함.

### 2. **단순한 코드가 낫다**

**원본 (10줄):**
- `joint().qpos` 직접 설정
- 명확하고 간단
- 동작함 ✅

**수정본 (50줄):**
- Stand keyframe 초기화
- qpos index mapping
- dm_control wrapper 처리
- 복잡하고 에러 많음 ❌

### 3. **Rendering ≠ Training Environment**

- **Rendering:** Stand keyframe 필요, qpos index 직접 접근 OK
- **Training Environment:** MuJoCo API 사용, wrapper 고려 필요
- **분리하자:** 각자 다른 스크립트 사용

---

## 🎬 최종 데이터 플로우

### 정상 동작 (현재):

```
[HDF5 파일]
    ↓ (convert/symmetrize)
[NPZ: q_ref + joint_names (with q_ prefix)]
    ↓ (environment_handler.py)
[series_data: {
    "pelvis_tx": [...],        ← q_ prefix 제거됨
    "pelvis_ty": [...],
    ...
}]
    ↓ (myoassist_leg_imitation_ver1_0.py)
[Environment에서 사용]
for key in ['pelvis_tx', 'pelvis_ty', ...]:
    joint(key).qpos = series_data[key][index]  ← 직접 사용 (prefix 없이)
```

### 문제 있던 플로우 (3ff92eb):

```
[HDF5 파일]
    ↓
[NPZ: q_ref + joint_names (with q_ prefix)]
    ↓
[series_data: {
    "pelvis_tx": [...],        ← q_ prefix 제거됨
    ...
}]
    ↓
[Environment에서 사용]
for key in ['pelvis_tx', ...]:
    joint(key).qpos = series_data[f"q_{key}"][index]  ← ❌ KeyError or 0!
                                    ^^^^^^^^^
                                    존재하지 않음!
```

---

## ✅ 결론

**정강이가 힙으로 가는 문제의 원인:**
1. **직접 원인:** `q_` prefix 불일치로 데이터 접근 실패
2. **2차 원인:** 초기화되지 않은 qpos 사용 (0 또는 쓰레기 값)
3. **시각적 결과:** Kinematic chain 붕괴 (femur 0, tibia 0 → 정강이가 힙에 붙음)

**해결책:**
- 원본 코드로 복구
- `series_data[key]` 사용 (prefix 없이)
- 렌더링은 별도 스크립트 사용

**교훈:**
- 데이터 계약 준수
- 단순함이 최고
- 관심사의 분리 (Separation of Concerns)

---

**작성:** GitHub Copilot  
**날짜:** 2024-11-18  
**상태:** ✅ 문제 해결됨, 근본 원인 파악됨
