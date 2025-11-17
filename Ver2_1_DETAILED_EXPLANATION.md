# Ver2_1 (Karico + Balancing) 상세 설명

## 🎯 **목표**
Ver2_0 Karico (stable training release) + 3D Balancing Rewards

---

## 📦 **1. Ver2_0 Karico Base (GitHub main 브랜치)**

### **핵심 개선사항:**
✅ **FIXED Reference Motion** 
- `convert_hdf5_direct.py`: HDF5 → MyoAssist NPZ 직접 변환
- 파일: `rl_train/reference_data/S004_trial01_08mps_3D_HDF5_v7.npz`
- 특징:
  - OpenSim 좌표계 직접 매핑 (transform 없음)
  - Relative positions (centered at 0)
  - 정확한 pelvis rotation order (tilt, list, rotation)

✅ **Add Resume 기능**
- 학습 중단 후 재개 가능
- Checkpoint 자동 로드

✅ **Training Stability**
- 안정적인 학습 파이프라인
- WandB 통합 (네트워크 에러 처리 포함)

---

## 🔥 **2. Ver2_1 Balancing Extensions (ghlee-lab 브랜치)**

### **추가된 파일:**

#### A. **환경 파일: `myoassist_leg_imitation_ver2_1.py`**
```python
class MyoAssistLegImitation_ver2_1(MyoAssistLegImitation):
    """Ver2_0 Karico + 3D Balancing Rewards"""
```

**구현 위치 및 내용:**

**1) `_setup()` 메서드 (Line 53-68)**
```python
def _setup(self, *, env_params, reference_data, **kwargs):
    # Rotation termination threshold 저장
    self._max_rot = kwargs.get('max_rot', 0.6)  # cos(53°)
    self.safe_height = env_params.safe_height
    
    # 부모 클래스 (Ver2_0 Karico) setup 호출
    super()._setup(...)
```
- **역할**: Balancing 파라미터 초기화
- **Ver2_0와의 차이**: `_max_rot` 파라미터 추가

**2) `_calculate_balancing_rewards()` 메서드 (Line 70-94)**
```python
def _calculate_balancing_rewards(self):
    balance_rewards = {}
    
    # [HIGH] Pelvis roll penalty
    pelvis_list = self.sim.data.joint('pelvis_list').qpos[0]
    pelvis_list_penalty = self.dt * (-np.square(pelvis_list))
    balance_rewards['pelvis_list_penalty'] = float(pelvis_list_penalty)
    
    # [MEDIUM] Pelvis height reward
    pelvis_height = self.sim.data.body('pelvis').xpos[2]
    target_height = 0.9
    height_reward = self.dt * np.exp(-2.0 * np.square(pelvis_height - target_height))
    balance_rewards['pelvis_height_reward'] = float(height_reward)
    
    return balance_rewards
```
- **역할**: 3D 균형 유지를 위한 보상 계산
- **pelvis_list_penalty**: 
  - 좌우 기울기(roll) 페널티
  - Quadratic 형태 (NaN 방지)
  - 가중치: config에서 0.1
- **pelvis_height_reward**:
  - 서있는 자세 유지 보상
  - Target: 0.9m
  - 가중치: config에서 0.02

**3) `_check_rotation_termination()` 메서드 (Line 96-117)**
```python
def _check_rotation_termination(self):
    # Pelvis orientation을 quaternion에서 rotation matrix로 변환
    pelvis_quat = self.sim.data.body('pelvis').xquat
    rot_mat = quat2mat(pelvis_quat)
    
    # Forward direction (Z-axis)
    forward_dir = rot_mat[:, 2]  # 3rd column
    reference_forward = np.array([0, 0, 1])  # World Z-axis
    
    # Cosine similarity (내적)
    cos_sim = np.dot(forward_dir, reference_forward)
    
    # Termination if rotated too much
    if cos_sim < self._max_rot:
        return True
    return False
```
- **역할**: 과도한 회전 시 episode 종료
- **Threshold**: cos(53°) = 0.6
- **효과**: 넘어지기 전에 episode 종료 → 불안정한 policy 학습 방지

**4) `get_reward_dict()` 메서드 오버라이드 (Line 119-137)**
```python
def get_reward_dict(self, obs_dict):
    # 부모 클래스 (Ver2_0) reward 계산
    rwd_dict = super().get_reward_dict(obs_dict)
    
    # Balancing rewards 추가
    balance_rewards = self._calculate_balancing_rewards()
    rwd_dict.update(balance_rewards)
    
    # Dense reward 재계산 (balancing 포함)
    rwd_dict['dense'] = np.sum([
        wt * rwd_dict[key] 
        for key, wt in self.rwd_keys_wt.items() 
        if key in rwd_dict
    ], axis=0)
    
    return rwd_dict
```
- **역할**: Ver2_0 rewards + Balancing rewards 통합
- **Ver2_0와의 차이**: `balance_rewards` 추가 및 `dense` 재계산

**5) `_get_done()` 메서드 오버라이드 (Line 139-152)**
```python
def _get_done(self):
    # 기존 termination (pelvis height)
    pelvis_height = self.sim.data.joint('pelvis_ty').qpos[0].copy()
    if pelvis_height < self.safe_height:
        return True
    
    # 새로운 termination (rotation)
    if self._check_rotation_termination():
        return True
    
    return False
```
- **역할**: 기존 높이 기반 + 새로운 회전 기반 termination
- **Ver2_0와의 차이**: `_check_rotation_termination()` 체크 추가

---

#### B. **Config 파일 수정**

**1) `config_imitation.py` (Line 16-19)**
```python
class RewardWeights:
    # ... 기존 rewards ...
    
    # 251117_Ver2_1: Balancing rewards
    pelvis_list_penalty: float = 0.0
    pelvis_height_reward: float = 0.0
```
- **역할**: RewardWeights dataclass에 balancing 필드 추가
- **기본값**: 0.0 (Ver2_0 호환성 유지)

**2) `config_imiatation_exo.py`**
- Ver2_1 config type 명시적 정의 (필요시)

**3) `S004_3D_IL_ver2_1_BALANCE.json`**
```json
{
    "env_params": {
        "env_id": "myoAssistLegImitationExo-v2_1",
        "reference_data_path": "rl_train/reference_data/S004_trial01_08mps_3D_HDF5_v7.npz",
        
        "reward_keys_and_weights": {
            // ... Ver2_0 기존 rewards ...
            
            "pelvis_list_penalty": 0.1,    // Balancing: Roll penalty
            "pelvis_height_reward": 0.02   // Balancing: Height reward
        },
        
        "max_rot": 0.6  // Rotation termination threshold
    },
    "total_timesteps": 3e7,
    "ppo_params": {
        "device": "cuda",
        "learning_rate": 3e-05
    }
}
```

---

#### C. **환경 등록**

**1) `__init__.py`**
```python
register_env_myoassist(
    id='myoAssistLegImitationExo-v2_1',
    entry_point='rl_train.envs.myoassist_leg_imitation_ver2_1:MyoAssistLegImitation_ver2_1',
    max_episode_steps=1000,
    kwargs={},
)
```

**2) `environment_handler.py`**
```python
def get_config_type_from_session_id(session_id):
    # ...
    elif session_id in ['myoAssistLegImitationExo-v2_1']:
        return ExoImitationTrainSessionConfig
```

---

## 🔬 **3. 코드 구조 비교**

### **Ver2_0 Karico (Base)**
```
MyoAssistLegImitation (ver1_0)
├── _setup()              # 기본 초기화
├── get_reward_dict()     # Ver2_0 rewards
│   ├── qpos_imitation
│   ├── qvel_imitation
│   ├── forward_reward
│   ├── muscle_activation
│   └── foot_force
├── _get_done()           # pelvis_height < safe_height
└── step()                # 기본 step logic
```

### **Ver2_1 (Karico + Balancing)**
```
MyoAssistLegImitation_ver2_1 (extends ver1_0)
├── _setup()              # + max_rot parameter
├── get_reward_dict()     # Ver2_0 + Balancing
│   ├── [Ver2_0 rewards]
│   ├── pelvis_list_penalty     [NEW]
│   └── pelvis_height_reward    [NEW]
├── _calculate_balancing_rewards()  [NEW]
│   ├── pelvis_list penalty (quadratic)
│   └── pelvis_height reward (exponential)
├── _check_rotation_termination()   [NEW]
│   └── quaternion → rotation matrix → cosine similarity
├── _get_done()           # + rotation termination
│   ├── [pelvis_height check]
│   └── [rotation check]           [NEW]
└── step()                # Inherited from ver1_0
```

---

## 📊 **4. Reward 구조 상세**

### **Ver2_0 Rewards (Base)**
```python
rwd_dict = {
    # Imitation
    'qpos_imitation': weighted_sum(qpos_errors),
    'qvel_imitation': weighted_sum(qvel_errors),
    'end_effector_imitation': COM_tracking,
    
    # Task
    'forward_reward': pelvis_tx_velocity,
    
    # Regularization
    'muscle_activation_penalty': -sum(activations^2),
    'muscle_activation_diff_penalty': -sum(diff^2),
    'foot_force_penalty': -sum(excessive_forces),
    
    # Total
    'dense': weighted_sum(all_rewards)
}
```

### **Ver2_1 Rewards (Karico + Balancing)**
```python
rwd_dict = {
    # [Ver2_0 모든 rewards] +
    
    # Balancing (3D Stability)
    'pelvis_list_penalty': -square(roll_angle) * 0.1,
    'pelvis_height_reward': exp(-2*(height-0.9)^2) * 0.02,
    
    # Total (recalculated)
    'dense': weighted_sum(ver2_0_rewards + balancing_rewards)
}
```

### **Reward Weights**
```
qpos_imitation:
  - pelvis_tilt: 1.0 (가장 중요)
  - knee: 1.0
  - pelvis_list: 0.5
  - hip_flexion: 0.5
  - hip_adduction/rotation: 0.3
  - ankle: 0.2
  - pelvis translations: 0.1

qvel_imitation: 0.1-0.2 (position보다 낮음)

forward_reward: 0.2
muscle penalties: 0.1

foot_force_penalty: 0.5

** Balancing (Ver2_1) **
pelvis_list_penalty: 0.1   # 3D stability
pelvis_height_reward: 0.02 # 서있기 유지
```

---

## 🎮 **5. Termination 조건**

### **Ver2_0**
```python
if pelvis_height < safe_height (0.7m):
    terminate = True
```

### **Ver2_1 (+ Rotation)**
```python
if pelvis_height < safe_height (0.7m):
    terminate = True

OR

if cos_similarity(pelvis_forward, world_Z) < 0.6:  # ~53° rotation
    terminate = True
```

**효과:**
- 과도하게 기울어진 상태에서 계속 학습하는 것 방지
- 넘어지기 직전 상태를 bad example로 학습하지 않음

---

## 🔧 **6. WandB 설정**

### **Online 모드 (기본)**
```python
# rl_train/envs/myoassist_leg_imitation_ver1_0.py
wandb.init(
    project='myoassist-3D-balancing',
    name='S004_ver2_1_karico_balance',
    settings=wandb.Settings(
        _disable_stats=True,   # Network traffic 감소
        _disable_meta=True,    # Network traffic 감소
    )
)
```

### **네트워크 에러 처리**
```python
try:
    wandb.init(...)
    wandb.log(...)
except Exception as e:
    print(f"⚠️ WandB 에러: {e}")
    print("   로컬 로그만 사용합니다.")
    self._wandb_enabled = False
    # 학습은 계속 진행 (중단되지 않음)
```

**특징:**
- 네트워크 끊겨도 학습 계속
- Reduced logging frequency (1000 steps마다)
- Local logs 항상 저장됨

---

## 📂 **7. 파일 위치 요약**

```
rl_train/
├── envs/
│   ├── myoassist_leg_imitation_ver1_0.py    # Ver2_0 Karico base
│   ├── myoassist_leg_imitation_ver2_1.py    # Ver2_0 + Balancing ⭐
│   ├── __init__.py                          # 환경 등록
│   └── environment_handler.py               # Config type mapping
│
├── train/train_configs/
│   ├── config_imitation.py                  # RewardWeights 정의 (수정됨)
│   ├── config_imiatation_exo.py             # ExoConfig
│   ├── S004_3D_IL_ver1_0_BASE.json          # Ver2_0 base config
│   └── S004_3D_IL_ver2_1_BALANCE.json       # Ver2_1 config ⭐
│
└── reference_data/
    └── S004_trial01_08mps_3D_HDF5_v7.npz    # FIXED reference motion
```

---

## ⚙️ **8. 학습 실행 커맨드**

```bash
# ghlee-lab 브랜치에서
conda activate myoassist

python -m rl_train.run_train \
    --config_file_path rl_train/train/train_configs/S004_3D_IL_ver2_1_BALANCE.json \
    --wandb_project myoassist-3D-balancing \
    --wandb_name S004_ver2_1_karico_balance_8envs \
    --config.env_params.num_envs 8 \
    --config.ppo_params.device cuda
```

**Parameters:**
- `total_timesteps`: 30,000,000
- `num_envs`: 8 (parallel)
- `device`: cuda (RTX 3060 Ti)
- `learning_rate`: 3e-05
- `estimated_time`: ~27-28시간

---

## 🔍 **9. Ver2_0 vs Ver2_1 핵심 차이**

| 항목 | Ver2_0 Karico | Ver2_1 (Karico + Balance) |
|------|---------------|---------------------------|
| **Base** | FIXED reference motion | ✓ Same |
| **Imitation Rewards** | qpos, qvel, end_effector | ✓ Same |
| **Task Rewards** | forward, muscle penalties | ✓ Same |
| **Balancing** | ❌ None | ✅ pelvis_list_penalty, pelvis_height_reward |
| **Termination** | Height only | ✅ Height + Rotation |
| **3D Stability** | 학습 과정에서 자연스럽게 | ✅ 명시적 보상/페널티 |
| **Use Case** | 2D-like 학습 | **3D 균형 필수 환경** |

---

## 🎯 **10. 기대 효과**

### **Ver2_0만 사용할 때:**
- ✅ Reference motion 정확히 모방
- ⚠️ 3D 균형이 학습 후반에 자연스럽게 나타남 (느림)
- ⚠️ 초기 학습 시 자주 넘어짐

### **Ver2_1 사용할 때:**
- ✅ Reference motion 정확히 모방
- ✅ 3D 균형을 명시적으로 학습 (빠름)
- ✅ 초기부터 안정적인 자세 유지
- ✅ Rotation termination으로 bad examples 방지

---

## 📝 **11. 다음 단계**

1. ✅ ghlee-lab 브랜치로 전환 완료
2. ✅ Ver2_1 코드 작성 완료
3. ⏳ Git commit & push
4. 🚀 학습 시작
5. 📊 Ver2_0 vs Ver2_1 비교 분석

---

**Created**: 2024-11-17
**Branch**: ghlee-lab
**Status**: Ready for training
**Base**: Ver2_0 Karico (main branch)
**Extension**: 3D Balancing Rewards
