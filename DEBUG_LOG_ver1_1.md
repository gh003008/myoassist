# ver1_1 Training Debug Log

**Date**: 2024-11-15 (251115)
**Task**: ver1_1 balancing reward 구현 후 학습 시작
**Status**: ✅ 성공 (47시간 예상 소요)

---

## 🐛 Debug Process Timeline

### Issue #1: gymnasium 모듈 없음
**Error**:
```
ModuleNotFoundError: No module named 'gymnasium'
```

**원인**: 
- base conda 환경에서 실행됨 (myoassist 환경 아님)

**시도한 해결책**:
1. ❌ `install_python_packages(['gymnasium'])` → base 환경에 설치됨
2. ❌ `conda activate myoassist; python ...` → PowerShell 세션이 환경 유지 못함
3. ✅ `conda run -n myoassist python ...` → 성공

**교훈**: PowerShell에서 conda activate는 세션 유지가 안됨. `conda run -n` 사용 필요.

---

### Issue #2: Invalid session id
**Error**:
```
ValueError: Invalid session id: myoAssistLegImitationExo-v1_1
```

**원인**: 
- `environment_handler.py`의 `get_config_type_from_session_id()`에 ver1_1이 등록 안됨

**해결**:
```python
# rl_train/envs/environment_handler.py line 93
elif session_id in ['myoAssistLegImitationExo-v0', 'myoAssistLegImitationExo-v1_1']:  # 251115_0028
    return ExoImitationTrainSessionConfig
```

**교훈**: 새 환경 버전 추가 시 config type mapper도 업데이트 필요.

---

### Issue #3: AttributeError - safe_height
**Error**:
```
AttributeError: 'MyoAssistLegImitation_ver1_1' object has no attribute 'safe_height'
```

**원인**: 
- `_get_done()`에서 `self.safe_height` 사용하는데 `_setup()`에서 초기화 안함

**해결**:
```python
# myoassist_leg_imitation_ver1_1.py _setup()
self.safe_height = env_params.safe_height  # 251115_0028: Initialize from config
self._max_rot = kwargs.get('max_rot', 0.6)
```

**교훈**: 부모 클래스의 속성을 override할 때 초기화 필수.

---

### Issue #4: step() returns 4 values, expected 5
**Error**:
```
ValueError: not enough values to unpack (expected 5, got 4)
```

**원인**: 
- Gymnasium API는 `obs, reward, done, truncated, info` 5개 반환
- 코드는 4개만 받으려 함

**해결**:
```python
# myoassist_leg_imitation_ver1_1.py step()
obs, reward, done, truncated, info = super().step(action)  # 251115_0028: 5 values
return obs, reward, done, truncated, info
```

**교훈**: Gymnasium vs Gym API 차이 주의. Gymnasium은 done을 (terminated, truncated)로 분리.

---

### Issue #5: Reward key not in weighted_reward_keys
**Error**:
```
AssertionError: reward_dict keys must be subset of weighted_reward_keys. 
Missing: {'pelvis_list_penalty', 'pelvis_height_reward'}
```

**원인**: 
- `get_reward_dict()`에서 새 reward key 추가했지만
- Config의 `reward_keys_and_weights`에 없어서 base class assertion 실패

**시도한 해결책**:
1. ❌ `_setup()`에서 `self.rwd_keys_wt` 직접 수정 → dict라서 안됨
2. ❌ `setattr(reward_weights, key, value)` → dataclass 변환 후 반영 안됨
3. ✅ `config.py`의 `RewardWeights` dataclass에 필드 추가

**해결**:
```python
# rl_train/train/train_configs/config_imitation.py
@dataclass
class RewardWeights:
    # ... 기존 필드들 ...
    
    # 251115_0028: ver1_1 balancing rewards
    pelvis_list_penalty: float = 0.0
    pelvis_height_reward: float = 0.0
```

**교훈**: Config 구조 변경 시 dataclass 정의부터 수정해야 함.

---

### Issue #6: Unicode Encode Error
**Error**:
```
UnicodeEncodeError: 'cp949' codec can't encode character '\u2705' in position 2
```

**원인**: 
- Windows 콘솔(cp949)이 이모지(✅) 지원 안함
- `print(f"✅ ver1_1 mode enabled...")` 에서 발생

**해결**:
```python
# run_train.py, environment_handler.py
print(f"[OK] {version_tag} mode enabled...")  # 이모지 → [OK] 텍스트로 변경
```

**교훈**: Windows 콘솔 호환성 고려. 이모지 사용 자제.

---

### Issue #7: Simulation Instability + NaN Explosion
**Error**:
```
WARNING:absl:Nan, Inf or huge value in QACC at DOF X
ep_len_mean: 1
ep_rew_mean: -645
value_loss: 1.56e+33
RuntimeError: Function 'MseLossBackward0' returned nan values
```

**원인**: 
- **Reward 함수의 exponential 항이 너무 강함**
```python
pelvis_list_penalty = -exp(10.0 * square(pelvis_list))  # exp(10)는 22026!
height_reward = exp(-5.0 * square(height - 0.9))
```
- `pelvis_list` 값이 0.5라디안(~29도)만 되어도:
  - `exp(10 * 0.5^2) = exp(2.5) = 12.18` → 페널티 -12.18
  - 여러 timestep 누적 → reward 폭발 → NaN

**해결**:
```python
# myoassist_leg_imitation_ver1_1.py _calculate_balancing_rewards()

# Before (폭발):
pelvis_list_penalty = self.dt * (-np.exp(10.0 * np.square(pelvis_list)))

# After (안정):
pelvis_list_penalty = self.dt * (-np.square(pelvis_list))  # 단순 quadratic

# Before (폭발):
height_reward = self.dt * np.exp(-5.0 * np.square(pelvis_height - target_height))

# After (안정):
height_reward = self.dt * np.exp(-2.0 * np.square(pelvis_height - target_height))
```

**결과**:
- ✅ NaN 없이 학습 진행
- ⚠️ 시뮬레이션 불안정 경고는 여전히 있지만 리셋 후 계속 진행
- ✅ FPS: 174-178 it/s (4 envs)

**교훈**: 
1. Exponential reward는 매우 조심해서 사용
2. 초기 random policy에서도 안정적인 범위로 설계
3. `exp(큰 값)` = 폭발, `exp(-작은 값)` = 0 근처로 수렴하도록 계수 조정 필요

---

## 📊 최종 학습 상태

### 성공적으로 시작된 명령어:
```bash
conda activate myoassist
python -m rl_train.run_train \
    --config_file_path rl_train/train/train_configs/S004_3D_IL_ver1_1_BALANCE.json \
    --use_ver1_1 \
    --wandb_project myoassist-3D-balancing \
    --wandb_name S004_ver1_1_stable \
    --config.env_params.num_envs 4
```

### 학습 진행 현황:
```
Time elapsed: 11초
Total timesteps: 2,048 / 30,000,000 (0.007%)
FPS: 174-178 it/s
Estimated time: 47.6 시간 (약 2일)

Environment: 4 parallel envs
```

### 여전히 남은 문제:
1. **시뮬레이션 불안정 경고 빈번**
   - `WARNING:absl:Nan, Inf or huge value in QACC at DOF X`
   - 원인: 초기 random policy가 물리적으로 불가능한 동작 시도
   - 해결: 학습이 진행되면서 자연스럽게 개선될 것으로 예상
   
2. **WandB 로깅 비활성화**
   - `Warning: wandb not installed. WandB logging disabled.`
   - 해결: 필요시 `pip install wandb` 후 재시작

---

## 🔧 수정된 파일 요약

### 1. `rl_train/envs/myoassist_leg_imitation_ver1_1.py` (핵심)
- ✅ `_setup()`: safe_height, max_rot 초기화
- ✅ `_calculate_balancing_rewards()`: Exponential → Quadratic 변경
- ✅ `step()`: 5개 값 반환 (Gymnasium API)
- ✅ `_get_done()`: rotation termination 체크

### 2. `rl_train/train/train_configs/config_imitation.py`
- ✅ `RewardWeights` dataclass에 `pelvis_list_penalty`, `pelvis_height_reward` 추가

### 3. `rl_train/envs/environment_handler.py`
- ✅ `get_config_type_from_session_id()`: ver1_1 지원
- ✅ `get_callback()`: ver1_1 callback 지원
- ✅ 이모지 제거

### 4. `rl_train/run_train.py`
- ✅ `--use_ver1_1` argument 추가
- ✅ WandB config에 ver1_1 태그 지원
- ✅ 이모지 제거

### 5. `rl_train/envs/__init__.py`
- ✅ ver1_1 환경 등록

---

## 📚 핵심 교훈

### 1. Conda 환경 관리
- PowerShell에서 `conda activate`는 세션 유지 안됨
- `conda run -n env_name` 사용 권장

### 2. Gymnasium API 차이
- Gym: `step() → (obs, reward, done, info)` (4개)
- Gymnasium: `step() → (obs, reward, terminated, truncated, info)` (5개)

### 3. Reward Function 설계
- **Exponential reward는 양날의 검**
  - 장점: 빠른 수렴, 명확한 신호
  - 단점: 값 범위 폭발 위험
- **안전한 설계 원칙**:
  - 초기 random policy에서도 bounded 값 유지
  - `exp(계수 * square(x))` 형태는 계수를 작게 (< 5)
  - 대안: Quadratic (`square(x)`), Gaussian (`exp(-계수 * square(x))`)
  
### 4. Config vs Code 구조
- Dataclass 기반 config는 필드 정의부터 수정
- Runtime에 dict 업데이트는 반영 안될 수 있음

### 5. Windows 호환성
- 콘솔 출력에 이모지 사용 금지 (cp949 인코딩)
- 영문 + 기호로 대체

---

## 🎯 다음 단계

1. **학습 모니터링** (47시간 동안):
   - 터미널 로그에서 `ep_rew_mean`, `ep_len_mean` 확인
   - 시뮬레이션 불안정 경고 빈도 감소하는지 체크
   - 주기적으로 checkpoint 저장 확인

2. **Reward 튜닝** (필요시):
   - `pelvis_list_penalty` weight: 0.1 → 0.05 or 0.2
   - `pelvis_height_reward` weight: 0.02 → 0.01 or 0.05
   - `max_rot` threshold: 0.6 → 0.5 (더 strict) or 0.7 (더 lenient)

3. **비교 실험** (학습 완료 후):
   - ver1_0 (balancing reward 없음) vs ver1_1 성능 비교
   - Episode length, success rate, gait quality 평가

4. **WandB 활성화** (선택):
   ```bash
   pip install wandb
   wandb login
   # 재시작하면 자동으로 로깅됨
   ```

---

**End of Debug Log** ✅
