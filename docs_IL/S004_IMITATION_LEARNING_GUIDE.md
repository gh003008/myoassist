# S004 Motion Imitation Learning Pipeline

이 가이드는 OpenSim 모션 데이터를 사용하여 MyoAssist에서 모방학습(Imitation Learning)을 실행하는 전체 워크플로우를 설명합니다.

## 📋 목차

1. [모방학습이란?](#모방학습이란)
2. [데이터 변환](#데이터-변환)
3. [환경 설정 검증](#환경-설정-검증)
4. [학습 실행](#학습-실행)
5. [결과 평가](#결과-평가)
6. [트러블슈팅](#트러블슈팅)

---

## 🎯 모방학습이란?

### MyoAssist의 모방학습 방식

이 프레임워크는 **"Reference Motion Tracking with Reward Shaping"** 방식을 사용합니다.

- **방식**: PPO (Proximal Policy Optimization) + Imitation Reward
- **차이점**: GAIL이나 AIRL이 아님 (Discriminator 없음)
- **특징**:
  - Reference trajectory를 직접 보상 함수에 포함
  - 관절 위치/속도 차이를 exponential reward로 계산
  - 계산이 가볍고 구현이 단순
  - 로보틱스 분야에서 널리 사용됨

### 보상 함수 구조

```python
# 관절 위치 보상
q_reward = dt * exp(-8 * (qpos_diff)²)

# 관절 속도 보상  
dq_reward = dt * exp(-8 * (qvel_diff)²)

# 총 보상 = 모방 보상 + 전진 보상 + 페널티
```

---

## 🔄 데이터 변환

### 1. OpenSim 데이터 구조

OpenSim NPZ 파일은 다음 구조를 가집니다:

```
- model_states: (N, 63) - 전체 상태 데이터
- model_states_columns: 컬럼 이름 (63개)
  - pelvis_tx, pelvis_ty, pelvis_tz
  - knee_angle_r/l, ankle_angle_r/l
  - hip_r/l_0~5 (6DOF 회전/이동)
  - 각속도, 접촉력 등
- sampling_rate: 샘플링 레이트 (Hz)
- height_m, weight_kg: 신체 정보
```

### 2. MyoAssist 형식

MyoAssist는 다음 구조를 요구합니다:

```python
{
    'metadata': {
        'sample_rate': 100,
        'data_length': 12028,
        'height_m': 1.74,
        'weight_kg': 70.56
    },
    'series_data': {
        'q_pelvis_tx': array[N],
        'q_pelvis_ty': array[N],
        'q_pelvis_tilt': array[N],
        'q_hip_flexion_r': array[N],
        'q_hip_flexion_l': array[N],
        'q_knee_angle_r': array[N],
        'q_knee_angle_l': array[N],
        'q_ankle_angle_r': array[N],
        'q_ankle_angle_l': array[N],
        'dq_*': array[N]  # 속도 데이터
    }
}
```

### 3. 변환 실행

```bash
# 기본 변환
python opensim2myoassist_converter.py "C:/workspace_home/opensim data/LD_gdp/S004/level_08mps/trial_01.npz" "rl_train/reference_data/S004_trial01_08mps.npz"

# 샘플링 레이트 지정
python opensim2myoassist_converter.py input.npz output.npz --sample_rate 30
```

### 4. 변환 결과 확인

```bash
python inspect_data_structures.py
```

---

## ✅ 환경 설정 검증

학습 전에 환경이 제대로 설정되었는지 확인:

```bash
python verify_S004_setup.py
```

이 스크립트는 다음을 확인합니다:
- ✅ Reference data 로드 가능
- ✅ 필수 키 존재 여부
- ✅ 환경 생성 가능
- ✅ 환경 reset/step 작동

---

## 🚀 학습 실행

### 방법 1: 간편 스크립트 (권장)

```bash
# 1. 빠른 테스트 (설정 확인용)
python train_S004_motion.py --quick_test

# 2. 기본 학습 (16개 병렬 환경)
python train_S004_motion.py

# 3. 환경 개수 조정 (PC 사양에 맞춤)
python train_S004_motion.py --num_envs 8

# 4. GPU 사용
python train_S004_motion.py --device cuda --num_envs 32

# 5. 렌더링 포함
python train_S004_motion.py --render
```

### 방법 2: 직접 실행

```bash
python rl_train/run_train.py --config_file_path rl_train/train/train_configs/S004_trial01_08mps_config.json
```

### 학습 중 모니터링

학습 중에는 다음 정보가 표시됩니다:
- Rollout 진행 상황
- 평균 보상
- Episode 길이
- 학습 통계

결과는 `rl_train/results/train_session_[timestamp]/`에 저장됩니다.

---

## 📊 결과 평가

### 1. 학습된 정책 평가

```bash
python rl_train/run_policy_eval.py rl_train/results/train_session_[timestamp]
```

### 2. 생성되는 결과물

- `analyze_results_[timesteps]_[num]/`
  - 보행 분석 그래프
  - 관절 궤적 비교
  - 근육 활성화 패턴
  - 영상 파일 (MP4)

### 3. 실시간 시각화

```bash
python rl_train/run_train.py \
    --config_file_path rl_train/results/train_session_[timestamp]/session_config.json \
    --config.env_params.prev_trained_policy_path rl_train/results/train_session_[timestamp]/trained_models/model_[steps] \
    --flag_realtime_evaluate
```

---

## 🎛️ 설정 커스터마이징

### Config 파일 수정

`rl_train/train/train_configs/S004_trial01_08mps_config.json`:

```json
{
    "total_timesteps": 3e7,  // 총 학습 스텝
    "env_params": {
        "num_envs": 16,  // 병렬 환경 개수
        "min_target_velocity": 0.8,  // 목표 속도 (m/s)
        "reference_data_path": "rl_train/reference_data/S004_trial01_08mps.npz",
        "reward_keys_and_weights": {
            "qpos_imitation_rewards": {
                "knee_angle_l": 1.0,  // 무릎 보상 가중치
                "hip_flexion_l": 0.2,  // 고관절 보상 가중치
                ...
            }
        }
    },
    "ppo_params": {
        "learning_rate": 0.0001,
        "n_steps": 1024,  // 환경 개수에 따라 조정
        "batch_size": 8192,
        "device": "cpu"  // 'cuda' for GPU
    }
}
```

### 보상 가중치 튜닝

중요한 관절에 더 높은 가중치 부여:
- `knee_angle_*`: 1.0 (무릎이 중요)
- `pelvis_tilt`: 1.0 (자세 유지)
- `hip_flexion_*`: 0.2 (미세 조정)
- `ankle_angle_*`: 0.2 (발목 움직임)

---

## 🔧 트러블슈팅

### 1. 메모리 부족

```bash
# 병렬 환경 개수 줄이기
python train_S004_motion.py --num_envs 4

# n_steps 조정 (batch_size 유지)
# num_envs * n_steps ≈ 16384
```

### 2. 학습이 불안정할 때

Config 파일에서 조정:
```json
{
    "ppo_params": {
        "learning_rate": 0.00005,  // 학습률 감소
        "clip_range": 0.1,  // 클리핑 범위 감소
        "target_kl": 0.005  // KL divergence 제한 강화
    }
}
```

### 3. 보상이 개선되지 않을 때

- Reference motion 품질 확인
- 보상 가중치 재조정
- 목표 속도가 reference와 일치하는지 확인
- `out_of_trajectory_threshold` 조정

### 4. 환경 생성 실패

```bash
# 의존성 확인
pip install -r requirements.txt

# 환경 검증
python verify_S004_setup.py
```

### 5. NumPy 버전 문제

```bash
# NumPy 호환성 확인
pip install numpy==1.23.5
```

---

## 📚 추가 리소스

### 튜토리얼 노트북

- `docs/tutorial/rl_imitation_tutorial.ipynb` - 모방학습 기초
- `docs/tutorial/rl_analyze_tutorial.ipynb` - 결과 분석

### 관련 문서

- [MyoAssist 공식 문서](https://myoassist.neumove.org/)
- [Reinforcement Learning](https://myoassist.neumove.org/reinforcement-learning/)
- [Configuration Guide](https://myoassist.neumove.org/reinforcement-learning/02_configuration)

### 참고 논문

이 방식은 다음과 유사합니다:
- DeepMimic (SIGGRAPH 2018)
- Motion Imitation via Deep RL with Reward Shaping

---

## 🎉 요약

```bash
# 1단계: 데이터 변환
python opensim2myoassist_converter.py "input.npz" "output.npz"

# 2단계: 환경 검증
python verify_S004_setup.py

# 3단계: 학습 실행
python train_S004_motion.py

# 4단계: 결과 평가
python rl_train/run_policy_eval.py rl_train/results/train_session_[timestamp]
```

---

**문제가 발생하면:**
1. `verify_S004_setup.py` 실행
2. Config 파일의 경로 확인
3. Reference data 형식 확인
4. GitHub Issues 검색: https://github.com/neumovelab/myoassist/issues
