# 2D vs 3D 모델 비교 가이드

## 🎯 개요

MyoAssist는 **2D (Sagittal Plane)** 와 **3D (Full Kinematics)** 두 가지 모델을 지원합니다.

---

## 📊 2D vs 3D 모델 비교

| 특성 | 2D 모델 (22 근육) | 3D 모델 (26 근육) |
|------|-------------------|-------------------|
| **운동 평면** | Sagittal만 (앞뒤) | 전체 3D |
| **DOF** | 8 | 16 |
| **근육 수** | 22 | 26 |
| **Pelvis** | 3 DOF (tx, ty, tilt) | 6 DOF (tx, ty, tz, list, tilt, rotation) |
| **Hip** | 1 DOF/side (flexion) | 3 DOF/side (flexion, adduction, rotation) |
| **Knee** | 1 DOF/side | 1 DOF/side |
| **Ankle** | 1 DOF/side | 1 DOF/side |
| **계산 부하** | 가벼움 ⚡ | 무거움 🔥 |
| **학습 속도** | 빠름 | 느림 |
| **권장 환경 수** | 16-32 | 4-8 |
| **적합한 작업** | 평지 보행, 빠른 프로토타이핑 | 복잡한 동작, 측면 안정성 |

---

## 📁 파일 및 설정 비교

### 2D 모델
```
모델: models/22muscle_2D/myoLeg22_2D_BASELINE.xml
설정: rl_train/train/train_configs/S004_trial01_08mps_config.json
데이터: rl_train/reference_data/S004_trial01_08mps.npz
학습: python train_S004_motion.py

변환기: opensim2myoassist_converter.py
```

### 3D 모델
```
모델: models/26muscle_3D/myoLeg26_BASELINE.xml
설정: rl_train/train/train_configs/S004_trial01_08mps_3D_config.json
데이터: rl_train/reference_data/S004_trial01_08mps_3D.npz
학습: python train_S004_motion_3D.py

변환기: opensim2myoassist_3D_converter.py
```

---

## 🔄 데이터 변환 비교

### 2D 데이터 변환
```bash
python opensim2myoassist_converter.py \
    "C:/workspace_home/opensim data/LD_gdp/S004/level_08mps/trial_01.npz" \
    "rl_train/reference_data/S004_trial01_08mps.npz"
```

**추출 데이터 (20개 신호):**
- `q_pelvis_tx, q_pelvis_ty, q_pelvis_tilt`
- `q_hip_flexion_r/l`
- `q_knee_angle_r/l`
- `q_ankle_angle_r/l`
- `dq_*` (속도)

### 3D 데이터 변환
```bash
python opensim2myoassist_3D_converter.py \
    "C:/workspace_home/opensim data/LD_gdp/S004/level_08mps/trial_01.npz" \
    "rl_train/reference_data/S004_trial01_08mps_3D.npz"
```

**추출 데이터 (32개 신호):**
- `q_pelvis_tx, q_pelvis_ty, q_pelvis_tz`
- `q_pelvis_list, q_pelvis_tilt, q_pelvis_rotation`
- `q_hip_flexion_r/l, q_hip_adduction_r/l, q_hip_rotation_r/l`
- `q_knee_angle_r/l`
- `q_ankle_angle_r/l`
- `qd*` (속도)

---

## ⚙️ 학습 설정 비교

### 2D 모델 설정
```json
{
    "env_params": {
        "model_path": "models/22muscle_2D/myoLeg22_2D_BASELINE.xml",
        "num_envs": 16,
        "reference_data_keys": [
            "ankle_angle_l", "ankle_angle_r",
            "hip_flexion_l", "hip_flexion_r",
            "knee_angle_l", "knee_angle_r",
            "pelvis_tilt", "pelvis_tx", "pelvis_ty"
        ],
        "reward_keys_and_weights": {
            "qpos_imitation_rewards": {
                "pelvis_ty": 0.1,
                "pelvis_tilt": 1.0,
                "hip_flexion_l": 0.2,
                "hip_flexion_r": 0.2,
                "knee_angle_l": 1.0,
                "knee_angle_r": 1.0,
                "ankle_angle_l": 0.2,
                "ankle_angle_r": 0.2
            }
        }
    },
    "policy_params": {
        "custom_policy_params": {
            "net_arch": {
                "human_actor": [64, 64],
                "exo_actor": [8, 8],
                "common_critic": [64, 64]
            }
        }
    }
}
```

### 3D 모델 설정
```json
{
    "env_params": {
        "model_path": "models/26muscle_3D/myoLeg26_BASELINE.xml",
        "num_envs": 8,
        "reference_data_keys": [
            "ankle_angle_l", "ankle_angle_r",
            "hip_flexion_l", "hip_flexion_r",
            "hip_adduction_l", "hip_adduction_r",
            "hip_rotation_l", "hip_rotation_r",
            "knee_angle_l", "knee_angle_r",
            "pelvis_list", "pelvis_tilt", "pelvis_rotation",
            "pelvis_tx", "pelvis_ty", "pelvis_tz"
        ],
        "reward_keys_and_weights": {
            "qpos_imitation_rewards": {
                "pelvis_tx": 0.1, "pelvis_ty": 0.1, "pelvis_tz": 0.1,
                "pelvis_list": 0.5, "pelvis_tilt": 1.0, "pelvis_rotation": 0.5,
                "hip_flexion_l": 0.5, "hip_flexion_r": 0.5,
                "hip_adduction_l": 0.3, "hip_adduction_r": 0.3,
                "hip_rotation_l": 0.3, "hip_rotation_r": 0.3,
                "knee_angle_l": 1.0, "knee_angle_r": 1.0,
                "ankle_angle_l": 0.2, "ankle_angle_r": 0.2
            }
        }
    },
    "policy_params": {
        "custom_policy_params": {
            "net_arch": {
                "human_actor": [128, 128],
                "exo_actor": [16, 16],
                "common_critic": [128, 128]
            }
        }
    }
}
```

---

## 🚀 학습 실행 비교

### 2D 모델 학습
```bash
# 빠른 테스트
python train_S004_motion.py --quick_test

# 전체 학습 (16 환경)
python train_S004_motion.py

# 커스텀
python train_S004_motion.py --num_envs 32 --device cuda
```

### 3D 모델 학습
```bash
# 빠른 테스트
python train_S004_motion_3D.py --quick_test

# 전체 학습 (8 환경)
python train_S004_motion_3D.py

# 커스텀 (GPU 권장)
python train_S004_motion_3D.py --num_envs 16 --device cuda
```

---

## 💡 언제 어떤 모델을 사용해야 할까?

### 2D 모델을 사용하세요 ✅

- ✅ **평지 보행** 연구
- ✅ **빠른 프로토타이핑** 필요
- ✅ **제한된 계산 자원** (노트북 등)
- ✅ **sagittal plane 동작**만 관심
- ✅ **빠른 학습**이 필요
- ✅ 초기 개념 검증

**예시:**
- 평지 보행 속도 제어
- 외골격 기본 제어 전략
- 근육 활성화 패턴 분석
- 알고리즘 테스트

### 3D 모델을 사용하세요 ✅

- ✅ **복잡한 동작** (계단, 경사, 회전)
- ✅ **측면 안정성** 중요
- ✅ **완전한 운동학** 필요
- ✅ **현실적인 시뮬레이션** 목표
- ✅ **충분한 계산 자원** (워크스테이션, 서버)
- ✅ 최종 연구 결과

**예시:**
- 불규칙한 지형 보행
- 회전이 포함된 동작
- 측면 균형 유지
- 실제 인간 동작 재현

---

## ⚡ 성능 및 자원 요구사항

### 2D 모델
| 항목 | 사양 |
|------|------|
| **RAM** | 8-16 GB |
| **CPU 코어** | 4-8 코어 |
| **학습 시간** | ~12-24시간 (3천만 스텝) |
| **환경당 시간** | ~20-30 ms/step |
| **권장 병렬 환경** | 16-32 |

### 3D 모델
| 항목 | 사양 |
|------|------|
| **RAM** | 16-32 GB |
| **CPU 코어** | 8-16 코어 (또는 GPU) |
| **학습 시간** | ~24-48시간 (3천만 스텝) |
| **환경당 시간** | ~50-80 ms/step |
| **권장 병렬 환경** | 4-8 (CPU), 8-16 (GPU) |

---

## 🎓 학습 팁

### 2D 모델
```python
# 빠른 학습을 위한 설정
{
    "num_envs": 32,           # 많은 병렬 환경
    "n_steps": 256,           # 작은 스텝
    "batch_size": 8192,       # 일정한 배치 크기 유지
    "learning_rate": 0.0001,
}
```

### 3D 모델
```python
# 안정적인 학습을 위한 설정
{
    "num_envs": 8,            # 적은 환경 (무거움)
    "n_steps": 1024,          # 큰 스텝
    "batch_size": 8192,       # 일정한 배치 크기 유지
    "learning_rate": 0.00005, # 작은 학습률 (안정성)
    "net_arch": {
        "human_actor": [128, 128],  # 큰 네트워크
        "common_critic": [128, 128]
    }
}
```

---

## 🔧 보상 가중치 튜닝

### 2D 모델
```json
{
    "qpos_imitation_rewards": {
        "pelvis_tilt": 1.0,    // 자세 유지 (중요!)
        "knee_angle_*": 1.0,   // 무릎 (중요!)
        "hip_flexion_*": 0.2,  // 고관절
        "ankle_angle_*": 0.2,  // 발목
        "pelvis_ty": 0.1       // 높이
    }
}
```

### 3D 모델
```json
{
    "qpos_imitation_rewards": {
        "pelvis_tilt": 1.0,        // 전후 자세 (중요!)
        "knee_angle_*": 1.0,       // 무릎 (중요!)
        "pelvis_list": 0.5,        // 측면 자세 (중요!)
        "pelvis_rotation": 0.5,    // 회전 자세
        "hip_flexion_*": 0.5,      // 고관절 굴곡
        "hip_adduction_*": 0.3,    // 고관절 내전
        "hip_rotation_*": 0.3,     // 고관절 회전
        "ankle_angle_*": 0.2       // 발목
    }
}
```

---

## 📊 기대 결과

### 2D 모델
- ✅ 빠른 수렴 (5-10M 스텝)
- ✅ 안정적인 sagittal plane 보행
- ✅ 명확한 보상 신호
- ⚠️ 측면 안정성 제한적

### 3D 모델
- ✅ 느린 수렴 (10-20M 스텝)
- ✅ 완전한 3D 보행
- ✅ 측면 안정성 포함
- ⚠️ 학습 불안정 가능성 높음

---

## 🎯 요약

| 시나리오 | 권장 모델 | 이유 |
|----------|----------|------|
| **빠른 개발** | 2D | 학습 속도 2배 빠름 |
| **평지 보행** | 2D | 충분히 정확 |
| **노트북** | 2D | 자원 효율적 |
| **복잡한 지형** | 3D | 측면 안정성 필수 |
| **현실적 시뮬레이션** | 3D | 완전한 운동학 |
| **강력한 워크스테이션/GPU** | 3D | 자원 활용 가능 |
| **논문 연구** | 3D | 더 완전한 모델 |
| **프로토타입** | 2D | 빠른 검증 |

---

## 🚀 추천 워크플로우

1. **2D로 시작** → 빠른 프로토타이핑 및 개념 검증
2. **알고리즘 검증** → 2D에서 하이퍼파라미터 튜닝
3. **3D로 확장** → 최종 결과 및 완전한 시뮬레이션
4. **비교 분석** → 2D vs 3D 결과 비교

```bash
# Step 1: 2D 빠른 테스트
python train_S004_motion.py --quick_test

# Step 2: 2D 전체 학습
python train_S004_motion.py

# Step 3: 3D 빠른 테스트
python train_S004_motion_3D.py --quick_test

# Step 4: 3D 전체 학습 (검증 후)
python train_S004_motion_3D.py --device cuda
```

---

**결론: 빠른 개발은 2D, 완전한 시뮬레이션은 3D! 🎯**
