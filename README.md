# MyoAssist Imitation Learning Pipeline

빠르고 깔끔한 모방학습(Imitation Learning) 파이프라인

---

## 📁 프로젝트 구조

```
myoassist/
├── convert_motion_data.py          # 🔧 모션 데이터 변환 (OpenSim → MyoAssist)
├── train_imitation_learning.py    # 🚀 IL 학습 실행
├── verify_S004_setup.py           # ✅ 환경 검증
│
├── rl_train/                      # 학습 관련 코드
│   ├── run_train.py              # 실제 학습 실행기
│   ├── run_policy_eval.py        # 정책 평가
│   └── reference_data/           # 변환된 모션 데이터 저장
│
├── models/                        # MuJoCo 모델
│   ├── 22muscle_2D/              # 2D 모델 (빠름)
│   └── 26muscle_3D/              # 3D 모델 (완전함)
│
├── docs_IL/                       # 📚 문서 (상세 가이드)
│   ├── README_S004_IMITATION_LEARNING.md
│   ├── S004_IMITATION_LEARNING_GUIDE.md
│   └── 2D_vs_3D_COMPARISON.md
│
└── deprecated/                    # 🗑️ 구버전 스크립트
```

---

## 🚀 빠른 시작 (3단계)

### 1️⃣ 데이터 변환

`convert_motion_data.py` 파일 상단 CONFIG 수정:

```python
CONFIG = {
    'input_file': r"C:\your\opensim\data\trial.npz",
    'output_file_2d': r"rl_train\reference_data\my_motion_2D.npz",
    'output_file_3d': r"rl_train\reference_data\my_motion_3D.npz",
    'model_type': 'both',  # '2D', '3D', or 'both'
}
```

실행:
```bash
python convert_motion_data.py
```

### 2️⃣ 학습 설정

`train_imitation_learning.py` 파일 상단 CONFIG 수정:

```python
CONFIG = {
    'experiment_name': 'my_experiment',
    'model_type': '2D',  # '2D' or '3D'
    'reference_data_path': 'rl_train/reference_data/my_motion_2D.npz',
    'training': {
        'num_envs': 16,  # 병렬 환경 개수
        'target_velocity': 0.8,  # 목표 속도 (m/s)
        'device': 'cpu',  # 'cpu' or 'cuda'
    },
}
```

### 3️⃣ 학습 실행

```bash
# 빠른 테스트
python train_imitation_learning.py --quick_test

# 전체 학습
python train_imitation_learning.py

# 3D 모델로 학습
python train_imitation_learning.py --model 3D --device cuda
```

---

## 📊 주요 스크립트 설명

### `convert_motion_data.py` - 모션 데이터 변환기

**기능:**
- OpenSim NPZ → MyoAssist NPZ 형식 변환
- 2D/3D 모두 지원
- 커스터마이징 가능한 joint 선택

**설정 가능 항목:**
```python
# 입출력 경로
'input_file': "OpenSim 데이터 경로"
'output_file_2d': "2D 출력 경로"
'output_file_3d': "3D 출력 경로"

# 변환할 관절 선택 (2D)
'joints_2d': {
    'pelvis': ['tx', 'ty', 'tilt'],
    'hip': ['flexion'],
    'knee': ['angle'],
    'ankle': ['angle'],
}

# 변환할 관절 선택 (3D)
'joints_3d': {
    'pelvis': ['tx', 'ty', 'tz', 'list', 'tilt', 'rotation'],
    'hip': ['flexion', 'adduction', 'rotation'],
    'knee': ['angle'],
    'ankle': ['angle'],
}

# OpenSim 컬럼 매핑 (데이터 구조에 맞게 수정)
'opensim_mapping': {
    'hip_flexion_r': 'hip_r_1',  # OpenSim 컬럼 이름
    ...
}
```

**사용법:**
```bash
# CONFIG 사용
python convert_motion_data.py

# 명령줄 인자 사용
python convert_motion_data.py \
    --input "data.npz" \
    --output_2d "out_2d.npz" \
    --model_type 2D
```

---

### `train_imitation_learning.py` - 학습 실행기

**기능:**
- 2D/3D 모델 모두 지원
- 자동 config 생성
- 하이퍼파라미터 커스터마이징

**설정 가능 항목:**
```python
# 실험 설정
'experiment_name': "실험 이름"
'model_type': '2D' or '3D'
'reference_data_path': "변환된 데이터 경로"

# 학습 파라미터
'training': {
    'total_timesteps': 3e7,
    'num_envs': 16,  # 2D: 16, 3D: 8 권장
    'target_velocity': 0.8,
    'device': 'cpu',
    'learning_rate': 0.0001,
}

# 보상 가중치 (중요!)
'reward_weights': {
    '2D': {
        'qpos_imitation': {
            'knee_angle_l': 1.0,  # 무릎 중요
            'pelvis_tilt': 1.0,   # 자세 중요
            'hip_flexion_l': 0.2,
            ...
        }
    }
}

# 네트워크 구조
'network_arch': {
    '2D': {'human_actor': [64, 64], ...},
    '3D': {'human_actor': [128, 128], ...},
}
```

**사용법:**
```bash
# 기본 (CONFIG 사용)
python train_imitation_learning.py

# 빠른 테스트
python train_imitation_learning.py --quick_test

# 3D 모델
python train_imitation_learning.py --model 3D

# GPU 사용
python train_imitation_learning.py --device cuda --num_envs 32

# 렌더링 포함
python train_imitation_learning.py --render
```

---

### `verify_S004_setup.py` - 환경 검증

**기능:**
- Reference data 로드 확인
- 환경 생성 테스트
- 필수 키 검증

**사용법:**
```bash
python verify_S004_setup.py
```

---

## ⚙️ 커스터마이징 가이드

### 1. 새로운 모션 데이터 사용

```python
# convert_motion_data.py의 CONFIG 수정
CONFIG = {
    'input_file': r"C:\your\new\trial.npz",
    'output_file_2d': r"rl_train\reference_data\new_trial_2D.npz",
    'model_type': 'both',
}
```

```python
# train_imitation_learning.py의 CONFIG 수정
CONFIG = {
    'experiment_name': 'new_trial_experiment',
    'reference_data_path': 'rl_train/reference_data/new_trial_2D.npz',
}
```

### 2. 관절 선택 커스터마이징

```python
# convert_motion_data.py
CONFIG = {
    'joints_2d': {
        'pelvis': ['tx', 'ty'],  # tilt 제외
        'hip': ['flexion'],
        'knee': ['angle'],
        # ankle 제외 가능
    },
}
```

### 3. 보상 가중치 튜닝

```python
# train_imitation_learning.py
CONFIG = {
    'reward_weights': {
        '2D': {
            'qpos_imitation': {
                'knee_angle_l': 2.0,  # 무릎에 더 집중
                'pelvis_tilt': 1.5,   # 자세 강화
                'hip_flexion_l': 0.1, # 고관절 완화
            }
        }
    }
}
```

### 4. 학습 속도 조정

```python
# train_imitation_learning.py
CONFIG = {
    'training': {
        'num_envs': 32,  # 더 많은 병렬 환경 (빠름)
        'learning_rate': 0.0002,  # 더 높은 학습률
    },
}
```

---

## 🎯 2D vs 3D 선택 가이드

| 사용 목적 | 권장 모델 | 이유 |
|----------|----------|------|
| 평지 보행 | 2D | 충분히 정확, 빠름 |
| 빠른 프로토타이핑 | 2D | 학습 속도 2배 빠름 |
| 제한된 자원 (노트북) | 2D | 가벼운 계산 |
| 복잡한 지형 | 3D | 측면 안정성 필요 |
| 완전한 시뮬레이션 | 3D | 16 DOF |
| 논문 연구 | 3D | 더 현실적 |

**권장 워크플로우:**
1. 2D로 빠른 검증 → 알고리즘 확인
2. 하이퍼파라미터 튜닝 → 2D에서 최적화
3. 3D로 최종 학습 → 완전한 결과

---

## 📈 학습 결과 확인

```bash
# 결과 위치
rl_train/results/train_session_[timestamp]/

# 평가 실행
python rl_train/run_policy_eval.py rl_train/results/train_session_[timestamp]

# 생성 파일
├── session_config.json          # 사용된 설정
├── train_log.json              # 학습 로그
├── trained_models/             # 저장된 모델
└── analyze_results_*/          # 분석 결과 (그래프, 영상)
```

---

## 🔧 트러블슈팅

### Q: "No module named 'gymnasium'" 에러
```bash
pip install gymnasium stable-baselines3 torch
```

### Q: 메모리 부족
```python
# train_imitation_learning.py CONFIG 수정
'training': {'num_envs': 4}  # 환경 개수 줄이기
```

### Q: 학습이 불안정
```python
# train_imitation_learning.py CONFIG 수정
'training': {'learning_rate': 0.00005}  # 학습률 낮추기
```

### Q: OpenSim 컬럼 매핑 오류
```python
# convert_motion_data.py에서 컬럼 이름 확인
# 'opensim_mapping' 딕셔너리 수정
```

---

## 📚 추가 자료

**상세 문서:** `docs_IL/` 폴더 참조
- `README_S004_IMITATION_LEARNING.md` - 전체 가이드
- `2D_vs_3D_COMPARISON.md` - 모델 비교
- `S004_IMITATION_LEARNING_GUIDE.md` - 단계별 튜토리얼

**공식 문서:** https://myoassist.neumove.org/

---

## ✅ 체크리스트

학습 시작 전 확인:
- [ ] OpenSim 데이터 준비
- [ ] `convert_motion_data.py` CONFIG 수정
- [ ] 데이터 변환 실행 및 확인
- [ ] `train_imitation_learning.py` CONFIG 수정
- [ ] 빠른 테스트 실행 (`--quick_test`)
- [ ] 전체 학습 실행

---

**모든 설정은 스크립트 상단 CONFIG에서!** 🎯
