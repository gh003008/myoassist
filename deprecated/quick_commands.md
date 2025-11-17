# MyoAssist 빠른 명령어 모음

## 🔧 환경 세팅

### 1. 처음 설치 (한 번만)
```powershell
# Conda 환경 생성
conda create -n myoassist python=3.11 -y

# 환경 활성화
conda activate myoassist

# MyoAssist 설치
pip install -e .

# 설치 확인
python test_setup.py
```

### 2. 환경 활성화 (매번 터미널 열 때마다)
```powershell
conda activate myoassist
```

---

## 🎯 모방학습 (Imitation Learning)

### 모방학습 종류
이 프레임워크는 **Reward Shaping 기반 Imitation Learning**을 사용합니다.
- GAIL/AIRL 같은 adversarial 방식이 **아닙니다**
- Reference motion을 reward function에 직접 포함하는 방식
- DeepMimic 스타일의 motion tracking

### 훈련 실행

#### 방법 1: 스크립트 사용 (권장)
```powershell
# Partial observation (기본)
.\run_imitation_training.ps1 partial_obs

# Full observation
.\run_imitation_training.ps1 full_obs

# Speed control
.\run_imitation_training.ps1 speed_control
```

#### 방법 2: 직접 명령어
```powershell
conda activate myoassist
python rl_train/run_train.py --config_file_path rl_train/train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json
```

#### 방법 3: 테스트 훈련 (빠른 확인용)
```powershell
conda activate myoassist
python rl_train/run_train.py --config_file_path rl_train/train/train_configs/test.json --flag_rendering
```

### 훈련 중단
```
Ctrl + C
```

---

## 📊 평가 (Evaluation)

### 방법 1: 스크립트 사용
```powershell
# 가장 최근 훈련 결과 평가
.\run_imitation_eval.ps1

# 특정 세션 평가
.\run_imitation_eval.ps1 rl_train/results/train_session_20250112-123456
```

### 방법 2: 직접 명령어
```powershell
conda activate myoassist

# Pretrained 모델 평가
python rl_train/run_policy_eval.py docs/assets/tutorial_rl_models/train_session_20250728-161129_tutorial_partial_obs

# 내 훈련 결과 평가
python rl_train/run_policy_eval.py rl_train/results/train_session_YYYYMMDD-HHMMSS
```

---

## 🎬 실시간 시뮬레이션 (Realtime Evaluation)

```powershell
conda activate myoassist

# Windows
python rl_train/run_train.py --config_file_path docs/assets/tutorial_rl_models/train_session_20250728-161129_tutorial_partial_obs/session_config.json --config.env_params.prev_trained_policy_path docs/assets/tutorial_rl_models/train_session_20250728-161129_tutorial_partial_obs/trained_models/model_19939328 --flag_realtime_evaluate

# 내 모델로 실행 (경로 수정 필요)
python rl_train/run_train.py --config_file_path rl_train/results/train_session_YYYYMMDD-HHMMSS/session_config.json --config.env_params.prev_trained_policy_path rl_train/results/train_session_YYYYMMDD-HHMMSS/trained_models/model_XXXXX --flag_realtime_evaluate
```

---

## 📁 결과 확인

### 훈련 결과 위치
```
rl_train/results/train_session_[날짜-시간]/
├── session_config.json          # 사용한 설정
├── train_log.json               # 훈련 로그
├── trained_models/              # 저장된 모델들
│   ├── model_1000000.zip
│   └── model_2000000.zip
└── analyze_results_*/           # 평가 결과
    ├── plots/                   # 그래프들
    └── videos/                  # 비디오들
```

### 결과 열기
```powershell
# 폴더 열기
explorer rl_train\results

# 최신 결과 확인
ls rl_train\results | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

---

## 🔍 참고 자료

### Config 파일 위치
```
rl_train/train/train_configs/
├── imitation_tutorial_22_separated_net_partial_obs.json   # 부분 관측
├── imitation_tutorial_22_separated_net_full_obs.json      # 전체 관측
├── imitation_tutorial_22_separated_net_speed_control.json # 속도 제어
├── imitation_tutorial_22_separated_net_exo_off.json       # 외골격 OFF
├── imitation.json                                          # 기본
└── test.json                                               # 테스트용 (빠름)
```

### Reference Data (모방할 모션)
```
rl_train/reference_data/
├── short_reference_gait.npz     # 짧은 보행 데이터
└── segmented.npz                # 세그먼트 보행 데이터
```

### 튜토리얼 노트북
```
docs/tutorial/
├── rl_imitation_tutorial.ipynb       # 모방학습 튜토리얼
├── rl_terrain_tutorial.ipynb         # 지형 튜토리얼
├── rl_analyze_tutorial.ipynb         # 분석 튜토리얼
└── rl_analyze_transfer_tutorial.ipynb # 전이학습 튜토리얼
```

---

## 🛠️ 문제 해결

### 환경이 안 보일 때
```powershell
conda env list
# myoassist가 없으면 다시 생성
conda create -n myoassist python=3.11 -y
```

### 패키지 에러 발생 시
```powershell
conda activate myoassist
pip install -e . --force-reinstall
```

### ModuleNotFoundError: No module named 'flatten_dict'
```powershell
# 그냥 다시 실행하면 보통 해결됨
# 또는
pip install flatten_dict
```

### MuJoCo 관련 에러
```powershell
pip install mujoco==3.3.3
```

---

## 📝 Config 파라미터 수정

### 환경 수 조절 (성능에 따라)
```powershell
# num_envs를 16으로 줄이고 n_steps를 1024로 증가
python rl_train/run_train.py --config_file_path rl_train/train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json --config.env_params.num_envs 16 --config.ppo_params.n_steps 1024
```

### 훈련 스텝 수 조절
```powershell
# 3천만 스텝 대신 1천만 스텝만
python rl_train/run_train.py --config_file_path rl_train/train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json --config.total_timesteps 10000000
```

---

## 💡 유용한 팁

1. **처음 실행 시 렌더링 켜기**: `--flag_rendering` 추가하면 진행 상황 볼 수 있음
2. **GPU 사용**: Config에서 `"device": "cuda"` (GPU 있을 때만)
3. **결과 비교**: 여러 config로 훈련 후 `run_policy_eval.py`로 비교
4. **Transfer Learning**: 이전 모델에서 계속 학습 가능 (`prev_trained_policy_path` 설정)

---

## 🔗 공식 문서
- 메인: https://myoassist.neumove.org/
- RL 가이드: https://myoassist.neumove.org/reinforcement-learning/
- GitHub: https://github.com/neumovelab/myoassist
