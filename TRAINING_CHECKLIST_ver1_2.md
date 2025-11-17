# Ver1_2 Training Checklist

## ✅ Completed

### 1. Reference Motion (FIXED)
- ✅ HDF5 파일 위치 확인: `C:\workspace\opensim data\LD\S004.h5`
- ✅ convert_hdf5_direct.py로 변환 완료
- ✅ 출력: `rl_train/reference_data/S004_trial01_08mps_3D_HDF5_v8_FIXED.npz`
- ✅ 12,028 프레임, 120.28초, 100Hz 샘플링

### 2. Ver1_2 Environment
- ✅ `myoassist_leg_imitation_ver1_2.py` 생성
- ✅ Curriculum Learning 구현
  - Stage 1 (0-30%): Double support only
  - Stage 2 (30-60%): Double support + heel strikes
  - Stage 3 (60-100%): All high-quality poses
- ✅ Phase Detection 구현
  - Heel strike detection
  - Double support detection
  - Quality-based pose filtering
- ✅ Balancing rewards 유지 (ver1_1에서)

### 3. Environment Registration
- ✅ `rl_train/envs/__init__.py`에 ver1_2 등록
- ✅ `environment_handler.py`에 ver1_2 session_id 추가
- ✅ `total_timesteps` 파라미터 전달 구현

### 4. Training Configuration
- ✅ `S004_3D_IL_ver1_2_CURRICULUM.json` 생성
- ✅ Reference path 업데이트 (v8_FIXED)
- ✅ env_id 업데이트 (v1_2)
- ✅ 8 parallel environments
- ✅ GPU (cuda) 사용 설정

### 5. Import Test
- ✅ Ver1_2 환경 import 성공

## 🚀 Ready to Launch

### Training Command
```bash
conda activate myoassist
python -m rl_train.run_train \
    --config_file_path rl_train/train/train_configs/S004_3D_IL_ver1_2_CURRICULUM.json \
    --use_ver1_2 \
    --wandb_project myoassist-3D-curriculum \
    --wandb_name S004_ver1_2_FIXED_ref_curriculum
```

## 📊 Expected Improvements

### From Ver1_1 to Ver1_2:
1. **FIXED Reference Motion** 
   - Ver1_1: 잘못된 reference → 부정확한 모방
   - Ver1_2: HDF5에서 직접 변환 → 정확한 모방

2. **Curriculum Learning**
   - Ver1_1: Random initialization → 불안정한 시작
   - Ver1_2: Progressive difficulty → 안정적 학습

3. **Phase-Aware Initialization**
   - Ver1_1: 모든 프레임에서 랜덤 샘플링 (swing phase 포함)
   - Ver1_2: 안정적인 자세만 선택 (double support, heel strike)

4. **Quality Filtering**
   - Ver1_1: 모든 자세 사용
   - Ver1_2: 상위 50% quality 자세만 사용

## 🎯 Training Stages (Curriculum)

### Stage 1: Beginner (0-9M timesteps, 0-30%)
- **Initialization**: Double support only (가장 안정적)
- **Expected**: 서기 자세 학습, 기본 균형 유지
- **Duration**: ~8시간

### Stage 2: Intermediate (9M-18M timesteps, 30-60%)  
- **Initialization**: Double support + heel strikes
- **Expected**: 걷기 transition 학습, 보행 사이클 이해
- **Duration**: ~8시간

### Stage 3: Advanced (18M-30M timesteps, 60-100%)
- **Initialization**: All high-quality poses
- **Expected**: 완전한 보행, 다양한 시작 자세 대응
- **Duration**: ~11시간

**Total Estimated Time**: ~27-28시간

## 📈 Monitoring Metrics

### Key Metrics to Watch:
1. **value_loss**: 처음에 높다가 점진적으로 감소 (< 5.0 목표)
2. **explained_variance**: 0 → 0.5+ 증가 (학습 진행도)
3. **episode_length**: 점진적으로 증가 (더 오래 서있음)
4. **qpos_imitation_reward**: 증가 (더 정확한 모방)
5. **pelvis_list_penalty**: 감소 (더 안정적인 균형)

### Curriculum Progress Check:
- 콘솔에서 "🎓 Curriculum [beginner/intermediate/advanced]" 메시지 확인
- Stage transition: 9M, 18M timesteps 근처

## 🐛 Potential Issues & Solutions

### Issue 1: scipy.signal import error
- **Solution**: scipy 이미 설치됨 (1.16.3) - OK

### Issue 2: "Simulation unstable" 경고 많이 발생
- **Expected**: Ver1_2에서는 감소할 것 (안정적인 초기화 덕분)
- **Action**: 처음 1-2시간 모니터링, 여전히 많으면 support_duration_frames 증가

### Issue 3: Curriculum stage가 안 바뀜
- **Check**: Callback이 `update_curriculum_progress()` 호출하는지 확인
- **Debug**: 콘솔에서 🎓 메시지 확인

### Issue 4: Reference motion 여전히 이상함
- **Check**: v8_FIXED.npz 파일 정상 로드되는지 확인
- **Verify**: Training log에서 reference_data_path 확인

## 📝 Next Steps After This Training

1. **Evaluation**: Best model로 평가, video 생성
2. **Comparison**: Ver1_1 vs Ver1_2 학습 곡선 비교
3. **Stage 2**: Fine-tuning with increased foot_contact_reward
4. **Analysis**: Curriculum effectiveness 분석 (각 stage별 성능)

## 🎓 Research Questions to Answer

1. Curriculum이 실제로 학습 속도를 높였는가?
2. 각 stage에서 policy가 무엇을 학습했는가?
3. FIXED reference가 모방 품질을 얼마나 개선했는가?
4. Phase-aware initialization이 "Simulation unstable" 빈도를 줄였는가?

---

**Created**: 2024-11-17
**Status**: ✅ Ready to Launch
**Estimated Completion**: 27-28 hours
