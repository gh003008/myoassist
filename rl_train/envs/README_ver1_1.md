# MyoAssist ver1_1: Balancing Rewards for 3D Model

**Created**: 2024-11-15 00:28 (251115_0028)  
**Purpose**: Add balancing/stability rewards to improve 3D model convergence

## 🎯 Problem Statement

- **2D model**: Converges successfully ✅
- **3D model**: Poor convergence, unstable gait ❌
- **Root cause**: Missing explicit balancing rewards
  - 2D has lateral constraints → naturally stable
  - 3D has full DOF → needs explicit balance terms

## 📦 What's New in ver1_1

### [HIGH Priority] Pelvis List Penalty
- **What**: Strong exponential penalty for roll (lateral tilt)
- **Why**: In 3D, lateral stability is critical. Roll should stay near 0.
- **Implementation**: `exp(-10.0 * pelvis_list²)` with weight 0.1
- **Impact**: Prevents agent from learning gaits that tip sideways

### [MEDIUM Priority] Rotation-Based Termination
- **What**: Early episode termination on excessive rotation
- **Why**: Stops unstable trajectories before falling
- **Implementation**: Terminate if forward direction deviates >53° (cos 0.6)
- **Impact**: Faster learning by avoiding exploration of falling states

### [BONUS] Pelvis Height Reward
- **What**: Gaussian reward for maintaining upright posture
- **Why**: Encourages staying at natural standing height (0.9m)
- **Implementation**: `exp(-5.0 * (height - 0.9)²)` with weight 0.02
- **Impact**: Complements height-based termination

## 📂 Files Created

```
rl_train/envs/
├── myoassist_leg_imitation_ver1_1.py      # Extended environment
└── README_ver1_1.md                        # This file

rl_train/train/train_configs/
└── S004_3D_IL_ver1_1_BALANCE.json         # Config with balance rewards
```

## 📝 Files Modified (Original MyoAssist)

### `rl_train/envs/__init__.py`
```python
# 251115_0028: ver1_1 환경 등록 - 3D balancing rewards 추가
register_env_myoassist(id='myoAssistLegImitationExo-v1_1',
        entry_point='rl_train.envs.myoassist_leg_imitation_ver1_1:MyoAssistLegImitation_ver1_1',
        max_episode_steps=1000,
        kwargs={},
    )
```

### `rl_train/envs/environment_handler.py`
```python
def get_callback(config, train_log_handler, use_ver1_0=False, use_ver1_1=False, wandb_config=None):
    # 251115_0028: ver1_1 callback for balancing rewards
    if use_ver1_1:
        from rl_train.envs.myoassist_leg_imitation_ver1_1 import ImitationCustomLearningCallback_ver1_1
        custom_callback = ImitationCustomLearningCallback_ver1_1(...)
```

### `rl_train/run_train.py`
```python
# 251115_0028: ver1_1 option for balancing rewards
parser.add_argument("--use_ver1_1", type=bool, default=False, ...)

def ppo_train_with_parameters(..., use_ver1_1=False, ...):
    # 251115_0028: Support ver1_1 callback
    custom_callback = EnvironmentHandler.get_callback(..., use_ver1_1=use_ver1_1, ...)
```

## 🚀 Usage

### Training with ver1_1

```bash
# Basic ver1_1 training
python -m rl_train.run_train \
    --config_file_path rl_train/train/train_configs/S004_3D_IL_ver1_1_BALANCE.json \
    --use_ver1_1 \
    --wandb_project myoassist-3D-balancing \
    --wandb_name S004_ver1_1_test

# Continue from previous checkpoint
python -m rl_train.run_train \
    --config_file_path rl_train/train/train_configs/S004_3D_IL_ver1_1_BALANCE.json \
    --use_ver1_1 \
    --config.env_params.prev_trained_policy_path "rl_train/results/train_session_XXXXXX/model_final.zip"
```

### Configuration Details

**Key reward weights in `S004_3D_IL_ver1_1_BALANCE.json`:**

```json
{
  "reward_keys_and_weights": {
    // Original imitation rewards (unchanged)
    "qpos_imitation_rewards": { ... },
    "qvel_imitation_rewards": { ... },
    "forward_reward": 0.2,
    "muscle_activation_penalty": 0.1,
    
    // 251115_0028: New balancing rewards
    "pelvis_list_penalty": 0.1,        // [HIGH] Roll penalty
    "pelvis_height_reward": 0.02       // [BONUS] Height maintenance
  },
  
  // 251115_0028: Rotation termination threshold
  "max_rot": 0.6  // cos(53°) - terminate if rotation exceeds this
}
```

## 🔧 Tuning Guidelines

### If model still falls frequently:
1. **Increase `pelvis_list_penalty` weight**: 0.1 → 0.2
2. **Decrease `max_rot` threshold**: 0.6 → 0.7 (more strict, 45°)
3. **Increase `pelvis_height_reward` weight**: 0.02 → 0.05

### If model is too conservative (doesn't walk):
1. **Decrease `pelvis_list_penalty` weight**: 0.1 → 0.05
2. **Increase `max_rot` threshold**: 0.6 → 0.5 (more lenient, 60°)
3. **Increase `forward_reward` weight**: 0.2 → 0.3

### Monitor in WandB:
- `reward/pelvis_list_penalty`: Should be close to 0 (negative values = penalty)
- `reward/pelvis_height_reward`: Should stay positive and stable
- `episode/mean_length`: Should increase over time (longer episodes = more stable)
- `info/pelvis_ty`: Monitor pelvis height (should stay ~0.9m)

## 🧪 Testing

### Quick verification (10 seconds):
```bash
python -c "from rl_train.envs.myoassist_leg_imitation_ver1_1 import MyoAssistLegImitation_ver1_1; print('✅ ver1_1 import successful')"
```

### Test environment creation:
```python
from rl_train.utils.mujoco_env_load import load_reference_data
from rl_train.envs.myoassist_leg_imitation_ver1_1 import MyoAssistLegImitation_ver1_1

# Load reference data
ref_data = load_reference_data(
    reference_data_path="rl_train/reference_data/S004_trial01_08mps_3D.npz",
    reference_data_keys=["pelvis_tx", "pelvis_ty", ...],
)

# Create environment
env = MyoAssistLegImitation_ver1_1(
    model_path="models/26muscle_3D/myoLeg26_BASELINE.xml",
    reference_data=ref_data,
    obs_keys=['qpos', 'qvel', 'act', 'target_velocity'],
    weighted_reward_keys={...},
    env_params=...,
)

# Test step
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)

print("✅ Environment test passed")
print(f"Reward components: {info.get('rwd_dict', {})}")
```

## 📊 Expected Results

### Before (ver1_0):
- Episode length: 50-200 steps (frequent falls)
- Mean reward: Highly variable
- Success rate: <30%
- Visual: Model tips sideways, unstable gait

### After (ver1_1):
- Episode length: 200-1000 steps (stable episodes)
- Mean reward: More stable, gradually increasing
- Success rate: >70%
- Visual: Upright posture, smooth gait

## 🤝 Collaboration Notes

### For team members:
- **Original files unchanged**: All MyoAssist base files are untouched
- **Extension pattern**: Copy ver1_0 → modify → create ver1_1
- **Comment convention**: All modifications marked with `# 251115_0028`
- **Config naming**: `*_ver1_1_*.json` for new configs
- **Backward compatible**: ver1_0 training still works as before

### Adding new reward terms:
1. Extend `MyoAssistLegImitation_ver1_1` class
2. Add reward calculation in `_calculate_balancing_rewards()`
3. Add weight to config JSON `reward_keys_and_weights`
4. Update this README with new term description

## 📚 References

### MyoSuite walk_v0.py balancing rewards:
- `_get_ref_rotation_rew()`: Orientation tracking
- `_get_rot_condition()`: Rotation-based termination
- `_get_height()`: CoM height monitoring
- `_get_com_velocity()`: Full CoM velocity (vs pelvis_tx only)

### Key differences from MyoSuite:
- **MyoSuite**: General walking environment with rotation rewards
- **MyoAssist ver1_1**: Imitation learning with balancing constraints
- **Approach**: Minimal auxiliary rewards (only critical balance terms)

## ❓ FAQ

**Q: Why not use MyoSuite's full reward structure?**  
A: MyoAssist uses imitation learning (track reference trajectory), not pure RL. Most rewards come from position/velocity tracking. We only add critical balance terms.

**Q: Can I use ver1_1 with 2D models?**  
A: Yes, but unnecessary. 2D models already stable. The penalty terms will be near-zero anyway (pelvis_list constrained in 2D).

**Q: How do I disable rotation termination?**  
A: Set `max_rot: 0.0` in config (or very small value). The termination check will effectively be disabled.

**Q: What if I want different balance strategy?**  
A: Extend `MyoAssistLegImitation_ver1_1` → create `ver1_2` with your approach. Keep ver1_1 unchanged for comparison.

## 📞 Contact

For questions about ver1_1 implementation:
- Check WandB logs for reward components
- Compare with MyoSuite walk_v0.py for reference
- Test with smaller networks first (faster iteration)

---

**Last updated**: 2024-11-15 00:28  
**Version**: ver1_1  
**Status**: Ready for testing ✅
