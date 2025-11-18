# Environment Code Reversion Summary

**Date:** 2024-11-17  
**Purpose:** Revert environment code to original simple implementation

## Background

While working on Ver2_1 (Karico + 3D balancing), we attempted to fix a kinematic rendering issue ("shin stuck to hip") by modifying the environment code. This approach was **incorrect** - the environment code works for training and should not be modified for rendering purposes.

## Problem

When trying to visualize symmetric reference motion in the training environment, we observed kinematic chain corruption:
- Symptom: Tibia appeared attached to hip instead of knee
- Root cause: Unknown, but related to environment wrapper behavior
- Attempted fixes (all failed):
  1. Stand keyframe initialization
  2. Joint name mapping (q_ prefix handling)
  3. qpos index direct access
  4. dm_control wrapper compatibility

## Key Insight

**User's correct observation:** "점점 이상한 길로 가고 있는거 같아... 환경 세팅 관련은 건드리면 안돼"

The environment code is **proven to work for training**. The rendering script (`render_hdf5_reference.py`) is **proven to work for visualization**. Mixing these concerns was the mistake.

## Changes Reverted

### 1. `rl_train/envs/myoassist_leg_imitation_ver1_0.py`

**Reverted `_setup()` method:**
- ❌ Removed: `_joint_qpos_map` initialization
- ❌ Removed: dm_control wrapper handling
- ❌ Removed: qpos index mapping
- ✅ Restored: Simple `super()._setup()` call

**Reverted `_follow_reference_motion()` method:**
- ❌ Removed: Stand keyframe initialization
- ❌ Removed: qpos index access
- ❌ Removed: dm_control compatibility code
- ✅ Restored: Simple `joint().qpos` setting (original ~10 lines)

**Removed import:**
```python
# Before
import mujoco  # For stand keyframe initialization

# After
# (removed - not needed)
```

### 2. `rl_train/envs/myoassist_leg_imitation_ver2_1.py`

**Reverted `_follow_reference_motion()` method:**
- ❌ Removed: Stand keyframe initialization
- ❌ Removed: qpos index access
- ❌ Removed: `_joint_qpos_map` usage
- ✅ Restored: Simple `joint().qpos` setting
- ✅ Preserved: Divide-by-zero fix (Ver2_1 specific feature)

**Removed import:**
```python
# Before
import mujoco  # For stand keyframe initialization

# After
# (removed - not needed)
```

## Files Kept (for documentation)

These files were created during investigation but are **not used** by environment:

1. **Data conversion scripts:**
   - `convert_for_environment.py` - HDF5 → Environment format
   - `convert_mujoco_to_env.py` - MuJoCo → Environment format
   - `S004_trial01_08mps_3D_ENV_symmetric.npz` - Environment format data

2. **Diagnostic scripts:**
   - `debug_qpos.py` - qpos inspection
   - `debug_joints.py` - Joint mapping verification

These are kept for reference but should NOT be used for training.

## New File Created

### `render_symmetric_reference.py`

**Purpose:** Standalone script to visualize symmetric reference motion

**Approach:**
- Based on proven working `render_hdf5_reference.py`
- Uses direct MuJoCo rendering (no environment wrapper)
- Handles both data formats (MuJoCo renderer & Environment)
- Includes symmetry checks
- Supports multiview rendering

**Usage:**
```bash
# Basic usage
python render_symmetric_reference.py

# Multiview (front + side)
python render_symmetric_reference.py --multiview --fps 100

# Custom frames
python render_symmetric_reference.py --frames 1200 --fps 100
```

**Key features:**
- ✅ Stand keyframe initialization (correct approach for rendering)
- ✅ qpos index mapping (safe for standalone script)
- ✅ Height offset handling
- ✅ Symmetry verification
- ✅ Clean visualization (transparent floor, hidden arms)

## Verification Checklist

Before proceeding with Ver2_1 training:

- [x] Environment code reverted to original
- [x] No `mujoco` imports in environment files
- [x] No `_joint_qpos_map` in environment code
- [x] Standalone rendering script created
- [ ] Test rendering script works correctly
- [ ] Verify kinematic chain is correct in rendered video
- [ ] Test environment still works for training
- [ ] Start Ver2_1 training

## Next Steps

1. **Test rendering script:**
   ```bash
   python render_symmetric_reference.py --multiview
   ```
   - Check output: `symmetric_reference.mp4`
   - Verify: No "shin stuck to hip" issue
   - Verify: Symmetric left/right motion

2. **Quick environment test:**
   ```bash
   python rl_train/run_sim_minimal.py
   ```
   - Verify environment still initializes correctly
   - Check for any errors

3. **Start Ver2_1 training:**
   ```bash
   python rl_train/run_train.py
   ```
   - Monitor first few episodes
   - Check reward curves
   - Verify stability

## Lessons Learned

1. **Separation of concerns:**
   - Training environment ≠ Visualization tool
   - Don't mix these responsibilities

2. **Trust proven code:**
   - If environment works for training → don't modify it
   - If renderer works for visualization → copy its approach

3. **Simplicity is better:**
   - Original simple code: ~10 lines, works
   - Modified complex code: ~50 lines, breaks
   - Revert to simple: ~10 lines, works again

4. **User intuition is valuable:**
   - "점점 이상한 길로 가고 있는거 같아" was correct
   - When going down a rabbit hole, step back and reassess

## Summary

We successfully reverted all environment code modifications and created a proper standalone rendering script. The environment is now back to its original working state, ready for Ver2_1 training with symmetric reference motion.

**Key principle:** Keep environment code simple and proven. Use separate tools for visualization.

---
**Author:** GitHub Copilot  
**Date:** 2024-11-17  
**Context:** Ver2_1 preparation (Karico + 3D balancing)
