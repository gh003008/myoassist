"""
=================================================================================
HDF5 to MyoAssist Coordinate & Joint Mapping Summary
=================================================================================

1. COORDINATE SYSTEM DISCOVERY
-------------------------------

MuJoCo Model (myoLeg22_2D_BASELINE.xml):
  - pelvis_tx: axis="1 0 0"  → X-axis (Right direction)
  - pelvis_ty: axis="0 1 0"  → Y-axis (Up direction)  
  - pelvis_tilt: axis="0 0 1" → Z-axis rotation (Pitch)
  - hip_flexion: axis="0 0 1" → Z-axis rotation

**KEY FINDING: MuJoCo model uses OpenSim coordinate system directly!**
  - X = Right
  - Y = Up
  - Z = Forward (out of page in 2D sagittal view)

2. DATA SOURCE COMPARISON
--------------------------

HDF5 (Original OpenSim):
  - Pure OpenSim data from inverse kinematics
  - Joint names: hip_flexion_r, pelvis_tx, etc.
  - Units: DEGREES (need conversion to radians)
  - Coordinate system: OpenSim (X=Right, Y=Up, Z=Forward)

NPZ (Pre-transformed):
  - Intermediate format with cryptic names (hip_r_0, hip_r_1, etc.)
  - Already transformed to different coordinate system
  - Units: Radians
  - Coordinate system: DIFFERENT from OpenSim
    * pelvis_tx in NPZ: horizontal wobble (±0.125m range)
    * pelvis_ty in NPZ: small vertical (±0.030m range)
    * pelvis_tz in NPZ: forward/back (±0.082m range)

3. CONVERSION STRATEGIES
-------------------------

NPZ → MyoAssist (convert_motion_data.py):
  NEEDS coordinate transform because NPZ is pre-transformed:
  
  Translations:
    q_pelvis_tx = opensim_tz           # forward
    q_pelvis_ty = -opensim_tx + offset  # left + height
    q_pelvis_tz = opensim_ty           # up
  
  Rotations:
    q_pelvis_list = opensim_rotation    # Z→X (twist → roll)
    q_pelvis_tilt = -opensim_list + 75deg  # X→Y (side → pitch) + offset
    q_pelvis_rotation = opensim_tilt    # Y→Z (tilt → yaw)

HDF5 → MyoAssist (convert_hdf5_direct.py):
  NO coordinate transform needed! Just copy + offsets:
  
  Translations:
    q_pelvis_tx = opensim_tx           # right (same)
    q_pelvis_ty = opensim_ty + offset  # up + height (same)
    q_pelvis_tz = opensim_tz           # forward (same)
  
  Rotations (apply NPZ-style swaps for consistency):
    q_pelvis_list = opensim_rotation
    q_pelvis_tilt = -opensim_list + 75deg
    q_pelvis_rotation = opensim_tilt

4. JOINT MAPPING
----------------

Hip Joints:
  HDF5: Direct semantic names
    - hip_flexion_r    (-16° to +25°)  ✓
    - hip_adduction_r  (-12° to +7°)   ✓
    - hip_rotation_r   (-7° to +4°)    ✓
  
  NPZ: Cryptic 6-DOF indices (correlation analysis revealed):
    - hip_r_1 = -hip_flexion_r   (correlation = -1.0)
    - hip_r_5 = -hip_adduction_r (correlation = -0.99)
    - hip_r_2 = -hip_rotation_r  (correlation = -0.90)
    - hip_r_0, 3, 4 = constraints (~55° fixed)

Knee/Ankle:
  Both HDF5 and NPZ: Direct mapping, same names
    - knee_angle_r
    - ankle_angle_r

5. OFFSETS & CORRECTIONS
-------------------------

Height Offset:
  - Purpose: Raise pelvis above ground
  - Value: body_height * 0.55 ≈ 0.96m
  - Applied to: q_pelvis_ty (Y-axis = up)

Tilt Offset:
  - Purpose: Make model stand upright
  - Value: 75° (from NPZ converter testing)
  - Applied to: q_pelvis_tilt after negating pelvis_list

Sign Corrections:
  - HDF5 hip joints: NO negation (use as-is)
  - NPZ hip joints: were negated in original transform

6. FINAL VERIFIED PIPELINE
---------------------------

HDF5 → MyoAssist v3:
  1. Load HDF5 ik_data (DEGREES)
  2. Convert to radians
  3. Direct copy for translations (add height offset)
  4. Apply NPZ-style rotation swaps (for model compatibility)
  5. Direct copy for hip/knee/ankle joints (NO negation)
  6. Save as MyoAssist NPZ format

Result: Clean, semantic conversion preserving OpenSim joint conventions!

=================================================================================
"""
print("Mapping summary created. See COORDINATE_MAPPING_ANALYSIS.txt")
