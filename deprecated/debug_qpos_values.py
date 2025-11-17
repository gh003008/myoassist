#!/usr/bin/env python3
"""
Debug: Check actual qpos values being set
"""
import numpy as np
import mujoco

model_path = r'C:\workspace_home\myoassist\models\22muscle_2D\myoLeg22_2D_BASELINE.xml'
npz_path = r'C:\workspace_home\myoassist\rl_train\reference_data\S004_trial01_08mps_3D_HDF5_v4.npz'

print('='*80)
print('Debug qpos Values')
print('='*80)

# Load data
data = np.load(npz_path)
q_ref = data['q_ref']
joint_names = data['joint_names']

print(f'\nq_ref shape: {q_ref.shape}')
print(f'joint_names: {list(joint_names)}')

# Load model
model = mujoco.MjModel.from_xml_path(model_path)
data_mj = mujoco.MjData(model)

# Print what we're setting
print('\n' + '='*80)
print('Sample frame 1000 (middle of motion):')
print('='*80)

frame_idx = 1000
print(f'\n[q_ref values at frame {frame_idx}]:')
for i, name in enumerate(joint_names):
    val_rad = q_ref[frame_idx, i]
    val_deg = np.degrees(val_rad)
    print(f'  q_ref[{i:2d}] {str(name):25s} = {val_rad:+.4f} rad ({val_deg:+7.2f} deg)')

# Show the mapping we use
joint_to_qpos = {}
for i in range(model.njnt):
    jnt_name = model.joint(i).name
    qpos_addr = model.jnt_qposadr[i]
    joint_to_qpos[jnt_name] = qpos_addr

print('\n' + '='*80)
print('Mapping being applied:')
print('='*80)

ref_joint_order = [
    (0, 'q_pelvis_tx', 'pelvis_tx'),
    (1, 'q_pelvis_ty', 'pelvis_ty'),
    (4, 'q_pelvis_tilt', 'pelvis_tilt'),
    (6, 'hip_flexion_r', 'hip_flexion_r'),
    (9, 'hip_flexion_l', 'hip_flexion_l'),
    (12, 'knee_angle_r', 'knee_angle_r'),
    (13, 'knee_angle_l', 'knee_angle_l'),
    (14, 'ankle_angle_r', 'ankle_angle_r'),
    (15, 'ankle_angle_l', 'ankle_angle_l'),
]

print(f'\n{"q_ref[i]":<15s} {"value":<15s} {"→":<5s} {"qpos[j]":<15s} {"joint name":<20s}')
print('-'*80)

for ref_idx, ref_name, mj_name in ref_joint_order:
    if mj_name in joint_to_qpos:
        qpos_idx = joint_to_qpos[mj_name]
        val_rad = q_ref[frame_idx, ref_idx]
        val_deg = np.degrees(val_rad)
        print(f'q_ref[{ref_idx:2d}]    {val_rad:+8.4f} rad  {"→":<5s} qpos[{qpos_idx:2d}]      {mj_name:20s} ({val_deg:+7.2f}°)')

# Actually set and check
print('\n' + '='*80)
print('Setting qpos and checking result:')
print('='*80)

data_mj.qpos[:] = 0
for ref_idx, ref_name, mj_name in ref_joint_order:
    if mj_name in joint_to_qpos:
        qpos_idx = joint_to_qpos[mj_name]
        data_mj.qpos[qpos_idx] = q_ref[frame_idx, ref_idx]

mujoco.mj_forward(model, data_mj)

print(f'\nActual qpos values after setting:')
print(f'{"qpos[i]":<15s} {"value (rad)":<15s} {"value (deg)":<15s} {"joint name":<20s}')
print('-'*80)

for jnt_name in ['pelvis_tx', 'pelvis_ty', 'pelvis_tilt', 
                  'hip_flexion_r', 'hip_flexion_l',
                  'knee_angle_r', 'knee_angle_l',
                  'ankle_angle_r', 'ankle_angle_l']:
    if jnt_name in joint_to_qpos:
        qpos_idx = joint_to_qpos[jnt_name]
        val = data_mj.qpos[qpos_idx]
        print(f'qpos[{qpos_idx:2d}]      {val:+8.4f}       {np.degrees(val):+8.2f}       {jnt_name:20s}')

# Check knee specifically
print('\n' + '='*80)
print('KNEE ANALYSIS:')
print('='*80)

print(f'\nKnee_angle_r throughout motion:')
knee_r_data = q_ref[:, 12]  # q_ref[12] = knee_angle_r
print(f'  Min: {knee_r_data.min():.4f} rad ({np.degrees(knee_r_data.min()):+.1f}°)')
print(f'  Max: {knee_r_data.max():.4f} rad ({np.degrees(knee_r_data.max()):+.1f}°)')
print(f'  Mean: {knee_r_data.mean():.4f} rad ({np.degrees(knee_r_data.mean()):+.1f}°)')
print(f'  Std: {knee_r_data.std():.4f} rad ({np.degrees(knee_r_data.std()):+.1f}°)')

print(f'\nModel knee_angle_r limits:')
knee_r_jnt_id = model.joint('knee_angle_r').id
if model.jnt_limited[knee_r_jnt_id]:
    limits = model.jnt_range[knee_r_jnt_id]
    print(f'  Range: [{np.degrees(limits[0]):.1f}, {np.degrees(limits[1]):.1f}]°')
else:
    print(f'  Unlimited')

print('\n' + '='*80)
