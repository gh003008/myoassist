#!/usr/bin/env python3
"""
Check exact qpos mapping in MuJoCo model
"""
import mujoco
import numpy as np

model_path = r'C:\workspace_home\myoassist\models\22muscle_2D\myoLeg22_2D_BASELINE.xml'

print('='*80)
print('MuJoCo Model qpos Mapping')
print('='*80)

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

print(f'\nTotal qpos: {model.nq}')
print(f'Total joints: {model.njnt}')

print('\n[Joint → qpos mapping]:')
print(f'{"Joint Name":<30s} {"Type":<10s} {"qpos_addr":<10s} {"Range":<20s}')
print('-'*80)

for i in range(model.njnt):
    jnt_name = model.joint(i).name
    jnt_type = model.jnt_type[i]
    qpos_addr = model.jnt_qposadr[i]
    
    # Get joint type name
    type_names = {0: 'free', 1: 'ball', 2: 'slide', 3: 'hinge'}
    type_name = type_names.get(jnt_type, f'type{jnt_type}')
    
    # Get range if limited
    if model.jnt_limited[i]:
        jnt_range = model.jnt_range[i]
        range_str = f'[{np.degrees(jnt_range[0]):.1f}, {np.degrees(jnt_range[1]):.1f}]°'
    else:
        range_str = 'unlimited'
    
    # Highlight main motion joints
    if any(key in jnt_name for key in ['pelvis', 'hip', 'knee_angle', 'ankle_angle', 'mtp']):
        marker = '👉'
    else:
        marker = '  '
    
    print(f'{marker} {jnt_name:<28s} {type_name:<10s} {qpos_addr:<10d} {range_str:<20s}')

print('\n' + '='*80)
print('Key Joints for Reference Motion:')
print('='*80)

# List the joints we need to set
key_joints = [
    'pelvis_tx', 'pelvis_ty', 'pelvis_tilt',
    'hip_flexion_r', 'knee_angle_r', 'ankle_angle_r', 'mtp_angle_r',
    'hip_flexion_l', 'knee_angle_l', 'ankle_angle_l', 'mtp_angle_l'
]

print('\nqpos indices to set:')
for jnt_name in key_joints:
    for i in range(model.njnt):
        if model.joint(i).name == jnt_name:
            qpos_addr = model.jnt_qposadr[i]
            print(f'  qpos[{qpos_addr:2d}] = {jnt_name}')
            break

print('\n' + '='*80)
