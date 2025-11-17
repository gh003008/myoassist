#!/usr/bin/env python3
"""
Complete Mapping Analysis: HDF5 → Model, NPZ → Model
"""
import numpy as np
import mujoco
import h5py

print('='*100)
print('COMPLETE JOINT MAPPING ANALYSIS')
print('='*100)

# ============================================================================
# 1. MuJoCo Model Structure
# ============================================================================
model_path = r'C:\workspace_home\myoassist\models\22muscle_2D\myoLeg22_2D_BASELINE.xml'
model = mujoco.MjModel.from_xml_path(model_path)

print('\n' + '='*100)
print('1. MuJoCo MODEL STRUCTURE')
print('='*100)
print(f'Total qpos: {model.nq}')
print(f'Total actuators (muscles): {model.nu}')
print(f'Total joints: {model.njnt}')

print('\n[Main Skeletal Joints - these need reference data]:')
print(f'{"qpos_idx":<10s} {"Joint Name":<30s} {"Type":<10s} {"Range":<30s}')
print('-'*100)

main_joints = []
for i in range(model.njnt):
    jnt_name = model.joint(i).name
    qpos_addr = model.jnt_qposadr[i]
    jnt_type = model.jnt_type[i]
    type_names = {0: 'free', 1: 'ball', 2: 'slide', 3: 'hinge'}
    type_name = type_names.get(jnt_type, f'type{jnt_type}')
    
    # Main skeletal joints (not muscle wrapping points)
    if any(key in jnt_name for key in ['pelvis_tx', 'pelvis_ty', 'pelvis_tilt', 
                                         'hip_flexion', 'knee_angle', 'ankle_angle', 'mtp_angle']):
        if model.jnt_limited[i]:
            jnt_range = model.jnt_range[i]
            range_str = f'[{np.degrees(jnt_range[0]):7.1f}, {np.degrees(jnt_range[1]):7.1f}]°'
        else:
            range_str = 'unlimited'
        
        print(f'qpos[{qpos_addr:2d}]  {jnt_name:<30s} {type_name:<10s} {range_str:<30s}')
        main_joints.append((qpos_addr, jnt_name))

# ============================================================================
# 2. HDF5 Data Structure
# ============================================================================
hdf5_path = r'C:\workspace_home\opensim data\LD\S004.h5'

print('\n' + '='*100)
print('2. HDF5 (OpenSim) DATA STRUCTURE')
print('='*100)

with h5py.File(hdf5_path, 'r') as f:
    ik_data = f['S004']['level_08mps']['trial_01']['MoCap']['ik_data']
    
    print(f'\nAvailable joints in HDF5:')
    print(f'{"HDF5 Joint Name":<30s} {"Shape":<15s} {"Range (degrees)":<40s}')
    print('-'*100)
    
    h5_joints = {}
    for key in sorted(ik_data.keys()):
        if key not in ['infos', 'time']:
            data = ik_data[key][()]
            h5_joints[key] = data
            print(f'{key:<30s} {str(data.shape):<15s} [{data.min():7.2f}, {data.max():7.2f}]')

# ============================================================================
# 3. NPZ Data Structure
# ============================================================================
npz_path = r'C:\workspace_home\opensim data\LD_gdp\S004\level_08mps\trial_01.npz'
npz_data = np.load(npz_path, allow_pickle=True)
model_states = npz_data['model_states']
columns = npz_data['model_states_columns']

print('\n' + '='*100)
print('3. NPZ (Intermediate) DATA STRUCTURE')
print('='*100)
print(f'\nRelevant columns in NPZ model_states:')
print(f'{"Col_idx":<10s} {"NPZ Column Name":<30s} {"Range":<40s}')
print('-'*100)

npz_joints = {}
for i, col_name in enumerate(columns):
    # Only show relevant joint data (not forces, velocities)
    if any(key in str(col_name) for key in ['pelvis_t', 'pelvis_0', 'pelvis_1', 'pelvis_2',
                                              'hip_', 'knee_angle', 'ankle_angle']):
        if 'vel' not in str(col_name) and 'force' not in str(col_name):
            data = model_states[:, i]
            npz_joints[str(col_name)] = (i, data)
            print(f'[{i:2d}]       {str(col_name):<30s} [{data.min():8.3f}, {data.max():8.3f}]')

# ============================================================================
# 4. Current HDF5 → Model Mapping (convert_hdf5_direct.py)
# ============================================================================
print('\n' + '='*100)
print('4. CURRENT HDF5 → MyoAssist MAPPING (convert_hdf5_direct.py)')
print('='*100)

from convert_hdf5_direct import MYOASSIST_JOINTS

print(f'\nMYOASSIST_JOINTS order (what we generate):')
for i, jnt_name in enumerate(MYOASSIST_JOINTS):
    print(f'  q_ref[{i:2d}] = {jnt_name}')

print('\n[HDF5 → MYOASSIST_JOINTS mapping]:')
print(f'{"q_ref_idx":<10s} {"MYOASSIST name":<25s} {"←":<5s} {"HDF5 name":<25s} {"Transform":<30s}')
print('-'*100)

# Pelvis translations
print(f'q_ref[0]   {"q_pelvis_tx":<25s} {"←":<5s} {"pelvis_tx":<25s} {"direct (right)":<30s}')
print(f'q_ref[1]   {"q_pelvis_ty":<25s} {"←":<5s} {"pelvis_ty":<25s} {"+ height_offset":<30s}')
print(f'q_ref[2]   {"q_pelvis_tz":<25s} {"←":<5s} {"pelvis_tz":<25s} {"direct (forward)":<30s}')

# Pelvis rotations
print(f'q_ref[3]   {"q_pelvis_list":<25s} {"←":<5s} {"pelvis_rotation":<25s} {"SWAP":<30s}')
print(f'q_ref[4]   {"q_pelvis_tilt":<25s} {"←":<5s} {"pelvis_list":<25s} {"-pelvis_list + 75deg":<30s}')
print(f'q_ref[5]   {"q_pelvis_rotation":<25s} {"←":<5s} {"pelvis_tilt":<25s} {"SWAP":<30s}')

# Hip/knee/ankle
print(f'q_ref[6]   {"hip_flexion_r":<25s} {"←":<5s} {"hip_flexion_r":<25s} {"direct":<30s}')
print(f'q_ref[7]   {"hip_adduction_r":<25s} {"←":<5s} {"hip_adduction_r":<25s} {"direct":<30s}')
print(f'q_ref[8]   {"hip_rotation_r":<25s} {"←":<5s} {"hip_rotation_r":<25s} {"direct":<30s}')
print(f'q_ref[9]   {"hip_flexion_l":<25s} {"←":<5s} {"hip_flexion_l":<25s} {"direct":<30s}')
print(f'q_ref[10]  {"hip_adduction_l":<25s} {"←":<5s} {"hip_adduction_l":<25s} {"direct":<30s}')
print(f'q_ref[11]  {"hip_rotation_l":<25s} {"←":<5s} {"hip_rotation_l":<25s} {"direct":<30s}')
print(f'q_ref[12]  {"knee_angle_r":<25s} {"←":<5s} {"knee_angle_r":<25s} {"direct":<30s}')
print(f'q_ref[13]  {"knee_angle_l":<25s} {"←":<5s} {"knee_angle_l":<25s} {"direct":<30s}')
print(f'q_ref[14]  {"ankle_angle_r":<25s} {"←":<5s} {"ankle_angle_r":<25s} {"direct":<30s}')
print(f'q_ref[15]  {"ankle_angle_l":<25s} {"←":<5s} {"ankle_angle_l":<25s} {"direct":<30s}')

# ============================================================================
# 5. Current Renderer Mapping (render_hdf5_reference.py)
# ============================================================================
print('\n' + '='*100)
print('5. CURRENT RENDERER MAPPING (render_hdf5_reference.py)')
print('='*100)
print('\n[q_ref → model.qpos mapping]:')
print(f'{"q_ref_idx":<10s} {"q_ref name":<25s} {"→":<5s} {"qpos_idx":<10s} {"model joint":<25s}')
print('-'*100)

ref_joint_order = [
    (0, 'q_pelvis_tx', 0, 'pelvis_tx'),
    (1, 'q_pelvis_ty', 1, 'pelvis_ty'),
    (4, 'q_pelvis_tilt', 2, 'pelvis_tilt'),
    (6, 'hip_flexion_r', 3, 'hip_flexion_r'),
    (9, 'hip_flexion_l', 18, 'hip_flexion_l'),
    (12, 'knee_angle_r', 6, 'knee_angle_r'),
    (13, 'knee_angle_l', 21, 'knee_angle_l'),
    (14, 'ankle_angle_r', 7, 'ankle_angle_r'),
    (15, 'ankle_angle_l', 22, 'ankle_angle_l'),
]

for ref_idx, ref_name, qpos_idx, mj_name in ref_joint_order:
    print(f'q_ref[{ref_idx:2d}]  {ref_name:<25s} {"→":<5s} qpos[{qpos_idx:2d}]   {mj_name:<25s}')

print('\n❗ SKIPPED (not in 2D model):')
print(f'  q_ref[2]  q_pelvis_tz')
print(f'  q_ref[3]  q_pelvis_list')
print(f'  q_ref[5]  q_pelvis_rotation')
print(f'  q_ref[7]  hip_adduction_r')
print(f'  q_ref[8]  hip_rotation_r')
print(f'  q_ref[10] hip_adduction_l')
print(f'  q_ref[11] hip_rotation_l')

# ============================================================================
# 6. NPZ → Model Mapping (convert_motion_data.py reference)
# ============================================================================
print('\n' + '='*100)
print('6. NPZ → MyoAssist MAPPING (convert_motion_data.py - for reference)')
print('='*100)
print('\n[NPZ model_states → series_data (3D) mapping]:')
print(f'{"NPZ column":<30s} {"→":<5s} {"series_data key":<30s} {"Transform":<40s}')
print('-'*100)

# From convert_motion_data.py convert_3d()
print(f'{"pelvis_tx":<30s} {"→":<5s} {"q_pelvis_tx":<30s} {"= opensim_tz (forward!)":<40s}')
print(f'{"pelvis_ty":<30s} {"→":<5s} {"q_pelvis_ty":<30s} {"= -opensim_tx + offset (left!)":<40s}')
print(f'{"pelvis_tz":<30s} {"→":<5s} {"q_pelvis_tz":<30s} {"= opensim_ty (up!)":<40s}')
print(f'{"pelvis_0":<30s} {"→":<5s} {"q_pelvis_list":<30s} {"= opensim_rotation":<40s}')
print(f'{"pelvis_1":<30s} {"→":<5s} {"q_pelvis_tilt":<30s} {"= -opensim_list + 75deg":<40s}')
print(f'{"pelvis_2":<30s} {"→":<5s} {"q_pelvis_rotation":<30s} {"= opensim_tilt":<40s}')

print('\n💡 KEY INSIGHT: NPZ uses DIFFERENT coordinate system than HDF5!')
print('   NPZ pelvis_tx/ty/tz are NOT the same as HDF5 pelvis_tx/ty/tz')

print('\n' + '='*100)
print('SUMMARY & RECOMMENDATIONS')
print('='*100)

print('\n🔍 PROBLEM DIAGNOSIS:')
print('   - HDF5 converter is applying NPZ-style rotation swaps')
print('   - But HDF5 is pure OpenSim, should use DIRECT mapping!')
print('   - Model uses OpenSim coordinate system (X=right, Y=up, Z=forward)')
print('   - Only need height offset, NO rotation swaps needed')

print('\n✅ CORRECT HDF5 → Model mapping should be:')
print('   Translations: Direct copy (tx, ty+offset, tz)')
print('   Rotations: Direct copy (list, tilt+offset, rotation)')
print('   Hip/knee/ankle: Direct copy')

print('\n' + '='*100)
