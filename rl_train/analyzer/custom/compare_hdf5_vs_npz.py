#!/usr/bin/env python3
"""
Compare HDF5 (original validated) vs NPZ (intermediate cryptic) data
"""
import h5py
import numpy as np

hdf5_path = r'C:\workspace_home\opensim data\LD\S004.h5'
npz_path = r'C:\workspace_home\opensim data\LD_gdp\S004\level_08mps\trial_01.npz'

print('='*80)
print('HDF5 vs NPZ Comparison')
print('='*80)

# Load HDF5
with h5py.File(hdf5_path, 'r') as f:
    ik_data = f['S004']['level_08mps']['trial_01']['MoCap']['ik_data']
    
    # Get key joints from HDF5
    print('\n[HDF5 Data - Proper Joint Names]:')
    h5_data = {}
    for joint in ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                  'knee_angle_r', 'ankle_angle_r',
                  'pelvis_tx', 'pelvis_ty', 'pelvis_tz', 'pelvis_tilt']:
        if joint in ik_data:
            data = ik_data[joint][()]
            h5_data[joint] = data
            print(f'  {joint:20s}: shape={data.shape}, range=[{data.min():7.3f}, {data.max():7.3f}], mean={data.mean():7.3f}')

# Load NPZ
npz_data = np.load(npz_path)
print(f'\n[NPZ Data - Cryptic Indexed Names]:')
print(f'  Keys: {sorted(npz_data.keys())}')

# Compare hip joints
print('\n' + '='*80)
print('Hip Joint Comparison (Degrees)')
print('='*80)

# HDF5 hip joints
hip_flex_h5 = np.degrees(h5_data['hip_flexion_r'])
hip_add_h5 = np.degrees(h5_data['hip_adduction_r'])
hip_rot_h5 = np.degrees(h5_data['hip_rotation_r'])

print('\n[HDF5]:')
print(f'  hip_flexion_r  : [{hip_flex_h5.min():7.2f}, {hip_flex_h5.max():7.2f}] deg, mean={hip_flex_h5.mean():7.2f}')
print(f'  hip_adduction_r: [{hip_add_h5.min():7.2f}, {hip_add_h5.max():7.2f}] deg, mean={hip_add_h5.mean():7.2f}')
print(f'  hip_rotation_r : [{hip_rot_h5.min():7.2f}, {hip_rot_h5.max():7.2f}] deg, mean={hip_rot_h5.mean():7.2f}')

# NPZ hip joints (all 6 DOF)
print('\n[NPZ - All 6 Hip DOF]:')
for i in range(6):
    key = f'hip_r_{i}'
    if key in npz_data:
        data = np.degrees(npz_data[key])
        print(f'  hip_r_{i}: [{data.min():7.2f}, {data.max():7.2f}] deg, mean={data.mean():7.2f}, std={data.std():6.2f}')

# Check pelvis translation direction
print('\n' + '='*80)
print('Pelvis Translation Analysis (Forward Walking Detection)')
print('='*80)

print('\n[HDF5 - OpenSim Coordinates (Y-up, Z-forward, X-right)]:')
pelvis_tx_h5 = h5_data['pelvis_tx']  # right
pelvis_ty_h5 = h5_data['pelvis_ty']  # up
pelvis_tz_h5 = h5_data['pelvis_tz']  # forward

print(f'  pelvis_tx (RIGHT)  : [{pelvis_tx_h5.min():7.3f}, {pelvis_tx_h5.max():7.3f}] m')
print(f'  pelvis_ty (UP)     : [{pelvis_ty_h5.min():7.3f}, {pelvis_ty_h5.max():7.3f}] m')
print(f'  pelvis_tz (FORWARD): [{pelvis_tz_h5.min():7.3f}, {pelvis_tz_h5.max():7.3f}] m')
print(f'\n  Forward progress (pelvis_tz): {pelvis_tz_h5[0]:.3f} -> {pelvis_tz_h5[-1]:.3f} m')
print(f'  Total distance: {pelvis_tz_h5[-1] - pelvis_tz_h5[0]:.3f} m')

print('\n[NPZ - Coordinate Unknown]:')
for axis in ['pelvis_0', 'pelvis_1', 'pelvis_2', 'pelvis_3', 'pelvis_4', 'pelvis_5']:
    if axis in npz_data:
        data = npz_data[axis]
        print(f'  {axis}: [{data.min():7.3f}, {data.max():7.3f}], first={data[0]:.3f}, last={data[-1]:.3f}')

# Check for sign/direction mismatch
print('\n' + '='*80)
print('Detecting Potential Sign/Direction Errors')
print('='*80)

# Compare hip_flexion_r with all 6 NPZ hip columns
print('\n[Correlation: HDF5 hip_flexion_r vs NPZ hip_r_X]:')
for i in range(6):
    key = f'hip_r_{i}'
    if key in npz_data:
        corr = np.corrcoef(h5_data['hip_flexion_r'], npz_data[key])[0, 1]
        print(f'  hip_flexion_r <-> hip_r_{i}: correlation = {corr:+.4f}')

print('\n[Correlation: HDF5 hip_adduction_r vs NPZ hip_r_X]:')
for i in range(6):
    key = f'hip_r_{i}'
    if key in npz_data:
        corr = np.corrcoef(h5_data['hip_adduction_r'], npz_data[key])[0, 1]
        print(f'  hip_adduction_r <-> hip_r_{i}: correlation = {corr:+.4f}')

print('\n[Correlation: HDF5 hip_rotation_r vs NPZ hip_r_X]:')
for i in range(6):
    key = f'hip_r_{i}'
    if key in npz_data:
        corr = np.corrcoef(h5_data['hip_rotation_r'], npz_data[key])[0, 1]
        print(f'  hip_rotation_r <-> hip_r_{i}: correlation = {corr:+.4f}')

print('\n' + '='*80)
print('DONE - Check correlations to identify NPZ column mappings')
print('='*80)
