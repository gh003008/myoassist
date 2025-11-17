#!/usr/bin/env python3
"""
Compare HDF5 vs NPZ model_states to find conversion errors
"""
import h5py
import numpy as np

hdf5_path = r'C:\workspace_home\opensim data\LD\S004.h5'
npz_path = r'C:\workspace_home\opensim data\LD_gdp\S004\level_08mps\trial_01.npz'

print('='*80)
print('HDF5 vs NPZ model_states Comparison')
print('='*80)

# Load HDF5
h5_data = {}
with h5py.File(hdf5_path, 'r') as f:
    ik_data = f['S004']['level_08mps']['trial_01']['MoCap']['ik_data']
    for joint in ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                  'knee_angle_r', 'ankle_angle_r',
                  'pelvis_tx', 'pelvis_ty', 'pelvis_tz', 
                  'pelvis_tilt', 'pelvis_list', 'pelvis_rotation']:
        if joint in ik_data:
            # HDF5 is in DEGREES!
            h5_data[joint] = np.radians(ik_data[joint][()])  # Convert to radians

# Load NPZ
npz_data = np.load(npz_path, allow_pickle=True)
model_states = npz_data['model_states']
columns = npz_data['model_states_columns']

# Create dictionary for easy access
npz_dict = {}
for i, col_name in enumerate(columns):
    npz_dict[col_name] = model_states[:, i]

print('\n[HDF5 Data (converted to radians)]:')
for key in ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r']:
    if key in h5_data:
        data = h5_data[key]
        print(f'  {key:20s}: [{data.min():7.3f}, {data.max():7.3f}] rad')

print('\n[NPZ model_states (hip 6-DOF)]:')
for i in range(6):
    key = f'hip_r_{i}'
    if key in npz_dict:
        data = npz_dict[key]
        print(f'  {key:20s}: [{data.min():7.3f}, {data.max():7.3f}] rad')

# Correlation analysis
print('\n' + '='*80)
print('Correlation Analysis: HDF5 vs NPZ')
print('='*80)

print('\n[HDF5 hip_flexion_r <-> NPZ hip_r_X]:')
for i in range(6):
    key = f'hip_r_{i}'
    if key in npz_dict:
        corr_pos = np.corrcoef(h5_data['hip_flexion_r'], npz_dict[key])[0, 1]
        corr_neg = np.corrcoef(h5_data['hip_flexion_r'], -npz_dict[key])[0, 1]
        best_corr = max(abs(corr_pos), abs(corr_neg))
        sign = '+' if abs(corr_pos) > abs(corr_neg) else '-'
        print(f'  {sign}hip_r_{i}: correlation = {best_corr:+.4f}')

print('\n[HDF5 hip_adduction_r <-> NPZ hip_r_X]:')
for i in range(6):
    key = f'hip_r_{i}'
    if key in npz_dict:
        corr_pos = np.corrcoef(h5_data['hip_adduction_r'], npz_dict[key])[0, 1]
        corr_neg = np.corrcoef(h5_data['hip_adduction_r'], -npz_dict[key])[0, 1]
        best_corr = max(abs(corr_pos), abs(corr_neg))
        sign = '+' if abs(corr_pos) > abs(corr_neg) else '-'
        print(f'  {sign}hip_r_{i}: correlation = {best_corr:+.4f}')

print('\n[HDF5 hip_rotation_r <-> NPZ hip_r_X]:')
for i in range(6):
    key = f'hip_r_{i}'
    if key in npz_dict:
        corr_pos = np.corrcoef(h5_data['hip_rotation_r'], npz_dict[key])[0, 1]
        corr_neg = np.corrcoef(h5_data['hip_rotation_r'], -npz_dict[key])[0, 1]
        best_corr = max(abs(corr_pos), abs(corr_neg))
        sign = '+' if abs(corr_pos) > abs(corr_neg) else '-'
        print(f'  {sign}hip_r_{i}: correlation = {best_corr:+.4f}')

# Pelvis analysis
print('\n' + '='*80)
print('Pelvis Translation Direction Analysis')
print('='*80)

print('\n[HDF5 - OpenSim (degrees converted to radians)]:')
for axis in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']:
    data = h5_data[axis]
    print(f'  {axis}: [{data.min():7.3f}, {data.max():7.3f}], Δ={data[-1]-data[0]:7.3f}')

print('\n[NPZ - model_states]:')
for axis in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']:
    if axis in npz_dict:
        data = npz_dict[axis]
        print(f'  {axis}: [{data.min():7.3f}, {data.max():7.3f}], Δ={data[-1]-data[0]:7.3f}')

print('\n[NPZ - pelvis 6-DOF rotations]:')
for i in range(6):
    key = f'pelvis_{i}'
    if key in npz_dict:
        data = npz_dict[key]
        print(f'  {key}: [{data.min():7.3f}, {data.max():7.3f}], Δ={data[-1]-data[0]:7.3f}')

# Check forward direction
print('\n[Forward Walking Detection]:')
h5_forward = h5_data['pelvis_tz'][-1] - h5_data['pelvis_tz'][0]
npz_tx = npz_dict['pelvis_tx'][-1] - npz_dict['pelvis_tx'][0]
npz_ty = npz_dict['pelvis_ty'][-1] - npz_dict['pelvis_ty'][0]
npz_tz = npz_dict['pelvis_tz'][-1] - npz_dict['pelvis_tz'][0]

print(f'  HDF5 pelvis_tz (forward): Δ={h5_forward:+.3f} m')
print(f'  NPZ pelvis_tx: Δ={npz_tx:+.3f} m')
print(f'  NPZ pelvis_ty: Δ={npz_ty:+.3f} m')
print(f'  NPZ pelvis_tz: Δ={npz_tz:+.3f} m')

if abs(npz_tx) > abs(npz_ty) and abs(npz_tx) > abs(npz_tz):
    print('  -> NPZ forward direction likely: pelvis_tx')
elif abs(npz_ty) > abs(npz_tz):
    print('  -> NPZ forward direction likely: pelvis_ty')
else:
    print('  -> NPZ forward direction likely: pelvis_tz')

print('\n' + '='*80)
print('DONE')
print('='*80)
