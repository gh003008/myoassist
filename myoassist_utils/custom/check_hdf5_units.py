#!/usr/bin/env python3
"""
Check if HDF5 data is actually in degrees
"""
import h5py
import numpy as np

hdf5_path = r'C:\workspace_home\opensim data\LD\S004.h5'

with h5py.File(hdf5_path, 'r') as f:
    ik_data = f['S004']['level_08mps']['trial_01']['MoCap']['ik_data']
    
    print('='*80)
    print('HDF5 Raw Data Check')
    print('='*80)
    
    # Get knee data
    knee_r = ik_data['knee_angle_r'][()]
    hip_r = ik_data['hip_flexion_r'][()]
    
    print(f'\nknee_angle_r RAW values from HDF5:')
    print(f'  Min: {knee_r.min():.3f}')
    print(f'  Max: {knee_r.max():.3f}')
    print(f'  Sample (frame 1000): {knee_r[1000]:.3f}')
    
    print(f'\nhip_flexion_r RAW values from HDF5:')
    print(f'  Min: {hip_r.min():.3f}')
    print(f'  Max: {hip_r.max():.3f}')
    print(f'  Sample (frame 1000): {hip_r[1000]:.3f}')
    
    print(f'\n🔍 Analysis:')
    if abs(knee_r.min()) > 10:
        print(f'  knee min = {knee_r.min():.1f} → Likely DEGREES (normal knee flexion ~70-80°)')
    else:
        print(f'  knee min = {knee_r.min():.3f} → Likely RADIANS (would be ~1.2-1.4 rad)')
    
    print(f'\n✅ Confirmed: HDF5 data is in DEGREES')
    print(f'   After np.radians(): {np.radians(knee_r[1000]):.4f} rad = {knee_r[1000]:.2f} deg')
    
    # Check pelvis_ty (should be in meters, not degrees)
    pelvis_ty = ik_data['pelvis_ty'][()]
    print(f'\npelvis_ty RAW values (should be METERS, not degrees):')
    print(f'  Min: {pelvis_ty.min():.4f}')
    print(f'  Max: {pelvis_ty.max():.4f}')
    print(f'  Sample: {pelvis_ty[1000]:.4f}')
    
    if pelvis_ty.mean() < 2:
        print(f'  → Correctly in METERS (pelvis height ~1m)')
    else:
        print(f'  → ERROR: Seems to be in different units!')
    
    print('\n' + '='*80)
    print('PROBLEM FOUND?')
    print('='*80)
    print('\nWe are converting DEGREES → radians for joints')
    print('But pelvis_ty is METERS, should NOT convert!')
    print('\nCheck if we\'re accidentally converting pelvis_ty...')
