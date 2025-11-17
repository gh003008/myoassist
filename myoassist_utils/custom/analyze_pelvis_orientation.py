#!/usr/bin/env python3
"""
Analyze pelvis orientation and height from HDF5
"""
import h5py
import numpy as np

hdf5_path = r'C:\workspace_home\opensim data\LD\S004.h5'

print('='*80)
print('Pelvis Orientation & Height Analysis')
print('='*80)

with h5py.File(hdf5_path, 'r') as f:
    ik_data = f['S004']['level_08mps']['trial_01']['MoCap']['ik_data']
    
    # Get pelvis data (in DEGREES)
    pelvis_tilt = ik_data['pelvis_tilt'][()]  # Forward/backward lean
    pelvis_list = ik_data['pelvis_list'][()]  # Side lean
    pelvis_rotation = ik_data['pelvis_rotation'][()]  # Twisting
    
    pelvis_tx = ik_data['pelvis_tx'][()]  # Right
    pelvis_ty = ik_data['pelvis_ty'][()]  # Up
    pelvis_tz = ik_data['pelvis_tz'][()]  # Forward
    
    print('\n[Pelvis Rotations - DEGREES]:')
    print(f'  pelvis_tilt (forward/back): [{pelvis_tilt.min():7.2f}, {pelvis_tilt.max():7.2f}] deg, mean={pelvis_tilt.mean():7.2f}')
    print(f'  pelvis_list (side lean):    [{pelvis_list.min():7.2f}, {pelvis_list.max():7.2f}] deg, mean={pelvis_list.mean():7.2f}')
    print(f'  pelvis_rotation (twist):    [{pelvis_rotation.min():7.2f}, {pelvis_rotation.max():7.2f}] deg, mean={pelvis_rotation.mean():7.2f}')
    
    print('\n[Pelvis Translations - METERS]:')
    print(f'  pelvis_tx (right):   [{pelvis_tx.min():7.3f}, {pelvis_tx.max():7.3f}] m, mean={pelvis_tx.mean():7.3f}')
    print(f'  pelvis_ty (up):      [{pelvis_ty.min():7.3f}, {pelvis_ty.max():7.3f}] m, mean={pelvis_ty.mean():7.3f}')
    print(f'  pelvis_tz (forward): [{pelvis_tz.min():7.3f}, {pelvis_tz.max():7.3f}] m, mean={pelvis_tz.mean():7.3f}')
    
    print('\n[Key Statistics]:')
    print(f'  Average pelvis height (pelvis_ty): {pelvis_ty.mean():.3f} m')
    print(f'  Average forward tilt (pelvis_tilt): {pelvis_tilt.mean():.2f} deg')
    
    # Check if we need offset
    print('\n[Recommended Offsets]:')
    print(f'  Height offset: +{pelvis_ty.mean():.3f} m (to bring pelvis_ty to this height)')
    print(f'  Tilt offset: {-pelvis_tilt.mean():.2f} deg (to straighten upright)')

print('\n' + '='*80)
print('DONE')
print('='*80)
