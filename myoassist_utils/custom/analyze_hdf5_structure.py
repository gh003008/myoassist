#!/usr/bin/env python3
"""
Analyze HDF5 structure to find proper joint names
"""
import h5py
import numpy as np

hdf5_path = r'C:\workspace_home\opensim data\LD\S004.h5'

print('='*80)
print('HDF5 Structure Analysis')
print('='*80)

with h5py.File(hdf5_path, 'r') as f:
    # Navigate to MoCap
    mocap = f['S004']['level_08mps']['trial_01']['MoCap']
    print(f'\n[MoCap keys]: {list(mocap.keys())}')
    
    # Get ik_data
    ik_data_group = mocap['ik_data']
    print(f'\n[ik_data type]: {type(ik_data_group)}')
    
    # If it's a group, list its keys
    if isinstance(ik_data_group, h5py.Group):
        print(f'[ik_data keys]: {list(ik_data_group.keys())}')
        
        # Check each sub-item
        for key in ik_data_group.keys():
            item = ik_data_group[key]
            if isinstance(item, h5py.Dataset):
                print(f'\n  Dataset: {key}')
                print(f'    Shape: {item.shape}')
                print(f'    Dtype: {item.dtype}')
                
                # If small or contains strings, print it
                if item.shape[0] < 100 or item.dtype.kind in ['S', 'U', 'O']:
                    data = item[()]
                    if isinstance(data, bytes):
                        data = data.decode('utf-8')
                    elif isinstance(data, np.ndarray) and data.dtype.kind in ['S', 'U', 'O']:
                        data = [d.decode('utf-8') if isinstance(d, bytes) else str(d) for d in data[:20]]
                    print(f'    Data (first 20): {data}')
    else:
        print(f'[ik_data shape]: {ik_data_group.shape}')
        print(f'[ik_data dtype]: {ik_data_group.dtype}')
    
    # Look for header-related keys
    print(f'\n[Looking for header/column info in MoCap]:')
    for key in mocap.keys():
        if any(keyword in key.lower() for keyword in ['header', 'column', 'label', 'name']):
            print(f'  Found key: {key}')
            dset = mocap[key]
            print(f'    Shape: {dset.shape}')
            if dset.shape[0] < 100:  # If small, print it
                data = dset[()]
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                elif isinstance(data, np.ndarray) and data.dtype.kind in ['S', 'U', 'O']:
                    data = [d.decode('utf-8') if isinstance(d, bytes) else str(d) for d in data]
                print(f'    Data: {data}')
    
    # Try known OpenSim joint names in any text fields
    print(f'\n[Searching for OpenSim joint names]:')
    target_joints = ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 
                     'knee_angle_r', 'ankle_angle_r', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz']
    
    for key in mocap.keys():
        try:
            data = mocap[key][()]
            if isinstance(data, np.ndarray):
                # Check if it contains string data
                if data.dtype.kind in ['S', 'U', 'O']:
                    strings = [d.decode('utf-8') if isinstance(d, bytes) else str(d) for d in data]
                    for target in target_joints:
                        if any(target in s for s in strings):
                            print(f'  Found "{target}" in key: {key}')
                            print(f'    All strings: {strings[:10]}...')  # Show first 10
                            break
        except:
            pass

print('\n' + '='*80)
print('DONE')
print('='*80)
