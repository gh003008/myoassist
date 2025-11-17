#!/usr/bin/env python3
"""
Inspect NPZ structure in detail
"""
import numpy as np

npz_path = r'C:\workspace_home\opensim data\LD_gdp\S004\level_08mps\trial_01.npz'

print('='*80)
print('NPZ Structure Inspection')
print('='*80)

npz_data = np.load(npz_path, allow_pickle=True)

print(f'\nKeys: {list(npz_data.keys())}')

for key in npz_data.keys():
    data = npz_data[key]
    print(f'\n[{key}]:')
    print(f'  Type: {type(data)}')
    print(f'  Dtype: {data.dtype}')
    print(f'  Shape: {data.shape}')
    
    if key == 'model_states_columns':
        print(f'  Columns:')
        for i, col in enumerate(data):
            print(f'    [{i:2d}] {col}')
    elif key == 'model_states':
        print(f'  Data shape: {data.shape}')
        print(f'  Sample (first frame):')
        for i in range(min(10, data.shape[1])):
            print(f'    [{i:2d}] {data[0, i]:10.6f}')
    elif data.size < 20:
        print(f'  Data: {data}')
    else:
        print(f'  Range: [{data.min():.6f}, {data.max():.6f}]')
        print(f'  First 5: {data[:5]}')
        print(f'  Last 5: {data[-5:]}')

print('\n' + '='*80)
print('DONE')
print('='*80)
