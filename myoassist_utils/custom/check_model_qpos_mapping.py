#!/usr/bin/env python3
"""
Find which NPZ axis corresponds to height by checking the model
"""
import numpy as np
import mujoco

model_path = r'C:\workspace_home\myoassist\models\22muscle_2D\myoLeg22_2D_BASELINE.xml'
npz_path = r'C:\workspace_home\opensim data\LD_gdp\S004\level_08mps\trial_01.npz'

print('='*80)
print('Model qpos Analysis - Find Height Axis')
print('='*80)

# Load model
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

print(f'\n[MuJoCo Model Info]:')
print(f'  nq (position DOF): {model.nq}')
print(f'  Joint names:')
for i in range(model.njnt):
    jnt_name = model.joint(i).name
    jnt_type = model.jnt_type[i]
    qpos_addr = model.jnt_qposadr[i]
    print(f'    [{i}] {jnt_name:20s} type={jnt_type} qpos_addr={qpos_addr}')

print(f'\n[First 20 qpos indices]:')
for i in range(min(20, model.nq)):
    print(f'    qpos[{i:2d}] = {data.qpos[i]:.6f}')

# Load NPZ
npz_data = np.load(npz_path, allow_pickle=True)
model_states = npz_data['model_states']
columns = npz_data['model_states_columns']

print(f'\n[NPZ model_states first 20 columns]:')
for i in range(min(20, len(columns))):
    col_name = columns[i]
    data_sample = model_states[0, i]
    data_range = [model_states[:, i].min(), model_states[:, i].max()]
    print(f'    [{i:2d}] {col_name:30s}: first={data_sample:+.6f}, range=[{data_range[0]:+.3f}, {data_range[1]:+.3f}]')

# Check which axis varies most (likely height during walking)
print(f'\n[Pelvis translation variation (std dev)]:')
for axis in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']:
    idx = np.where(columns == axis)[0]
    if len(idx) > 0:
        idx = idx[0]
        std = model_states[:, idx].std()
        mean = model_states[:, idx].mean()
        print(f'  {axis}: std={std:.4f} m, mean={mean:+.4f} m')

print('\n' + '='*80)
print('Check which pelvis_t* has largest variation')
print('='*80)
