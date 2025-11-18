#!/usr/bin/env python3
"""
Convert MuJoCo renderer format NPZ to Environment format NPZ

Takes existing NPZ with q_ref/joint_names (q_ prefix) 
and converts to series_data/metadata (NO q_ prefix) format.

Usage:
    python convert_mujoco_to_env.py
"""
import numpy as np
import os

# Input: MuJoCo renderer format (with q_ prefix)
INPUT_NPZ = r'rl_train/reference_data/S004_trial01_08mps_3D_HDF5_v7_symmetric.npz'

# Output: Environment format (without q_ prefix)
OUTPUT_NPZ = r'rl_train/reference_data/S004_trial01_08mps_3D_ENV_symmetric.npz'

def convert_mujoco_to_env(input_path, output_path):
    """
    Convert MuJoCo renderer format to Environment format
    
    Input format:
        - q_ref: [n_frames, n_dof] array
        - joint_names: ['q_pelvis_tx', 'q_hip_flexion_r', ...] (with q_ prefix)
        - series_data: {'q_pelvis_tx': [...], 'dq_pelvis_tx': [...]} (with q_ and dq_ prefix)
        
    Output format:
        - series_data: {'pelvis_tx': [...], 'dpelvis_tx': [...]} (NO q_ prefix, 'd' not 'dq')
        - metadata: {...}
    """
    
    print('='*80)
    print('Converting MuJoCo Renderer Format → Environment Format')
    print('='*80)
    
    # Load input
    print(f'\n📂 Loading: {input_path}')
    data = np.load(input_path, allow_pickle=True)
    
    print(f'   Keys found: {list(data.keys())}')
    
    # Check format
    if 'q_ref' not in data or 'joint_names' not in data:
        raise ValueError('Input must be MuJoCo renderer format (q_ref, joint_names)')
    
    q_ref = data['q_ref']
    joint_names = data['joint_names']
    
    print(f'\n📊 Input data:')
    print(f'   Frames: {q_ref.shape[0]}')
    print(f'   DOF: {q_ref.shape[1]}')
    print(f'   Joint names (first 5): {[str(n) for n in joint_names[:5]]}')
    
    # Create environment series_data
    print(f'\n🔄 Converting to environment format...')
    series_data = {}
    
    for i, joint_name in enumerate(joint_names):
        joint_name_str = str(joint_name)
        
        # Remove q_ prefix
        if joint_name_str.startswith('q_'):
            env_joint_name = joint_name_str[2:]
        else:
            env_joint_name = joint_name_str
        
        # Position data (NO modification to pelvis_ty - already ground-relative)
        # environment_handler.py will add +0.91m offset when loading
        series_data[env_joint_name] = q_ref[:, i]
        
        # Velocity data (d prefix, NOT dq)
        dq = np.gradient(q_ref[:, i], axis=0) * 100  # 100 Hz
        series_data[f'd{env_joint_name}'] = dq
    
    print(f'   ✅ Created {len(series_data)} series_data keys')
    
    # Create metadata
    metadata = {
        'sample_rate': 100.0,
        'data_length': q_ref.shape[0],
        'duration': q_ref.shape[0] / 100.0,
        'dof': q_ref.shape[1],
        'model_type': '3D',
        'source': 'converted_from_mujoco_format',
        'format': 'environment',
    }
    
    # Save
    print(f'\n💾 Saving: {output_path}')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.savez(
        output_path,
        series_data=series_data,
        metadata=metadata,
    )
    
    print(f'\n{"="*80}')
    print(f'✅ CONVERSION COMPLETE!')
    print(f'{"="*80}')
    print(f'Output format: ENVIRONMENT (series_data, metadata)')
    print(f'  Frames: {metadata["data_length"]}')
    print(f'  DOF: {metadata["dof"]}')
    print(f'  Duration: {metadata["duration"]:.2f} sec')
    print(f'  Sample rate: {metadata["sample_rate"]} Hz')
    print(f'\nSeries data keys (NO q_ prefix):')
    position_keys = sorted([k for k in series_data.keys() if not k.startswith('d')])
    for i, key in enumerate(position_keys[:8]):
        print(f'  {key}')
    print(f'  ... ({len(position_keys)} position keys total)')
    print(f'\nVelocity keys (d prefix, NOT dq):')
    velocity_keys = sorted([k for k in series_data.keys() if k.startswith('d')])
    for i, key in enumerate(velocity_keys[:8]):
        print(f'  {key}')
    print(f'  ... ({len(velocity_keys)} velocity keys total)')
    
    # Verification
    print(f'\n🔍 Verification:')
    
    # Re-load and check
    verify = np.load(output_path, allow_pickle=True)
    v_series = verify['series_data'].item()
    v_meta = verify['metadata'].item()
    
    print(f'   ✅ File can be loaded')
    print(f'   ✅ series_data has {len(v_series)} keys')
    print(f'   ✅ metadata has {len(v_meta)} keys')
    print(f'   ✅ Sample key "pelvis_tx" shape: {v_series["pelvis_tx"].shape}')
    print(f'   ✅ Sample key "dpelvis_tx" shape: {v_series["dpelvis_tx"].shape}')
    
    # Check for q_ prefix (should NOT exist)
    q_prefix_keys = [k for k in v_series.keys() if k.startswith('q_')]
    if q_prefix_keys:
        print(f'   ⚠️  WARNING: Found keys with q_ prefix: {q_prefix_keys}')
    else:
        print(f'   ✅ No q_ prefix found (correct!)')
    
    # Check for dq_ prefix (should NOT exist, should be just 'd')
    dq_prefix_keys = [k for k in v_series.keys() if k.startswith('dq_')]
    if dq_prefix_keys:
        print(f'   ⚠️  WARNING: Found keys with dq_ prefix: {dq_prefix_keys}')
    else:
        print(f'   ✅ No dq_ prefix found (correct! Using d prefix)')

def main():
    convert_mujoco_to_env(INPUT_NPZ, OUTPUT_NPZ)
    
    print(f'\n{"="*80}')
    print(f'✅ DONE!')
    print(f'{"="*80}')
    print(f'\nNext steps:')
    print(f'1. Use visualize_symmetric_in_env.py with new NPZ:')
    print(f'   python -m rl_train.analyzer.custom.visualize_symmetric_in_env \\')
    print(f'       --reference {OUTPUT_NPZ} \\')
    print(f'       --speed 0.5 --steps 400')
    print(f'\n2. Update training config to use new NPZ:')
    print(f'   "reference_data_path": "{OUTPUT_NPZ}"')

if __name__ == '__main__':
    main()
