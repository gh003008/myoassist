#!/usr/bin/env python3
"""
Direct HDF5 to MyoAssist converter
Bypasses problematic NPZ intermediate format
"""
import h5py
import numpy as np
import os

# ============================================================================
# CONFIG
# ============================================================================
CONFIG = {
    'input_hdf5': r'C:\workspace_home\opensim data\LD\S004.h5',
    'subject': 'S004',
    'speed': '08mps',
    'trial': 'trial_01',
    'output_dir': r'C:\workspace_home\myoassist\rl_train\reference_data',
    'output_name': 'S004_trial01_08mps_3D_HDF5_v7',
    
    # HDF5 = OpenSim, Model = OpenSim ??DIRECT MAPPING (NO coordinate transform!)
    # Use RELATIVE positions (centered at 0) like NPZ data
    # FIXED: pelvis rotation order now matches MuJoCo qpos order (tilt, list, rotation)
    
    # Offsets
    'height_offset_m': 0.0,   # NOT USED - using relative positions
    'tilt_offset_deg': 0.0,   # NO SWAP! Just offset if needed
}

# ============================================================================
# HDF5 ??MyoAssist joint mapping
# ============================================================================
# MyoAssist joint order (3D model with full DOF)
# CRITICAL: Order must match MuJoCo model qpos indices!
# MuJoCo model has: qpos[3]=pelvis_tilt, qpos[4]=pelvis_list
MYOASSIST_JOINTS = [
    'q_pelvis_tx',        # 0 ??qpos[0] - Right (X)
    'q_pelvis_ty',        # 1 ??qpos[1] - Up (Y)
    'q_pelvis_tz',        # 2 ??qpos[2] - Forward (Z)
    'q_pelvis_tilt',      # 3 ??qpos[3] - Pitch (forward/back tilt)
    'q_pelvis_list',      # 4 ??qpos[4] - Roll (side lean)
    'q_pelvis_rotation',  # 5 ??qpos[5] - Yaw (twist)
    'q_hip_flexion_r',    # 6 ??qpos[6]
    'q_hip_adduction_r',  # 7 ??qpos[7]
    'q_hip_rotation_r',   # 8 ??qpos[8]
    'q_hip_flexion_l',    # 9 ??qpos[23]
    'q_hip_adduction_l',  # 10 ??qpos[24]
    'q_hip_rotation_l',   # 11 ??qpos[25]
    'q_knee_angle_r',     # 12 ??qpos[11]
    'q_knee_angle_l',     # 13 ??qpos[28]
    'q_ankle_angle_r',    # 14 ??qpos[12]
    'q_ankle_angle_l',    # 15 ??qpos[29]
]

def load_hdf5_data(hdf5_path, subject, speed, trial):
    """Load motion data from HDF5 file
    
    IMPORTANT: HDF5 has mixed units!
    - Joint angles (hip, knee, ankle, pelvis rotations): DEGREES ??convert to radians
    - Translations (pelvis_tx, pelvis_ty, pelvis_tz): METERS ??keep as-is
    """
    print(f'Loading HDF5: {hdf5_path}')
    
    with h5py.File(hdf5_path, 'r') as f:
        path = f[subject][f'level_{speed}'][trial]['MoCap']['ik_data']
        
        data = {}
        translation_keys = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']
        
        for key in path.keys():
            if key != 'infos' and key != 'time':
                raw_data = path[key][()]
                
                # Translations are in METERS - keep as-is
                if key in translation_keys:
                    data[key] = raw_data
                # Rotations/angles are in DEGREES - convert to radians
                else:
                    data[key] = np.radians(raw_data)
        
        # Get time
        data['time'] = path['time'][()]
        
    print(f'  Loaded {len(data)} datasets, {len(data["time"])} frames')
    return data

def convert_to_myoassist(hdf5_data, config):
    """Convert HDF5 OpenSim data to MyoAssist format
    
    Key insight: MuJoCo model uses OpenSim coordinate system directly!
    - OpenSim & MuJoCo both use: X=Right, Y=Up, Z=Forward
    - NO coordinate transform needed (unlike NPZ converter)
    - Only apply offsets to match model's default pose
    """
    
    n_frames = len(hdf5_data['time'])
    series_data = {}
    
    # Get body height for offset calculation
    # From analysis: pelvis_ty average = 0.996m for 1.74m person
    # Ratio: 0.996 / 1.74 = 0.57 (use 0.55 to match NPZ converter)
    body_height = 1.74  # Default, should be in HDF5 if available
    height_offset = body_height * config.get('height_offset_m', 0.55)
    
    print(f'Using height offset: {height_offset:.3f} m (body_height * {config.get("height_offset_m", 0.55)})')
    
    # ========================================================================
    # Pelvis Translations - CONVERT TO RELATIVE POSITIONS
    # ========================================================================
    # CRITICAL: MyoAssist expects RELATIVE positions (centered around 0)
    # Not absolute positions! NPZ data shows q_pelvis_ty ~ 0.0 ± 0.03m
    
    # Calculate mean positions to center the data
    pelvis_tx_mean = np.mean(hdf5_data['pelvis_tx'])
    pelvis_ty_mean = np.mean(hdf5_data['pelvis_ty'])
    pelvis_tz_mean = np.mean(hdf5_data['pelvis_tz'])
    
    series_data['pelvis_tx'] = hdf5_data['pelvis_tx'] - pelvis_tx_mean  # Right (relative)
    series_data['pelvis_ty'] = hdf5_data['pelvis_ty'] - pelvis_ty_mean  # Up (relative)
    series_data['pelvis_tz'] = hdf5_data['pelvis_tz'] - pelvis_tz_mean  # Forward (relative)
    
    print(f'\nPelvis translation (converted to RELATIVE):')
    print(f'  TX (right):   [{series_data["pelvis_tx"].min():8.4f}, {series_data["pelvis_tx"].max():8.4f}] m (mean subtracted: {pelvis_tx_mean:.4f})')
    print(f'  TY (up):      [{series_data["pelvis_ty"].min():8.4f}, {series_data["pelvis_ty"].max():8.4f}] m (mean subtracted: {pelvis_ty_mean:.4f})')
    print(f'  TZ (forward): [{series_data["pelvis_tz"].min():8.4f}, {series_data["pelvis_tz"].max():8.4f}] m (mean subtracted: {pelvis_tz_mean:.4f})')
    
    # ========================================================================
    # Pelvis Rotations - DIRECT MAPPING (OpenSim = MuJoCo coordinate system)
    # ========================================================================
    tilt_offset_rad = np.radians(config.get('tilt_offset_deg', 0.0))
    
    # CRITICAL: Match MuJoCo qpos order! qpos[3]=tilt, qpos[4]=list
    series_data['pelvis_tilt'] = hdf5_data['pelvis_tilt'] + tilt_offset_rad  # Direct: forward/back tilt + offset
    series_data['pelvis_list'] = hdf5_data['pelvis_list']  # Direct: side lean
    series_data['pelvis_rotation'] = hdf5_data['pelvis_rotation']  # Direct: twist
    
    print(f'\nPelvis rotation ranges (DIRECT mapping, no swaps):')
    print(f'  tilt:     [{np.degrees(series_data["pelvis_tilt"].min()):+.1f}, {np.degrees(series_data["pelvis_tilt"].max()):+.1f}] deg (offset={config.get("tilt_offset_deg", 0.0)}deg)')
    print(f'  list:     [{np.degrees(series_data["pelvis_list"].min()):+.1f}, {np.degrees(series_data["pelvis_list"].max()):+.1f}] deg')
    print(f'  rotation: [{np.degrees(series_data["pelvis_rotation"].min()):+.1f}, {np.degrees(series_data["pelvis_rotation"].max()):+.1f}] deg')
    
    # ========================================================================
    # Hip joints - DIRECT MAPPING (OpenSim names = MuJoCo names)
    # Add 'q_' prefix for consistency with training pipeline
    # ========================================================================
    series_data['hip_flexion_r'] = hdf5_data['hip_flexion_r']
    series_data['hip_adduction_r'] = hdf5_data['hip_adduction_r']
    series_data['hip_rotation_r'] = hdf5_data['hip_rotation_r']
    
    series_data['hip_flexion_l'] = hdf5_data['hip_flexion_l']
    series_data['hip_adduction_l'] = hdf5_data['hip_adduction_l']
    series_data['hip_rotation_l'] = hdf5_data['hip_rotation_l']
    
    print(f'\nHip joint ranges (radians, DIRECT mapping):')
    print(f'  hip_flexion_r:   [{series_data["hip_flexion_r"].min():+.3f}, {series_data["hip_flexion_r"].max():+.3f}] ({np.degrees(series_data["hip_flexion_r"].min()):+.1f}, {np.degrees(series_data["hip_flexion_r"].max()):+.1f} deg)')
    print(f'  hip_adduction_r: [{series_data["hip_adduction_r"].min():+.3f}, {series_data["hip_adduction_r"].max():+.3f}]')
    print(f'  hip_rotation_r:  [{series_data["hip_rotation_r"].min():+.3f}, {series_data["hip_rotation_r"].max():+.3f}]')
    
    # ========================================================================
    # Knee and Ankle - DIRECT MAPPING
    # Add 'q_' prefix for consistency
    # ========================================================================
    series_data['knee_angle_r'] = hdf5_data['knee_angle_r']
    series_data['knee_angle_l'] = hdf5_data['knee_angle_l']
    series_data['ankle_angle_r'] = hdf5_data['ankle_angle_r']
    series_data['ankle_angle_l'] = hdf5_data['ankle_angle_l']
    
    print(f'\nKnee/Ankle ranges (radians):')
    print(f'  knee_angle_r:  [{series_data["knee_angle_r"].min():+.3f}, {series_data["knee_angle_r"].max():+.3f}]')
    print(f'  ankle_angle_r: [{series_data["ankle_angle_r"].min():+.3f}, {series_data["ankle_angle_r"].max():+.3f}]')
    
    # ========================================================================
    # Compute velocities (dq) via numerical differentiation
    # ========================================================================
    dt = 0.01  # 100 Hz sampling rate
    
    # Calculate velocities for all position data
    for key in list(series_data.keys()):
        # Use central differences for better accuracy
        dq = np.gradient(series_data[key], dt)
        series_data[f'd{key}'] = dq  # Will become 'dq_pelvis_tx', etc. after prefix addition
    
    print(f'\n✅ Computed velocities for {len(series_data)//2} position channels')
    
    # ========================================================================
    # Assemble final array in MyoAssist order
    # ========================================================================
    q_ref = np.zeros((n_frames, len(MYOASSIST_JOINTS)))
    
    for i, joint_name in enumerate(MYOASSIST_JOINTS):
        # Remove 'q_' prefix to match series_data keys
        key = joint_name[2:] if joint_name.startswith('q_') else joint_name
        if key in series_data:
            q_ref[:, i] = series_data[key]
        else:
            print(f'WARNING: {joint_name} not found in converted data!')
    
    return q_ref, series_data

def save_myoassist_format(q_ref, series_data, config):
    """Save in MyoAssist NPZ format compatible with training pipeline"""
    
    output_path = os.path.join(config['output_dir'], f"{config['output_name']}.npz")
    
    # Create output directory if needed
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Create metadata (required by training pipeline)
    metadata = {
        'sample_rate': 100.0,
        'duration': q_ref.shape[0] / 100.0,
        'num_frames': q_ref.shape[0],
        'num_dof': q_ref.shape[1],
        'source': 'HDF5_direct_v7',
        'subject': config['subject'],
        'motion_type': f"level_{config['speed']}",
        'trial': config['trial']
    }
    
    # Environment expects series_data keys with proper prefixes:
    # - Position: 'q_pelvis_tx', 'q_hip_flexion_r', etc.
    # - Velocity: 'dq_pelvis_tx', 'dq_hip_flexion_r', etc.
    series_data_with_prefix = {}
    for key, value in series_data.items():
        if key.startswith('d'):  # velocity (dpelvis_tx -> dq_pelvis_tx)
            new_key = f'd{key[1:]}'  # Strip 'd', will add back below
            series_data_with_prefix[f'dq_{key[1:]}'] = value
        else:  # position (pelvis_tx -> q_pelvis_tx)
            series_data_with_prefix[f'q_{key}'] = value
    
    # Save with complete structure for training compatibility
    np.savez(
        output_path,
        q_ref=q_ref,
        series_data=series_data_with_prefix,
        metadata=metadata,
        joint_names=MYOASSIST_JOINTS,
        num_dof=q_ref.shape[1],
        sampling_rate=100.0,
        duration=q_ref.shape[0] / 100.0
    )
    
    print(f'\nSaved: {output_path}')
    print(f'  Shape: {q_ref.shape}')
    print(f'  DOF: {q_ref.shape[1]}')
    print(f'  Frames: {q_ref.shape[0]}')
    print(f'  Duration: {q_ref.shape[0]/100.0:.2f} sec')
    print(f'  Metadata: {metadata}')

def main():
    print('='*80)
    print('HDF5 ??MyoAssist Direct Converter')
    print('='*80)
    
    # Load
    hdf5_data = load_hdf5_data(
        CONFIG['input_hdf5'],
        CONFIG['subject'],
        CONFIG['speed'],
        CONFIG['trial']
    )
    
    # Convert
    print('\nConverting to MyoAssist format...')
    q_ref, series_data = convert_to_myoassist(hdf5_data, CONFIG)
    
    # Save
    save_myoassist_format(q_ref, series_data, CONFIG)
    
    print('\n' + '='*80)
    print('DONE')
    print('='*80)

if __name__ == '__main__':
    main()
