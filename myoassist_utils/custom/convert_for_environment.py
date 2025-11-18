#!/usr/bin/env python3
"""
HDF5 to Environment Format Converter
Converts HDF5 data directly to training environment format WITHOUT q_ prefix.

This is separate from convert_hdf5_direct.py which is for MuJoCo renderer visualization.

Pipeline separation:
- convert_hdf5_direct.py → NPZ with q_ prefix → render_hdf5_reference.py (MuJoCo renderer)
- convert_for_environment.py → NPZ without q_ prefix → Training environment (THIS FILE)
"""
import h5py
import numpy as np
import os

# ============================================================================
# CONFIG
# ============================================================================
CONFIG = {
    'input_hdf5': r'C:\workspace\opensim data\LD\S004.h5',
    'subject': 'S004',
    'speed': '08mps',
    'trial': 'trial_01',
    'output_dir': r'C:\workspace\myoassist\rl_train\reference_data',
    'output_name': 'S004_trial01_08mps_3D_ENV',  # Different name to distinguish from MuJoCo version
    
    # Offsets
    'height_offset_m': 0.0,   # NOT USED - using relative positions
    'tilt_offset_deg': 0.0,   # NO SWAP! Just offset if needed
}

# ============================================================================
# Environment joint order (NO q_ prefix!)
# ============================================================================
# CRITICAL: These are the exact keys expected by environment series_data
ENV_JOINTS = [
    'pelvis_tx',        # 0 - Right (X)
    'pelvis_ty',        # 1 - Up (Y)
    'pelvis_tz',        # 2 - Forward (Z)
    'pelvis_tilt',      # 3 - Pitch (forward/back tilt)
    'pelvis_list',      # 4 - Roll (side lean)
    'pelvis_rotation',  # 5 - Yaw (twist)
    'hip_flexion_r',    # 6
    'hip_adduction_r',  # 7
    'hip_rotation_r',   # 8
    'hip_flexion_l',    # 9
    'hip_adduction_l',  # 10
    'hip_rotation_l',   # 11
    'knee_angle_r',     # 12
    'knee_angle_l',     # 13
    'ankle_angle_r',    # 14
    'ankle_angle_l',    # 15
]

def load_hdf5_data(hdf5_path, subject, speed, trial):
    """Load motion data from HDF5 file
    
    IMPORTANT: HDF5 has mixed units!
    - Joint angles (hip, knee, ankle, pelvis rotations): DEGREES → convert to radians
    - Translations (pelvis_tx, pelvis_ty, pelvis_tz): METERS → keep as-is
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

def convert_to_environment_format(hdf5_data, config):
    """Convert HDF5 OpenSim data to Environment series_data format
    
    Key differences from MuJoCo renderer converter:
    - NO q_ prefix on joint names
    - NO dq_ prefix on velocities (just 'd')
    - Direct series_data format (no separate q_ref array)
    - Ground-relative pelvis_ty (environment_handler adds +0.91m offset)
    """
    
    n_frames = len(hdf5_data['time'])
    series_data = {}
    
    print(f'\nConverting to ENVIRONMENT format (NO q_ prefix)...')
    
    # ========================================================================
    # Pelvis Translations - CONVERT TO RELATIVE POSITIONS
    # ========================================================================
    # CRITICAL: Environment expects RELATIVE positions (centered around 0)
    # Ground-relative (environment_handler will add +0.91m to pelvis_ty)
    
    pelvis_tx_mean = np.mean(hdf5_data['pelvis_tx'])
    pelvis_ty_mean = np.mean(hdf5_data['pelvis_ty'])
    pelvis_tz_mean = np.mean(hdf5_data['pelvis_tz'])
    
    series_data['pelvis_tx'] = hdf5_data['pelvis_tx'] - pelvis_tx_mean  # Right (relative)
    series_data['pelvis_ty'] = hdf5_data['pelvis_ty'] - pelvis_ty_mean  # Up (relative, ground-based)
    series_data['pelvis_tz'] = hdf5_data['pelvis_tz'] - pelvis_tz_mean  # Forward (relative)
    
    print(f'\nPelvis translation (RELATIVE, ground-based):')
    print(f'  pelvis_tx:   [{series_data["pelvis_tx"].min():8.4f}, {series_data["pelvis_tx"].max():8.4f}] m')
    print(f'  pelvis_ty:   [{series_data["pelvis_ty"].min():8.4f}, {series_data["pelvis_ty"].max():8.4f}] m (environment_handler adds +0.91m)')
    print(f'  pelvis_tz:   [{series_data["pelvis_tz"].min():8.4f}, {series_data["pelvis_tz"].max():8.4f}] m')
    
    # ========================================================================
    # Pelvis Rotations - DIRECT MAPPING
    # ========================================================================
    tilt_offset_rad = np.radians(config.get('tilt_offset_deg', 0.0))
    
    series_data['pelvis_tilt'] = hdf5_data['pelvis_tilt'] + tilt_offset_rad
    series_data['pelvis_list'] = hdf5_data['pelvis_list']
    series_data['pelvis_rotation'] = hdf5_data['pelvis_rotation']
    
    print(f'\nPelvis rotation (radians):')
    print(f'  pelvis_tilt:     [{np.degrees(series_data["pelvis_tilt"].min()):+.1f}, {np.degrees(series_data["pelvis_tilt"].max()):+.1f}] deg')
    print(f'  pelvis_list:     [{np.degrees(series_data["pelvis_list"].min()):+.1f}, {np.degrees(series_data["pelvis_list"].max()):+.1f}] deg')
    print(f'  pelvis_rotation: [{np.degrees(series_data["pelvis_rotation"].min()):+.1f}, {np.degrees(series_data["pelvis_rotation"].max()):+.1f}] deg')
    
    # ========================================================================
    # Hip, Knee, Ankle - DIRECT MAPPING (NO q_ prefix!)
    # ========================================================================
    series_data['hip_flexion_r'] = hdf5_data['hip_flexion_r']
    series_data['hip_adduction_r'] = hdf5_data['hip_adduction_r']
    series_data['hip_rotation_r'] = hdf5_data['hip_rotation_r']
    
    series_data['hip_flexion_l'] = hdf5_data['hip_flexion_l']
    series_data['hip_adduction_l'] = hdf5_data['hip_adduction_l']
    series_data['hip_rotation_l'] = hdf5_data['hip_rotation_l']
    
    series_data['knee_angle_r'] = hdf5_data['knee_angle_r']
    series_data['knee_angle_l'] = hdf5_data['knee_angle_l']
    series_data['ankle_angle_r'] = hdf5_data['ankle_angle_r']
    series_data['ankle_angle_l'] = hdf5_data['ankle_angle_l']
    
    print(f'\nJoint ranges (radians):')
    print(f'  hip_flexion_r:   [{series_data["hip_flexion_r"].min():+.3f}, {series_data["hip_flexion_r"].max():+.3f}]')
    print(f'  knee_angle_r:    [{series_data["knee_angle_r"].min():+.3f}, {series_data["knee_angle_r"].max():+.3f}]')
    print(f'  ankle_angle_r:   [{series_data["ankle_angle_r"].min():+.3f}, {series_data["ankle_angle_r"].max():+.3f}]')
    
    # ========================================================================
    # Compute velocities (d prefix, NOT dq!)
    # ========================================================================
    dt = 0.01  # 100 Hz sampling rate
    
    for key in list(series_data.keys()):
        dq = np.gradient(series_data[key], dt)
        series_data[f'd{key}'] = dq  # 'dpelvis_tx', 'dhip_flexion_r', etc.
    
    print(f'\n✅ Computed velocities (d prefix) for {len(series_data)//2} position channels')
    
    return series_data

def save_environment_format(series_data, config):
    """Save in Environment NPZ format (series_data only, NO q_ref/joint_names)"""
    
    output_path = os.path.join(config['output_dir'], f"{config['output_name']}.npz")
    
    # Create output directory if needed
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Get frame count from any series_data key
    n_frames = len(series_data['pelvis_tx'])
    
    # Create metadata
    metadata = {
        'sample_rate': 100.0,
        'data_length': n_frames,
        'duration': n_frames / 100.0,
        'dof': len(ENV_JOINTS),
        'model_type': '3D',
        'source': 'HDF5_environment_v1',
        'subject': config['subject'],
        'motion_type': f"level_{config['speed']}",
        'trial': config['trial'],
        'format': 'environment',  # Mark as environment format
    }
    
    # Save in environment format (series_data + metadata)
    np.savez(
        output_path,
        series_data=series_data,
        metadata=metadata,
    )
    
    print(f'\n{"="*80}')
    print(f'✅ Saved: {output_path}')
    print(f'{"="*80}')
    print(f'Format: ENVIRONMENT (series_data)')
    print(f'  Frames: {n_frames}')
    print(f'  DOF: {len(ENV_JOINTS)}')
    print(f'  Duration: {n_frames/100.0:.2f} sec')
    print(f'  Sample rate: 100 Hz')
    print(f'\nSeries data keys (NO q_ prefix):')
    position_keys = [k for k in series_data.keys() if not k.startswith('d')]
    for i, key in enumerate(position_keys[:5]):
        print(f'  {i}: {key}')
    print(f'  ... ({len(position_keys)} position keys total)')
    print(f'\nVelocity keys (d prefix, NOT dq):')
    velocity_keys = [k for k in series_data.keys() if k.startswith('d')]
    for i, key in enumerate(velocity_keys[:5]):
        print(f'  {i}: {key}')
    print(f'  ... ({len(velocity_keys)} velocity keys total)')

def main():
    print('='*80)
    print('HDF5 → Environment Format Converter')
    print('Creates NPZ compatible with training environment (NO q_ prefix)')
    print('='*80)
    
    # Load
    hdf5_data = load_hdf5_data(
        CONFIG['input_hdf5'],
        CONFIG['subject'],
        CONFIG['speed'],
        CONFIG['trial']
    )
    
    # Convert
    series_data = convert_to_environment_format(hdf5_data, CONFIG)
    
    # Save
    save_environment_format(series_data, CONFIG)
    
    print('\n' + '='*80)
    print('✅ DONE! Use this NPZ file with training environment.')
    print('='*80)

if __name__ == '__main__':
    main()
