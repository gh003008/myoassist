"""
Universal OpenSim to MyoAssist Motion Data Converter
====================================================

A flexible converter that works for both 2D and 3D models.
Configure settings at the top of the file for different trials/models.

Author: 
Date: 2025-11-14
"""

import numpy as np
import argparse
from pathlib import Path

# ============================================================================
# USER CONFIGURATION - EDIT THESE SETTINGS
# ============================================================================

CONFIG = {
    # Input/Output paths
    'input_file': r"C:\workspace_home\opensim data\LD_gdp\S004\level_08mps\trial_01.npz",
    'output_file_2d': r"rl_train\reference_data\S004_trial01_08mps_2D_ver9_0.npz",
    'output_file_3d': r"rl_train\reference_data\S004_trial01_08mps_3D_ver9_0.npz",
    
    # Model type: '2D', '3D', or 'both'
    'model_type': '3D',  # ver9: CORRECT HIP! (1,2,5 not 3,4,5)
    
    # Joint selections for 2D model (sagittal plane only)
    'joints_2d': {
        'pelvis': ['tx', 'ty', 'tilt'],  # Translation X, Y and pitch rotation
        'hip': ['flexion'],               # Only flexion (sagittal)
        'knee': ['angle'],                # Knee flexion
        'ankle': ['angle'],               # Ankle dorsiflexion
    },
    
    # Joint selections for 3D model (full kinematics)
    'joints_3d': {
        'pelvis': ['tx', 'ty', 'tz', 'list', 'tilt', 'rotation'],  # 6 DOF
        'hip': ['flexion', 'adduction', 'rotation'],                # 3 DOF per side
        'knee': ['angle'],                                          # 1 DOF per side
        'ankle': ['angle'],                                         # 1 DOF per side
    },
    
    # OpenSim column mapping (adjust based on your OpenSim model)
    'opensim_mapping': {
        # Pelvis translations
        'pelvis_tx': 'pelvis_tx',
        'pelvis_ty': 'pelvis_ty',
        'pelvis_tz': 'pelvis_tz',
        
        # Pelvis rotations (X=list, Y=tilt, Z=rotation)
        'pelvis_list': 'pelvis_0',
        'pelvis_tilt': 'pelvis_1',
        'pelvis_rotation': 'pelvis_2',
        
        # Hip joints: OpenSim 6-DOF but only 1,2,5 are active (0,3,4 are ~55deg fixed)
        # Analysis shows: hip_r_1 = flexion, hip_r_2 = adduction, hip_r_5 = rotation
        'hip_flexion_r': 'hip_r_1',      # -25~16 deg ✓
        'hip_adduction_r': 'hip_r_2',    # -8~4 deg ✓
        'hip_rotation_r': 'hip_r_5',     # -8~12 deg ✓ (NOT hip_r_2!)
        'hip_flexion_l': 'hip_l_1',
        'hip_adduction_l': 'hip_l_2',
        'hip_rotation_l': 'hip_l_5',
        
        # Knee and ankle
        'knee_angle_r': 'knee_angle_r',
        'knee_angle_l': 'knee_angle_l',
        'ankle_angle_r': 'ankle_angle_r',
        'ankle_angle_l': 'ankle_angle_l',
    },
    
    # Velocity columns (if available in OpenSim data)
    'velocity_columns': {
        'pelvis_tx': None,  # Will compute from position
        'pelvis_ty': None,
        'pelvis_tz': None,
        'pelvis_list': None,
        'pelvis_tilt': None,
        'pelvis_rotation': None,
        'hip_flexion_r': 'hip_r_y_angular_vel',
        'hip_flexion_l': 'hip_l_y_angular_vel',
        'hip_adduction_r': 'hip_r_x_angular_vel',
        'hip_adduction_l': 'hip_l_x_angular_vel',
        'hip_rotation_r': 'hip_r_z_angular_vel',
        'hip_rotation_l': 'hip_l_z_angular_vel',
        'knee_angle_r': 'knee_angle_r_vel',
        'knee_angle_l': 'knee_angle_l_vel',
        'ankle_angle_r': 'ankle_angle_r_vel',
        'ankle_angle_l': 'ankle_angle_l_vel',
    },
    
    # Resampling (None = keep original rate)
    'target_sample_rate': None,
}

# ============================================================================
# CONVERTER CLASS
# ============================================================================

class UniversalMotionConverter:
    """Universal converter for OpenSim to MyoAssist format"""
    
    def __init__(self, config):
        self.config = config
        self.data = None
        self.model_states = None
        self.columns = None
        self.col_idx = {}
        self.sampling_rate = None
        
    def load_opensim_data(self, filepath):
        """Load OpenSim NPZ file"""
        print(f"\n{'='*80}")
        print(f"📂 Loading OpenSim data: {filepath}")
        print(f"{'='*80}")
        
        self.data = np.load(filepath, allow_pickle=True)
        self.model_states = self.data['model_states']
        self.columns = self.data['model_states_columns']
        self.sampling_rate = int(self.data['sampling_rate'])
        self.col_idx = {col: i for i, col in enumerate(self.columns)}
        
        print(f"✅ Loaded {self.model_states.shape[0]} frames at {self.sampling_rate} Hz")
        print(f"   Available columns: {len(self.columns)}")
        
    def _get_column_data(self, col_name):
        """Get data for a specific column"""
        if col_name in self.col_idx:
            return self.model_states[:, self.col_idx[col_name]]
        else:
            print(f"⚠️  Column '{col_name}' not found, using zeros")
            return np.zeros(self.model_states.shape[0])
    
    def _compute_velocity(self, position_data, vel_col_name=None):
        """Compute velocity from position or use provided column"""
        if vel_col_name and vel_col_name in self.col_idx:
            return self._get_column_data(vel_col_name)
        else:
            dt = 1.0 / self.sampling_rate
            return np.gradient(position_data, dt)
    
    def convert_2d(self):
        """Convert to 2D format"""
        print(f"\n🔄 Converting to 2D format...")
        
        series_data = {}
        joints_config = self.config['joints_2d']
        mapping = self.config['opensim_mapping']
        vel_cols = self.config['velocity_columns']
        
        # Extract positions
        for joint_type, components in joints_config.items():
            for component in components:
                if joint_type == 'pelvis':
                    key = f'pelvis_{component}'
                    opensim_col = mapping.get(key)
                    if opensim_col:
                        pos_data = self._get_column_data(opensim_col)
                        series_data[f'q_{key}'] = pos_data
                        vel_data = self._compute_velocity(pos_data, vel_cols.get(key))
                        series_data[f'dq_{key}'] = vel_data
                        
                elif joint_type in ['hip', 'knee', 'ankle']:
                    for side in ['r', 'l']:
                        key = f'{joint_type}_{component}_{side}'
                        opensim_col = mapping.get(key)
                        if opensim_col:
                            pos_data = self._get_column_data(opensim_col)
                            series_data[f'q_{key}'] = pos_data
                            vel_data = self._compute_velocity(pos_data, vel_cols.get(key))
                            series_data[f'dq_{key}'] = vel_data
        
        metadata = self._create_metadata('2D', len(series_data) // 2)
        return metadata, series_data
    
    def convert_3d(self):
        """Convert to 3D format with OpenSim -> MuJoCo coordinate transform"""
        print(f"\n🔄 Converting to 3D format (with coordinate transform)...")
        
        series_data = {}
        joints_config = self.config['joints_3d']
        mapping = self.config['opensim_mapping']
        vel_cols = self.config['velocity_columns']
        
        # Coordinate system transformation flags
        # OpenSim: Y-up, Z-forward, X-right
        # MuJoCo: Z-up, X-forward, Y-left
        apply_transform = True
        
        # Extract positions
        for joint_type, components in joints_config.items():
            for component in components:
                if joint_type == 'pelvis':
                    key = f'pelvis_{component}'
                    opensim_col = mapping.get(key)
                    if opensim_col:
                        pos_data = self._get_column_data(opensim_col)
                        
                        # Apply coordinate transformation
                        if apply_transform:
                            if component == 'tx':  # OpenSim X → MuJoCo X (keep)
                                pos_data = pos_data
                            elif component == 'ty':  # OpenSim Y (up) → MuJoCo Z (up)
                                pos_data = pos_data  # Will swap with tz below
                            elif component == 'tz':  # OpenSim Z (forward) → MuJoCo X (forward)
                                pos_data = pos_data  # Will swap below
                        
                        series_data[f'q_{key}'] = pos_data
                        vel_data = self._compute_velocity(pos_data, vel_cols.get(key))
                        series_data[f'dq_{key}'] = vel_data
                        
                elif joint_type in ['hip', 'knee', 'ankle']:
                    for side in ['r', 'l']:
                        key = f'{joint_type}_{component}_{side}'
                        opensim_col = mapping.get(key)
                        if opensim_col:
                            pos_data = self._get_column_data(opensim_col)
                        series_data[f'q_{key}'] = pos_data
                        vel_data = self._compute_velocity(pos_data, vel_cols.get(key))
                        series_data[f'dq_{key}'] = vel_data
        
        # Swap pelvis coordinates: OpenSim (X,Y,Z) -> MuJoCo (Z,Y,X)
        # OpenSim: tx=right, ty=up, tz=forward
        # MuJoCo: tx=forward, ty=left, tz=up
        if apply_transform and 'q_pelvis_tx' in series_data and 'q_pelvis_tz' in series_data:
            print("  🔄 Applying coordinate transform: OpenSim → MuJoCo")
            print("     OpenSim (X=right, Y=up, Z=forward) → MuJoCo (X=forward, Y=left, Z=up)")
            
            # Save original translations
            opensim_tx = series_data['q_pelvis_tx'].copy()  # right
            opensim_ty = series_data['q_pelvis_ty'].copy()  # up
            opensim_tz = series_data['q_pelvis_tz'].copy()  # forward
            
            # Transform translations: MuJoCo tx=forward(OpenSim tz), ty=-right(OpenSim -tx), tz=up(OpenSim ty)
            series_data['q_pelvis_tx'] = opensim_tz           # forward
            series_data['q_pelvis_ty'] = -opensim_tx          # left (negate right)
            series_data['q_pelvis_tz'] = opensim_ty           # up
            
            # Add height offset: Tested pelvis_tz=forward, pelvis_tx=rightward
            # Therefore pelvis_ty must be height (up)!
            body_height = self.data.get('height_m', 1.75)
            pelvis_height_offset = body_height * 0.55  # Typical pelvis height ratio
            series_data['q_pelvis_ty'] = series_data['q_pelvis_ty'] + pelvis_height_offset
            
            print(f"     ✅ Transformed pelvis translations (height offset on TY: +{pelvis_height_offset:.3f}m)")
            
            # Recompute velocities after transform
            series_data['dq_pelvis_tx'] = self._compute_velocity(series_data['q_pelvis_tx'], None)
            series_data['dq_pelvis_ty'] = self._compute_velocity(series_data['q_pelvis_ty'], None)
            series_data['dq_pelvis_tz'] = self._compute_velocity(series_data['q_pelvis_tz'], None)
            
            # Transform rotations (Euler angles)
            # OpenSim: list=X(side-to-side), tilt=Y(pitch), rotation=Z(yaw)
            # MuJoCo: list=X(roll), tilt=Y(pitch), rotation=Z(yaw)
            # Need to swap axes: OpenSim XYZ → MuJoCo ZYX (same as translation)
            if 'q_pelvis_list' in series_data:
                opensim_list = series_data['q_pelvis_list'].copy()      # X rotation (side bend)
                opensim_tilt = series_data['q_pelvis_tilt'].copy()      # Y rotation (forward/back tilt)
                opensim_rotation = series_data['q_pelvis_rotation'].copy()  # Z rotation (twist)
                
                # Transform: Swap rotation axes to match new coordinate system
                # OpenSim tilt (Y-axis rotation around up) → MuJoCo rotation (Z-axis rotation around up)
                # OpenSim rotation (Z-axis rotation around forward) → MuJoCo tilt (Y-axis rotation around left)
                # OpenSim list (X-axis rotation around right) → MuJoCo list (X-axis rotation around forward)
                
                # Adjust tilt offset: 80deg → 75deg (5deg less backward lean)
                import numpy as np
                tilt_offset = np.pi / 2 - 0.262  # 90deg - 15deg = 75deg
                
                series_data['q_pelvis_list'] = opensim_rotation      # Z→X (twist → roll)
                series_data['q_pelvis_tilt'] = -opensim_list + tilt_offset  # X→Y (side bend → pitch) + offset
                series_data['q_pelvis_rotation'] = opensim_tilt      # Y→Z (tilt → yaw)
                
                # Recompute velocities
                series_data['dq_pelvis_list'] = self._compute_velocity(series_data['q_pelvis_list'], None)
                series_data['dq_pelvis_tilt'] = self._compute_velocity(series_data['q_pelvis_tilt'], None)
                series_data['dq_pelvis_rotation'] = self._compute_velocity(series_data['q_pelvis_rotation'], None)
                
                print(f"     ✅ Transformed pelvis rotations (tilt offset: {np.degrees(tilt_offset):.1f}deg)")
        
        metadata = self._create_metadata('3D', len(series_data) // 2)
        return metadata, series_data
    
    def _create_metadata(self, model_type, dof):
        """Create metadata dictionary"""
        target_rate = self.config['target_sample_rate'] or self.sampling_rate
        
        metadata = {
            'sample_rate': target_rate,
            'original_sample_rate': self.sampling_rate,
            'data_length': self.model_states.shape[0],
            'model_type': model_type,
            'dof': dof,
            'height_m': float(self.data['height_m']) if 'height_m' in self.data else None,
            'weight_kg': float(self.data['weight_kg']) if 'weight_kg' in self.data else None,
        }
        return metadata
    
    def save(self, output_path, metadata, series_data):
        """Save converted data to NPZ file"""
        print(f"\n💾 Saving to: {output_path}")
        
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.savez(output_path, metadata=metadata, series_data=series_data)
        
        print(f"✅ Saved successfully!")
        print(f"\n📋 Summary:")
        print(f"   Model type: {metadata['model_type']}")
        print(f"   DOF: {metadata['dof']}")
        print(f"   Data length: {metadata['data_length']} frames")
        print(f"   Duration: {metadata['data_length'] / metadata['sample_rate']:.2f} seconds")
        print(f"   Signals: {len(series_data)}")
        
    def verify(self, output_path):
        """Verify the converted file"""
        print(f"\n🔍 Verifying: {output_path}")
        try:
            data = np.load(output_path, allow_pickle=True)
            metadata = data['metadata'].item()
            series_data = data['series_data'].item()
            print(f"✅ Verification successful!")
            print(f"   Keys: {list(series_data.keys())[:5]}... ({len(series_data)} total)")
        except Exception as e:
            print(f"❌ Verification failed: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Convert OpenSim motion data to MyoAssist format')
    parser.add_argument('--input', type=str, help='Input OpenSim NPZ file (overrides config)')
    parser.add_argument('--output_2d', type=str, help='Output 2D NPZ file (overrides config)')
    parser.add_argument('--output_3d', type=str, help='Output 3D NPZ file (overrides config)')
    parser.add_argument('--model_type', type=str, choices=['2D', '3D', 'both'], 
                        help='Model type to convert (overrides config)')
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    if args.input:
        CONFIG['input_file'] = args.input
    if args.output_2d:
        CONFIG['output_file_2d'] = args.output_2d
    if args.output_3d:
        CONFIG['output_file_3d'] = args.output_3d
    if args.model_type:
        CONFIG['model_type'] = args.model_type
    
    # Create converter
    converter = UniversalMotionConverter(CONFIG)
    converter.load_opensim_data(CONFIG['input_file'])
    
    # Convert based on model type
    if CONFIG['model_type'].lower() in ['2d', 'both']:
        metadata_2d, series_2d = converter.convert_2d()
        converter.save(CONFIG['output_file_2d'], metadata_2d, series_2d)
        converter.verify(CONFIG['output_file_2d'])
    
    if CONFIG['model_type'].lower() in ['3d', 'both']:
        metadata_3d, series_3d = converter.convert_3d()
        converter.save(CONFIG['output_file_3d'], metadata_3d, series_3d)
        converter.verify(CONFIG['output_file_3d'])
    
    print(f"\n{'='*80}")
    print(f"✅ Conversion complete!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
