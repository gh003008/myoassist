"""
OpenSim 3D Motion Data to MyoAssist 3D Reference Format Converter
==================================================================

Converts OpenSim NPZ motion data (with full 3D kinematics) to MyoAssist 3D IL-compatible format.

Usage:
    python opensim2myoassist_3D_converter.py <input_npz> <output_npz> [--sample_rate RATE]

Example:
    python opensim2myoassist_3D_converter.py "C:/workspace_home/opensim data/LD_gdp/S004/level_08mps/trial_01.npz" "rl_train/reference_data/S004_trial01_08mps_3D.npz"
"""

import numpy as np
import sys
import argparse
from pathlib import Path


class OpenSimToMyoAssist3DConverter:
    """Converts OpenSim motion data to MyoAssist 3D reference format"""
    
    def __init__(self, opensim_file_path):
        """Initialize converter with OpenSim data file"""
        print(f"📂 Loading OpenSim 3D data: {opensim_file_path}")
        self.data = np.load(opensim_file_path, allow_pickle=True)
        self.model_states = self.data['model_states']
        self.columns = self.data['model_states_columns']
        self.sampling_rate = int(self.data['sampling_rate'])
        
        # Create column name to index mapping
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
    
    def _extract_hip_3dof(self):
        """
        Extract 3DOF hip rotations from OpenSim data.
        OpenSim uses 6DOF joints (3 rotations + 3 translations).
        For hip: hip_X_0~2 are rotations (flexion, adduction, rotation)
        """
        # Hip right: flexion, adduction, rotation
        hip_r_flexion = self._get_column_data('hip_r_1')    # Y rotation (flexion/extension)
        hip_r_adduction = self._get_column_data('hip_r_0')  # X rotation (adduction/abduction)
        hip_r_rotation = self._get_column_data('hip_r_2')   # Z rotation (internal/external)
        
        # Hip left: flexion, adduction, rotation
        hip_l_flexion = self._get_column_data('hip_l_1')
        hip_l_adduction = self._get_column_data('hip_l_0')
        hip_l_rotation = self._get_column_data('hip_l_2')
        
        return {
            'hip_flexion_r': hip_r_flexion,
            'hip_adduction_r': hip_r_adduction,
            'hip_rotation_r': hip_r_rotation,
            'hip_flexion_l': hip_l_flexion,
            'hip_adduction_l': hip_l_adduction,
            'hip_rotation_l': hip_l_rotation,
        }
    
    def _extract_pelvis_6dof(self):
        """Extract pelvis 6DOF (3 translations + 3 rotations)"""
        # Translations
        pelvis_tx = self._get_column_data('pelvis_tx')
        pelvis_ty = self._get_column_data('pelvis_ty')
        pelvis_tz = self._get_column_data('pelvis_tz')
        
        # Rotations: list, tilt, rotation
        # pelvis_0 = X rotation (list - lateral tilt)
        # pelvis_1 = Y rotation (tilt - forward/backward)
        # pelvis_2 = Z rotation (rotation - axial)
        pelvis_list = self._get_column_data('pelvis_0')
        pelvis_tilt = self._get_column_data('pelvis_1')
        pelvis_rotation = self._get_column_data('pelvis_2')
        
        return {
            'pelvis_tx': pelvis_tx,
            'pelvis_ty': pelvis_ty,
            'pelvis_tz': pelvis_tz,
            'pelvis_list': pelvis_list,
            'pelvis_tilt': pelvis_tilt,
            'pelvis_rotation': pelvis_rotation,
        }
    
    def _compute_velocities_3d(self):
        """Compute velocities for 3D joints"""
        dt = 1.0 / self.sampling_rate
        velocities = {}
        
        # Pelvis velocities
        pelvis_data = self._extract_pelvis_6dof()
        for key, values in pelvis_data.items():
            velocities[f'd{key}'] = np.gradient(values, dt)
        
        # Hip velocities (can use angular velocity columns if available)
        hip_data = self._extract_hip_3dof()
        
        # Try to use provided angular velocities first
        hip_vel_mapping = {
            'hip_flexion_r': 'hip_r_y_angular_vel',
            'hip_adduction_r': 'hip_r_x_angular_vel',
            'hip_rotation_r': 'hip_r_z_angular_vel',
            'hip_flexion_l': 'hip_l_y_angular_vel',
            'hip_adduction_l': 'hip_l_x_angular_vel',
            'hip_rotation_l': 'hip_l_z_angular_vel',
        }
        
        for hip_key, vel_col in hip_vel_mapping.items():
            if vel_col in self.col_idx:
                velocities[f'd{hip_key}'] = self._get_column_data(vel_col)
            else:
                # Compute from position
                velocities[f'd{hip_key}'] = np.gradient(hip_data[hip_key], dt)
        
        # Knee and ankle velocities
        for joint in ['knee_angle_r', 'knee_angle_l', 'ankle_angle_r', 'ankle_angle_l']:
            vel_col = f'{joint}_vel'
            if vel_col in self.col_idx:
                velocities[f'd{joint}'] = self._get_column_data(vel_col)
            else:
                pos = self._get_column_data(joint)
                velocities[f'd{joint}'] = np.gradient(pos, dt)
        
        return velocities
    
    def convert(self, output_path=None, target_sample_rate=None):
        """
        Convert OpenSim 3D data to MyoAssist 3D reference format
        
        Args:
            output_path: Path to save the converted NPZ file
            target_sample_rate: Target sampling rate (if None, use original)
        """
        print("\n🔄 Converting to MyoAssist 3D format...")
        
        # Use original sampling rate if not specified
        if target_sample_rate is None:
            target_sample_rate = self.sampling_rate
        
        # Create series_data dictionary
        series_data = {}
        
        # Extract pelvis 6DOF
        print("\n📊 Extracting pelvis 6DOF...")
        pelvis_data = self._extract_pelvis_6dof()
        for key, values in pelvis_data.items():
            series_data[f'q_{key}'] = values
        
        # Extract hip 3DOF
        print("📊 Extracting hip 3DOF...")
        hip_data = self._extract_hip_3dof()
        for key, values in hip_data.items():
            series_data[f'q_{key}'] = values
        
        # Extract knee and ankle (1DOF each)
        print("📊 Extracting knee and ankle...")
        series_data['q_knee_angle_r'] = self._get_column_data('knee_angle_r')
        series_data['q_knee_angle_l'] = self._get_column_data('knee_angle_l')
        series_data['q_ankle_angle_r'] = self._get_column_data('ankle_angle_r')
        series_data['q_ankle_angle_l'] = self._get_column_data('ankle_angle_l')
        
        # Extract velocities
        print("📊 Computing velocities...")
        velocities = self._compute_velocities_3d()
        for key, values in velocities.items():
            series_data[f'q{key}'] = values  # dq_* format
        
        # Create metadata
        metadata = {
            'sample_rate': target_sample_rate,
            'original_sample_rate': self.sampling_rate,
            'data_length': self.model_states.shape[0],
            'model_type': '3D',
            'dof': {
                'pelvis': 6,  # 3 translations + 3 rotations
                'hip': 3,     # flexion, adduction, rotation
                'knee': 1,    # flexion
                'ankle': 1,   # dorsiflexion
            },
            'source_file': str(self.data),
            'height_m': float(self.data['height_m']) if 'height_m' in self.data else None,
            'weight_kg': float(self.data['weight_kg']) if 'weight_kg' in self.data else None,
        }
        
        # Print summary
        print("\n📋 Conversion Summary:")
        print(f"   Model type: 3D (full kinematics)")
        print(f"   Data length: {metadata['data_length']} frames")
        print(f"   Sampling rate: {metadata['sample_rate']} Hz")
        print(f"   Duration: {metadata['data_length'] / metadata['sample_rate']:.2f} seconds")
        print(f"   Number of signals: {len(series_data)}")
        print(f"\n   DOF breakdown:")
        print(f"     - Pelvis: 6 DOF (tx, ty, tz, list, tilt, rotation)")
        print(f"     - Hip (each): 3 DOF (flexion, adduction, rotation)")
        print(f"     - Knee (each): 1 DOF")
        print(f"     - Ankle (each): 1 DOF")
        print(f"     - Total: 16 DOF")
        
        print(f"\n   Available signals:")
        for key in sorted(series_data.keys()):
            print(f"     - {key}: shape={series_data[key].shape}")
        
        # Save if output path is provided
        if output_path:
            print(f"\n💾 Saving to: {output_path}")
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            np.savez(
                output_path,
                metadata=metadata,
                series_data=series_data
            )
            print(f"✅ Conversion complete! Saved to: {output_path}")
            
            # Verification
            self._verify_output(output_path)
        
        return metadata, series_data
    
    def _verify_output(self, output_path):
        """Verify the converted file can be loaded properly"""
        print("\n🔍 Verifying output file...")
        try:
            data = np.load(output_path, allow_pickle=True)
            metadata = data['metadata'].item()
            series_data = data['series_data'].item()
            
            print(f"✅ Verification successful!")
            print(f"   - Metadata keys: {list(metadata.keys())}")
            print(f"   - Series data keys: {len(series_data)} signals")
            print(f"   - Model type: {metadata['model_type']}")
            
            # Check for required keys for MyoAssist 3D IL
            required_keys_3d = [
                'q_pelvis_tx', 'q_pelvis_ty', 'q_pelvis_tz',
                'q_pelvis_list', 'q_pelvis_tilt', 'q_pelvis_rotation',
                'q_hip_flexion_l', 'q_hip_flexion_r',
                'q_hip_adduction_l', 'q_hip_adduction_r',
                'q_hip_rotation_l', 'q_hip_rotation_r',
                'q_knee_angle_l', 'q_knee_angle_r',
                'q_ankle_angle_l', 'q_ankle_angle_r',
            ]
            
            missing_keys = [key for key in required_keys_3d if key not in series_data]
            if missing_keys:
                print(f"⚠️  Warning: Missing recommended keys: {missing_keys}")
            else:
                print(f"✅ All 3D model keys present!")
                
        except Exception as e:
            print(f"❌ Verification failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert OpenSim 3D motion data to MyoAssist 3D reference format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to OpenSim NPZ file'
    )
    
    parser.add_argument(
        'output_file',
        type=str,
        help='Path to output MyoAssist 3D NPZ file'
    )
    
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=None,
        help='Target sampling rate (Hz). If not specified, use original rate.'
    )
    
    args = parser.parse_args()
    
    # Convert
    converter = OpenSimToMyoAssist3DConverter(args.input_file)
    converter.convert(
        output_path=args.output_file,
        target_sample_rate=args.sample_rate
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, show help
        print(__doc__)
        print("\n" + "="*80)
        print("Example usage:")
        print("="*80)
        print('python opensim2myoassist_3D_converter.py "C:/workspace_home/opensim data/LD_gdp/S004/level_08mps/trial_01.npz" "rl_train/reference_data/S004_trial01_08mps_3D.npz"')
        sys.exit(0)
    
    main()
