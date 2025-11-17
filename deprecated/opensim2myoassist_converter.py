"""
OpenSim Motion Data to MyoAssist Reference Format Converter
============================================================

Converts OpenSim NPZ motion data to MyoAssist IL-compatible reference trajectory format.

Usage:
    python opensim2myoassist_converter.py <input_npz> <output_npz> [--sample_rate RATE]

Example:
    python opensim2myoassist_converter.py "C:/workspace_home/opensim data/LD_gdp/S004/level_08mps/trial_01.npz" "rl_train/reference_data/S004_trial01_08mps.npz"
"""

import numpy as np
import sys
import argparse
from pathlib import Path


class OpenSimToMyoAssistConverter:
    """Converts OpenSim motion data to MyoAssist reference format"""
    
    # Joint name mapping: OpenSim column -> MyoAssist joint name
    JOINT_MAPPING = {
        # Position (q_*)
        'pelvis_tx': 'pelvis_tx',
        'pelvis_ty': 'pelvis_ty',
        'pelvis_tz': 'pelvis_tz',
        'knee_angle_r': 'knee_angle_r',
        'knee_angle_l': 'knee_angle_l',
        'ankle_angle_r': 'ankle_angle_r',
        'ankle_angle_l': 'ankle_angle_l',
        'subtalar_angle_r': 'subtalar_angle_r',
        'subtalar_angle_l': 'subtalar_angle_l',
        # Add pelvis orientation if available
        'pelvis_tilt': 'pelvis_tilt',  # May need to compute from rotation
    }
    
    # Velocity mapping: OpenSim column -> MyoAssist joint velocity
    VELOCITY_MAPPING = {
        'knee_angle_r_vel': 'knee_angle_r',
        'knee_angle_l_vel': 'knee_angle_l',
        'ankle_angle_r_vel': 'ankle_angle_r',
        'ankle_angle_l_vel': 'ankle_angle_l',
        'subtalar_angle_r_vel': 'subtalar_angle_r',
        'subtalar_angle_l_vel': 'subtalar_angle_l',
    }
    
    def __init__(self, opensim_file_path):
        """Initialize converter with OpenSim data file"""
        print(f"📂 Loading OpenSim data: {opensim_file_path}")
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
    
    def _extract_hip_flexion(self):
        """
        Extract hip flexion from pelvis and hip rotation data.
        OpenSim uses 6DOF joints (3 rotations + 3 translations).
        Hip flexion is typically the Y-axis rotation.
        """
        # Try to find hip flexion from the rotation data
        # Hip flexion is often hip_r_1 or hip_l_1 (Y-axis rotation)
        hip_r_flexion = self._get_column_data('hip_r_1')  # Y rotation
        hip_l_flexion = self._get_column_data('hip_l_1')  # Y rotation
        
        return hip_r_flexion, hip_l_flexion
    
    def _extract_pelvis_tilt(self):
        """Extract pelvis tilt (pitch rotation around Y-axis)"""
        # Pelvis tilt is typically pelvis_1 (Y-axis rotation)
        return self._get_column_data('pelvis_1')
    
    def _compute_pelvis_velocity(self):
        """Compute pelvis velocities from position if not available"""
        dt = 1.0 / self.sampling_rate
        
        # Get or compute pelvis_tx velocity
        if 'pelvis_tx_vel' in self.col_idx:
            pelvis_tx_vel = self._get_column_data('pelvis_tx_vel')
        else:
            pelvis_tx = self._get_column_data('pelvis_tx')
            pelvis_tx_vel = np.gradient(pelvis_tx, dt)
        
        # Get or compute pelvis_ty velocity
        if 'pelvis_ty_vel' in self.col_idx:
            pelvis_ty_vel = self._get_column_data('pelvis_ty_vel')
        else:
            pelvis_ty = self._get_column_data('pelvis_ty')
            pelvis_ty_vel = np.gradient(pelvis_ty, dt)
        
        # Get or compute pelvis_tz velocity
        if 'pelvis_tz_vel' in self.col_idx:
            pelvis_tz_vel = self._get_column_data('pelvis_tz_vel')
        else:
            pelvis_tz = self._get_column_data('pelvis_tz')
            pelvis_tz_vel = np.gradient(pelvis_tz, dt)
        
        # Pelvis tilt velocity
        pelvis_tilt = self._extract_pelvis_tilt()
        pelvis_tilt_vel = np.gradient(pelvis_tilt, dt)
        
        return pelvis_tx_vel, pelvis_ty_vel, pelvis_tz_vel, pelvis_tilt_vel
    
    def convert(self, output_path=None, target_sample_rate=None):
        """
        Convert OpenSim data to MyoAssist reference format
        
        Args:
            output_path: Path to save the converted NPZ file
            target_sample_rate: Target sampling rate (if None, use original)
        """
        print("\n🔄 Converting to MyoAssist format...")
        
        # Use original sampling rate if not specified
        if target_sample_rate is None:
            target_sample_rate = self.sampling_rate
        
        # Create series_data dictionary
        series_data = {}
        
        # Extract positions (q_*)
        print("\n📊 Extracting joint positions...")
        series_data['q_pelvis_tx'] = self._get_column_data('pelvis_tx')
        series_data['q_pelvis_ty'] = self._get_column_data('pelvis_ty')
        series_data['q_pelvis_tz'] = self._get_column_data('pelvis_tz')
        series_data['q_pelvis_tilt'] = self._extract_pelvis_tilt()
        
        # Extract hip flexion
        hip_r_flexion, hip_l_flexion = self._extract_hip_flexion()
        series_data['q_hip_flexion_r'] = hip_r_flexion
        series_data['q_hip_flexion_l'] = hip_l_flexion
        
        # Extract other joints
        series_data['q_knee_angle_r'] = self._get_column_data('knee_angle_r')
        series_data['q_knee_angle_l'] = self._get_column_data('knee_angle_l')
        series_data['q_ankle_angle_r'] = self._get_column_data('ankle_angle_r')
        series_data['q_ankle_angle_l'] = self._get_column_data('ankle_angle_l')
        
        # Extract velocities (dq_*)
        print("📊 Extracting joint velocities...")
        pelvis_tx_vel, pelvis_ty_vel, pelvis_tz_vel, pelvis_tilt_vel = self._compute_pelvis_velocity()
        series_data['dq_pelvis_tx'] = pelvis_tx_vel
        series_data['dq_pelvis_ty'] = pelvis_ty_vel
        series_data['dq_pelvis_tz'] = pelvis_tz_vel
        series_data['dq_pelvis_tilt'] = pelvis_tilt_vel
        
        # Extract hip velocities
        hip_r_flexion_vel = np.gradient(hip_r_flexion, 1.0 / self.sampling_rate)
        hip_l_flexion_vel = np.gradient(hip_l_flexion, 1.0 / self.sampling_rate)
        series_data['dq_hip_flexion_r'] = hip_r_flexion_vel
        series_data['dq_hip_flexion_l'] = hip_l_flexion_vel
        
        # Extract other joint velocities
        series_data['dq_knee_angle_r'] = self._get_column_data('knee_angle_r_vel')
        series_data['dq_knee_angle_l'] = self._get_column_data('knee_angle_l_vel')
        series_data['dq_ankle_angle_r'] = self._get_column_data('ankle_angle_r_vel')
        series_data['dq_ankle_angle_l'] = self._get_column_data('ankle_angle_l_vel')
        
        # Create metadata
        metadata = {
            'sample_rate': target_sample_rate,
            'original_sample_rate': self.sampling_rate,
            'data_length': self.model_states.shape[0],
            'source_file': str(self.data),
            'height_m': float(self.data['height_m']) if 'height_m' in self.data else None,
            'weight_kg': float(self.data['weight_kg']) if 'weight_kg' in self.data else None,
        }
        
        # Print summary
        print("\n📋 Conversion Summary:")
        print(f"   Data length: {metadata['data_length']} frames")
        print(f"   Sampling rate: {metadata['sample_rate']} Hz")
        print(f"   Duration: {metadata['data_length'] / metadata['sample_rate']:.2f} seconds")
        print(f"   Number of signals: {len(series_data)}")
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
            
            # Check for required keys for MyoAssist IL
            required_keys = [
                'q_pelvis_tx', 'q_pelvis_ty', 'q_pelvis_tilt',
                'q_hip_flexion_l', 'q_hip_flexion_r',
                'q_knee_angle_l', 'q_knee_angle_r',
                'q_ankle_angle_l', 'q_ankle_angle_r',
                'dq_pelvis_tx', 'dq_pelvis_ty', 'dq_pelvis_tilt',
            ]
            
            missing_keys = [key for key in required_keys if key not in series_data]
            if missing_keys:
                print(f"⚠️  Warning: Missing recommended keys: {missing_keys}")
            else:
                print(f"✅ All recommended keys present!")
                
        except Exception as e:
            print(f"❌ Verification failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert OpenSim motion data to MyoAssist reference format',
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
        help='Path to output MyoAssist NPZ file'
    )
    
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=None,
        help='Target sampling rate (Hz). If not specified, use original rate.'
    )
    
    args = parser.parse_args()
    
    # Convert
    converter = OpenSimToMyoAssistConverter(args.input_file)
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
        print('python opensim2myoassist_converter.py "C:/workspace_home/opensim data/LD_gdp/S004/level_08mps/trial_01.npz" "rl_train/reference_data/S004_trial01_08mps.npz"')
        sys.exit(0)
    
    main()
