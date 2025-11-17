"""
Verify S004 Motion Data and Environment Setup
==============================================

This script verifies that:
1. The converted reference data is loaded correctly
2. The environment can be created successfully
3. Basic simulation runs without errors

Usage:
    python verify_S004_setup.py
"""

import numpy as np
import sys
from pathlib import Path

def verify_reference_data():
    """Verify the converted reference data"""
    print("\n" + "="*80)
    print("📊 Verifying Reference Data")
    print("="*80)
    
    ref_data_path = "rl_train/reference_data/S004_trial01_08mps.npz"
    
    if not Path(ref_data_path).exists():
        print(f"❌ Reference data not found: {ref_data_path}")
        return False
    
    try:
        data = np.load(ref_data_path, allow_pickle=True)
        metadata = data['metadata'].item()
        series_data = data['series_data'].item()
        
        print(f"✅ Reference data loaded successfully!")
        print(f"\n📋 Metadata:")
        print(f"   - Sample rate: {metadata['sample_rate']} Hz")
        print(f"   - Data length: {metadata['data_length']} frames")
        print(f"   - Duration: {metadata['data_length'] / metadata['sample_rate']:.2f} seconds")
        print(f"   - Height: {metadata['height_m']:.2f} m")
        print(f"   - Weight: {metadata['weight_kg']:.2f} kg")
        
        print(f"\n📊 Series data:")
        print(f"   - Number of signals: {len(series_data)}")
        
        # Check required keys
        required_keys = [
            'q_pelvis_tx', 'q_pelvis_ty', 'q_pelvis_tilt',
            'q_hip_flexion_l', 'q_hip_flexion_r',
            'q_knee_angle_l', 'q_knee_angle_r',
            'q_ankle_angle_l', 'q_ankle_angle_r',
            'dq_pelvis_tx', 'dq_pelvis_ty',
        ]
        
        missing = [k for k in required_keys if k not in series_data]
        if missing:
            print(f"⚠️  Missing keys: {missing}")
        else:
            print(f"✅ All required keys present!")
        
        # Show data ranges
        print(f"\n📈 Data ranges:")
        for key in ['q_hip_flexion_r', 'q_knee_angle_r', 'q_ankle_angle_r']:
            if key in series_data:
                values = series_data[key]
                print(f"   {key:25s}: [{values.min():7.3f}, {values.max():7.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading reference data: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_environment():
    """Verify that the environment can be created"""
    print("\n" + "="*80)
    print("🌍 Verifying Environment Setup")
    print("="*80)
    
    try:
        from rl_train.utils.data_types import DictionableDataclass
        from rl_train.utils.environment_handler import EnvironmentHandler
        from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
        
        print("✅ Imports successful!")
        
        # Load config
        config_path = "rl_train/train/train_configs/S004_trial01_08mps_config.json"
        config = EnvironmentHandler.get_session_config_from_path(
            config_path, 
            ExoImitationTrainSessionConfig
        )
        
        print(f"✅ Config loaded successfully!")
        print(f"   - Environment ID: {config.env_params.env_id}")
        print(f"   - Model path: {config.env_params.model_path}")
        print(f"   - Reference data: {config.env_params.reference_data_path}")
        
        # Try to create environment with 1 env for testing
        config.env_params.num_envs = 1
        print(f"\n🔧 Creating test environment...")
        
        env = EnvironmentHandler.create_environment(config, is_rendering_on=False)
        
        print(f"✅ Environment created successfully!")
        print(f"   - Observation space: {env.observation_space.shape}")
        print(f"   - Action space: {env.action_space.shape}")
        
        # Try a reset
        print(f"\n🔄 Testing environment reset...")
        obs, info = env.reset()
        print(f"✅ Environment reset successful!")
        print(f"   - Observation shape: {obs.shape}")
        
        # Try a few steps
        print(f"\n👟 Testing environment steps...")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"✅ Environment steps successful!")
        print(f"   - Last reward: {reward:.4f}")
        print(f"   - Terminated: {terminated}")
        print(f"   - Truncated: {truncated}")
        
        env.close()
        print(f"\n✅ Environment closed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with environment: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_training_instructions():
    """Print instructions for training"""
    print("\n" + "="*80)
    print("🎓 Training Instructions")
    print("="*80)
    
    print("\n✅ Setup verified! You can now start training.")
    
    print("\n📚 Training options:")
    print("\n1️⃣  Quick test (to verify everything works):")
    print("   python train_S004_motion.py --quick_test")
    
    print("\n2️⃣  Full training with default settings:")
    print("   python train_S004_motion.py")
    
    print("\n3️⃣  Custom training (adjust parameters):")
    print("   python train_S004_motion.py --num_envs 8 --device cuda")
    
    print("\n4️⃣  Direct training command:")
    print("   python rl_train/run_train.py --config_file_path rl_train/train/train_configs/S004_trial01_08mps_config.json")
    
    print("\n💡 After training completes:")
    print("   python rl_train/run_policy_eval.py rl_train/results/train_session_[timestamp]")
    
    print("\n" + "="*80)


def main():
    print("\n🔍 MyoAssist S004 Motion Setup Verification")
    
    # Verify reference data
    ref_ok = verify_reference_data()
    
    if not ref_ok:
        print("\n❌ Reference data verification failed!")
        sys.exit(1)
    
    # Verify environment
    env_ok = verify_environment()
    
    if not env_ok:
        print("\n❌ Environment verification failed!")
        sys.exit(1)
    
    # Print instructions
    print_training_instructions()
    
    print("\n✅ All verifications passed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
