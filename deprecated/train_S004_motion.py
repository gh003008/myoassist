"""
Quick Training Script for MyoAssist Imitation Learning
=======================================================

This script trains the MyoAssist agent to imitate the S004 trial 01 motion data.

Usage:
    python train_S004_motion.py [--quick_test] [--num_envs N] [--device DEVICE]

Arguments:
    --quick_test    Run a quick test with minimal timesteps (for testing setup)
    --num_envs N    Number of parallel environments (default: 16)
    --device DEVICE Device to use: 'cpu' or 'cuda' (default: cpu)
    --render        Enable rendering during training
"""

import subprocess
import sys
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Train MyoAssist IL agent with S004 motion data'
    )
    
    parser.add_argument(
        '--quick_test',
        action='store_true',
        help='Run quick test with minimal timesteps'
    )
    
    parser.add_argument(
        '--num_envs',
        type=int,
        default=16,
        help='Number of parallel environments (default: 16)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for training (default: cpu)'
    )
    
    parser.add_argument(
        '--render',
        action='store_true',
        help='Enable rendering during training'
    )
    
    args = parser.parse_args()
    
    # Config file path
    config_file = "rl_train/train/train_configs/S004_trial01_08mps_config.json"
    
    # Modify config if needed
    if args.quick_test or args.num_envs != 16 or args.device != 'cpu':
        print(f"\n🔧 Modifying config:")
        print(f"   - num_envs: {args.num_envs}")
        print(f"   - device: {args.device}")
        if args.quick_test:
            print(f"   - total_timesteps: 50000 (quick test)")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        config['env_params']['num_envs'] = args.num_envs
        config['ppo_params']['device'] = args.device
        
        # Adjust n_steps based on num_envs to keep batch size similar
        config['ppo_params']['n_steps'] = int(8192 / args.num_envs)
        
        if args.quick_test:
            config['total_timesteps'] = 50000
            config['logger_params']['logging_frequency'] = 1
        
        # Save temporary config
        temp_config_file = "rl_train/train/train_configs/S004_trial01_08mps_config_temp.json"
        with open(temp_config_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        config_file = temp_config_file
    
    # Build command
    cmd = [
        "python",
        "rl_train/run_train.py",
        "--config_file_path",
        config_file
    ]
    
    if args.render:
        cmd.append("--flag_rendering")
    
    print("\n" + "="*80)
    print("🚀 Starting MyoAssist Imitation Learning Training")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"   Config file: {config_file}")
    print(f"   Number of environments: {args.num_envs}")
    print(f"   Device: {args.device}")
    print(f"   Rendering: {args.render}")
    print(f"   Quick test mode: {args.quick_test}")
    
    print(f"\n🎯 Reference motion: S004 trial 01 (0.8 m/s)")
    print(f"   Duration: 120.28 seconds")
    print(f"   Sampling rate: 100 Hz → resampled to 30 Hz")
    
    print("\n" + "="*80)
    print("💻 Running command:")
    print("   " + " ".join(cmd))
    print("="*80 + "\n")
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "="*80)
        print("✅ Training completed successfully!")
        print("="*80)
        print("\n📊 Check results in: rl_train/results/train_session_[timestamp]/")
        print("\n💡 To evaluate the trained policy, run:")
        print("   python rl_train/run_policy_eval.py rl_train/results/train_session_[timestamp]")
        
    except subprocess.CalledProcessError as e:
        print("\n" + "="*80)
        print(f"❌ Training failed with error code: {e.returncode}")
        print("="*80)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("⚠️  Training interrupted by user")
        print("="*80)
        sys.exit(0)


if __name__ == "__main__":
    main()
