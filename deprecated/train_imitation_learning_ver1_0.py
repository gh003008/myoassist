"""
ver1_0: MyoAssist Imitation Learning Training Script
====================================================

Features:
- WandB integration for logging
- 10% interval evaluation with rendering
- Compatible with original MyoAssist structure
- Based on original code with minimal modifications

Version: ver1_0
Date: 2025-11-14
"""

import subprocess
import sys
import json
import argparse
from pathlib import Path

# ============================================================================
# USER CONFIGURATION - EDIT THESE SETTINGS
# ============================================================================

CONFIG = {
    # Experiment name (will be used in WandB and output folder names)
    'experiment_name': 'S004_3D_IL_ver1_0',
    
    # Model selection: '2D' or '3D'
    'model_type': '3D',
    
    # Model paths
    'model_paths': {
        '2D': 'models/22muscle_2D/myoLeg22_2D_BASELINE.xml',
        '3D': 'models/26muscle_3D/myoLeg26_BASELINE.xml',
    },
    
    # Reference data path
    'reference_data_path': 'rl_train/reference_data/S004_trial01_08mps_3D.npz',
    
    # Joint keys to track (must match your reference data)
    'reference_joint_keys': {
        '2D': [
            "ankle_angle_l", "ankle_angle_r",
            "hip_flexion_l", "hip_flexion_r",
            "knee_angle_l", "knee_angle_r",
            "pelvis_tilt", "pelvis_tx", "pelvis_ty"
        ],
        '3D': [
            "ankle_angle_l", "ankle_angle_r",
            "hip_flexion_l", "hip_flexion_r",
            "hip_adduction_l", "hip_adduction_r",
            "hip_rotation_l", "hip_rotation_r",
            "knee_angle_l", "knee_angle_r",
            "pelvis_list", "pelvis_tilt", "pelvis_rotation",
            "pelvis_tx", "pelvis_ty", "pelvis_tz"
        ],
    },
    
    # Training parameters
    'training': {
        'total_timesteps': 3e7,
        'num_envs': 8,  # Will auto-adjust: 16 for 2D, 8 for 3D
        'target_velocity': 0.8,  # m/s
        'device': 'cpu',  # 'cpu' or 'cuda'
        'learning_rate': 0.0001,
    },
    
    # WandB settings
    'wandb': {
        'enabled': True,
        'project': 'myoassist-imitation',
        'entity': None,  # Your WandB username or team
        'tags': ['ver1_0', 'imitation', 'S004'],
    },
    
    # Reward weights (customize based on your task)
    'reward_weights': {
        '2D': {
            'qpos_imitation': {
                'pelvis_ty': 0.1,
                'pelvis_tilt': 1.0,
                'hip_flexion_l': 0.2,
                'hip_flexion_r': 0.2,
                'knee_angle_l': 1.0,
                'knee_angle_r': 1.0,
                'ankle_angle_l': 0.2,
                'ankle_angle_r': 0.2,
            },
            'qvel_imitation': {
                'pelvis_ty': 0.1,
                'pelvis_tilt': 0.1,
                'hip_flexion_l': 0.2,
                'hip_flexion_r': 0.2,
                'knee_angle_l': 0.1,
                'knee_angle_r': 0.1,
                'ankle_angle_l': 0.1,
                'ankle_angle_r': 0.1,
            },
        },
        '3D': {
            'qpos_imitation': {
                'pelvis_tx': 0.1, 'pelvis_ty': 0.1, 'pelvis_tz': 0.1,
                'pelvis_list': 0.5, 'pelvis_tilt': 1.0, 'pelvis_rotation': 0.5,
                'hip_flexion_l': 0.5, 'hip_flexion_r': 0.5,
                'hip_adduction_l': 0.3, 'hip_adduction_r': 0.3,
                'hip_rotation_l': 0.3, 'hip_rotation_r': 0.3,
                'knee_angle_l': 1.0, 'knee_angle_r': 1.0,
                'ankle_angle_l': 0.2, 'ankle_angle_r': 0.2,
            },
            'qvel_imitation': {
                'pelvis_tx': 0.1, 'pelvis_ty': 0.1, 'pelvis_tz': 0.1,
                'pelvis_list': 0.1, 'pelvis_tilt': 0.1, 'pelvis_rotation': 0.1,
                'hip_flexion_l': 0.2, 'hip_flexion_r': 0.2,
                'hip_adduction_l': 0.1, 'hip_adduction_r': 0.1,
                'hip_rotation_l': 0.1, 'hip_rotation_r': 0.1,
                'knee_angle_l': 0.1, 'knee_angle_r': 0.1,
                'ankle_angle_l': 0.1, 'ankle_angle_r': 0.1,
            },
        },
    },
    
    # Network architecture
    'network_arch': {
        '2D': {
            'human_actor': [64, 64],
            'exo_actor': [8, 8],
            'common_critic': [64, 64],
        },
        '3D': {
            'human_actor': [128, 128],
            'exo_actor': [16, 16],
            'common_critic': [128, 128],
        },
    },
}

# ============================================================================
# TRAINING SCRIPT
# ============================================================================

def create_config_file(model_type, quick_test=False):
    """Generate config JSON file based on settings"""
    
    # Auto-adjust num_envs for 3D
    num_envs = CONFIG['training']['num_envs']
    if model_type == '3D' and num_envs > 8:
        num_envs = 8
        print(f"⚙️  Auto-adjusted num_envs to {num_envs} for 3D model")
    
    # Calculate n_steps to maintain batch size
    n_steps = int(8192 / num_envs)
    
    config_dict = {
        "total_timesteps": 50000 if quick_test else CONFIG['training']['total_timesteps'],
        "evaluate_param_list": [{
            "num_timesteps": 200,
            "min_target_velocity": CONFIG['training']['target_velocity'],
            "max_target_velocity": CONFIG['training']['target_velocity'],
            "target_velocity_period": 2,
            "velocity_mode": "UNIFORM",
        }],
        "num_envs": num_envs,
        "n_steps": n_steps,
        "learning_rate": CONFIG['training']['learning_rate'],
        "device": CONFIG['training']['device'],
        
        "env_params": {
            "env_id": "myoAssistLegImitationExo-v0",
            "model_path": CONFIG['model_paths'][model_type],
            "reference_data_path": CONFIG['reference_data_path'],
            "reference_data_keys": CONFIG['reference_joint_keys'][model_type],
            "flag_random_ref_index": False,
            "out_of_trajectory_threshold": 0.3,
            "prev_trained_policy_path": None,
            "reward_keys_and_weights": {
                "qpos_imitation_rewards": CONFIG['reward_weights'][model_type]['qpos_imitation'],
                "qvel_imitation_rewards": CONFIG['reward_weights'][model_type]['qvel_imitation'],
            }
        },
        
        "policy_params": {
            "network_architecture": CONFIG['network_arch'][model_type]
        },
        
        "logger_params": {
            "logging_frequency": 10,
            "evaluate_frequency": 50
        },
        
        "auto_reward_adjust_params": {
            "learning_rate": 0.0
        }
    }
    
    # Save config
    config_dir = Path("rl_train/train/train_configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{CONFIG['experiment_name']}_{model_type}_ver1_0.json"
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    return str(config_path)


def main():
    parser = argparse.ArgumentParser(description='ver1_0: Train MyoAssist Imitation Learning')
    parser.add_argument('--model', type=str, default=CONFIG['model_type'],
                       choices=['2D', '3D'], help='Model type')
    parser.add_argument('--device', type=str, default=CONFIG['training']['device'],
                       choices=['cpu', 'cuda'], help='Training device')
    parser.add_argument('--render', action='store_true', help='Enable rendering during training')
    parser.add_argument('--quick_test', action='store_true', help='Quick test with 50k timesteps')
    parser.add_argument('--num_envs', type=int, default=None, help='Number of parallel environments')
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging')
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    model_type = args.model
    CONFIG['training']['device'] = args.device
    if args.num_envs:
        CONFIG['training']['num_envs'] = args.num_envs
    
    print("=" * 80)
    print("🚀 MyoAssist Imitation Learning Training (ver1_0)")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Experiment: {CONFIG['experiment_name']}")
    print(f"   Model type: {model_type}")
    print(f"   Reference data: {CONFIG['reference_data_path']}")
    print(f"   Quick test: {args.quick_test}")
    print(f"   Device: {CONFIG['training']['device']}")
    print(f"   WandB: {'Disabled' if args.no_wandb else 'Enabled'}")
    
    # Create config file
    config_path = create_config_file(model_type, args.quick_test)
    print(f"   Config: {config_path}")
    
    # Prepare command
    cmd = [
        sys.executable,
        "rl_train/run_train_ver1_0.py",
        "--config_file_path", config_path,
    ]
    
    if args.render:
        cmd.append("--flag_rendering")
    
    # Add WandB arguments
    if not args.no_wandb and CONFIG['wandb']['enabled']:
        cmd.extend([
            "--enable_wandb",
            "--wandb_project", CONFIG['wandb']['project'],
            "--wandb_name", CONFIG['experiment_name'],
        ])
    
    print(f"\n💻 Running: {' '.join(cmd)}")
    print("=" * 80)
    print()
    
    # Run training
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("✅ Training completed successfully!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print(f"❌ Training failed: {result.returncode}")
        print("=" * 80)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
