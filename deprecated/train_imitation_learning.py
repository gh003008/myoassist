"""
Universal MyoAssist Imitation Learning Training Script
======================================================

Flexible training script for any motion data and model configuration.
Edit the CONFIG section at the top for different experiments.

Author: GitHub Copilot
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
    # Experiment name (will be used in output folder names)
    'experiment_name': 'S004_3D_IL',
    
    # Model selection: '2D' or '3D'
    'model_type': '3D',
    
    # Model paths
    'model_paths': {
        '2D': 'models/22muscle_2D/myoLeg22_2D_BASELINE.xml',
        '3D': 'models/26muscle_3D/myoLeg26_BASELINE.xml',
    },
    
    # Reference data path
    'reference_data_path': 'rl_train/reference_data/S004_trial01_08mps_3D.npz',
    
    # WandB settings (set None to disable)
    'wandb': {
        'enabled': True,
        'project': 'myoassist-imitation',
        'entity': None,  # Your WandB username/team (None = default)
        'log_freq': 100,  # Log every N steps
    },
    
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
        'num_envs': 16,  # Will auto-adjust: 16 for 2D, 8 for 3D
        'target_velocity': 0.8,  # m/s
        'device': 'cpu',  # 'cpu' or 'cuda'
        'learning_rate': 0.0001,
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
            "cam_type": "follow",
            "cam_distance": 2.5,
            "visualize_activation": True
        }],
        "logger_params": {
            "logging_frequency": 1 if quick_test else 8
        },
        "env_params": {
            "env_id": "myoAssistLegImitationExo-v0",
            "num_envs": num_envs,
            "seed": 1234,
            "safe_height": 0.7,
            "out_of_trajectory_threshold": 0.2,
            "flag_random_ref_index": True,
            "control_framerate": 30,
            "physics_sim_framerate": 1200,
            "min_target_velocity": CONFIG['training']['target_velocity'],
            "max_target_velocity": CONFIG['training']['target_velocity'],
            "min_target_velocity_period": 2,
            "max_target_velocity_period": 10,
            "enable_lumbar_joint": False,
            "lumbar_joint_fixed_angle": -0.13,
            "lumbar_joint_damping_value": 0.05,
            "observation_joint_pos_keys": CONFIG['reference_joint_keys'][model_type],
            "observation_joint_vel_keys": CONFIG['reference_joint_keys'][model_type],
            "observation_sensor_keys": ["r_foot", "l_foot", "r_toes", "l_toes"],
            "joint_limit_sensor_keys": [
                "r_knee_sensor", "l_knee_sensor", "r_hip_sensor", "l_hip_sensor",
                "r_ankle_sensor", "l_ankle_sensor", "r_mtp_sensor", "l_mtp_sensor"
            ],
            "terrain_type": "flat",
            "reward_keys_and_weights": {
                "qpos_imitation_rewards": CONFIG['reward_weights'][model_type]['qpos_imitation'],
                "qvel_imitation_rewards": CONFIG['reward_weights'][model_type]['qvel_imitation'],
                "end_effector_imitation_reward": 0.0,
                "forward_reward": 1.0,
                "muscle_activation_penalty": 0.1,
                "muscle_activation_diff_penalty": 0.1,
                "footstep_delta_time": 0.0,
                "average_velocity_per_step": 0.0,
                "muscle_activation_penalty_per_step": 0.0,
                "joint_constraint_force_penalty": 1.0,
                "foot_force_penalty": 0.5
            },
            "custom_max_episode_steps": 1000,
            "model_path": CONFIG['model_paths'][model_type],
            "reference_data_path": CONFIG['reference_data_path'],
            "reference_data_keys": CONFIG['reference_joint_keys'][model_type],
            "prev_trained_policy_path": None
        },
        "ppo_params": {
            "learning_rate": CONFIG['training']['learning_rate'],
            "n_steps": n_steps,
            "batch_size": 8192,
            "n_epochs": 30,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "clip_range_vf": 100,
            "ent_coef": 0.001,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "sde_sample_freq": -1,
            "target_kl": 0.01,
            "device": CONFIG['training']['device']
        },
        "policy_params": {
            "custom_policy_params": {
                "net_arch": CONFIG['network_arch'][model_type],
                "net_indexing_info": {},  # Will be auto-generated by environment
                "log_std_init": 0.0,
                "reset_shared_net_after_load": False,
                "reset_policy_net_after_load": False,
                "reset_value_net_after_load": False
            }
        },
        "auto_reward_adjust_params": {
            "learning_rate": 0.0
        }
    }
    
    # Save config
    config_path = f"rl_train/train/train_configs/{CONFIG['experiment_name']}_{model_type}_temp.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    return config_path


def run_training(model_type, quick_test=False, render=False):
    """Execute training"""
    
    print(f"\n{'='*80}")
    print(f"🚀 MyoAssist Imitation Learning Training")
    print(f"{'='*80}")
    print(f"\n📋 Configuration:")
    print(f"   Experiment: {CONFIG['experiment_name']}")
    print(f"   Model type: {model_type}")
    print(f"   Reference data: {CONFIG['reference_data_path']}")
    print(f"   Quick test: {quick_test}")
    print(f"   Device: {CONFIG['training']['device']}")
    
    if CONFIG['wandb']['enabled']:
        print(f"   WandB: Enabled (project: {CONFIG['wandb']['project']})")
    else:
        print(f"   WandB: Disabled")
    
    # Create config file
    config_path = create_config_file(model_type, quick_test)
    print(f"   Config: {config_path}")
    
    # Build command
    if CONFIG['wandb']['enabled']:
        # Use our WandB-integrated training script
        cmd = [
            "python", "train_with_wandb.py",
            "--config_file_path", config_path,
            "--wandb_project", CONFIG['wandb']['project'],
            "--experiment_name", CONFIG['experiment_name'],
            "--model_type", model_type,
            "--reference_data", CONFIG['reference_data_path'],
        ]
        if CONFIG['wandb']['entity']:
            cmd.extend(["--wandb_entity", CONFIG['wandb']['entity']])
    else:
        # Use original MyoAssist training script
        cmd = ["python", "rl_train/run_train.py", "--config_file_path", config_path]
        if render:
            cmd.append("--flag_rendering")
    
    print(f"\n💻 Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Run
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Training completed!")
        print(f"📊 Results in: rl_train/results/train_session_[timestamp]/")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='Train MyoAssist IL agent')
    parser.add_argument('--model', type=str, choices=['2D', '3D'], 
                        default=CONFIG['model_type'], help='Model type')
    parser.add_argument('--quick_test', action='store_true', help='Quick test mode')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--num_envs', type=int, help='Override num_envs')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Override device')
    
    args = parser.parse_args()
    
    # Override config with command line args
    if args.num_envs:
        CONFIG['training']['num_envs'] = args.num_envs
    if args.device:
        CONFIG['training']['device'] = args.device
    
    # Run training
    run_training(args.model, args.quick_test, args.render)


if __name__ == "__main__":
    main()
