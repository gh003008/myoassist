"""
Training Runner with WandB Integration
======================================
Runs MyoAssist training with WandB logging without modifying original code.

Usage:
    conda activate myoassist
    python train_with_wandb.py
"""

import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import MyoAssist components
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass
from rl_train.envs.environment_handler import EnvironmentHandler
import rl_train.utils.train_log_handler as train_log_handler

# Import our WandB callback
from wandb_callback import WandBCallback, CombinedCallback

# Import WandB
import wandb


def get_git_info():
    import subprocess
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
        return {"commit": commit, "branch": branch}
    except:
        return {"commit": "unknown", "branch": "unknown"}


VERSION = {
    "version": "0.3.0",
    **get_git_info()
}


def train_with_wandb(config_file_path, wandb_config):
    """Run training with WandB integration"""
    
    seed = 1234
    np.random.seed(seed)
    
    # Load config
    print(f"Loading config from {config_file_path}")
    config = EnvironmentHandler.get_session_config_from_path(
        config_file_path,
        ImitationTrainSessionConfig
    )
    
    # Create log directory
    log_dir = os.path.join("rl_train", "results", f"train_session_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize train log handler
    train_log = train_log_handler.TrainLogHandler(log_dir)
    
    # Create environment
    print("Creating training environment...")
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False)
    
    # Create model
    print("Creating PPO model...")
    model = EnvironmentHandler.get_stable_baselines3_model(config, env)
    
    # Update config from model policy
    EnvironmentHandler.updateconfig_from_model_policy(config, model)
    
    # Save session config
    session_config_dict = DictionableDataclass.to_dict(config)
    session_config_dict["env_params"].pop("reference_data", None)
    session_config_dict["code_version"] = VERSION
    
    with open(os.path.join(log_dir, 'session_config.json'), 'w', encoding='utf-8') as file:
        json.dump(session_config_dict, file, ensure_ascii=False, indent=4)
    
    # Get MyoAssist callback
    myoassist_callback = EnvironmentHandler.get_callback(config, train_log)
    
    # Create WandB callback
    wandb_callback = WandBCallback(
        config=wandb_config,
        log_dir=log_dir,
        total_timesteps=config.total_timesteps,
        verbose=1
    )
    
    # Combine callbacks
    combined_callback = CombinedCallback([myoassist_callback, wandb_callback])
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting training")
    print(f"   Total timesteps: {config.total_timesteps:,.0f}")
    print(f"   Log directory: {log_dir}")
    print(f"{'='*80}\n")
    
    # Train!
    model.learn(
        reset_num_timesteps=False,
        total_timesteps=config.total_timesteps,
        log_interval=1,
        callback=combined_callback,
        progress_bar=True
    )
    
    env.close()
    print("\n✅ Training complete!")
    
    return log_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="myoassist-imitation")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="training")
    parser.add_argument("--model_type", type=str, default="3D")
    parser.add_argument("--reference_data", type=str, default="")
    
    args = parser.parse_args()
    
    # WandB config
    wandb_config = {
        'wandb_project': args.wandb_project,
        'wandb_entity': args.wandb_entity,
        'experiment_name': args.experiment_name,
        'model_type': args.model_type,
        'reference_data_path': args.reference_data,
    }
    
    # Run training
    log_dir = train_with_wandb(args.config_file_path, wandb_config)
    print(f"📊 Results saved to: {log_dir}")
