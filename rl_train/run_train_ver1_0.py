"""
ver1_0: Training runner with WandB integration
"""
import numpy as np
import rl_train.train.train_configs.config as myoassist_config
import rl_train.utils.train_log_handler as train_log_handler
from rl_train.utils.data_types import DictionableDataclass
import json
import os
from datetime import datetime
from rl_train.envs.environment_handler import EnvironmentHandler
import subprocess

# ver1_0: Import ver1_0 callback
from rl_train.envs.myoassist_leg_imitation_ver1_0 import ImitationCustomLearningCallback_ver1_0
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig

def get_git_info():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
        return {
            "commit": commit,
            "branch": branch
        }
    except:
        return {
            "commit": "unknown",
            "branch": "unknown"
        }

# Version information
VERSION = {
    "version": "ver1_0",  # ver1_0: WandB + 10% evaluation
    **get_git_info()
}

def ppo_evaluate_with_rendering(config):
    """Evaluate trained model with rendering"""
    seed = 1234
    np.random.seed(seed)

    env = EnvironmentHandler.create_environment(config, is_rendering_on=True, is_evaluate_mode=True)
    model = EnvironmentHandler.get_stable_baselines3_model(config, env)

    EnvironmentHandler.updateconfig_from_model_policy(config, model)

    obs, info = env.reset()
    for _ in range(config.evaluate_param_list[0]["num_timesteps"]):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, truncated, info = env.step(action)
        if truncated:
            obs, info = env.reset()

    env.close()

def ppo_train_with_parameters(config, train_time_step, is_rendering_on, train_log_handler, wandb_config=None):
    """ver1_0: Training with WandB support"""
    seed = 1234
    np.random.seed(seed)

    env = EnvironmentHandler.create_environment(config, is_rendering_on)
    model = EnvironmentHandler.get_stable_baselines3_model(config, env)

    EnvironmentHandler.updateconfig_from_model_policy(config, model)

    session_config_dict = DictionableDataclass.to_dict(config)
    session_config_dict["env_params"].pop("reference_data", None)

    session_config_dict["code_version"] = VERSION
    with open(os.path.join(log_dir, 'session_config.json'), 'w', encoding='utf-8') as file:
        json.dump(session_config_dict, file, ensure_ascii=False, indent=4)

    # ver1_0: Use ver1_0 callback with WandB
    if isinstance(config, ImitationTrainSessionConfig):
        custom_callback = ImitationCustomLearningCallback_ver1_0(
            log_rollout_freq=config.logger_params.logging_frequency,
            evaluate_freq=config.logger_params.evaluate_frequency,
            log_handler=train_log_handler,
            original_reward_weights=config.env_params.reward_keys_and_weights,
            auto_reward_adjust_params=config.auto_reward_adjust_params,
            config=config,
            wandb_config=wandb_config,
        )
    else:
        custom_callback = EnvironmentHandler.get_callback(config, train_log_handler)


    model.learn(reset_num_timesteps=False, total_timesteps=train_time_step, log_interval=1, callback=custom_callback, progress_bar=True)
    env.close()
    print("learning done!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_file_path", type=str, default="", help="path to train config file")
    parser.add_argument("--flag_rendering", type=bool, default=False, action=argparse.BooleanOptionalAction, help="rendering(True/False)")
    parser.add_argument("--flag_realtime_evaluate", type=bool, default=False, action=argparse.BooleanOptionalAction, help="realtime evaluate(True/False)")
    
    # ver1_0: WandB arguments
    parser.add_argument("--wandb_project", type=str, default="myoassist-rl", help="WandB project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--enable_wandb", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Enable WandB logging")
    
    # ver1_0: Resume training arguments
    parser.add_argument("--resume_from", type=str, default=None, help="Path to previous training session directory to resume from (e.g., rl_train/results/train_session_20251117-143446)")

    args, unknown_args = parser.parse_known_args()
    if args.config_file_path is None:
        raise ValueError("config_file_path is required")

    default_config = EnvironmentHandler.get_session_config_from_path(args.config_file_path, myoassist_config.TrainSessionConfigBase)
    DictionableDataclass.add_arguments(default_config, parser, prefix="config.")
    args = parser.parse_args()

    config_type = EnvironmentHandler.get_config_type_from_session_id(default_config.env_params.env_id)
    config = EnvironmentHandler.get_session_config_from_path(args.config_file_path, config_type)

    DictionableDataclass.set_from_args(config, args, prefix="config.")

    # ver1_0: Resume training if specified
    if args.resume_from:
        print(f"🔄 Resuming training from: {args.resume_from}")
        log_dir = args.resume_from
        train_log_handler = train_log_handler.TrainLogHandler(log_dir)
        
        # Load existing log data with appropriate checkpoint type
        from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData
        train_log_handler.load_log_data(ImitationTrainCheckpointData)
        
        # Find the latest model checkpoint
        model_dir = os.path.join(log_dir, "trained_models")
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
            if model_files:
                # Extract timesteps from filename (e.g., model_593920.zip -> 593920)
                latest_model = max(model_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
                latest_timestep = int(latest_model.split('_')[1].split('.')[0])
                config.env_params.prev_trained_policy_path = os.path.join(model_dir, latest_model)
                print(f"📦 Loading checkpoint: {latest_model} ({latest_timestep:,} timesteps)")
            else:
                print("⚠️ No model checkpoints found, starting from scratch")
        else:
            print("⚠️ Model directory not found, starting from scratch")
    else:
        log_dir = os.path.join("rl_train","results", f"train_session_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(log_dir, exist_ok=True)
        train_log_handler = train_log_handler.TrainLogHandler(log_dir)
    
    # ver1_0: Prepare WandB config
    wandb_config = None
    if args.enable_wandb:
        # Extract num_envs from config or use default
        num_envs = getattr(config, 'num_envs', getattr(config.env_params, 'num_envs', 8))
        
        wandb_config = {
            'project': args.wandb_project,
            'name': args.wandb_name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'config': {
                'model_type': '3D' if '3D' in config.env_params.model_path else '2D',
                'total_timesteps': config.total_timesteps,
                'num_envs': num_envs,
                'learning_rate': getattr(config, 'learning_rate', 0.0001),
                'device': getattr(config, 'device', 'cpu'),
            }
        }
        print(f"✅ WandB enabled: {wandb_config['project']}/{wandb_config['name']}")

    if args.flag_realtime_evaluate:
        ppo_evaluate_with_rendering(config)
    else:
        ppo_train_with_parameters(config,
                                train_time_step=config.total_timesteps,
                                is_rendering_on=args.flag_rendering,
                                train_log_handler=train_log_handler,
                                wandb_config=wandb_config)
    