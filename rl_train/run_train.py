import numpy as np
import rl_train.train.train_configs.config as myoassist_config
import rl_train.utils.train_log_handler as train_log_handler
from rl_train.utils.data_types import DictionableDataclass
import json
import os
from datetime import datetime
from rl_train.envs.environment_handler import EnvironmentHandler
import subprocess
from pathlib import Path

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
    "version": "0.3.0",  # MAJOR.MINOR.PATCH
    **get_git_info()
}

def visualize_reference_motion(config, output_dir):
    """
    Visualize reference motion before training starts
    
    Args:
        config: Training configuration
        output_dir: Directory to save visualization video
    """
    # Check if config has reference data path
    if not hasattr(config.env_params, 'reference_data_path'):
        print("⏭️  No reference data to visualize (not an imitation task)")
        return
    
    if not config.env_params.reference_data_path:
        print("⏭️  No reference data to visualize (not an imitation task)")
        return
    
    ref_data_path = Path(config.env_params.reference_data_path)
    if not ref_data_path.exists():
        print(f"⚠️  Reference data not found: {ref_data_path}")
        return
    
    model_path = Path(config.env_params.model_path)
    if not model_path.exists():
        print(f"⚠️  Model file not found: {model_path}")
        return
    
    print("\n" + "="*80)
    print("🎬 REFERENCE MOTION VISUALIZATION")
    print("="*80)
    print(f"📁 Reference data: {ref_data_path.name}")
    print(f"🤖 Model: {model_path.name}")
    print(f"📊 Total timesteps: {config.total_timesteps:,}")
    print(f"⏱️  This will take ~20-30 seconds (high quality + multiview)...")
    print("="*80 + "\n")
    
    # Import render function
    try:
        from rl_train.analyzer.custom.render_hdf5_reference import render_reference_motion
    except ImportError:
        print("⚠️  Could not import render_hdf5_reference module")
        return
    
    # Generate output path
    output_path = Path(output_dir) / f"ref_{ref_data_path.stem}_preview_multiview.mp4"
    
    # Render with more frames for smoother motion + multiview
    try:
        render_reference_motion(
            npz_path=str(ref_data_path),
            model_path=str(model_path),
            output_path=str(output_path),
            num_frames=600,  # 600 frames for smooth 20-second video at 30fps
            height_offset=0.95,
            fps=30,  # 30 FPS for smooth playback
            multiview=True  # Front + Side views
        )
        print(f"\n✅ Reference motion saved: {output_path}")
        print(f"👀 Review this video to confirm you're using the correct reference data!")
        print(f"   🎥 Front view (left) + Side view (right) for symmetry check")
        print("="*80 + "\n")
    except Exception as e:
        print(f"⚠️  Failed to render reference motion: {e}")
        print(f"   Training will continue anyway...")
        print("="*80 + "\n")

def ppo_evaluate_with_rendering(config):
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
def ppo_train_with_parameters(config, train_time_step, is_rendering_on, train_log_handler, use_ver1_0=False, wandb_config=None):
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

    custom_callback = EnvironmentHandler.get_callback(config, train_log_handler, use_ver1_0=use_ver1_0, wandb_config=wandb_config)


    model.learn(reset_num_timesteps=False, total_timesteps=train_time_step, log_interval=1, callback=custom_callback, progress_bar=True)
    env.close()
    print("learning done!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_file_path", type=str, default="", help="path to train config file")
    parser.add_argument("--flag_rendering", type=bool, default=False, action=argparse.BooleanOptionalAction, help="rendering(True/False)")
    parser.add_argument("--flag_realtime_evaluate", type=bool, default=False, action=argparse.BooleanOptionalAction, help="realtime evaluate(True/False)")
    
    # ver1_0 options
    parser.add_argument("--use_ver1_0", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Use ver1_0 callback with WandB (True/False)")
    parser.add_argument("--wandb_project", type=str, default="myoassist-imitation", help="WandB project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="WandB run name")

    args, unknown_args = parser.parse_known_args()
    if args.config_file_path is None:
        raise ValueError("config_file_path is required")

    default_config = EnvironmentHandler.get_session_config_from_path(args.config_file_path, myoassist_config.TrainSessionConfigBase)
    DictionableDataclass.add_arguments(default_config, parser, prefix="config.")
    args = parser.parse_args()

    config_type = EnvironmentHandler.get_config_type_from_session_id(default_config.env_params.env_id)
    config = EnvironmentHandler.get_session_config_from_path(args.config_file_path, config_type)


    DictionableDataclass.set_from_args(config, args, prefix="config.")


    log_dir = os.path.join("rl_train","results", f"train_session_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    train_log_handler = train_log_handler.TrainLogHandler(log_dir)
    
    # 🎬 Visualize reference motion BEFORE training starts
    if not args.flag_realtime_evaluate:
        visualize_reference_motion(config, log_dir)
    
    # Prepare WandB config if ver1_0 is enabled
    wandb_config = None
    if args.use_ver1_0:
        wandb_config = {
            'project': args.wandb_project,
            'name': args.wandb_name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'config': {
                'model_type': '3D' if '3D' in config.env_params.model_path else '2D',
                'total_timesteps': config.total_timesteps,
            },
            'tags': ['ver1_0', 'imitation'],
        }
        print(f"\n✅ ver1_0 mode enabled with WandB: {wandb_config['project']}/{wandb_config['name']}\n")

    if args.flag_realtime_evaluate:
        ppo_evaluate_with_rendering(config)
    else:
        ppo_train_with_parameters(config,
                                train_time_step=config.total_timesteps,
                                is_rendering_on=args.flag_rendering,
                                train_log_handler=train_log_handler,
                                use_ver1_0=args.use_ver1_0,
                                wandb_config=wandb_config)
    