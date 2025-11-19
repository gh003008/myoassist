"""
run_train_ver3_0.py
===================
SIMPLIFIED Training Script for Ver 3.0 System

Changes from original run_train.py:
- Only imports environment_handler_ver3_0 (simplified handler)
- Always uses Ver 3.0 callback (no version branching)
- Simplified command-line arguments (removed use_ver1_0 flag)
- Keeps LiveRenderToggleCallback (keyboard control during training)
- Keeps visualization functionality
- Keeps resume training functionality
- Always enables WandB for Ver 3.0

Created: 2024
"""

import numpy as np
import rl_train.train.train_configs.config as myoassist_config
import rl_train.utils.train_log_handler as train_log_handler
from rl_train.utils.data_types import DictionableDataclass
import json
import os
from datetime import datetime
from rl_train.envs.environment_handler_ver3_0 import EnvironmentHandler
import subprocess
from pathlib import Path

################################################################################################################### Rendering
import os, sys, time, threading
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter, configure
import numbers
################################################################################################################### Rendering


################################################################################################################### Rendering - LiveRenderToggleCallback
def _getch_nonblock():
    if os.name == "nt":  # Windows
        import msvcrt
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            return ch
        return None
    else:  # POSIX
        import select, sys, termios, tty
        dr, _, _ = select.select([sys.stdin], [], [], 0.03)
        if dr:
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            return ch
        return None
    
class LiveRenderToggleCallback(BaseCallback):
    """
    Thread-safe keyboard-controlled live rendering callback
    
    Keys:
      - 'o' : Toggle window ON/OFF (open/close viewer)
      - 'v' : Toggle pause/resume (sync off/on)
      - 'm' : Next env
      - 'n' : Previous env
      - 'q' : Stop key listener
    
    CRITICAL: All env_method calls happen ONLY in _on_step() (main learning thread)
              Key thread only pushes commands to queue -> prevents race conditions
    """
    def __init__(self, num_envs:int, start_index:int=0, render_every_n_steps:int=1, verbose:int=1):
        super().__init__(verbose)
        self.num_envs = int(num_envs)
        self.curr_idx = int(start_index)

        self.enabled_window = False
        self.paused = False

        self.render_interval = max(1, int(render_every_n_steps))
        self._last_render_step = -1

        self._stop = False
        self._th = threading.Thread(target=self._key_loop, daemon=True)

        self._cmd_q = deque()
        self._lock = threading.Lock()

    def _key_loop(self):
        """Key listener thread - only pushes commands to queue"""
        def _getch_nonblock():
            if os.name == "nt":
                import msvcrt
                if msvcrt.kbhit(): return msvcrt.getwch()
                return None
            else:
                import select, termios, tty
                dr, _, _ = select.select([sys.stdin], [], [], 0.02)
                if not dr: return None
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    ch = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
                return ch

        while not self._stop:
            ch = _getch_nonblock()
            if not ch:
                time.sleep(0.01)
                continue
            c = ch.lower()
            with self._lock:
                if c == 'o':
                    self._cmd_q.append(("TOGGLE_WINDOW", None))
                elif c == 'v':
                    self._cmd_q.append(("TOGGLE_PAUSE", None))
                elif c == 'm':
                    self._cmd_q.append(("SWITCH_ENV", +1))
                elif c == 'n':
                    self._cmd_q.append(("SWITCH_ENV", -1))
                elif c == 'q':
                    self._cmd_q.append(("STOP_KEYS", None))
            time.sleep(0.005)

    def _drain_cmds(self):
        """Safely drain command queue"""
        cmds = []
        with self._lock:
            while self._cmd_q:
                cmds.append(self._cmd_q.popleft())
        return cmds

    def _on_training_start(self) -> None:
        """Start key listener thread"""
        self._th.start()
        if self.verbose:
            print("[LiveRender] keys: o(open/close), v(pause), m/n(next/prev env), q(stop)")

    def _on_training_end(self) -> None:
        """Cleanup on training end"""
        self._stop = True
        if self.enabled_window:
            try:
                self.training_env.env_method("close_live_view", indices=[self.curr_idx])
            except Exception:
                pass

    def _on_step(self) -> bool:
        """
        Main step callback - processes commands and renders
        All env_method calls happen HERE (main learning thread only)
        """
        # 1) Process keyboard commands
        for typ, arg in self._drain_cmds():
            try:
                if typ == "TOGGLE_WINDOW":
                    if self.enabled_window:
                        # Close window
                        try:
                            self.training_env.env_method("close_live_view", indices=[self.curr_idx])
                        except Exception:
                            pass
                        self.enabled_window = False
                        if self.verbose: print("[LiveRender] window OFF")
                    else:
                        # Open window
                        self.enabled_window = True
                        if self.verbose: print("[LiveRender] window ON (env %d)" % self.curr_idx)
                        try:
                            self.training_env.env_method("render_live", paused=self.paused, indices=[self.curr_idx])
                        except Exception as e:
                            if self.verbose: print(f"[LiveRender] open failed: {e}")
                            self.enabled_window = False

                elif typ == "TOGGLE_PAUSE":
                    self.paused = not self.paused
                    if self.verbose: print(f"[LiveRender] paused={'ON' if self.paused else 'OFF'}")

                elif typ == "SWITCH_ENV":
                    delta = int(arg)
                    prev = self.curr_idx
                    self.curr_idx = (self.curr_idx + delta) % self.num_envs
                    if self.verbose: print(f"[LiveRender] env {prev} -> {self.curr_idx}")
                    if self.enabled_window:
                        # Close prev -> short sleep -> open new
                        try:
                            self.training_env.env_method("close_live_view", indices=[prev])
                        except Exception:
                            pass
                        time.sleep(0.01)
                        try:
                            self.training_env.env_method("render_live", paused=self.paused, indices=[self.curr_idx])
                        except Exception as e:
                            if self.verbose: print(f"[LiveRender] switch open failed: {e}")
                            self.enabled_window = False

                elif typ == "STOP_KEYS":
                    if self.verbose: print("[LiveRender] key listener stopped")
                    self._stop = True

            except Exception as e:
                # Never let errors kill training
                if self.verbose: print(f"[LiveRender] cmd {typ} error: {e}")

        # 2) Periodic frame update (only if window ON and not paused)
        if self.enabled_window and (not self.paused):
            if (self.num_timesteps - self._last_render_step) >= self.render_interval:
                try:
                    self.training_env.env_method("render_live", paused=False, indices=[self.curr_idx])
                    self._last_render_step = self.num_timesteps
                except Exception as e:
                    if self.verbose: print(f"[LiveRender] render tick failed: {e}")
        return True
################################################################################################################### Rendering


def get_git_info():
    """Get current git commit and branch information"""
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
    "version": "3.0.0",  # Ver 3.0 - Simplified system
    **get_git_info()
}


def visualize_training_environment(env, config, output_dir, num_frames=300):
    """
    Visualize the ACTUAL training environment with reference motion
    
    Args:
        env: The actual training environment (can be VecEnv or single env)
        config: Training configuration
        output_dir: Directory to save visualization video
        num_frames: Number of frames to render (default: 300 = 10 seconds @ 30 Hz)
    """
    # Check if this is an imitation task
    if not hasattr(config.env_params, 'reference_data_path'):
        print("⏭️  No reference data to visualize (not an imitation task)")
        return
    
    if not config.env_params.reference_data_path:
        print("⏭️  No reference data to visualize (not an imitation task)")
        return
    
    print("\n" + "="*80)
    print("🎬 TRAINING ENVIRONMENT VISUALIZATION")
    print("="*80)
    print(f"📁 Reference data: {Path(config.env_params.reference_data_path).name}")
    print(f"🤖 Model: {Path(config.env_params.model_path).name}")
    print(f"📊 Rendering {num_frames} frames from the ACTUAL training environment")
    print(f"⏱️  This will take ~10-20 seconds...")
    print("="*80 + "\n")
    
    try:
        import imageio
        from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
        
        # Check if this is a VecEnv (parallel environments)
        if isinstance(env, VecEnv):
            print("   🔄 Detected VecEnv (parallel environments)")
            print("   Creating a separate single environment for visualization...")
            
            # Create a fresh single environment instance
            original_num_envs = config.env_params.num_envs
            config.env_params.num_envs = 1
            
            temp_env = EnvironmentHandler.create_environment(config, load_reference_data=False)
            config.env_params.num_envs = original_num_envs
            
            # Unwrap if it's DummyVecEnv
            if isinstance(temp_env, DummyVecEnv):
                render_env = temp_env.envs[0]
            else:
                render_env = temp_env
        else:
            render_env = env
        
        width, height = 1920, 720
        frames = []
        
        print("🎥 Rendering frames...")
        
        # Check if render_env has .sim (unwrapped) or needs VecEnv methods
        if hasattr(render_env, 'sim'):
            # Direct access to environment
            print("   Using direct environment rendering...")
            render_env.reset()
            
            for frame_idx in range(num_frames):
                render_env.step(np.zeros(render_env.action_space.shape))
                
                # Progress indicator
                if frame_idx % 30 == 0:
                    print(f"   Frame {frame_idx}/{num_frames} ({100*frame_idx//num_frames}%)")
                
                # Try dm_control style rendering
                try:
                    img = render_env.sim.physics.render(
                        width=width, 
                        height=height,
                        camera_id=-1
                    )
                except AttributeError:
                    # Fallback: try mujoco-py style rendering
                    try:
                        img = render_env.sim.render(
                            width=width, 
                            height=height,
                            camera_id=-1,
                            mode='offscreen'
                        )
                    except:
                        print(f"   ⚠️  Could not render frame {frame_idx}")
                        continue
                
                frames.append(img)
        else:
            # VecEnv - use render method
            print("   Using VecEnv rendering...")
            render_env.reset()
            
            for frame_idx in range(num_frames):
                render_env.step(np.zeros((render_env.num_envs, render_env.action_space.shape[0])))
                
                if frame_idx % 30 == 0:
                    print(f"   Frame {frame_idx}/{num_frames} ({100*frame_idx//num_frames}%)")
                
                imgs = render_env.get_images()
                if imgs is not None and len(imgs) > 0:
                    frames.append(imgs[0])
        
        # Generate output path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{timestamp}_training_env_reference_ver3_0.mp4"
        output_path = Path(output_dir) / output_filename
        
        # Save video
        if len(frames) > 0:
            print(f"\n💾 Saving video...")
            imageio.mimwrite(str(output_path), frames, fps=30, quality=8)
            
            print(f"\n✅ Training environment visualization saved: {output_path}")
            print(f"📹 Video info: {len(frames)} frames @ 30 fps = {len(frames)/30:.1f} seconds")
            print(f"👀 This is the EXACT environment that will be used for training!")
            print(f"   ⚠️  Check pelvis height - should NOT be underground!")
            print("="*80 + "\n")
        else:
            print(f"\n⚠️  No frames rendered - skipping video save")
            print("="*80 + "\n")
        
        # Clean up temporary environment
        if isinstance(env, VecEnv) and render_env != env:
            render_env.close()
        
    except Exception as e:
        print(f"⚠️  Failed to visualize training environment: {e}")
        import traceback
        traceback.print_exc()
        print(f"   Training will continue anyway...")
        print("="*80 + "\n")


def ppo_train_with_parameters_ver3_0(config, train_time_step, train_log_handler, 
                                      wandb_config=None, visualize_before_training=True, 
                                      enable_live_render=True, enable_balance_reward=False):
    """
    Ver 3.0 training function - simplified with unified callback system
    
    Args:
        config: Training session configuration
        train_time_step: Number of timesteps to train
        train_log_handler: Log handler for training
        wandb_config: WandB configuration dict (required for Ver 3.0)
        visualize_before_training: Whether to visualize environment before training
        enable_live_render: Whether to enable keyboard-controlled live rendering
        enable_balance_reward: Whether to enable balance reward (Ver 3.0 feature)
    """
    seed = 1234
    np.random.seed(seed)

    # Step 1: Create training environment
    print("\n" + "="*80)
    print("🏗️  CREATING TRAINING ENVIRONMENT (Ver 3.0)")
    print("="*80)
    env = EnvironmentHandler.create_environment(
        config, 
        load_reference_data=True,
        enable_balance_reward=enable_balance_reward
    )
    print("✅ Ver 3.0 environment created successfully!")
    print("="*80 + "\n")
    
    # Step 2: Visualize environment before training
    if visualize_before_training:
        visualize_training_environment(env, config, log_dir, num_frames=300)
    
    # Step 3: Create RL model
    print("\n" + "="*80)
    print("🤖 CREATING RL MODEL (PPO)")
    print("="*80)
    model = EnvironmentHandler.get_stable_baselines3_model(config, env)
    print("✅ Model created successfully!")
    print("="*80 + "\n")

    EnvironmentHandler.updateconfig_from_model_policy(config, model)

    # Save session config
    session_config_dict = DictionableDataclass.to_dict(config)
    session_config_dict["env_params"].pop("reference_data", None)
    session_config_dict["code_version"] = VERSION

    with open(os.path.join(log_dir, 'session_config.json'), 'w', encoding='utf-8') as file:
        json.dump(session_config_dict, file, ensure_ascii=False, indent=4)

    # Step 4: Get Ver 3.0 callback (always uses ver3_0)
    custom_callback = EnvironmentHandler.get_callback(
        config, 
        train_log_handler, 
        wandb_config=wandb_config,
        enable_live_render=enable_live_render,
        enable_balance_reward=enable_balance_reward
    )

    # Step 5: Train!
    model.learn(
        reset_num_timesteps=False, 
        total_timesteps=train_time_step, 
        log_interval=1, 
        callback=custom_callback, 
        progress_bar=True
    )
    env.close()
    print("✅ Learning done!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_file_path", type=str, default="", help="path to train config file")
    
    # Ver 3.0: Simplified arguments (always uses WandB, always uses ver3_0 callback)
    parser.add_argument("--wandb_project", type=str, default="myoassist-imitation-ver3_0", 
                        help="WandB project name (default: myoassist-imitation-ver3_0)")
    parser.add_argument("--wandb_name", type=str, default=None, 
                        help="WandB run name (auto-generated if not specified)")
    
    # Ver 3.0 feature: Balance reward
    parser.add_argument("--enable_balance_reward", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Enable balance reward (Ver 3.0 feature)")
    
    # Live rendering during training
    parser.add_argument("--enable_live_render", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Enable keyboard-controlled live rendering during training (default: True, press 'o' to toggle)")
    
    # Resume training
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to previous training session directory to resume from")

    args, unknown_args = parser.parse_known_args()
    if args.config_file_path is None:
        raise ValueError("config_file_path is required")

    default_config = EnvironmentHandler.get_session_config_from_path(args.config_file_path, myoassist_config.TrainSessionConfigBase)
    DictionableDataclass.add_arguments(default_config, parser, prefix="config.")
    args = parser.parse_args()

    config_type = EnvironmentHandler.get_config_type_from_session_id(default_config.env_params.env_id)
    config = EnvironmentHandler.get_session_config_from_path(args.config_file_path, config_type)

    DictionableDataclass.set_from_args(config, args, prefix="config.")

    # Resume training if specified
    if args.resume_from:
        print("\n" + "="*80)
        print(f"🔄 RESUMING TRAINING FROM: {args.resume_from}")
        print("="*80)
        log_dir = args.resume_from
        train_log_handler = train_log_handler.TrainLogHandler(log_dir)
        
        # Load existing log data
        from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData
        train_log_handler.load_log_data(ImitationTrainCheckpointData)
        
        # Find the latest model checkpoint
        model_dir = os.path.join(log_dir, "trained_models")
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
            if model_files:
                latest_model = max(model_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
                latest_timestep = int(latest_model.split('_')[1].split('.')[0])
                model_path = os.path.join(model_dir, latest_model)
                config.env_params.prev_trained_policy_path = model_path
                
                completed_timesteps = latest_timestep
                remaining_timesteps = config.total_timesteps - completed_timesteps
                progress_pct = (completed_timesteps / config.total_timesteps) * 100
                
                print(f"📦 Loading checkpoint: {latest_model}")
                print(f"   Completed: {completed_timesteps:,} timesteps ({progress_pct:.1f}%)")
                print(f"   Remaining: {remaining_timesteps:,} timesteps")
                print(f"   Model path: {model_path}")
                print("="*80 + "\n")
            else:
                print("⚠️  No model checkpoints found in trained_models/ directory")
                print("   Starting from scratch...")
                print("="*80 + "\n")
        else:
            print("⚠️  Model directory not found, starting from scratch")
            print("="*80 + "\n")
    else:
        # Create new timestamped log directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_name = Path(args.config_file_path).stem
        session_name = f"{timestamp}_{config_name}_ver3_0"
        
        log_dir = os.path.join("rl_train", "results", session_name)
        os.makedirs(log_dir, exist_ok=True)
        train_log_handler = train_log_handler.TrainLogHandler(log_dir)

    print(f"\n📁 Session directory: {log_dir}")
    print(f"   All results (videos, models, logs) will be saved here.\n")
    
    # Ver 3.0: Always use WandB
    wandb_config = {
        'project': args.wandb_project,
        'name': args.wandb_name or f"train_ver3_0_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'config': {
            'model_type': '3D' if '3D' in config.env_params.model_path else '2D',
            'total_timesteps': config.total_timesteps,
            'env_id': config.env_params.env_id,
            'version': '3.0',
            'balance_reward': args.enable_balance_reward,
        },
        'tags': ['ver3_0', 'imitation', 'simplified'],
    }
    
    print(f"\n✅ Ver 3.0 training with WandB: {wandb_config['project']}/{wandb_config['name']}")
    if args.enable_balance_reward:
        print(f"   🎯 Balance reward ENABLED (Ver 3.0 feature)")
    print()

    # Run Ver 3.0 training
    ppo_train_with_parameters_ver3_0(
        config,
        train_time_step=config.total_timesteps,
        train_log_handler=train_log_handler,
        wandb_config=wandb_config,
        enable_live_render=args.enable_live_render,
        enable_balance_reward=args.enable_balance_reward
    )
