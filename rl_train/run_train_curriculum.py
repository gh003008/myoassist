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

# ============ CURRICULUM LEARNING EXTENSION ============
from rl_train.utils.curriculum_scheduler import CurriculumScheduler, interpolate_reward_weights
# =======================================================

################################################################################################################### Rendering
import os, sys, time, threading
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter, configure
import numbers
################################################################################################################### Rendering


################################################################################################################### Rendering
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
    키:
      - 'o' : 창 ON/OFF (viewer open/close)
      - 'v' : 일시정지/재개 (sync 끔/켬)
      - 'm' : 다음 env
      - 'n' : 이전 env
      - 'q' : 키 리스너 종료
    주의: 키 스레드에서는 '명령만 큐에 push'하고, 실제 env_method 호출은
         _on_step() (메인 학습 스레드)에서만 수행 -> 경쟁/랜덤 크래시 방지
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

    # ---------- 키 스레드: 명령만 큐에 넣는다 ----------
    def _key_loop(self):
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
        cmds = []
        with self._lock:
            while self._cmd_q:
                cmds.append(self._cmd_q.popleft())
        return cmds

    # ---------- SB3 콜백 라이프사이클 ----------
    def _on_training_start(self) -> None:
        self._th.start()
        if self.verbose:
            print("[LiveRender] keys: o(open/close), v(pause), m/n(next/prev env), q(stop)")

    def _on_training_end(self) -> None:
        self._stop = True
        if self.enabled_window:
            try:
                self.training_env.env_method("close_live_view", indices=[self.curr_idx])
            except Exception:
                pass

    # ---------- 메인 학습 스레드에서만 env_method를 호출 ----------
    def _on_step(self) -> bool:
        # 1) 키 명령 처리
        for typ, arg in self._drain_cmds():
            try:
                if typ == "TOGGLE_WINDOW":
                    if self.enabled_window:
                        # close
                        try:
                            self.training_env.env_method("close_live_view", indices=[self.curr_idx])
                        except Exception:
                            pass
                        self.enabled_window = False
                        if self.verbose: print("[LiveRender] window OFF")
                    else:
                        # open
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
                        # close prev -> short sleep -> open new
                        try:
                            self.training_env.env_method("close_live_view", indices=[prev])
                        except Exception:
                            pass
                        time.sleep(0.01)
                        try:
                            self.training_env.env_method("render_live", paused=self.paused, indices=[self.curr_idx])
                        except Exception as e:
                            if self.verbose: print(f"[LiveRender] switch open failed: {e}")
                            # 창 상태를 정합하게 유지
                            self.enabled_window = False

                elif typ == "STOP_KEYS":
                    if self.verbose: print("[LiveRender] key listener stopped")
                    self._stop = True

            except Exception as e:
                # 어떤 에러도 학습을 죽이지 않도록 삼킨다
                if self.verbose: print(f"[LiveRender] cmd {typ} error: {e}")

        # 2) 주기적 프레임 업데이트 (창 ON & pause OFF 일 때만)
        if self.enabled_window and (not self.paused):
            if (self.num_timesteps - self._last_render_step) >= self.render_interval:
                try:
                    self.training_env.env_method("render_live", paused=False, indices=[self.curr_idx])
                    self._last_render_step = self.num_timesteps
                except Exception as e:
                    if self.verbose: print(f"[LiveRender] render tick failed: {e}")
        return True
################################################################################################################### Rendering

# ============================================================================
# CURRICULUM LEARNING CALLBACK (EXTENSION)
# ============================================================================
class CurriculumLearningCallback(BaseCallback):
    """
    Curriculum Learning Callback
    
    학습 중 주기적으로 curriculum stage를 체크하고,
    stage가 변경되면 환경 파라미터와 reward weights를 업데이트합니다.
    
    주요 기능:
    - Stage 전환 자동 감지
    - 환경 velocity range 동적 변경
    - Episode length 동적 변경
    - Reward weights 동적 변경 (config 업데이트)
    
    Args:
        scheduler: CurriculumScheduler 인스턴스
        check_freq: Stage 체크 주기 (timesteps)
        verbose: 로그 레벨
    """
    def __init__(self, scheduler: CurriculumScheduler, config, check_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.scheduler = scheduler
        self.config = config
        self.check_freq = check_freq
        self.last_check_timestep = 0
        
        # Store original reward weights for interpolation
        self.base_reward_weights = dict(config.env_params.reward_keys_and_weights)
    
    def _on_step(self) -> bool:
        """매 스텝마다 호출되는 콜백"""
        
        # Check frequency에 따라 stage 확인
        if self.num_timesteps - self.last_check_timestep >= self.check_freq:
            self.last_check_timestep = self.num_timesteps
            
            # Update scheduler with elapsed timesteps
            stage_changed = self.scheduler.update(self.check_freq)
            
            if stage_changed:
                self._apply_new_stage()
        
        return True
    
    def _apply_new_stage(self):
        """새로운 curriculum stage 적용"""
        stage = self.scheduler.get_current_stage()
        params = self.scheduler.get_current_stage_params()
        
        print("\n" + "🎓"*40)
        print(f"🎓 CURRICULUM STAGE {stage.stage_id}: {stage.name}")
        print(f"   Timestep: {self.num_timesteps:,}")
        print("🎓"*40)
        
        # 1. Update velocity range
        if 'target_velocity_range' in params:
            vel_min, vel_max = params['target_velocity_range']
            try:
                # VecEnv method to update all environments
                self.training_env.env_method('update_velocity_range', vel_min, vel_max)
                print(f"   ✅ Velocity range updated: {vel_min:.2f} ~ {vel_max:.2f} m/s")
            except Exception as e:
                print(f"   ⚠️  Could not update velocity range: {e}")
        
        # 2. Update max episode steps
        if 'max_episode_steps' in params:
            max_steps = params['max_episode_steps']
            try:
                self.training_env.env_method('update_max_episode_steps', max_steps)
                print(f"   ✅ Max episode steps updated: {max_steps}")
            except Exception as e:
                print(f"   ⚠️  Could not update max episode steps: {e}")
        
        # 3. Update reward weights
        if 'reward_weights' in params and params['reward_weights'] is not None:
            try:
                # Interpolate between base and stage weights
                progress = self.scheduler.get_progress()
                new_weights = interpolate_reward_weights(
                    self.base_reward_weights,
                    params['reward_weights'],
                    progress=1.0  # Immediately apply full stage weights
                )
                
                # Update config (this affects new episodes)
                self.config.env_params.reward_keys_and_weights.update(new_weights)
                
                # Update existing environments
                self.training_env.env_method('update_reward_weights', new_weights)
                
                print(f"   ✅ Reward weights updated ({len(params['reward_weights'])} weights)")
                for key, value in params['reward_weights'].items():
                    print(f"      {key}: {value:.3f}")
            except Exception as e:
                print(f"   ⚠️  Could not update reward weights: {e}")
        
        print("🎓"*40 + "\n")
# ============================================================================

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

def visualize_training_environment(env, config, output_dir, num_frames=300):
    """
    Visualize the ACTUAL training environment with reference motion
    This uses the same environment that will be used for training!
    
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
            
            # Create a fresh single environment instance (not wrapped in VecEnv)
            from rl_train.envs.environment_handler import EnvironmentHandler
            
            # Temporarily disable num_envs to create single env
            original_num_envs = config.env_params.num_envs  # num_envs is in env_params!
            config.env_params.num_envs = 1
            
            # Create single environment (will be DummyVecEnv with 1 env)
            temp_env = EnvironmentHandler.create_environment(config, is_rendering_on=False)
            
            # Restore original num_envs
            config.env_params.num_envs = original_num_envs
            
            # Unwrap if it's DummyVecEnv
            if isinstance(temp_env, DummyVecEnv):
                render_env = temp_env.envs[0]  # DummyVecEnv has .envs attribute
            else:
                # For SubprocVecEnv or other, just use temp_env directly
                # We'll use get_images() method instead
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
            
            # Reset environment to start
            render_env.reset()
            
            for frame_idx in range(num_frames):
                # Step through reference motion
                render_env.step(np.zeros(render_env.action_space.shape))
                
                # Progress indicator
                if frame_idx % 30 == 0:
                    print(f"   Frame {frame_idx}/{num_frames} ({100*frame_idx//num_frames}%)")
                
                # Render frame using dm_control's physics
                # dm_control uses physics.render() not sim.render()
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
                # Step through reference motion
                render_env.step(np.zeros((render_env.num_envs, render_env.action_space.shape[0])))
                
                # Progress indicator
                if frame_idx % 30 == 0:
                    print(f"   Frame {frame_idx}/{num_frames} ({100*frame_idx//num_frames}%)")
                
                # Get image from first environment
                imgs = render_env.get_images()
                if imgs is not None and len(imgs) > 0:
                    frames.append(imgs[0])
        
        # Generate output path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{timestamp}_training_env_reference.mp4"
        output_path = Path(output_dir) / output_filename
        
        # Save video only if we got frames
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
        
        # Clean up temporary environment if created
        if isinstance(env, VecEnv) and render_env != env:
            render_env.close()
        
    except Exception as e:
        print(f"⚠️  Failed to visualize training environment: {e}")
        import traceback
        traceback.print_exc()
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
def ppo_train_with_parameters(config, train_time_step, is_rendering_on, train_log_handler, 
                             use_ver1_0=False, wandb_config=None, visualize_before_training=True,
                             curriculum_scheduler=None):  # ============ CURRICULUM EXTENSION ============
    seed = 1234
    np.random.seed(seed)

    # 🎬 Step 1: Create training environment
    print("\n" + "="*80)
    print("🏗️  CREATING TRAINING ENVIRONMENT")
    print("="*80)
    env = EnvironmentHandler.create_environment(config, is_rendering_on)
    print("✅ Environment created successfully!")
    print("="*80 + "\n")
    
    # 🎬 Step 2: Visualize the environment BEFORE training (optional)
    if visualize_before_training:
        visualize_training_environment(env, config, log_dir, num_frames=300)
    
    # 🎬 Step 3: Create RL model
    print("\n" + "="*80)
    print("🤖 CREATING RL MODEL (PPO)")
    print("="*80)
    model = EnvironmentHandler.get_stable_baselines3_model(config, env)
    print("✅ Model created successfully!")
    print("="*80 + "\n")

    EnvironmentHandler.updateconfig_from_model_policy(config, model)

    session_config_dict = DictionableDataclass.to_dict(config)
    session_config_dict["env_params"].pop("reference_data", None)

    session_config_dict["code_version"] = VERSION
    with open(os.path.join(log_dir, 'session_config.json'), 'w', encoding='utf-8') as file:
        json.dump(session_config_dict, file, ensure_ascii=False, indent=4)

    # ============ CURRICULUM LEARNING EXTENSION ============
    # Apply initial curriculum stage if enabled
    if curriculum_scheduler is not None:
        initial_stage_params = curriculum_scheduler.get_current_stage_params()
        if 'target_velocity_range' in initial_stage_params:
            vel_min, vel_max = initial_stage_params['target_velocity_range']
            config.env_params.min_target_velocity = vel_min
            config.env_params.max_target_velocity = vel_max
            print(f"🎯 Curriculum Stage 1: Velocity set to {vel_min:.2f}~{vel_max:.2f} m/s")
        
        if 'max_episode_steps' in initial_stage_params:
            config.env_params.custom_max_episode_steps = initial_stage_params['max_episode_steps']
            print(f"🎯 Curriculum Stage 1: Max episode steps = {initial_stage_params['max_episode_steps']}")
    # =======================================================

    custom_callback = EnvironmentHandler.get_callback(config, train_log_handler, use_ver1_0=use_ver1_0, wandb_config=wandb_config)

    # ============ CURRICULUM LEARNING EXTENSION ============
    # Add curriculum callback if enabled
    if curriculum_scheduler is not None:
        curriculum_callback = CurriculumLearningCallback(
            scheduler=curriculum_scheduler,
            config=config,
            check_freq=10000,  # Check every 10k timesteps
            verbose=1
        )
        
        # Combine callbacks
        from stable_baselines3.common.callbacks import CallbackList
        if isinstance(custom_callback, list):
            all_callbacks = custom_callback + [curriculum_callback]
        else:
            all_callbacks = [custom_callback, curriculum_callback]
        custom_callback = CallbackList(all_callbacks)
        
        print("✅ Curriculum callback added to training pipeline\n")
    # =======================================================

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
    parser.add_argument("--wandb_project", type=str, default="myoassist-3D-imitation", help="WandB project name (default: myoassist-3D-imitation)")
    parser.add_argument("--wandb_name", type=str, default=None, help="WandB run name (auto-generated if not specified)")
    
    # Resume training
    parser.add_argument("--resume_from", type=str, default=None, help="Path to previous training session directory to resume from (e.g., rl_train/results/20251118_144143_S004_3D_IL_ver2_1_BALANCE)")
    
    # ============ CURRICULUM LEARNING EXTENSION ============
    parser.add_argument("--enable_curriculum", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Enable curriculum learning (True/False)")
    parser.add_argument("--curriculum_config", type=str, default="rl_train/train/train_configs/curriculum_treadmill_default.json", help="Path to curriculum config JSON file")
    # =======================================================

    args, unknown_args = parser.parse_known_args()
    if args.config_file_path is None:
        raise ValueError("config_file_path is required")

    default_config = EnvironmentHandler.get_session_config_from_path(args.config_file_path, myoassist_config.TrainSessionConfigBase)
    DictionableDataclass.add_arguments(default_config, parser, prefix="config.")
    args = parser.parse_args()

    config_type = EnvironmentHandler.get_config_type_from_session_id(default_config.env_params.env_id)
    config = EnvironmentHandler.get_session_config_from_path(args.config_file_path, config_type)


    DictionableDataclass.set_from_args(config, args, prefix="config.")


    # 🔄 Resume training if specified
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
                # Extract timesteps from filename (e.g., model_3162112.zip -> 3162112)
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
        # Create timestamped log directory with clear naming
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Extract meaningful name from config
        config_name = Path(args.config_file_path).stem  # e.g., S004_3D_IL_ver2_1_BALANCE
        session_name = f"{timestamp}_{config_name}"
        
        log_dir = os.path.join("rl_train", "results", session_name)
        os.makedirs(log_dir, exist_ok=True)
        train_log_handler = train_log_handler.TrainLogHandler(log_dir)

    
    print(f"\n📁 Session directory: {log_dir}")
    print(f"   All results (videos, models, logs) will be saved here.\n")
    
    # ============ CURRICULUM LEARNING EXTENSION ============
    # Initialize Curriculum Scheduler
    curriculum_scheduler = None
    if args.enable_curriculum:
        if os.path.exists(args.curriculum_config):
            curriculum_scheduler = CurriculumScheduler.from_config(
                args.curriculum_config,
                enable=True
            )
        else:
            print(f"⚠️  Curriculum config not found: {args.curriculum_config}")
            print("   Using default treadmill curriculum")
            curriculum_scheduler = CurriculumScheduler.create_default_treadmill_curriculum(enable=True)
        
        # Save curriculum config to log directory
        curriculum_save_path = os.path.join(log_dir, "curriculum_config.json")
        if os.path.exists(args.curriculum_config):
            import shutil
            shutil.copy(args.curriculum_config, curriculum_save_path)
        print(f"📚 Curriculum learning ENABLED - config saved to {curriculum_save_path}\n")
    else:
        print("📚 Curriculum learning DISABLED\n")
    # =======================================================
    
    # 🎬 Visualization will happen INSIDE ppo_train_with_parameters
    # after environment creation, before training starts
    
    # 🔍 Auto-detect if this is Ver2_1 environment
    is_ver2_1 = config.env_params.env_id == "myoAssistLegImitationExo-v2_1"
    
    # Prepare WandB config
    # Ver2_1 environments automatically use WandB unless explicitly disabled
    # Other versions require --use_ver1_0 flag
    use_wandb = args.use_ver1_0 or is_ver2_1
    
    wandb_config = None
    if use_wandb:
        wandb_config = {
            'project': args.wandb_project,
            'name': args.wandb_name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'config': {
                'model_type': '3D' if '3D' in config.env_params.model_path else '2D',
                'total_timesteps': config.total_timesteps,
                'env_id': config.env_params.env_id,
            },
            'tags': ['ver2_1' if is_ver2_1 else 'ver1_0', 'imitation'],
        }
        
        if is_ver2_1:
            print(f"\n✅ Ver2_1 detected - WandB auto-enabled: {wandb_config['project']}/{wandb_config['name']}")
            print(f"   (Ver2_1 environments always use WandB for tracking)\n")
        else:
            print(f"\n✅ ver1_0 mode enabled with WandB: {wandb_config['project']}/{wandb_config['name']}\n")

    if args.flag_realtime_evaluate:
        ppo_evaluate_with_rendering(config)
    else:
        ppo_train_with_parameters(config,
                                train_time_step=config.total_timesteps,
                                is_rendering_on=args.flag_rendering,
                                train_log_handler=train_log_handler,
                                use_ver1_0=use_wandb,  # Pass WandB flag
                                wandb_config=wandb_config,
                                curriculum_scheduler=curriculum_scheduler)  # ============ CURRICULUM EXTENSION ============

    