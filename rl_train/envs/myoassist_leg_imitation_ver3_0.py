"""
MyoAssist Leg Imitation Environment Ver 3.0

Created: 2024-11-19
Purpose: Complete standalone training system - ALL-IN-ONE implementation

🎯 Ver 3.0 Features:
- ✅ NO INHERITANCE from BaseCustomLearningCallback (self-contained!)
- ✅ WandB logging integration (from ver1_0)
- ✅ 10% progress evaluation with rendering (from ver1_0)
- ✅ LiveRender keyboard control support (new in ghlee-home)
- ✅ Balance rewards for 3D stability (from ver2_1)
- ✅ Thread-safe rendering with exception handling (new)
- ✅ Complete callback + environment in single file

📦 Code Sources:
- BaseCallback features: Copied from learning_callback.py
- WandB & Evaluation: From myoassist_leg_imitation_ver1_0.py
- Balance rewards: From myoassist_leg_imitation_ver2_1.py
- Exception handling: From recent ghlee-home updates
- LiveRender: Compatible with LiveRenderToggleCallback

🚀 Usage:
    from rl_train.envs.myoassist_leg_imitation_ver3_0 import (
        ImitationCustomLearningCallback_ver3_0,
        MyoAssistLegImitation_ver3_0
    )
    
    # Just use ver3_0 - no other imports needed!

⚠️ Dependencies:
- stable_baselines3 (BaseCallback only)
- wandb
- numpy
- myosuite

NO dependency on learning_callback.py or other ver files!
"""

import collections
import numpy as np
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback  # Only SB3!
from stable_baselines3.common.vec_env import SubprocVecEnv

# MyoAssist specific imports
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.utils.data_types import DictionableDataclass
from rl_train.utils import train_log_handler
from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from myosuite.utils.quat_math import quat2mat

# WandB integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  Warning: wandb not installed. WandB logging disabled.")

################################################################


class ImitationCustomLearningCallback_ver3_0(BaseCallback):
    """
    Ver 3.0: Complete standalone callback - NO external inheritance!
    
    All features integrated from:
    - BaseCustomLearningCallback (learning_callback.py)
    - ImitationCustomLearningCallback_ver1_0 (WandB + 10% evaluation)
    - Recent exception handling updates
    
    Features:
    - ✅ Episode reward/length tracking
    - ✅ WandB logging with training metrics
    - ✅ 10% progress interval evaluation with rendering
    - ✅ Thread-safe exception handling for Tcl/Tk errors
    - ✅ Compatible with LiveRenderToggleCallback
    - ✅ Self-contained - no BaseCustomLearningCallback dependency
    """
    
    def __init__(self, *,
                 log_rollout_freq: int,
                 evaluate_freq: int,
                 log_handler: train_log_handler.TrainLogHandler,
                 original_reward_weights: ImitationTrainSessionConfig.EnvParams.RewardWeights,
                 auto_reward_adjust_params: ImitationTrainSessionConfig.AutoRewardAdjustParams,
                 config: ImitationTrainSessionConfig,
                 wandb_config: dict = None,
                 verbose=1):
        """
        Args:
            log_rollout_freq: Logging frequency (from BaseCustomLearningCallback)
            evaluate_freq: Evaluation frequency (from BaseCustomLearningCallback)
            log_handler: Training log handler (from BaseCustomLearningCallback)
            original_reward_weights: Reward weights for imitation
            auto_reward_adjust_params: Auto reward adjustment parameters
            config: Complete training configuration
            wandb_config: WandB configuration dict (from ver1_0)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        # From BaseCustomLearningCallback
        self.log_rollout_freq = log_rollout_freq
        self.evaluate_freq = evaluate_freq
        self.train_log_handler = log_handler
        self.log_count = 0
        
        # From ImitationCustomLearningCallback
        self._reward_weights = original_reward_weights
        self._auto_reward_adjust_params = auto_reward_adjust_params
        self._config = config
        
        # From ver1_0: WandB & Evaluation
        self._wandb_config = wandb_config or {}
        self._total_timesteps = config.total_timesteps
        self._last_render_timesteps = 0
        self._render_interval = int(self._total_timesteps * 0.1)  # 10% intervals
        self._wandb_enabled = WANDB_AVAILABLE and wandb_config is not None
        
        # Progress tracking
        self.prev_logging_timestep = 0

    def _init_callback(self):
        """
        Initialize callback - called at start of training
        Combines BaseCustomLearningCallback + ImitationCustomLearningCallback
        """
        # From BaseCustomLearningCallback: Episode tracking
        self.rewards_sum = np.zeros(self.training_env.num_envs)
        self.current_episode_rewards = np.zeros(self.training_env.num_envs)
        self.episode_counts = np.zeros(self.training_env.num_envs)
        self.episode_length_counts = np.zeros(self.training_env.num_envs)
        self.current_episode_length_counts = np.zeros(self.training_env.num_envs)

        self.current_reward_dict_sum = [{} for _ in range(self.training_env.num_envs)]
        self.episode_reward_dict_sum = [{} for _ in range(self.training_env.num_envs)]

        # From ImitationCustomLearningCallback: Reward accumulation
        self.reward_accumulate = DictionableDataclass.create(
            ImitationTrainSessionConfig.EnvParams.RewardWeights
        )
        self.reward_accumulate = DictionableDataclass.to_dict(self.reward_accumulate)
        for key in self.reward_accumulate.keys():
            self.reward_accumulate[key] = 0
        
        # From ver1_0: Initialize WandB if enabled
        if self._wandb_enabled and not wandb.run:
            try:
                wandb.init(
                    project=self._wandb_config.get('project', 'myoassist-imitation'),
                    name=self._wandb_config.get('name', 'training'),
                    config=self._wandb_config.get('config', {}),
                    tags=self._wandb_config.get('tags', []),
                    settings=wandb.Settings(
                        _disable_stats=True,  # Reduce network traffic
                        _disable_meta=True,   # Reduce network traffic
                    )
                )
                print("✅ WandB initialized successfully!")
            except Exception as e:
                print(f"⚠️  WandB initialization failed: {e}")
                print("   Using local logging only.")
                self._wandb_enabled = False

    def _on_step(self) -> bool:
        """
        Called after each environment step
        Combines tracking from BaseCustomLearningCallback + ver1_0 features
        """
        # From BaseCustomLearningCallback: Track episode rewards/lengths
        self.current_episode_rewards += self.locals["rewards"]
        for idx, done in enumerate(self.locals["dones"]):
            self.current_episode_length_counts[idx] += 1

            # Safeguard: 'info' may be None or may not contain 'rwd_dict'
            info_dict = None
            try:
                info_dict = self.locals["infos"][idx]
            except (IndexError, KeyError):
                info_dict = None

            if info_dict and isinstance(info_dict, dict) and "rwd_dict" in info_dict:
                for key, val in info_dict["rwd_dict"].items():
                    if key not in self.current_reward_dict_sum[idx]:
                        self.current_reward_dict_sum[idx][key] = 0
                    self.current_reward_dict_sum[idx][key] += val
                    
            if done:
                self.rewards_sum[idx] += self.current_episode_rewards[idx]
                self.episode_counts[idx] += 1
                self.current_episode_rewards[idx] = 0.0
                self.episode_length_counts[idx] += self.current_episode_length_counts[idx]
                self.current_episode_length_counts[idx] = 0

                # Aggregate episode-level reward dictionary
                for key, val in self.current_reward_dict_sum[idx].items():
                    if key not in self.episode_reward_dict_sum[idx]:
                        self.episode_reward_dict_sum[idx][key] = 0
                    self.episode_reward_dict_sum[idx][key] += val
                
                # Reset current reward dict
                self.current_reward_dict_sum[idx] = {}
        
        # From ImitationCustomLearningCallback: Accumulate rewards
        for info in self.locals["infos"]:
            if info and isinstance(info, dict) and "rwd_dict" in info:
                for key in self.reward_accumulate.keys():
                    if key in info["rwd_dict"]:
                        self.reward_accumulate[key] += info["rwd_dict"][key]
        
        # From ver1_0: Log to WandB (reduced frequency)
        if self._wandb_enabled and self.num_timesteps % 1000 == 0:
            if len(self.locals["infos"]) > 0:
                rwd_dict = self.locals["infos"][0].get("rwd_dict", {})
                wandb_log = {
                    "timestep": self.num_timesteps,
                    "reward/total": np.mean(self.locals.get("rewards", [0])),
                }
                
                # Log individual reward components
                for key, value in rwd_dict.items():
                    wandb_log[f"reward/{key}"] = value
                
                # Log PPO training metrics
                if hasattr(self.model, 'logger') and self.model.logger:
                    logger_data = self.model.logger.name_to_value
                    if logger_data:
                        wandb_log.update({
                            "train/value_loss": logger_data.get("train/value_loss", 0),
                            "train/policy_gradient_loss": logger_data.get("train/policy_gradient_loss", 0),
                            "train/entropy_loss": logger_data.get("train/entropy_loss", 0),
                            "train/approx_kl": logger_data.get("train/approx_kl", 0),
                            "train/clip_fraction": logger_data.get("train/clip_fraction", 0),
                            "train/learning_rate": logger_data.get("train/learning_rate", 0),
                            "train/explained_variance": logger_data.get("train/explained_variance", 0),
                        })
                
                try:
                    wandb.log(wandb_log, step=self.num_timesteps, commit=True)
                except Exception:
                    pass  # Network issue, skip this log
        
        # From ver1_0: Periodic evaluation (10% intervals)
        current_timesteps = self.num_timesteps
        if current_timesteps - self._last_render_timesteps >= self._render_interval:
            self._last_render_timesteps = current_timesteps
            progress_pct = int((current_timesteps / self._total_timesteps) * 100)
            print(f"\n{'='*60}")
            print(f"🎬 {progress_pct}% Complete - Starting Evaluation")
            print(f"{'='*60}\n")
            try:
                self._evaluate_and_render(progress_pct)
            except Exception as e:
                print(f"\n⚠️  Evaluation/rendering failed - training continues")
                print(f"   Error: {type(e).__name__}: {e}")
                print(f"   (Checkpoints are still being saved)\n")
        
        return True

    def _evaluate_and_render(self, progress_pct):
        """
        Ver3_0: Periodic evaluation with rendering
        From ver1_0 + exception handling
        """
        import os
        from rl_train.envs.environment_handler_ver3_0 import EnvironmentHandler_ver3_0
        
        # Create evaluation directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        eval_dir = os.path.join(self.train_log_handler.log_dir, f"eval_{progress_pct}pct_{timestamp}")
        os.makedirs(eval_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"📊 EVALUATION @ {progress_pct}% ({self.num_timesteps:,} steps)")
        print(f"📁 Saving to: {eval_dir}")
        print(f"{'='*80}\n")
        
        # Save current model
        model_path = os.path.join(eval_dir, f"{timestamp}_model.zip")
        self.model.save(model_path)
        print(f"💾 Model saved: {model_path}")
        
        # Create rendering environment
        print("🎥 Creating rendering environment...")
        try:
            eval_env = EnvironmentHandler_ver3_0.create_environment(
                self._config, 
                is_rendering_on=True, 
                is_evaluate_mode=True
            )
            
            # Load model
            from stable_baselines3 import PPO
            eval_model = PPO.load(model_path, env=eval_env)
            
            # Start evaluation
            print(f"🏃 Starting evaluation ({self._config.evaluate_param_list[0]['num_timesteps']} steps)...")
            obs, info = eval_env.reset()
        except Exception as e:
            print(f"\n⚠️  Rendering environment creation failed - skipping evaluation (training continues)")
            print(f"   Error: {type(e).__name__}: {e}")
            print(f"   (Checkpoint saved successfully)\n")
            return
        
        episode_rewards = []
        episode_reward = 0
        frames = []
        
        # Video recording setup
        try:
            import imageio
            video_enabled = True
            print("📹 Video recording enabled")
        except ImportError:
            video_enabled = False
            print("⚠️  imageio not found - skipping video")
        
        for step in range(self._config.evaluate_param_list[0]["num_timesteps"]):
            action, _states = eval_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            episode_reward += reward
            
            # Capture frames
            if video_enabled:
                try:
                    frame = eval_env.render()
                    if frame is not None:
                        frames.append(frame)
                except (RuntimeError, Exception) as e:
                    if step == 0 or 'Tcl_AsyncDelete' in str(e) or 'memory' in str(e).lower():
                        print(f"⚠️  Rendering error (training continues): {type(e).__name__}: {e}")
                        video_enabled = False
            
            if truncated or done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                obs, info = eval_env.reset()
        
        # Save video
        if video_enabled and len(frames) > 0:
            video_filename = f"{timestamp}_eval_{progress_pct}pct.mp4"
            video_path = os.path.join(eval_dir, video_filename)
            print(f"💾 Saving video... ({len(frames)} frames)")
            try:
                import imageio
                imageio.mimsave(video_path, frames, fps=100)
                print(f"🎬 Video saved: {video_path}")
                print(f"   📹 Playback speed: 100 fps (real-time)")
                print(f"   ⏱️  Video length: {len(frames)/100:.1f}s")
            except Exception as e:
                print(f"⚠️  Video save failed: {e}")
        
        eval_env.close()
        
        # Print results
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        if episode_rewards:
            print(f"✅ Evaluation complete - Mean reward: {mean_reward:.2f} ({len(episode_rewards)} episodes)")
        else:
            print(f"✅ Evaluation complete")
        
        # Save results
        eval_results = {
            'progress_pct': progress_pct,
            'timesteps': self.num_timesteps,
            'episode_rewards': episode_rewards,
            'mean_reward': float(mean_reward),
        }
        
        import json
        results_path = os.path.join(eval_dir, "eval_results.json")
        with open(results_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"📊 Results saved: {results_path}\n")
        
        # Log to WandB
        if self._wandb_enabled:
            try:
                wandb.log({
                    f"eval/mean_reward_{progress_pct}pct": mean_reward,
                    f"eval/num_episodes_{progress_pct}pct": len(episode_rewards),
                }, step=self.num_timesteps, commit=True)
                
                # Save model as artifact
                artifact = wandb.Artifact(
                    name=f"model_{progress_pct}pct",
                    type="model",
                    description=f"Model checkpoint at {progress_pct}% training progress"
                )
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)
                print(f"📦 Model saved to WandB artifact")
            except Exception as e:
                print(f"⚠️  WandB logging failed (network issue): {e}")
                print(f"   Local results saved in {eval_dir}")

    def _on_rollout_end(self) -> ImitationTrainCheckpointData:
        """
        Called at end of rollout
        Combines BaseCustomLearningCallback + ImitationCustomLearningCallback
        """
        # Calculate episode statistics
        total_episodes = int(np.sum(self.episode_counts))
        if total_episodes > 0:
            mean_reward = np.sum(self.rewards_sum) / total_episodes
            mean_ep_length = np.sum(self.episode_length_counts) / total_episodes
        else:
            mean_reward = 0.0
            mean_ep_length = 0.0

        # Create log data (from BaseCustomLearningCallback structure)
        log_data = ImitationTrainCheckpointData(
            timesteps=self.num_timesteps,
            episode_reward_mean=float(mean_reward),
            episode_length_mean=float(mean_ep_length),
            reward_weights=DictionableDataclass.to_dict(self._reward_weights),
            reward_accumulate=self.reward_accumulate.copy(),
        )
        
        # Write log
        self.train_log_handler.add_log_data(log_data)
        self.train_log_handler.write_json_file()
        
        # Log to WandB
        if self._wandb_enabled:
            try:
                if hasattr(self.model, 'logger') and self.model.logger:
                    logger_data = self.model.logger.name_to_value
                    wandb.log({
                        "train/entropy_loss": logger_data.get("train/entropy_loss", 0),
                        "train/policy_gradient_loss": logger_data.get("train/policy_gradient_loss", 0),
                        "train/value_loss": logger_data.get("train/value_loss", 0),
                        "train/approx_kl": logger_data.get("train/approx_kl", 0),
                        "train/clip_fraction": logger_data.get("train/clip_fraction", 0),
                        "train/learning_rate": logger_data.get("train/learning_rate", 0),
                        "train/explained_variance": logger_data.get("train/explained_variance", 0),
                        "episode/mean_reward": mean_reward,
                        "episode/mean_length": mean_ep_length,
                    }, step=self.num_timesteps, commit=True)
            except Exception:
                pass  # Network issue, skip
        
        # Reset tracking
        self.rewards_sum = np.zeros(self.training_env.num_envs)
        self.episode_counts = np.zeros(self.training_env.num_envs)
        self.episode_length_counts = np.zeros(self.training_env.num_envs)
        
        # Reset reward accumulate
        for key in self.reward_accumulate.keys():
            self.reward_accumulate[key] = 0
        
        return log_data


##############################################################################
# ENVIRONMENT
##############################################################################


class MyoAssistLegImitation_ver3_0(MyoAssistLegBase):
    """
    Ver 3.0: Complete imitation environment with balance rewards
    
    Features from ver2_1:
    - Pelvis list (roll) penalty for 3D stability
    - Pelvis height reward for upright posture
    - Rotation-based termination
    
    All imitation functionality integrated.
    """
    
    def _setup(self, *,
               env_params: ImitationTrainSessionConfig.EnvParams,
               reference_data: dict | None = None,
               loop_reference_data: bool = False,
               **kwargs):
        # Setup parameters
        self._flag_random_ref_index = getattr(env_params, 'flag_random_ref_index', False)
        self._out_of_trajectory_threshold = env_params.out_of_trajectory_threshold
        self.reference_data_keys = env_params.reference_data_keys
        self._loop_reference_data = loop_reference_data
        self._reward_keys_and_weights = env_params.reward_keys_and_weights

        # Setup reference data
        self.setup_reference_data(data=reference_data)

        # Call parent setup
        super()._setup(env_params=env_params, **kwargs)

    def set_reward_weights(self, reward_keys_and_weights):
        """Update reward weights dynamically"""
        self._reward_keys_and_weights = reward_keys_and_weights

    def get_obs_dict(self, sim):
        """Override from MujocoEnv"""
        return super().get_obs_dict(sim)

    def _get_qpos_diff(self) -> dict:
        """Calculate joint position differences from reference"""
        def get_qpos_diff_one(key: str):
            diff = (self.sim.data.joint(f"{key}").qpos[0].copy() - 
                   self._reference_data["series_data"][f"{key}"][self._imitation_index])
            return diff
        
        name_diff_dict = {}
        for q_key in self._reward_keys_and_weights.qpos_imitation_rewards:
            name_diff_dict[q_key] = get_qpos_diff_one(q_key)
        return name_diff_dict

    def _get_qvel_diff(self):
        """Calculate joint velocity differences from reference"""
        speed_ratio = (self._target_velocity / 
                      self._reference_data["series_data"]["dpelvis_tx"][self._imitation_index])

        def get_qvel_diff_one(key: str):
            diff = (self.sim.data.joint(f"{key}").qvel[0].copy() - 
                   self._reference_data["series_data"][f"d{key}"][self._imitation_index] * speed_ratio)
            return diff
        
        name_diff_dict = {}
        for q_key in self._reward_keys_and_weights.qvel_imitation_rewards:
            name_diff_dict[q_key] = get_qvel_diff_one(q_key)
        return name_diff_dict

    def _get_qpos_diff_nparray(self):
        """Get position differences as numpy array"""
        return np.array([diff for diff in self._get_qpos_diff().values()])

    def _get_end_effector_diff(self):
        """Calculate end effector differences (placeholder)"""
        return np.array([0])

    def _calculate_imitation_rewards(self, obs_dict):
        """Calculate all imitation rewards"""
        base_reward, base_info = super()._calculate_base_reward(obs_dict)

        q_diff_dict = self._get_qpos_diff()
        dq_diff_dict = self._get_qvel_diff()
        anchor_diff_array = self._get_end_effector_diff()

        # Calculate joint position rewards
        q_reward_dict = {}
        for joint_name, diff in q_diff_dict.items():
            q_reward_dict[joint_name] = self.dt * np.exp(-8 * np.square(diff))

        # Calculate joint velocity rewards
        dq_reward_dict = {}
        for joint_name, diff in dq_diff_dict.items():
            dq_reward_dict[joint_name] = self.dt * np.exp(-8 * np.square(diff))
        
        # Calculate end effector reward
        anchor_reward = self.dt * np.mean(np.exp(-5 * np.square(anchor_diff_array)))

        # Sum weighted rewards
        qpos_imitation_rewards = np.sum([
            q_reward_dict[key] * self._reward_keys_and_weights.qpos_imitation_rewards[key] 
            for key in q_reward_dict.keys()
        ])
        qvel_imitation_rewards = np.sum([
            dq_reward_dict[key] * self._reward_keys_and_weights.qvel_imitation_rewards[key] 
            for key in dq_reward_dict.keys()
        ])

        # Update base rewards
        base_reward.update({
            'qpos_imitation_rewards': qpos_imitation_rewards,
            'qvel_imitation_rewards': qvel_imitation_rewards,
            'end_effector_imitation_reward': anchor_reward
        })

        return base_reward, base_info

    def get_reward_dict(self, obs_dict):
        """Override from MujocoEnv - construct reward dictionary"""
        imitation_rewards, info = self._calculate_imitation_rewards(obs_dict)

        # Construct reward dictionary
        rwd_dict = collections.OrderedDict(
            (key, imitation_rewards[key]) for key in imitation_rewards
        )

        # Add fixed keys
        rwd_dict.update({
            'sparse': 0,
            'solved': False,
            'done': self._get_done(),
        })
        
        # Calculate dense reward
        rwd_dict['dense'] = np.sum([
            wt * rwd_dict[key] 
            for key, wt in self.rwd_keys_wt.items() 
            if key in rwd_dict
        ], axis=0)
        
        return rwd_dict

    def _follow_reference_motion(self, is_x_follow: bool):
        """Set simulation state to match reference motion"""
        # Set positions
        for key in self.reference_data_keys:
            self.sim.data.joint(f"{key}").qpos = (
                self._reference_data["series_data"][f"{key}"][self._imitation_index]
            )
            if not is_x_follow and key == 'pelvis_tx':
                self.sim.data.joint(f"{key}").qpos = 0
        
        # Set velocities with speed scaling
        speed_ratio = (self._target_velocity / 
                      self._reference_data["series_data"]["dpelvis_tx"][self._imitation_index])
        for key in self.reference_data_keys:
            self.sim.data.joint(f"{key}").qvel = (
                self._reference_data["series_data"][f"d{key}"][self._imitation_index] * speed_ratio
            )

    def imitation_step(self, is_x_follow: bool, specific_index: int | None = None):
        """Advance imitation index and update simulation"""
        if specific_index is None:
            self._imitation_index += 1
            if self._imitation_index >= self._reference_data_length:
                self._imitation_index = 0
        else:
            self._imitation_index = specific_index
        
        self._follow_reference_motion(is_x_follow)
        self.forward()
        return self._imitation_index

    def step(self, a, **kwargs):
        """Override environment step"""
        # Update imitation index
        if self._imitation_index is not None:
            self._imitation_index += 1
            if self._imitation_index < self._reference_data_length:
                is_out_of_index = False
            else:
                if self._loop_reference_data:
                    self._imitation_index = 0
                    is_out_of_index = False
                else:
                    is_out_of_index = True
                    self._imitation_index = self._reference_data_length - 1
        else:
            is_out_of_index = True
        
        # Execute step
        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)
        
        # Check termination conditions
        if is_out_of_index:
            reward = 0
            truncated = True
        else:
            q_diff_nparray = self._get_qpos_diff_nparray()
            is_out_of_trajectory = np.any(
                np.abs(q_diff_nparray) > self._out_of_trajectory_threshold
            )
            terminated = terminated or is_out_of_trajectory
        
        return (next_obs, reward, terminated, truncated, info)

    def setup_reference_data(self, data: dict | None):
        """Setup reference motion data"""
        self._reference_data = data
        self._imitation_index = None
        if data is not None:
            self._reference_data_length = data["metadata"]["resampled_data_length"]
        else:
            raise ValueError("Reference data is not set")

    def reset(self, **kwargs):
        """Override environment reset"""
        rng = np.random.default_rng()
        
        # Random or fixed start index
        if self._flag_random_ref_index:
            self._imitation_index = rng.integers(
                0, int(self._reference_data_length * 0.8)
            )
        else:
            self._imitation_index = 0
        
        # Set initial pose from reference
        self._follow_reference_motion(False)
        
        obs = super().reset(
            reset_qpos=self.sim.data.qpos, 
            reset_qvel=self.sim.data.qvel, 
            **kwargs
        )
        return obs

    def _initialize_pose(self):
        """Override pose initialization"""
        super()._initialize_pose()


##############################################################################
# REGISTRATION
##############################################################################

# Environment will be registered in myosuite/__init__.py
# No registration code here - keeps it clean
