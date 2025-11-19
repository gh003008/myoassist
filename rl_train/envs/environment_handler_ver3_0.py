"""
environment_handler_ver3_0.py
=============================
SIMPLIFIED Environment Handler for Ver 3.0 System

Changes from original environment_handler.py:
- Only imports myoassist_leg_imitation_ver3_0 (removes ver1_0, original versions)
- Simplified get_callback() to only use Ver 3.0 callback
- Removed use_ver1_0 parameter (always uses Ver 3.0)
- Keeps all reference data loading logic (NPZ/JSON, resampling)
- Keeps all model creation logic
- Simplified callback creation (no version branching)

Created: 2024
"""

import gymnasium as gym
import numpy as np
import os
import json
import warnings

from myosuite.utils.quat_math import quat2mat, mat2quat, euler2quat
from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.utils.data_types import DictionableDataclass


class EnvironmentHandler:
    """
    Ver 3.0 Environment Handler - Simplified for single version system
    """

    @staticmethod
    def create_environment(config: TrainSessionConfigBase, load_reference_data=False, enable_balance_reward=False):
        """
        Create environment from config
        
        Args:
            config: Training session configuration
            load_reference_data: Whether to load reference data
            enable_balance_reward: Whether to enable balance reward (Ver 3.0 feature)
        """
        from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig

        # Load reference data if needed
        ref_data_dict = None
        if load_reference_data:
            ref_data_dict = EnvironmentHandler.load_reference_data(config)

        # Prepare gym.make arguments
        gym_make_args = {
            'seed': config.env_params.seed,
            'model_path': config.env_params.model_path,
            'env_params': config.env_params,
            'is_evaluate_mode': False
        }
        
        # Add reference data if it exists
        if ref_data_dict is not None:
            gym_make_args['reference_data'] = ref_data_dict
        
        # Add Ver 3.0 specific args
        if isinstance(config, ImitationTrainSessionConfig):
            gym_make_args['enable_balance_reward'] = enable_balance_reward
        
        # Create environment using gym.make (supports both VecEnv and single env)
        try:
            from stable_baselines3.common.vec_env import SubprocVecEnv
            
            if config.env_params.num_envs == 1:
                env = gym.make(config.env_params.env_id, **gym_make_args).unwrapped
                config.ppo_params.n_steps = config.ppo_params.batch_size
            else:
                env = SubprocVecEnv([
                    lambda: gym.make(config.env_params.env_id, **gym_make_args).unwrapped 
                    for _ in range(config.env_params.num_envs)
                ])
            
            print("✅ Created Ver 3.0 environment (simplified callback system)")
            if enable_balance_reward:
                print("   🎯 Balance reward ENABLED (Ver 3.0 feature)")
            
            return env
        except Exception as e:
            new_message = str(e)[:1000]
            e.args = (new_message,)
            raise e

    @staticmethod
    def load_reference_data(config: TrainSessionConfigBase):
        """
        Load reference data from NPZ or JSON file
        
        Handles:
        - NPZ format: Environment format (series_data, metadata) or MuJoCo format (q_ref, joint_names)
        - JSON format: Direct dictionary
        - Automatic format detection and conversion
        - Resampling to match control_framerate
        - Critical fixes: pelvis_ty offset, hip_adduction sign flip
        """
        from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
        
        if not isinstance(config, ImitationTrainSessionConfig):
            raise ValueError("Reference data loading only supported for ImitationTrainSessionConfig")

        print("=" * 80)
        print("📂 LOADING REFERENCE DATA")
        print(f"   Path: {config.env_params.reference_data_path}")
        
        if config.env_params.reference_data_path.endswith(".npz"):
            ref_data_dict = dict(np.load(config.env_params.reference_data_path, allow_pickle=True))
            
            # Convert arrays to proper format
            for key in ref_data_dict.keys():
                if isinstance(ref_data_dict[key], np.ndarray) and ref_data_dict[key].ndim == 0:
                    ref_data_dict[key] = ref_data_dict[key].item()
            
            # Format detection and conversion
            if 'series_data' in ref_data_dict and 'metadata' in ref_data_dict:
                # Format B: Environment format - already correct
                print("📦 Detected ENVIRONMENT format (series_data, metadata)")
                
                # CRITICAL: Remove q_ prefix from environment format keys!
                # Some environment format files still have q_ prefix in keys
                series_data = ref_data_dict['series_data']
                first_key = list(series_data.keys())[0]
                
                if first_key.startswith('q_') or first_key.startswith('dq_'):
                    print("   ⚠️  Detected q_ prefix in environment format keys - removing...")
                    new_series_data = {}
                    for key, value in series_data.items():
                        if key.startswith('q_'):
                            new_key = key[2:]  # Remove 'q_'
                        elif key.startswith('dq_'):
                            new_key = 'd' + key[3:]  # 'dq_pelvis_tx' → 'dpelvis_tx'
                        else:
                            new_key = key
                        new_series_data[new_key] = value
                    ref_data_dict['series_data'] = new_series_data
                    print(f"   ✅ Removed q_ prefix from {len(new_series_data)} keys")
                else:
                    print("   NO prefix removal needed - keys already clean!")
                
                # CRITICAL: Apply pelvis_ty offset for environment format!
                # Environment format NPZ has ground-relative pelvis_ty
                # Need to add +0.91m offset to match MuJoCo "stand" keyframe height
                if 'pelvis_ty' in ref_data_dict['series_data']:
                    ref_data_dict['series_data']['pelvis_ty'] = ref_data_dict['series_data']['pelvis_ty'] + 0.91
                    print(f"   ⚠️  Applied pelvis_ty offset: +0.91m (ground-relative → MuJoCo model height)")
                elif 'q_pelvis_ty' in ref_data_dict['series_data']:
                    # Shouldn't happen after prefix removal, but just in case
                    ref_data_dict['series_data']['pelvis_ty'] = ref_data_dict['series_data']['q_pelvis_ty'] + 0.91
                    del ref_data_dict['series_data']['q_pelvis_ty']
                    print(f"   ⚠️  Applied pelvis_ty offset: +0.91m (found q_pelvis_ty, converted to pelvis_ty)")
                
                # CRITICAL: Fix left hip adduction sign!
                # Left hip adduction needs sign flip for correct visualization
                if 'hip_adduction_l' in ref_data_dict['series_data']:
                    ref_data_dict['series_data']['hip_adduction_l'] = -ref_data_dict['series_data']['hip_adduction_l']
                    print(f"   🔄 Applied sign flip to hip_adduction_l (left hip ab/adduction fix)")
                
            elif 'q_ref' in ref_data_dict and 'joint_names' in ref_data_dict:
                # Format A: MuJoCo renderer format - needs conversion
                print("📦 Detected MUJOCO RENDERER format (q_ref, joint_names with q_ prefix)")
                print("   Converting to environment format (series_data, metadata)...")
                
                q_ref = ref_data_dict['q_ref']
                joint_names = ref_data_dict['joint_names']
                
                # Create series_data dict
                series_data = {}
                for i, joint_name in enumerate(joint_names):
                    joint_name_str = str(joint_name)
                    
                    # CRITICAL: Environment expects joint names WITHOUT "q_" prefix
                    if joint_name_str.startswith('q_'):
                        env_joint_name = joint_name_str[2:]  # Remove "q_" prefix
                    else:
                        env_joint_name = joint_name_str
                    
                    # CRITICAL: HDF5 data has pelvis_ty relative to ground (0.0)
                    # MuJoCo model expects pelvis_ty relative to "stand" keyframe (~0.91m)
                    if env_joint_name == 'pelvis_ty':
                        series_data[env_joint_name] = q_ref[:, i] + 0.91
                        print(f"   ⚠️  Applied pelvis_ty offset: +0.91m (HDF5 ground-relative → MuJoCo model height)")
                    else:
                        series_data[env_joint_name] = q_ref[:, i]
                    
                    # Velocity data (approximate with finite difference)
                    dq = np.gradient(q_ref[:, i], axis=0) * 100  # 100 Hz sampling
                    series_data[f'd{env_joint_name}'] = dq
                
                # Create metadata
                metadata = {
                    'data_length': q_ref.shape[0],
                    'sample_rate': 100,  # HDF5 converted data is 100 Hz
                    'dof': q_ref.shape[1],
                    'model_type': '3D',
                    'resampled_data_length': q_ref.shape[0],
                    'resampled_sample_rate': 100,
                }
                
                # Replace with converted format
                ref_data_dict = {
                    'series_data': series_data,
                    'metadata': metadata
                }
                print(f"   ✅ Converted {q_ref.shape[0]} frames, {q_ref.shape[1]} DOF @ 100 Hz")
                
        elif config.env_params.reference_data_path.endswith(".json"):
            with open(config.env_params.reference_data_path, 'r') as f:
                ref_data_dict = json.load(f)
        else:
            raise ValueError("Unsupported file format. Please use either .npz or .json.")

        # CRITICAL: Only resample if control_framerate != sample_rate
        if ref_data_dict["metadata"]["sample_rate"] != config.env_params.control_framerate:
            print(f"   🔄 Resampling from {ref_data_dict['metadata']['sample_rate']} Hz to {config.env_params.control_framerate} Hz...")
            for key in ref_data_dict["series_data"].keys():
                original_data_length = len(ref_data_dict["series_data"][key])
                original_sample_rate = ref_data_dict["metadata"]["sample_rate"]
                original_x = np.linspace(0, original_data_length - 1, original_data_length)

                new_sample_rate = config.env_params.control_framerate
                new_length = int(original_data_length * new_sample_rate / original_sample_rate)
                new_x = np.linspace(0, original_data_length - 1, new_length)
                ref_data_dict["series_data"][key] = np.interp(new_x, original_x, ref_data_dict["series_data"][key])
            
            ref_data_dict["metadata"]["resampled_data_length"] = new_length
            ref_data_dict["metadata"]["resampled_sample_rate"] = new_sample_rate
            print(f"   ✅ Resampled to {new_length} frames @ {new_sample_rate} Hz")
        else:
            # No resampling needed
            print(f"   ℹ️  No resampling needed (already @ {config.env_params.control_framerate} Hz)")
            if "resampled_data_length" not in ref_data_dict["metadata"]:
                ref_data_dict["metadata"]["resampled_data_length"] = ref_data_dict["metadata"]["data_length"]
            if "resampled_sample_rate" not in ref_data_dict["metadata"]:
                ref_data_dict["metadata"]["resampled_sample_rate"] = ref_data_dict["metadata"]["sample_rate"]

        return ref_data_dict

    @staticmethod
    def get_config_type_from_session_id(session_id):
        """Get config class type from session ID"""
        from rl_train.train.train_configs.config import TrainSessionConfigBase
        from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
        from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
        
        print(f"session_id: {session_id}")
        if session_id == 'myoAssistLeg-v0':
            return TrainSessionConfigBase
        elif session_id in ['myoAssistLegImitation-v0']:
            return ImitationTrainSessionConfig
        elif session_id in ['myoAssistLegImitationExo-v0', 'myoAssistLegImitationExo-v2_1']:
            return ExoImitationTrainSessionConfig
        raise ValueError(f"Invalid session id: {session_id}")
        
    @staticmethod
    def get_session_config_from_path(config_path, class_type):
        """Load session config from JSON file"""
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            session_config = DictionableDataclass.create(class_type, config_dict)
        return session_config

    @staticmethod
    def get_callback(config, train_log_handler, wandb_config=None, enable_live_render=True, 
                     enable_balance_reward=False):
        """
        Get Ver 3.0 callback for training
        
        Args:
            config: Training session configuration
            train_log_handler: Log handler for training
            wandb_config: WandB configuration dict (required for Ver 3.0)
            enable_live_render: If True, add LiveRenderToggleCallback for keyboard control
            enable_balance_reward: If True, enable balance reward in callback
        
        Returns:
            Callback or CallbackList with Ver 3.0 callback and optional LiveRenderToggleCallback
        """
        from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
        from stable_baselines3.common.callbacks import CallbackList
        
        # Ver 3.0: Always use ver3_0 callback (no version branching)
        if isinstance(config, ImitationTrainSessionConfig):
            from rl_train.envs.myoassist_leg_imitation_ver3_0 import ImitationCustomLearningCallback_ver3_0
            
            # Ver 3.0 callback with all integrated features
            main_callback = ImitationCustomLearningCallback_ver3_0(
                log_rollout_freq=config.logger_params.logging_frequency,
                evaluate_freq=config.logger_params.evaluate_frequency,
                log_handler=train_log_handler,
                original_reward_weights=config.env_params.reward_keys_and_weights,
                auto_reward_adjust_params=config.auto_reward_adjust_params,
                config=config,
                wandb_config=wandb_config,
                enable_balance_reward=enable_balance_reward,  # Ver 3.0 feature
            )
            print("✅ Using Ver 3.0 callback (WandB + 10% evaluation + exception handling)")
            if enable_balance_reward:
                print("   🎯 Balance reward enabled in callback")
        else:
            # Non-imitation environments: use base callback
            from rl_train.utils import learning_callback
            main_callback = learning_callback.BaseCustomLearningCallback(
                log_rollout_freq=config.logger_params.logging_frequency,
                evaluate_freq=config.logger_params.evaluate_frequency,
                log_handler=train_log_handler,
                total_timesteps=config.total_timesteps,
            )
        
        # Add LiveRenderToggleCallback if enabled
        if enable_live_render:
            from rl_train.run_train import LiveRenderToggleCallback
            live_render_callback = LiveRenderToggleCallback(
                num_envs=config.env_params.num_envs,
                start_index=0,
                render_every_n_steps=1,
                verbose=1
            )
            print("✅ Live rendering enabled (press 'o' to toggle, 'v' to pause, 'm/n' to switch env)")
            return CallbackList([main_callback, live_render_callback])
        else:
            return main_callback

    @staticmethod
    def get_stable_baselines3_model(config: TrainSessionConfigBase, env, trained_model_path: str | None = None):
        """
        Create or load Stable Baselines3 PPO model
        
        Args:
            config: Training session configuration
            env: Gym environment
            trained_model_path: Path to trained model (optional)
        """
        import stable_baselines3
        from rl_train.train.policies.rl_agent_human import HumanActorCriticPolicy
        from rl_train.train.policies.rl_agent_exo import HumanExoActorCriticPolicy
        
        # Select policy class based on environment type
        if config.env_params.env_id in ["myoAssistLegImitationExo-v0", "myoAssistLegImitationExo-v2_1"]:
            policy_class = HumanExoActorCriticPolicy
            print(f"Using HumanExoActorCriticPolicy")
        else:
            policy_class = HumanActorCriticPolicy
            print(f"Using HumanActorCriticPolicy")
        
        # Load existing model or create new one
        if trained_model_path is not None:
            print(f"Loading trained model from {trained_model_path}")
            model = stable_baselines3.PPO.load(
                trained_model_path,
                env=env,
                custom_objects={"policy_class": policy_class},
            )
        elif config.env_params.prev_trained_policy_path:
            print(f"Loading previous trained policy from {config.env_params.prev_trained_policy_path}")
            model = stable_baselines3.PPO.load(
                config.env_params.prev_trained_policy_path,
                env=env,
                custom_objects={"policy_class": policy_class},
                verbose=2,
                **DictionableDataclass.to_dict(config.ppo_params),
            )
            # Reset network if specified
            model.policy.reset_network(
                reset_shared_net=config.policy_params.custom_policy_params.reset_shared_net_after_load,
                reset_policy_net=config.policy_params.custom_policy_params.reset_policy_net_after_load,
                reset_value_net=config.policy_params.custom_policy_params.reset_value_net_after_load
            )
        else:
            # Create new model from scratch
            model = stable_baselines3.PPO(
                policy=policy_class,
                env=env,
                policy_kwargs=DictionableDataclass.to_dict(config.policy_params),
                verbose=2,
                **DictionableDataclass.to_dict(config.ppo_params),
            )
        return model

    @staticmethod
    def updateconfig_from_model_policy(config, model):
        """Update config with model policy information (placeholder)"""
        pass
