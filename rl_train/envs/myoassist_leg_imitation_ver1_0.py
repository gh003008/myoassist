import collections
import numpy as np
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.utils.data_types import DictionableDataclass
from rl_train.utils import train_log_handler
from stable_baselines3.common.vec_env import SubprocVecEnv
from rl_train.utils.learning_callback import BaseCustomLearningCallback
from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig

# ver1_0: WandB integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. WandB logging disabled.")

################################################################


class ImitationCustomLearningCallback_ver1_0(BaseCustomLearningCallback):
    """
    ver1_0: Enhanced callback with WandB logging and periodic evaluation
    
    Features:
    - WandB logging for rewards, training metrics, and videos
    - 10% progress interval evaluation with rendering
    - Compatible with original MyoAssist structure
    """
    
    def __init__(self, *,
                 log_rollout_freq: int,
                 evaluate_freq: int,
                 log_handler:train_log_handler.TrainLogHandler,
                 original_reward_weights:ImitationTrainSessionConfig.EnvParams.RewardWeights,
                 auto_reward_adjust_params:ImitationTrainSessionConfig.AutoRewardAdjustParams,
                 config: ImitationTrainSessionConfig,
                 wandb_config: dict = None,
                 verbose=1):
        super().__init__(log_rollout_freq=log_rollout_freq, 
                         evaluate_freq=evaluate_freq,
                         log_handler=log_handler,
                         verbose=verbose)
        self._reward_weights = original_reward_weights
        self._auto_reward_adjust_params = auto_reward_adjust_params
        self._config = config
        self._wandb_config = wandb_config or {}
        self._total_timesteps = config.total_timesteps
        self._last_render_timesteps = 0
        self._render_interval = int(self._total_timesteps * 0.1)  # 10% 간격
        self._wandb_enabled = WANDB_AVAILABLE and wandb_config is not None
        

    def _init_callback(self):
        super()._init_callback()

        self.reward_accumulate = DictionableDataclass.create(ImitationTrainSessionConfig.EnvParams.RewardWeights)
        self.reward_accumulate = DictionableDataclass.to_dict(self.reward_accumulate)
        for key in self.reward_accumulate.keys():
            self.reward_accumulate[key] = 0
        
        # ver1_0: Initialize WandB if enabled
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
                print(f"⚠️ WandB 초기화 실패: {e}")
                print("   로컬 로그만 사용합니다.")
                self._wandb_enabled = False
    
    #called after all envs step done
    def _on_step(self) -> bool:
        subprocvec_env:SubprocVecEnv = self.model.get_env()
        
        # Accumulate rewards
        for info in self.locals["infos"]:
            for key in self.reward_accumulate.keys():
                self.reward_accumulate[key] += info["rwd_dict"][key]
        
        # ver1_0: Log to WandB (reduced frequency to avoid network issues)
        if self._wandb_enabled and self.num_timesteps % 1000 == 0:  # Log every 1000 steps
            # Get reward info from first environment
            if len(self.locals["infos"]) > 0:
                rwd_dict = self.locals["infos"][0].get("rwd_dict", {})
                wandb_log = {
                    "timestep": self.num_timesteps,
                    "reward/total": np.mean(self.locals.get("rewards", [0])),
                }
                
                # Log individual reward components
                for key, value in rwd_dict.items():
                    wandb_log[f"reward/{key}"] = value
                
                # Log PPO training metrics if available
                if hasattr(self.model, 'logger') and self.model.logger:
                    logger_data = self.model.logger.name_to_value
                    if logger_data:  # Check if logger has data
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
                except Exception as e:
                    # Network issue, skip this log
                    pass
        
        # ver1_0: Periodic evaluation and rendering (10% intervals)
        current_timesteps = self.num_timesteps
        if current_timesteps - self._last_render_timesteps >= self._render_interval:
            self._last_render_timesteps = current_timesteps
            progress_pct = int((current_timesteps / self._total_timesteps) * 100)
            print(f"\n{'='*60}")
            print(f"🎬 {progress_pct}% 완료 - 중간 평가 및 렌더링 시작")
            print(f"{'='*60}\n")
            self._evaluate_and_render(progress_pct)

        super()._on_step()
            
        return True
    
    def _evaluate_and_render(self, progress_pct):
        """ver1_0: 중간 평가 및 렌더링 수행"""
        import os
        from rl_train.envs.environment_handler import EnvironmentHandler
        
        # 저장 디렉토리 생성
        eval_dir = os.path.join(self.train_log_handler.log_dir, f"eval_{progress_pct}pct")
        os.makedirs(eval_dir, exist_ok=True)
        
        # 현재 모델 임시 저장
        model_path = os.path.join(eval_dir, "temp_model.zip")
        self.model.save(model_path)
        print(f"💾 모델 저장: {model_path}")
        
        # 렌더링 환경 생성
        print("🎥 렌더링 환경 생성 중...")
        eval_env = EnvironmentHandler.create_environment(
            self._config, 
            is_rendering_on=True, 
            is_evaluate_mode=True
        )
        
        # 모델 로드
        from stable_baselines3 import PPO
        eval_model = PPO.load(model_path, env=eval_env)
        
        # 평가 실행
        print(f"🏃 평가 시작 ({self._config.evaluate_param_list[0]['num_timesteps']} steps)...")
        obs, info = eval_env.reset()
        episode_rewards = []
        episode_reward = 0
        frames = []
        
        # 비디오 녹화 준비
        try:
            import imageio
            video_enabled = True
            print("📹 비디오 녹화 활성화")
        except ImportError:
            video_enabled = False
            print("⚠️ imageio 없음 - 비디오 저장 건너뜀")
        
        for step in range(self._config.evaluate_param_list[0]["num_timesteps"]):
            action, _states = eval_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            episode_reward += reward
            
            # 프레임 캡처 (2프레임마다, 30Hz → 15fps)
            if video_enabled and step % 2 == 0:
                try:
                    frame = eval_env.render()
                    if frame is not None:
                        frames.append(frame)
                except Exception as e:
                    if step == 0:
                        print(f"⚠️ 프레임 캡처 실패: {e}")
                        video_enabled = False
            
            if truncated or done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                obs, info = eval_env.reset()
        
        # 비디오 저장
        if video_enabled and len(frames) > 0:
            video_path = os.path.join(eval_dir, "evaluation.mp4")
            print(f"💾 비디오 저장 중... ({len(frames)} 프레임)")
            try:
                imageio.mimsave(video_path, frames, fps=15)
                print(f"🎬 비디오 저장 완료: {video_path}")
            except Exception as e:
                print(f"⚠️ 비디오 저장 실패: {e}")
        
        eval_env.close()
        
        # 결과 출력
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        if episode_rewards:
            print(f"✅ 평가 완료 - 평균 보상: {mean_reward:.2f} (에피소드 {len(episode_rewards)}개)")
        else:
            print(f"✅ 평가 완료")
        
        # 결과 저장
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
        print(f"📊 결과 저장: {results_path}\n")
        
        # ver1_0: Log to WandB
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
                print(f"📦 모델을 WandB artifact로 저장완료")
            except Exception as e:
                print(f"⚠️ WandB 로깅 실패 (네트워크 문제 가능성): {e}")
                print(f"   로컬 결과는 {eval_dir}에 저장됨")

    def _on_rollout_start(self) -> None:
        super()._on_rollout_start()

    def _on_rollout_end(self, write_log: bool = True) -> "ImitationTrainCheckpointData":
        log_data_base = super()._on_rollout_end(write_log=False)
        if log_data_base is None:
            return
        log_data = ImitationTrainCheckpointData(
            **log_data_base.__dict__,
            reward_weights=DictionableDataclass.to_dict(self._reward_weights),
            reward_accumulate=self.reward_accumulate.copy(),
        )
        if write_log:
            self.train_log_handler.add_log_data(log_data)
            self.train_log_handler.write_json_file()
        
        # ver1_0: Log PPO training metrics to WandB
        if self._wandb_enabled:
            try:
                # Get PPO logger info
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
                        "episode/mean_reward": log_data_base.episode_reward_mean,
                        "episode/mean_length": log_data_base.episode_length_mean,
                    }, step=self.num_timesteps, commit=True)
            except Exception as e:
                # Network issue, skip this log
                pass
        
        self.rewards_sum = np.zeros(self.training_env.num_envs)
        self.episode_counts = np.zeros(self.training_env.num_envs)
        self.episode_length_counts = np.zeros(self.training_env.num_envs)
        
        return log_data
        

        ## ARA (Disabled)

        # print(f"DEBUG:: {self.reward_accumulate=}")
        # joint_rewards = {}
        # for key in self.reward_accumulate.keys():
        #     # print(f"DEBUG:: {key=} {self.reward_accumulate[key]=}")
        #     if MyoLeg18Imitation.Q_POS_DIFF_REWARD_KEY_PREFIX in key:
        #         joint_rewards[key] = self.reward_accumulate[key]
        #     self.reward_accumulate[key] = 0
        # reward_mean = 0
        # for key in joint_rewards.keys():
        #     reward_mean += joint_rewards[key]
        # reward_mean /= len(joint_rewards)
        # joint_reward_deviations = {key: (joint_rewards[key] - reward_mean)/reward_mean for key in joint_rewards.keys()}

        # for key in joint_reward_deviations.keys():
        #     new_reward_weight = getattr(self._reward_weights, key) - self._auto_reward_adjust_params.learning_rate * joint_reward_deviations[key]
        #     setattr(self._reward_weights, key, new_reward_weight)
        # subprocvec_env:SubprocVecEnv = self.model.get_env()
        # subprocvec_env.env_method('set_reward_weights', self._reward_weights)
        # print(f"DEBUG:: {self._reward_weights=}")


##############################################################################



class MyoAssistLegImitation(MyoAssistLegBase):
    
    # automatically inherit from MyoAssistLegBase
    # DEFAULT_OBS_KEYS = ['qpos',
    #                     'qvel',
    #                     'act',
    #                     'target_velocity',
    #                     ]

    def _setup(self,*,
            env_params:ImitationTrainSessionConfig.EnvParams,
            reference_data:dict|None = None,
            loop_reference_data:bool = False,
            **kwargs,
        ):
        self._flag_random_ref_index = env_params.flag_random_ref_index
        self._out_of_trajectory_threshold = env_params.out_of_trajectory_threshold
        self.reference_data_keys = env_params.reference_data_keys
        self._loop_reference_data = loop_reference_data
        self._reward_keys_and_weights:ImitationTrainSessionConfig.EnvParams.RewardWeights = env_params.reward_keys_and_weights

        self.setup_reference_data(data=reference_data)

        super()._setup(env_params=env_params,
                       **kwargs,
                       )

        
        
    def set_reward_weights(self, reward_keys_and_weights:TrainSessionConfigBase.EnvParams.RewardWeights):
        self._reward_keys_and_weights = reward_keys_and_weights
    # override from MujocoEnv
    def get_obs_dict(self, sim):
        return super().get_obs_dict(sim)

    def _get_qpos_diff(self) -> dict:

        def get_qpos_diff_one(key:str):
            diff = self.sim.data.joint(f"{key}").qpos[0].copy() - self._reference_data["series_data"][f"q_{key}"][self._imitation_index]
            return diff
        name_diff_dict = {}
        for q_key in self._reward_keys_and_weights.qpos_imitation_rewards:
            name_diff_dict[q_key] = get_qpos_diff_one(q_key)
        return name_diff_dict
    def _get_qvel_diff(self):
        speed_ratio_to_target_velocity = self._target_velocity / self._reference_data["series_data"]["dq_pelvis_tx"][self._imitation_index]

        def get_qvel_diff_one(key:str):
            diff = self.sim.data.joint(f"{key}").qvel[0].copy() - self._reference_data["series_data"][f"dq_{key}"][self._imitation_index] * speed_ratio_to_target_velocity
            return diff
        name_diff_dict = {}
        for q_key in self._reward_keys_and_weights.qvel_imitation_rewards:
            # joint_weight = self._reward_keys_and_weights.qvel_imitation_rewards[q_key]
            name_diff_dict[q_key] = get_qvel_diff_one(q_key)
        return name_diff_dict
    def _get_qpos_diff_nparray(self):
        return np.array([diff for diff in self._get_qpos_diff().values()])
    def _get_end_effector_diff(self):
        # body_pos = self.sim.data.body('pelvis').xpos.copy()
        # diff_array = []
        # for mapping in self.ANCHOR_SIM_TO_REF.values():
        #     sim_anchor = self.sim.data.joint(mapping.sim_name).xanchor.copy() - body_pos
        #     ref_anchor = self._reference_data[mapping.ref_name][self._imitation_index]
        #     diff = np.linalg.norm(sim_anchor - ref_anchor)
        #     diff_array.append(diff)
        # return diff_array
        return np.array([0])
    
    def _calculate_imitation_rewards(self, obs_dict):
        base_reward, base_info = super()._calculate_base_reward(obs_dict)

        q_diff_dict = self._get_qpos_diff()
        dq_diff_dict = self._get_qvel_diff()
        anchor_diff_array = self._get_end_effector_diff()

        # Calculate joint position rewards
        q_reward_dict = {}
        for joint_name, diff in q_diff_dict.items():
            q_reward_dict[joint_name] = self.dt * np.exp(-8 * np.square(diff))

        dq_reward_dict = {}
        for joint_name, diff in dq_diff_dict.items():
            dq_reward_dict[joint_name] = self.dt * np.exp(-8 * np.square(diff))
        
        # Calculate end effector reward
        anchor_reward = self.dt * np.mean(np.exp(-5 * np.square(anchor_diff_array)))

        # Calculate joint imitation rewards sum
        qpos_imitation_rewards = np.sum([q_reward_dict[key] * self._reward_keys_and_weights.qpos_imitation_rewards[key] for key in q_reward_dict.keys()])
        qvel_imitation_rewards = np.sum([dq_reward_dict[key] * self._reward_keys_and_weights.qvel_imitation_rewards[key] for key in dq_reward_dict.keys()])

        # Add new key-value pairs to the base_reward dictionary
        base_reward.update({
            'qpos_imitation_rewards': qpos_imitation_rewards,
            'qvel_imitation_rewards': qvel_imitation_rewards,
            'end_effector_imitation_reward': anchor_reward
        })

        # Use the updated base_reward as imitation_rewards
        imitation_rewards = base_reward
        info = base_info
        return imitation_rewards, info
    

    # override from MujocoEnv
    def get_reward_dict(self, obs_dict):
        # Calculate common rewards
        imitation_rewards, info = self._calculate_imitation_rewards(obs_dict)

        # Construct reward dictionary
        # Automatically add all imitation_rewards items to rwd_dict
        rwd_dict = collections.OrderedDict((key, imitation_rewards[key]) for key in imitation_rewards)

        # Add additional fixed keys
        rwd_dict.update({
            'sparse': 0,
            'solved': False,
            'done': self._get_done(),
        })
        # Calculate final reward
        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items() if key in rwd_dict], axis=0)
        
        return rwd_dict
    

    def _follow_reference_motion(self, is_x_follow:bool):
        for key in self.reference_data_keys:
            self.sim.data.joint(f"{key}").qpos = self._reference_data["series_data"][f"q_{key}"][self._imitation_index]
            if not is_x_follow and key == 'pelvis_tx':
                self.sim.data.joint(f"{key}").qpos = 0
            # if key == 'pelvis_ty':
            #     self.sim.data.joint(f"{key}").qpos += 0.05
        speed_ratio_to_target_velocity = self._target_velocity / self._reference_data["series_data"]["dq_pelvis_tx"][self._imitation_index]
        for key in self.reference_data_keys:
            self.sim.data.joint(f"{key}").qvel = self._reference_data["series_data"][f"dq_{key}"][self._imitation_index] * speed_ratio_to_target_velocity
    def imitation_step(self, is_x_follow:bool, specific_index:int|None = None):
        if specific_index is None:
            self._imitation_index += 1
            if self._imitation_index >= self._reference_data_length:
                self._imitation_index = 0
        else:
            self._imitation_index = specific_index
        self._follow_reference_motion(is_x_follow)
        # should call this but I don't know why
        # next_obs, reward, terminated, truncated, info = super().step(np.zeros(self.sim.model.nu))
        # return (next_obs, reward, False, False, info)
        self.forward()
        return self._imitation_index
        # pass
    
    # override
    def step(self, a, **kwargs):
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
        
        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)
        if is_out_of_index:
            reward = 0
            truncated = True
        else:
            q_diff_nparray:np.ndarray = self._get_qpos_diff_nparray()
            is_out_of_trajectory = np.any(np.abs(q_diff_nparray) >self._out_of_trajectory_threshold)
            terminated = terminated or is_out_of_trajectory
        
        return (next_obs, reward, terminated, truncated, info)
        
    
    def setup_reference_data(self, data:dict|None):
        self._reference_data = data
        self._imitation_index = None
        if data is not None:
            # self._follow_reference_motion(False)
            self._reference_data_length = self._reference_data["metadata"]["resampled_data_length"]
        else:
            raise ValueError("Reference data is not set")

    def reset(self, **kwargs):
        rng = np.random.default_rng()# TODO: refactoring random to use seed
        
        if self._flag_random_ref_index:
            self._imitation_index = rng.integers(0, int(self._reference_data_length * 0.8))
        else:
            self._imitation_index = 0
        # generate random targets
        # new_qpos = self.generate_qpos()# TODO: should set qvel too.
        # self.sim.data.qpos = new_qpos
        self._follow_reference_motion(False)
        
        obs = super().reset(reset_qpos= self.sim.data.qpos, reset_qvel=self.sim.data.qvel, **kwargs)
        return obs

    # override
    def _initialize_pose(self):
        super()._initialize_pose()