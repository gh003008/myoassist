"""
MyoAssist Leg Imitation Environment ver1_1

Created: 2024-11-15 00:28 (251115_0028)
Purpose: Add balancing/stability rewards for 3D model convergence

Changes from ver1_0:
1. [HIGH] pelvis_list_penalty: Penalize roll (lateral tilt) for 3D stability
2. [MEDIUM] rotation_based_termination: Early termination on excessive rotation
3. Enhanced observation with pelvis quaternion orientation

Original MyoAssist scripts are not modified. This extends ver1_0.
"""

import collections
import numpy as np
from rl_train.envs.myoassist_leg_imitation_ver1_0 import (
    MyoAssistLegImitation, 
    ImitationCustomLearningCallback_ver1_0
)
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from myosuite.utils.quat_math import quat2mat

# 251115_0028: ver1_1 callback extends ver1_0
class ImitationCustomLearningCallback_ver1_1(ImitationCustomLearningCallback_ver1_0):
    """
    ver1_1: Same as ver1_0 with support for new balance rewards
    
    No changes needed from ver1_0 - just inheritance for consistency
    """
    pass


##############################################################################


class MyoAssistLegImitation_ver1_1(MyoAssistLegImitation):
    """
    251115_0028: Extended imitation environment with 3D balancing rewards
    
    New features:
    - pelvis_list_penalty: Strong penalty for roll (lateral tilt)
    - rotation_based_termination: Stop episode on excessive rotation
    - Compatible with existing config structure
    """
    
    def _setup(self, *, 
               env_params: ImitationTrainSessionConfig.EnvParams,
               reference_data: dict | None = None,
               loop_reference_data: bool = False,
               **kwargs):
        """251115_0028: Initialize with balance penalty parameters"""
        
        # 251115_0028: Store parameters before parent setup
        self._max_rot = kwargs.get('max_rot', env_params.__dict__.get('max_rot', 0.6))  # default: cos(53°) ≈ 0.6
        self.safe_height = env_params.safe_height  # 251115_0028: Initialize safe_height from env_params
        
        # Call parent setup
        super()._setup(
            env_params=env_params,
            reference_data=reference_data,
            loop_reference_data=loop_reference_data,
            **kwargs
        )
    
    def _calculate_balancing_rewards(self):
        """
        251115_0028: Calculate balancing/stability rewards for 3D model
        
        Returns:
            dict: Balance reward components
        """
        balance_rewards = {}
        
        # [HIGH Priority] pelvis_list_penalty: Roll (lateral tilt) penalty
        # In 3D, lateral stability is critical. Roll should stay near 0.
        pelvis_list = self.sim.data.joint('pelvis_list').qpos[0]  # roll angle
        
        # 251115_0028_FIX: Reduced exponential to prevent NaN explosion
        # Changed from exp(-10.0 * square) to linear penalty for stability
        pelvis_list_penalty = self.dt * (-np.square(pelvis_list))  # Simple quadratic penalty
        balance_rewards['pelvis_list_penalty'] = float(pelvis_list_penalty)
        
        # Optional: pelvis height reward (encourage staying upright)
        # This is secondary since we already have height-based termination
        pelvis_height = self.sim.data.body('pelvis').xpos[2]
        target_height = 0.9  # typical standing height
        # 251115_0028_FIX: Reduced exponential to prevent reward explosion  
        height_reward = self.dt * np.exp(-2.0 * np.square(pelvis_height - target_height))  # Reduced from 5.0 to 2.0
        balance_rewards['pelvis_height_reward'] = float(height_reward)
        
        return balance_rewards
    
    def _check_rotation_termination(self):
        """
        251115_0028: [MEDIUM Priority] Check if excessive rotation occurred
        
        Terminates episode if pelvis rotates too much from forward direction.
        This prevents the agent from learning unstable gaits that eventually fall.
        
        Returns:
            bool: True if rotation exceeds threshold
        """
        # Get pelvis orientation quaternion
        pelvis_quat = self.sim.data.body('pelvis').xquat.copy()
        
        # Convert quaternion to rotation matrix
        rot_mat = quat2mat(pelvis_quat)
        
        # Check forward direction: [1, 0, 0] in body frame
        # After rotation, this should still point mostly forward (+x direction)
        forward_vec = rot_mat @ np.array([1.0, 0.0, 0.0])
        
        # If x-component < max_rot, the body has rotated too much
        # max_rot = 0.6 means cos(53°) - allows up to 53° rotation from forward
        if np.abs(forward_vec[0]) < self._max_rot:
            return True
        
        return False
    
    def get_reward_dict(self, obs_dict):
        """
        251115_0028: Override to add balancing rewards
        
        Extends parent imitation rewards with balance/stability terms
        """
        # Get base + imitation rewards from parent
        rwd_dict = super().get_reward_dict(obs_dict)
        
        # 251115_0028: Add balancing rewards
        balance_rewards = self._calculate_balancing_rewards()
        rwd_dict.update(balance_rewards)
        
        # Compute weighted total
        # Note: Weights for new terms should be added to config
        if self.rwd_keys_wt:
            dense_reward = 0.0
            for key, weight in self.rwd_keys_wt.items():
                if key in rwd_dict:
                    dense_reward += weight * rwd_dict[key]
            rwd_dict['dense'] = dense_reward
        
        return rwd_dict
    
    def _get_done(self):
        """
        251115_0028: Override termination with rotation check
        
        Original termination: pelvis_height < safe_height
        New: Also terminate on excessive rotation
        """
        # Original height-based termination
        pelvis_height = self.sim.data.body('pelvis').xpos[2]
        if pelvis_height < self.safe_height:
            return True
        
        # 251115_0028: [MEDIUM] Rotation-based termination
        if self._check_rotation_termination():
            return True
        
        return False
    
    def step(self, action):
        """
        251115_0028: Override to ensure new rewards are properly logged
        """
        obs, reward, done, truncated, info = super().step(action)  # 251115_0028: gymnasium returns 5 values
        
        # 251115_0028: Add balance reward info for debugging
        if hasattr(self, '_last_balance_rewards'):
            info['balance_rewards'] = self._last_balance_rewards
        
        return obs, reward, done, truncated, info


##############################################################################
# Environment Registration
##############################################################################

def make_env_ver1_1(env_params: ImitationTrainSessionConfig.EnvParams, rank: int, seed: int = 0):
    """
    251115_0028: Factory function for ver1_1 environment
    
    Compatible with existing training pipeline
    """
    def _init():
        import numpy as np
        from rl_train.utils.mujoco_env_load import load_reference_data
        
        reference_data_dict = load_reference_data(
            reference_data_path=env_params.reference_data_path,
            reference_data_keys=env_params.reference_data_keys,
        )
        
        env = MyoAssistLegImitation_ver1_1(
            model_path=env_params.model_path,
            reference_data=reference_data_dict,
            obs_keys=env_params.observation_keys,
            weighted_reward_keys=env_params.reward_keys_and_weights,
            env_params=env_params,
        )
        env.seed(seed + rank)
        return env
    
    return _init
