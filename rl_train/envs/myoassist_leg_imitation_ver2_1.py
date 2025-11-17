"""
MyoAssist Leg Imitation Environment ver2_1

Created: 2024-11-17 (Ver2_1 based on Ver2_0 Karico)
Purpose: Add balancing/stability rewards for 3D model convergence

Changes from ver2_0 (Karico):
1. [HIGH] pelvis_list_penalty: Penalize roll (lateral tilt) for 3D stability
2. [MEDIUM] pelvis_height_reward: Encourage upright posture
3. [MEDIUM] rotation_based_termination: Early termination on excessive rotation

Base: Ver2_0 Karico (stable training release)
Extension: 3D balancing rewards from previous ver1_1 work
"""

import collections
import numpy as np
from rl_train.envs.myoassist_leg_imitation_ver1_0 import (
    MyoAssistLegImitation, 
    ImitationCustomLearningCallback_ver1_0
)
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from myosuite.utils.quat_math import quat2mat


# 251117_Ver2_1: Callback extends ver1_0 (Karico base)
class ImitationCustomLearningCallback_ver2_1(ImitationCustomLearningCallback_ver1_0):
    """
    ver2_1: Same as ver1_0 (Karico) with support for new balance rewards
    
    No changes needed from ver1_0 - just inheritance for consistency
    """
    pass


##############################################################################


class MyoAssistLegImitation_ver2_1(MyoAssistLegImitation):
    """
    251117_Ver2_1: Ver2_0 Karico + 3D balancing rewards
    
    New features:
    - pelvis_list_penalty: Strong penalty for roll (lateral tilt)
    - pelvis_height_reward: Encourage staying upright
    - rotation_based_termination: Stop episode on excessive rotation
    - Compatible with Ver2_0 Karico's stable training pipeline
    """
    
    def _setup(self, *, 
               env_params: ImitationTrainSessionConfig.EnvParams,
               reference_data: dict | None = None,
               loop_reference_data: bool = False,
               **kwargs):
        """251117_Ver2_1: Initialize with balance penalty parameters"""
        
        # Store parameters before parent setup
        self._max_rot = kwargs.get('max_rot', env_params.__dict__.get('max_rot', 0.6))  # default: cos(53°) ≈ 0.6
        self.safe_height = env_params.safe_height
        
        # Call parent setup (ver1_0 Karico)
        super()._setup(
            env_params=env_params,
            reference_data=reference_data,
            loop_reference_data=loop_reference_data,
            **kwargs
        )
    
    def _calculate_balancing_rewards(self):
        """
        251117_Ver2_1: Calculate balancing/stability rewards for 3D model
        
        Returns:
            dict: Balance reward components
        """
        balance_rewards = {}
        
        # [HIGH Priority] pelvis_list_penalty: Roll (lateral tilt) penalty
        # In 3D, lateral stability is critical. Roll should stay near 0.
        pelvis_list = self.sim.data.joint('pelvis_list').qpos[0]  # roll angle
        
        # Quadratic penalty for stability (no exp to avoid NaN)
        pelvis_list_penalty = self.dt * (-np.square(pelvis_list))
        balance_rewards['pelvis_list_penalty'] = float(pelvis_list_penalty)
        
        # [MEDIUM Priority] pelvis height reward (encourage staying upright)
        pelvis_height = self.sim.data.body('pelvis').xpos[2]
        target_height = 0.9  # typical standing height
        # Reduced exponential to prevent reward explosion
        height_reward = self.dt * np.exp(-2.0 * np.square(pelvis_height - target_height))
        balance_rewards['pelvis_height_reward'] = float(height_reward)
        
        return balance_rewards
    
    def _check_rotation_termination(self):
        """
        251117_Ver2_1: [MEDIUM Priority] Check if excessive rotation occurred
        
        Terminates episode if pelvis rotates too much from forward direction.
        This prevents the agent from learning unstable gaits that eventually fall.
        
        Returns:
            bool: True if episode should terminate
        """
        # Get pelvis orientation (quaternion)
        pelvis_quat = self.sim.data.body('pelvis').xquat  # [w, x, y, z]
        
        # Convert to rotation matrix
        rot_mat = quat2mat(pelvis_quat)
        
        # Forward vector in world frame (should be close to [0, 0, 1])
        forward_vec = rot_mat[:, 2]  # Third column = forward direction
        
        # Check if forward vector is pointing too far from [0, 0, 1]
        # Dot product with world forward [0, 0, 1]
        forward_alignment = forward_vec[2]  # Just the Z component
        
        # If alignment < threshold, pelvis is rotated too much
        if forward_alignment < self._max_rot:
            return True
        
        return False
    
    def get_reward_dict(self, obs_dict):
        """
        251117_Ver2_1: Override to add balancing rewards
        
        Extends parent reward calculation with 3D stability components
        """
        # Get base rewards from parent (ver1_0 Karico)
        rwd_dict = super().get_reward_dict(obs_dict)
        
        # Add balancing rewards
        balance_rewards = self._calculate_balancing_rewards()
        rwd_dict.update(balance_rewards)
        
        # Recalculate dense reward with new components
        rwd_dict['dense'] = np.sum([
            wt * rwd_dict[key] 
            for key, wt in self.rwd_keys_wt.items() 
            if key in rwd_dict
        ], axis=0)
        
        return rwd_dict
    
    def _get_done(self):
        """
        251117_Ver2_1: Override to add rotation-based termination
        
        Checks both height and rotation criteria
        """
        # Check height termination (from parent)
        pelvis_height = self.sim.data.joint('pelvis_ty').qpos[0].copy()
        if pelvis_height < self.safe_height:
            return True
        
        # Check rotation termination
        if self._check_rotation_termination():
            return True
        
        return False
