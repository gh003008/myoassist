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
# Ver2_1: Import from ver1_0 (Karico base)
from rl_train.envs.myoassist_leg_imitation_ver1_0 import (
    MyoAssistLegImitation,  # Ver1_0 Karico base class
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
        self._step_count = 0  # Track steps to skip rotation check during initialization
        
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
        
        Checks both height and rotation criteria.
        Skip rotation check for first 10 steps to allow initialization.
        """
        # Check height termination (from parent)
        pelvis_height = self.sim.data.joint('pelvis_ty').qpos[0].copy()
        if pelvis_height < self.safe_height:
            return True
        
        # Skip rotation check during initialization (first 10 steps)
        if self._step_count < 10:
            return False
        
        # Check rotation termination
        if self._check_rotation_termination():
            return True
        
        return False
    
    def step(self, a, **kwargs):
        """251117_Ver2_1: Override to track step count"""
        result = super().step(a, **kwargs)
        self._step_count += 1
        return result
    
    def reset(self, **kwargs):
        """251117_Ver2_1: Override to reset step count"""
        self._step_count = 0
        return super().reset(**kwargs)
    
    def _get_qvel_diff(self):
        """
        251117_Ver2_1: Override to handle divide-by-zero in velocity calculation
        
        Fixes issue where reference velocity can be zero at initialization.
        """
        # Get reference velocity with epsilon to prevent divide-by-zero
        # Access series_data without "q_" prefix (dpelvis_tx instead of dq_pelvis_tx)
        ref_velocity = self._reference_data["series_data"]["dpelvis_tx"][self._imitation_index]
        speed_ratio_to_target_velocity = self._target_velocity / (ref_velocity + 1e-8)

        def get_qvel_diff_one(key:str):
            # Access series_data without "q_" prefix (already removed in environment_handler)
            diff = self.sim.data.joint(f"{key}").qvel[0].copy() - self._reference_data["series_data"][f"d{key}"][self._imitation_index] * speed_ratio_to_target_velocity
            return diff
        
        name_diff_dict = {}
        for q_key in self._reward_keys_and_weights.qvel_imitation_rewards:
            name_diff_dict[q_key] = get_qvel_diff_one(q_key)
        
        return name_diff_dict
    
    def _follow_reference_motion(self, is_x_follow:bool):
        """
        251117_Ver2_1: Override to handle divide-by-zero in velocity setting
        
        Fixes issue where reference velocity can be zero at initialization.
        Reverted to simple implementation - works with existing pipeline.
        """
        # Original simple implementation
        for key in self.reference_data_keys:
            # Access series_data without "q_" prefix (environment_handler removes it)
            self.sim.data.joint(f"{key}").qpos = self._reference_data["series_data"][f"{key}"][self._imitation_index]
            if not is_x_follow and key == 'pelvis_tx':
                self.sim.data.joint(f"{key}").qpos = 0
        
        # Set velocities with epsilon to prevent divide-by-zero
        ref_velocity = self._reference_data["series_data"]["dpelvis_tx"][self._imitation_index]
        speed_ratio_to_target_velocity = self._target_velocity / (ref_velocity + 1e-8)
        
        for key in self.reference_data_keys:
            # Access series_data without "q_" prefix
            self.sim.data.joint(f"{key}").qvel = self._reference_data["series_data"][f"d{key}"][self._imitation_index] * speed_ratio_to_target_velocity
