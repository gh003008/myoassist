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
    251119_Update: + Reference-informed velocity constraints + Phase-aware muscle rewards
    
    New features:
    - pelvis_list_penalty: Strong penalty for roll (lateral tilt)
    - pelvis_height_reward: Encourage staying upright
    - rotation_based_termination: Stop episode on excessive rotation
    - [NEW] joint_velocity_constraint: Reference statistics-based velocity limits
    - [NEW] joint_acceleration_penalty: Smooth motion (D-gain style)
    - [NEW] phase_muscle_reward: Phase-aware muscle activation patterns (optional)
    - Compatible with Ver2_0 Karico's stable training pipeline
    """
    
    # Reference velocity statistics (from S004_trial01_08mps_3D_HDF5_v7.npz analysis)
    # Used for intelligent velocity constraints instead of arbitrary thresholds
    REF_VEL_STATS = {
        'hip_flexion': {'max_abs': 3.30, 'std': 1.25, 'safe_threshold': 5.80},   # max + 2*std
        'hip_adduction': {'max_abs': 2.0, 'std': 0.8, 'safe_threshold': 3.60},   # estimated
        'hip_rotation': {'max_abs': 2.0, 'std': 0.8, 'safe_threshold': 3.60},    # estimated
        'knee_angle': {'max_abs': 7.75, 'std': 2.97, 'safe_threshold': 13.69},   # max + 2*std
        'ankle_angle': {'max_abs': 5.14, 'std': 1.51, 'safe_threshold': 8.16},   # max + 2*std
    }
    
    # Simple phase-aware muscle activation patterns (Ext/Flex groups)
    # Based on standard gait biomechanics (Winter, Perry)
    PHASE_MUSCLE_PATTERN = {
        'stance': {  # 0.0~0.6: Support phase (foot on ground)
            'extensors': ['glutmax', 'vasti', 'soleus', 'gastroc', 'hamstrings'],  # Push body up
            'flexors': ['iliopsoas', 'tibant']  # Minimal during stance
        },
        'swing': {  # 0.6~1.0: Swing phase (foot in air)
            'flexors': ['iliopsoas', 'rectfem', 'tibant'],  # Leg forward + toe clearance
            'extensors': ['soleus', 'gastroc']  # Should rest during swing
        }
    }
    
    def _setup(self, *, 
               env_params: ImitationTrainSessionConfig.EnvParams,
               reference_data: dict | None = None,
               loop_reference_data: bool = False,
               **kwargs):
        """251117_Ver2_1: Initialize with balance penalty parameters + velocity history"""
        
        # Store parameters before parent setup
        self._max_rot = kwargs.get('max_rot', env_params.__dict__.get('max_rot', 0.6))  # default: cos(53°) ≈ 0.6
        self.safe_height = env_params.safe_height
        self._step_count = 0  # Track steps to skip rotation check during initialization
        
        # 251119: Velocity history for acceleration penalty
        self._prev_qvel = None
        
        # 251119: Phase detection - estimate stride length from reference data
        self._stride_length = None  # Will be computed from reference data after parent setup
        
        # Call parent setup (ver1_0 Karico)
        super()._setup(
            env_params=env_params,
            reference_data=reference_data,
            loop_reference_data=loop_reference_data,
            **kwargs
        )
        
        # 251119: Compute stride length from reference hip flexion peaks
        if self._reference_data is not None:
            self._compute_stride_length()
    
    def _compute_stride_length(self):
        """251119: Estimate stride length from reference hip flexion peaks"""
        try:
            from scipy.signal import find_peaks
            hip_flex_l = self._reference_data["series_data"]["q_hip_flexion_l"]
            peaks, _ = find_peaks(hip_flex_l, distance=200)  # Min 200 samples between peaks
            if len(peaks) > 1:
                self._stride_length = int(np.mean(np.diff(peaks)))
            else:
                self._stride_length = 600  # Default ~0.5s at 1200Hz
        except:
            self._stride_length = 600  # Fallback
    
    def _get_gait_phase(self):
        """
        251119: Estimate gait phase (0.0~1.0) from hip flexion pattern
        
        Returns:
            float: Phase in [0.0, 1.0], where 0.0=heel strike, 0.6=toe-off
        """
        if self._stride_length is None:
            return 0.0
        
        # Phase is position within stride cycle
        phase = (self._imitation_index % self._stride_length) / self._stride_length
        return phase
    
    def _calculate_balancing_rewards(self):
        """
        251119_Update: Calculate balancing + velocity constraint + phase-aware rewards
        
        Returns:
            dict: All reward components (balance, velocity, acceleration, phase)
        """
        balance_rewards = {}
        
        # ========== Original Ver2_1 Balance Rewards ==========
        
        # [HIGH Priority] pelvis_list_penalty: Roll (lateral tilt) penalty
        pelvis_list = self.sim.data.joint('pelvis_list').qpos[0]
        pelvis_list_penalty = self.dt * (-np.square(pelvis_list))
        balance_rewards['pelvis_list_penalty'] = float(pelvis_list_penalty)
        
        # [MEDIUM Priority] pelvis height reward
        pelvis_height = self.sim.data.body('pelvis').xpos[2]
        target_height = 0.9
        height_reward = self.dt * np.exp(-2.0 * np.square(pelvis_height - target_height))
        balance_rewards['pelvis_height_reward'] = float(height_reward)
        
        # ========== NEW: Phase 1 - Velocity Constraints ==========
        
        # Joint velocity constraint (reference statistics-based)
        vel_penalty = 0.0
        joint_groups = {
            'hip_flexion': ['hip_flexion_l', 'hip_flexion_r'],
            'hip_adduction': ['hip_adduction_l', 'hip_adduction_r'],
            'hip_rotation': ['hip_rotation_l', 'hip_rotation_r'],
            'knee_angle': ['knee_angle_l', 'knee_angle_r'],
            'ankle_angle': ['ankle_angle_l', 'ankle_angle_r'],
        }
        
        for group_name, joint_list in joint_groups.items():
            threshold = self.REF_VEL_STATS[group_name]['safe_threshold']
            for joint in joint_list:
                try:
                    vel_current = abs(self.sim.data.joint(joint).qvel[0])
                    if vel_current > threshold:
                        # Strong penalty for exceeding reference range
                        excessive = vel_current - threshold
                        vel_penalty -= (excessive ** 2) * 5.0  # Amplified penalty
                except:
                    pass  # Skip if joint not found
        
        balance_rewards['joint_velocity_constraint'] = self.dt * vel_penalty
        
        # Joint acceleration penalty (smooth motion, D-gain style)
        acc_penalty = 0.0
        if self._prev_qvel is not None:
            try:
                # Compute acceleration for key joints
                for joint in ['hip_flexion_l', 'hip_flexion_r', 'knee_angle_l', 'knee_angle_r']:
                    qvel_curr = self.sim.data.joint(joint).qvel[0]
                    qvel_prev = self._prev_qvel.get(joint, qvel_curr)
                    acceleration = (qvel_curr - qvel_prev) / self.dt
                    # Penalize large accelerations (prevents "jerky" motion)
                    acc_penalty -= (acceleration ** 2) * 0.1
            except:
                pass
        
        # Update velocity history
        self._prev_qvel = {}
        for joint in ['hip_flexion_l', 'hip_flexion_r', 'knee_angle_l', 'knee_angle_r', 
                      'ankle_angle_l', 'ankle_angle_r']:
            try:
                self._prev_qvel[joint] = self.sim.data.joint(joint).qvel[0]
            except:
                pass
        
        balance_rewards['joint_acceleration_penalty'] = self.dt * acc_penalty
        
        # ========== NEW: Phase 2 - Phase-Aware Muscle Activation (OPTIONAL) ==========
        
        # Get phase-muscle reward (will return 0.0 if disabled in config)
        phase_reward = self._calculate_phase_muscle_reward()
        balance_rewards['phase_muscle_reward'] = phase_reward
        
        return balance_rewards
    
    def _calculate_phase_muscle_reward(self):
        """
        251119: Phase-aware muscle activation reward (Simple Ext/Flex grouping)
        
        Encourages physiologically plausible muscle patterns based on gait phase.
        Can be disabled by setting weight to 0.0 in config.
        
        Returns:
            float: Phase-aware muscle reward
        """
        # Check if this reward is enabled (weight > 0)
        if not hasattr(self, '_reward_keys_and_weights'):
            return 0.0
        
        phase_weight = self.rwd_keys_wt.get('phase_muscle_reward', 0.0)
        if phase_weight == 0.0:
            return 0.0  # Disabled
        
        try:
            phase = self._get_gait_phase()
            
            # Get current muscle activations (0~1 range)
            # self.sim.data.act gives muscle activations
            muscle_act = self.sim.data.act.copy()  # shape: (26,)
            
            # Determine phase pattern
            if phase < 0.6:  # Stance phase
                pattern = self.PHASE_MUSCLE_PATTERN['stance']
                high_group = pattern['extensors']
                low_group = pattern['flexors']
            else:  # Swing phase
                pattern = self.PHASE_MUSCLE_PATTERN['swing']
                high_group = pattern['flexors']
                low_group = pattern['extensors']
            
            reward = 0.0
            
            # Reward for appropriate muscle activation
            for i in range(len(muscle_act)):
                muscle_name = self.sim.model.actuator(i).name
                activation = muscle_act[i]
                
                # Check if this muscle should be HIGH in this phase
                is_high_muscle = any(pattern_name in muscle_name for pattern_name in high_group)
                # Check if this muscle should be LOW in this phase
                is_low_muscle = any(pattern_name in muscle_name for pattern_name in low_group)
                
                if is_high_muscle:
                    # Encourage activation (but don't penalize too much if low)
                    if activation > 0.3:
                        reward += 0.02  # Small positive reward
                elif is_low_muscle:
                    # Encourage relaxation (but flexible)
                    if activation < 0.2:
                        reward += 0.01  # Smaller reward
            
            return self.dt * reward
            
        except Exception as e:
            # Fail gracefully if something goes wrong
            return 0.0
    
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
        """251119_Update: Reset step count + velocity history"""
        self._step_count = 0
        self._prev_qvel = None  # Reset velocity history for acceleration calculation
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
