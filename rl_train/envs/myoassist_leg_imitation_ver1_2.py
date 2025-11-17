"""
MyoAssist Leg Imitation Environment ver1_2

Created: 2024-11-17 (251117)
Purpose: FIXED reference motion + Curriculum learning + Phase-aware initialization

Changes from ver1_1:
1. [CRITICAL] Uses FIXED reference motion (HDF5_v8_FIXED.npz)
2. [HIGH] Curriculum learning: Progressive difficulty in initialization
3. [HIGH] Phase-aware initialization: Start from stable poses (heel strike, double support)
4. [MEDIUM] Quality-based pose filtering
5. [MEDIUM] Progressive reward weight scheduling (optional)

Key Features:
- Heel strike detection from reference data
- Double support phase identification
- Curriculum stages based on training progress
- Maintains balancing rewards from ver1_1
"""

import collections
import numpy as np
from rl_train.envs.myoassist_leg_imitation_ver1_1 import (
    MyoAssistLegImitation_ver1_1,
    ImitationCustomLearningCallback_ver1_1
)
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from myosuite.utils.quat_math import quat2mat


##############################################################################
# CURRICULUM LEARNING
##############################################################################

class CurriculumScheduler:
    """
    Manages curriculum learning stages
    
    Stage 1 (0-30%): Double support only (most stable)
    Stage 2 (30-60%): Heel strikes + double support
    Stage 3 (60-100%): Full gait cycle
    """
    
    def __init__(self, total_timesteps):
        self.total_timesteps = total_timesteps
        self.stages = {
            'beginner': (0.0, 0.3),      # 0-30%: Very stable poses only
            'intermediate': (0.3, 0.6),   # 30-60%: Stable poses
            'advanced': (0.6, 1.0),       # 60-100%: All poses
        }
    
    def get_stage(self, current_timestep):
        """Return current curriculum stage"""
        progress = min(1.0, current_timestep / self.total_timesteps)
        
        if progress < 0.3:
            return 'beginner'
        elif progress < 0.6:
            return 'intermediate'
        else:
            return 'advanced'
    
    def get_allowed_indices_mask(self, current_timestep, 
                                  double_support_mask, 
                                  heel_strike_mask,
                                  high_quality_mask):
        """
        Return boolean mask of allowed initialization indices
        
        Args:
            current_timestep: Current training timestep
            double_support_mask: Boolean array for double support phases
            heel_strike_mask: Boolean array for heel strike moments
            high_quality_mask: Boolean array for high-quality poses
            
        Returns:
            allowed_mask: Boolean array of allowed indices
        """
        stage = self.get_stage(current_timestep)
        
        if stage == 'beginner':
            # Only double support phases (양발 지지)
            return double_support_mask & high_quality_mask
        elif stage == 'intermediate':
            # Double support + heel strikes
            return (double_support_mask | heel_strike_mask) & high_quality_mask
        else:
            # All high-quality poses
            return high_quality_mask


##############################################################################
# PHASE DETECTION
##############################################################################

class GaitPhaseDetector:
    """
    Detects gait phases from reference motion data
    """
    
    @staticmethod
    def detect_heel_strikes(ref_data, threshold_velocity=-0.01):
        """
        Detect heel strike moments from foot vertical position
        
        Heel strike = foot moving down then stops
        
        Args:
            ref_data: Reference data dictionary
            threshold_velocity: Threshold for foot velocity (m/s)
            
        Returns:
            heel_strike_indices: Array of frame indices
        """
        # Try to get foot height data
        r_foot_height = ref_data["series_data"].get("q_ankle_angle_r", None)
        l_foot_height = ref_data["series_data"].get("q_ankle_angle_l", None)
        
        if r_foot_height is None:
            # Fallback: Use heuristic based on gait cycle
            # Typical gait: heel strike at 0% and 50% of cycle
            n_frames = len(ref_data["series_data"]["q_pelvis_tx"])
            return np.array([0, n_frames // 2])
        
        # Calculate vertical velocity (approximate with diff)
        # In reality, we'd use pelvis_ty or foot marker data
        # For now, use pelvis vertical motion as proxy
        pelvis_ty = ref_data["series_data"]["q_pelvis_ty"]
        pelvis_vy = np.diff(pelvis_ty, prepend=pelvis_ty[0])
        
        # Heel strike = local minima in pelvis height
        # (when foot makes contact, pelvis is lowest)
        from scipy.signal import find_peaks
        
        # Invert to find valleys (minima)
        peaks, _ = find_peaks(-pelvis_ty, distance=50, prominence=0.005)
        
        if len(peaks) == 0:
            # Fallback
            n_frames = len(pelvis_ty)
            return np.array([0, n_frames // 2])
        
        return peaks
    
    @staticmethod
    def detect_double_support(ref_data, support_duration_frames=10):
        """
        Detect double support phases (both feet on ground)
        
        Heuristic: Around heel strikes, there's double support
        
        Args:
            ref_data: Reference data dictionary
            support_duration_frames: Duration of double support (frames)
            
        Returns:
            double_support_mask: Boolean array (True = double support)
        """
        n_frames = len(ref_data["series_data"]["q_pelvis_tx"])
        heel_strikes = GaitPhaseDetector.detect_heel_strikes(ref_data)
        
        # Create mask
        double_support_mask = np.zeros(n_frames, dtype=bool)
        
        # Mark regions around each heel strike
        half_duration = support_duration_frames // 2
        for hs in heel_strikes:
            start = max(0, hs - half_duration)
            end = min(n_frames, hs + half_duration)
            double_support_mask[start:end] = True
        
        return double_support_mask
    
    @staticmethod
    def filter_quality_poses(ref_data, quality_percentile=50):
        """
        Filter poses by quality metrics
        
        Quality criteria:
        1. Pelvis height (not too low)
        2. Pelvis tilt/list (not too tilted)
        3. Velocity (reasonable walking speed)
        
        Args:
            ref_data: Reference data dictionary
            quality_percentile: Keep top X percentile (50 = top 50%)
            
        Returns:
            high_quality_mask: Boolean array
        """
        series = ref_data["series_data"]
        n_frames = len(series["q_pelvis_tx"])
        
        quality_scores = np.zeros(n_frames)
        
        # Criterion 1: Pelvis height (prefer upright)
        pelvis_ty = series["q_pelvis_ty"]
        height_score = 1.0 - np.abs(pelvis_ty) / (np.abs(pelvis_ty).max() + 1e-6)
        
        # Criterion 2: Pelvis orientation (prefer level)
        pelvis_tilt = np.abs(series["q_pelvis_tilt"])
        pelvis_list = np.abs(series["q_pelvis_list"])
        orientation_score = 1.0 - (pelvis_tilt + pelvis_list) / (np.pi / 4)  # Normalize by 45°
        orientation_score = np.clip(orientation_score, 0, 1)
        
        # Criterion 3: Reasonable velocity
        pelvis_vx = series.get("dq_pelvis_tx", np.zeros(n_frames))
        # Prefer velocities around 0.8 m/s (normal walking)
        velocity_score = 1.0 - np.abs(pelvis_vx - 0.8) / 1.0
        velocity_score = np.clip(velocity_score, 0, 1)
        
        # Combined score
        quality_scores = (height_score + orientation_score + velocity_score) / 3.0
        
        # Keep top percentile
        threshold = np.percentile(quality_scores, 100 - quality_percentile)
        high_quality_mask = quality_scores >= threshold
        
        return high_quality_mask


##############################################################################
# VER1_2 ENVIRONMENT
##############################################################################

class MyoAssistLegImitation_ver1_2(MyoAssistLegImitation_ver1_1):
    """
    251117: Ver1_2 with FIXED reference + curriculum + phase-aware init
    
    New features:
    - Curriculum learning scheduler
    - Heel strike detection
    - Double support detection
    - Quality-based pose filtering
    - Phase-aware initialization
    """
    
    def _setup(self, *, 
               env_params: ImitationTrainSessionConfig.EnvParams,
               reference_data: dict | None = None,
               loop_reference_data: bool = False,
               total_timesteps: int = 30_000_000,  # For curriculum
               **kwargs):
        """251117: Initialize with curriculum and phase detection"""
        
        # Store total timesteps for curriculum
        self._total_timesteps = total_timesteps
        self._current_timestep = 0  # Will be updated by callback
        
        # Call parent setup (ver1_1)
        super()._setup(
            env_params=env_params,
            reference_data=reference_data,
            loop_reference_data=loop_reference_data,
            **kwargs
        )
        
        # Initialize curriculum scheduler
        self.curriculum = CurriculumScheduler(total_timesteps)
        
        # Detect gait phases from reference data
        print("\n" + "="*80)
        print("[*] PHASE DETECTION & QUALITY FILTERING (ver1_2)")  # 251117: Remove emoji for Windows
        print("="*80)
        
        self.heel_strike_indices = GaitPhaseDetector.detect_heel_strikes(
            self._reference_data
        )
        print(f"[OK] Detected {len(self.heel_strike_indices)} heel strikes at frames: {self.heel_strike_indices[:5]}...")  # 251117: Remove emoji
        
        self.double_support_mask = GaitPhaseDetector.detect_double_support(
            self._reference_data,
            support_duration_frames=10
        )
        n_double_support = np.sum(self.double_support_mask)
        print(f"[OK] Double support frames: {n_double_support} / {self._reference_data_length} ({100*n_double_support/self._reference_data_length:.1f}%)")
        
        self.high_quality_mask = GaitPhaseDetector.filter_quality_poses(
            self._reference_data,
            quality_percentile=50
        )
        n_high_quality = np.sum(self.high_quality_mask)
        print(f"[OK] High-quality poses: {n_high_quality} / {self._reference_data_length} ({100*n_high_quality/self._reference_data_length:.1f}%)")
        
        # Create heel strike mask for curriculum
        self.heel_strike_mask = np.zeros(self._reference_data_length, dtype=bool)
        self.heel_strike_mask[self.heel_strike_indices] = True
        
        print("="*80 + "\n")
    
    def update_curriculum_progress(self, current_timestep):
        """
        Update current training progress for curriculum
        
        Called by callback during training
        """
        self._current_timestep = current_timestep
    
    def reset(self, **kwargs):
        """
        251117: Phase-aware curriculum reset
        
        Initialization strategy changes based on training progress:
        - Early (0-30%): Double support only (most stable)
        - Mid (30-60%): Double support + heel strikes
        - Late (60%+): All high-quality poses
        """
        rng = np.random.default_rng()
        
        if self._flag_random_ref_index:
            # Get allowed indices based on curriculum stage
            allowed_mask = self.curriculum.get_allowed_indices_mask(
                self._current_timestep,
                self.double_support_mask,
                self.heel_strike_mask,
                self.high_quality_mask
            )
            
            allowed_indices = np.where(allowed_mask)[0]
            
            # Fallback: if no allowed indices, use all high-quality
            if len(allowed_indices) == 0:
                allowed_indices = np.where(self.high_quality_mask)[0]
            
            # Fallback: if still no indices, use heel strikes
            if len(allowed_indices) == 0:
                allowed_indices = self.heel_strike_indices
            
            # Sample from allowed indices
            self._imitation_index = rng.choice(allowed_indices)
            
            # Debug: Print curriculum stage occasionally
            if rng.random() < 0.01:  # 1% chance
                stage = self.curriculum.get_stage(self._current_timestep)
                progress = self._current_timestep / self._total_timesteps
                print(f"[CURRICULUM] [{stage}] @ {progress*100:.1f}%: "  # 251117: Remove emoji
                      f"Init at frame {self._imitation_index} "
                      f"({len(allowed_indices)} allowed)")
        else:
            # Non-random: start at first heel strike
            self._imitation_index = self.heel_strike_indices[0]
        
        # Follow reference motion
        self._follow_reference_motion(False)
        
        # Reset - filter out reset_qpos/reset_qvel from kwargs to avoid conflict
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['reset_qpos', 'reset_qvel']}
        obs = super(MyoAssistLegImitation_ver1_1, self).reset(
            reset_qpos=self.sim.data.qpos, 
            reset_qvel=self.sim.data.qvel, 
            **filtered_kwargs
        )
        return obs


##############################################################################
# CALLBACK
##############################################################################

class ImitationCustomLearningCallback_ver1_2(ImitationCustomLearningCallback_ver1_1):
    """
    251117: Callback for ver1_2 with curriculum progress updates
    """
    
    def _on_rollout_start(self) -> None:
        """Update curriculum progress before each rollout"""
        # Update training progress in all environments
        if hasattr(self.training_env, 'env_method'):
            self.training_env.env_method(
                'update_curriculum_progress',
                self.num_timesteps
            )
        
        super()._on_rollout_start()
