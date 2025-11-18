"""
Curriculum Learning Scheduler for MyoAssist
============================================

Curriculum Learning: 학습 난이도를 점진적으로 증가시켜 더 효율적인 학습을 달성

주요 기능:
1. Stage-based curriculum: 단계별 난이도 조절
2. Reward weight scheduling: 보상 가중치 자동 조정
3. Environment parameter scheduling: 환경 파라미터 점진적 변경
4. 완전 모듈화: on/off 가능, 기존 코드 수정 최소화

Author: Generated for ghlee
Date: 2025-11-18
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json


@dataclass
class CurriculumStage:
    """단일 Curriculum Stage 정의"""
    stage_id: int
    name: str
    timesteps: int  # 이 단계에서 학습할 timestep 수
    
    # 환경 파라미터
    target_velocity_range: Tuple[float, float] = (0.4, 0.6)
    max_episode_steps: int = 500
    
    # Reward weights (None이면 변경하지 않음)
    reward_weights: Optional[Dict[str, float]] = None
    
    # 설명
    description: str = ""


class CurriculumScheduler:
    """
    Curriculum Learning Scheduler
    
    사용 예시:
    >>> scheduler = CurriculumScheduler.from_config("curriculum_config.json")
    >>> if scheduler.should_update(current_timestep):
    >>>     new_params = scheduler.get_current_stage_params()
    >>>     env.update_parameters(new_params)
    """
    
    def __init__(self, stages: List[CurriculumStage], enable: bool = True):
        """
        Args:
            stages: Curriculum stage 리스트 (난이도 순서대로)
            enable: Curriculum 활성화 여부 (False면 첫 번째 stage만 사용)
        """
        self.stages = stages
        self.enable = enable
        self.current_stage_idx = 0
        self.total_timesteps_processed = 0
        
        # Stage별 시작 timestep 계산
        self.stage_start_timesteps = []
        cumulative = 0
        for stage in stages:
            self.stage_start_timesteps.append(cumulative)
            cumulative += stage.timesteps
        
        print(f"📚 Curriculum Scheduler initialized: {'ENABLED' if enable else 'DISABLED'}")
        if enable:
            self._print_curriculum_plan()
    
    def _print_curriculum_plan(self):
        """Curriculum 계획 출력"""
        print("\n" + "="*80)
        print("📚 CURRICULUM LEARNING PLAN")
        print("="*80)
        for i, stage in enumerate(self.stages):
            start_ts = self.stage_start_timesteps[i]
            end_ts = start_ts + stage.timesteps
            print(f"\nStage {stage.stage_id}: {stage.name}")
            print(f"  Timesteps: {start_ts:,} → {end_ts:,} ({stage.timesteps:,} steps)")
            print(f"  Velocity: {stage.target_velocity_range[0]:.2f} ~ {stage.target_velocity_range[1]:.2f} m/s")
            print(f"  Max Episode: {stage.max_episode_steps} steps")
            if stage.reward_weights:
                print(f"  Reward Adjustments: {len(stage.reward_weights)} weights modified")
            print(f"  Description: {stage.description}")
        print("="*80 + "\n")
    
    def update(self, timesteps_delta: int) -> bool:
        """
        Timestep 업데이트 및 stage 전환 확인
        
        Args:
            timesteps_delta: 증가한 timestep 수
            
        Returns:
            bool: Stage가 변경되었으면 True
        """
        if not self.enable:
            return False
        
        self.total_timesteps_processed += timesteps_delta
        
        # 다음 stage로 전환해야 하는지 확인
        if self.current_stage_idx < len(self.stages) - 1:
            next_stage_start = self.stage_start_timesteps[self.current_stage_idx + 1]
            if self.total_timesteps_processed >= next_stage_start:
                self.current_stage_idx += 1
                self._on_stage_change()
                return True
        
        return False
    
    def _on_stage_change(self):
        """Stage 변경 시 호출되는 콜백"""
        stage = self.get_current_stage()
        print("\n" + "🎓"*40)
        print(f"🎓 CURRICULUM STAGE CHANGED → Stage {stage.stage_id}: {stage.name}")
        print(f"   Timestep: {self.total_timesteps_processed:,}")
        print(f"   {stage.description}")
        print("🎓"*40 + "\n")
    
    def get_current_stage(self) -> CurriculumStage:
        """현재 stage 반환"""
        return self.stages[self.current_stage_idx]
    
    def get_current_stage_params(self) -> Dict:
        """
        현재 stage의 파라미터 반환
        
        Returns:
            dict: 환경 및 보상에 적용할 파라미터
        """
        if not self.enable:
            # Curriculum disabled: 기본값 사용 (첫 stage)
            return {}
        
        stage = self.get_current_stage()
        params = {
            'target_velocity_range': stage.target_velocity_range,
            'max_episode_steps': stage.max_episode_steps,
        }
        
        if stage.reward_weights:
            params['reward_weights'] = stage.reward_weights
        
        return params
    
    def get_progress(self) -> float:
        """현재 stage 내 진행률 (0.0 ~ 1.0)"""
        if not self.enable:
            return 1.0
        
        stage_start = self.stage_start_timesteps[self.current_stage_idx]
        stage_duration = self.stages[self.current_stage_idx].timesteps
        progress_in_stage = self.total_timesteps_processed - stage_start
        return min(1.0, progress_in_stage / stage_duration)
    
    def should_update(self, current_timestep: int) -> bool:
        """
        지정된 timestep에서 업데이트가 필요한지 확인
        
        Args:
            current_timestep: 현재 총 timestep
            
        Returns:
            bool: 업데이트 필요 여부
        """
        if not self.enable:
            return False
        
        # Stage 경계에 도달했는지 확인
        for i in range(self.current_stage_idx + 1, len(self.stages)):
            if current_timestep >= self.stage_start_timesteps[i]:
                return True
        return False
    
    def to_dict(self) -> Dict:
        """Scheduler 상태를 dict로 저장 (체크포인트용)"""
        return {
            'enable': self.enable,
            'current_stage_idx': self.current_stage_idx,
            'total_timesteps_processed': self.total_timesteps_processed,
        }
    
    def from_dict(self, state: Dict):
        """저장된 상태로부터 복원 (체크포인트 로드)"""
        self.enable = state.get('enable', self.enable)
        self.current_stage_idx = state.get('current_stage_idx', 0)
        self.total_timesteps_processed = state.get('total_timesteps_processed', 0)
    
    @classmethod
    def from_config(cls, config_path: str, enable: bool = True) -> 'CurriculumScheduler':
        """
        JSON 설정 파일로부터 Scheduler 생성
        
        Args:
            config_path: Curriculum 설정 파일 경로
            enable: Curriculum 활성화 여부
            
        Returns:
            CurriculumScheduler 인스턴스
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        stages = []
        for stage_config in config['stages']:
            stage = CurriculumStage(
                stage_id=stage_config['stage_id'],
                name=stage_config['name'],
                timesteps=stage_config['timesteps'],
                target_velocity_range=tuple(stage_config['target_velocity_range']),
                max_episode_steps=stage_config.get('max_episode_steps', 1000),
                reward_weights=stage_config.get('reward_weights'),
                description=stage_config.get('description', '')
            )
            stages.append(stage)
        
        return cls(stages, enable=enable)
    
    @classmethod
    def create_default_treadmill_curriculum(cls, enable: bool = True) -> 'CurriculumScheduler':
        """
        Treadmill 보행을 위한 기본 Curriculum 생성
        
        단계별 학습 전략:
        1. Foundation (기초): 천천히, 안정성 집중
        2. Stabilization (안정화): 중간 속도, 균형 강화
        3. Target Performance (목표 달성): 정상 속도
        4. Robustness (강건성): 다양한 속도
        """
        stages = [
            # Stage 1: Foundation - 천천히 걷기 배우기
            CurriculumStage(
                stage_id=1,
                name="Foundation",
                timesteps=5_000_000,  # 5M steps
                target_velocity_range=(0.4, 0.5),
                max_episode_steps=300,
                reward_weights={
                    # Pelvis 안정성에 집중
                    'pelvis_list': 3.0,  # 더 높은 가중치
                    'pelvis_tilt': 3.0,
                    'pelvis_rotation': 3.0,
                    'pelvis_list_penalty': 0.8,  # 더 높은 페널티
                    'pelvis_height_reward': 0.1,
                    # 다리 움직임은 자유롭게
                    'hip_flexion_l': 1.0,
                    'hip_flexion_r': 1.0,
                    'knee_angle_l': 1.5,
                    'knee_angle_r': 1.5,
                },
                description="Learn basic stability and slow walking (0.4~0.5 m/s)"
            ),
            
            # Stage 2: Stabilization - 속도 증가 및 균형 강화
            CurriculumStage(
                stage_id=2,
                name="Stabilization",
                timesteps=8_000_000,  # 8M steps
                target_velocity_range=(0.5, 0.7),
                max_episode_steps=600,
                reward_weights={
                    # 균형잡힌 가중치
                    'pelvis_list': 2.0,
                    'pelvis_tilt': 2.0,
                    'pelvis_rotation': 2.0,
                    'pelvis_list_penalty': 0.5,
                    'pelvis_height_reward': 0.05,
                    # 관절 추적 강화
                    'hip_flexion_l': 1.5,
                    'hip_flexion_r': 1.5,
                    'hip_adduction_l': 1.0,
                    'hip_adduction_r': 1.0,
                    'knee_angle_l': 2.0,
                    'knee_angle_r': 2.0,
                },
                description="Increase speed and refine balance (0.5~0.7 m/s)"
            ),
            
            # Stage 3: Target Performance - 목표 속도 달성
            CurriculumStage(
                stage_id=3,
                name="Target Performance",
                timesteps=10_000_000,  # 10M steps
                target_velocity_range=(0.7, 0.8),
                max_episode_steps=1000,
                reward_weights=None,  # 기본 가중치 사용 (config 파일의 값)
                description="Achieve target walking speed (0.7~0.8 m/s)"
            ),
            
            # Stage 4: Robustness - 다양한 속도 적응
            CurriculumStage(
                stage_id=4,
                name="Robustness",
                timesteps=7_000_000,  # 7M steps
                target_velocity_range=(0.6, 0.9),
                max_episode_steps=1000,
                reward_weights={
                    # 미세 조정
                    'forward_reward': 0.1,  # 속도 변화 적응 장려
                    'muscle_activation_penalty': 0.15,  # 효율성 강조
                },
                description="Adapt to variable speeds for robustness (0.6~0.9 m/s)"
            ),
        ]
        
        return cls(stages, enable=enable)


def interpolate_reward_weights(
    base_weights: Dict[str, float],
    stage_weights: Optional[Dict[str, float]],
    progress: float = 1.0
) -> Dict[str, float]:
    """
    두 reward weight 딕셔너리 사이를 선형 보간
    
    Args:
        base_weights: 기본 가중치
        stage_weights: Stage별 가중치 (None이면 base_weights 반환)
        progress: 보간 진행률 (0.0 = base, 1.0 = stage)
        
    Returns:
        보간된 가중치 딕셔너리
    """
    if stage_weights is None:
        return base_weights.copy()
    
    result = base_weights.copy()
    for key, target_value in stage_weights.items():
        if key in result:
            base_value = base_weights[key]
            result[key] = base_value + (target_value - base_value) * progress
    
    return result


if __name__ == "__main__":
    # 테스트 코드
    print("Testing Curriculum Scheduler...")
    
    # 기본 curriculum 생성
    scheduler = CurriculumScheduler.create_default_treadmill_curriculum(enable=True)
    
    # 시뮬레이션
    print("\n시뮬레이션:")
    for timestep in [0, 3_000_000, 5_000_000, 10_000_000, 20_000_000, 30_000_000]:
        changed = scheduler.update(1_000_000)  # 1M씩 증가
        if changed or timestep == 0:
            stage = scheduler.get_current_stage()
            params = scheduler.get_current_stage_params()
            print(f"\nTimestep {timestep:,}: Stage {stage.stage_id} - {stage.name}")
            print(f"  Velocity: {params['target_velocity_range']}")
            print(f"  Max Episode: {params['max_episode_steps']}")
            print(f"  Progress: {scheduler.get_progress()*100:.1f}%")
