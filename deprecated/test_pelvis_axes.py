"""
Test which pelvis joint controls height (up direction)
"""
import numpy as np
from rl_train.envs.environment_handler import EnvironmentHandler
import rl_train.train.train_configs.config as myoassist_config

# Load config
config_path = 'rl_train/train/train_configs/S004_3D_IL_ver1_0_BASE.json'
default_config = EnvironmentHandler.get_session_config_from_path(
    config_path, 
    myoassist_config.TrainSessionConfigBase
)
config_type = EnvironmentHandler.get_config_type_from_session_id(
    default_config.env_params.env_id
)
config = EnvironmentHandler.get_session_config_from_path(config_path, config_type)

# Create environment
print("Creating environment...")
env = EnvironmentHandler.create_environment(
    config, 
    is_rendering_on=False, 
    is_evaluate_mode=True
)
env.reset()

# Get initial pelvis position
pelvis_body = env.sim.model.body('pelvis')
pelvis_pos_initial = env.sim.data.body(pelvis_body.id).xpos.copy()
print(f"\n초기 pelvis 위치 (world): {pelvis_pos_initial}")

# Test each joint with +1.0 offset
joints_to_test = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']

print("\n" + "="*60)
print("각 joint에 +1.0 적용 후 pelvis world 위치 변화:")
print("="*60)

for joint_name in joints_to_test:
    env.reset()
    
    # Apply +1.0 to this joint
    joint = env.sim.data.joint(joint_name)
    joint.qpos[0] = 1.0
    
    # Forward kinematics
    env.sim.forward()
    
    # Get new pelvis position
    pelvis_pos_new = env.sim.data.body(pelvis_body.id).xpos.copy()
    delta = pelvis_pos_new - pelvis_pos_initial
    
    print(f"\n{joint_name} = +1.0:")
    print(f"  World position: {pelvis_pos_new}")
    print(f"  Delta: {delta}")
    print(f"  → X변화: {delta[0]:+.3f}, Y변화: {delta[1]:+.3f}, Z변화: {delta[2]:+.3f}")
    
    # Identify direction
    abs_delta = np.abs(delta)
    max_axis = np.argmax(abs_delta)
    axis_names = ['X(forward)', 'Y(left)', 'Z(up)']
    print(f"  ✅ 주 변화 방향: {axis_names[max_axis]}")

print("\n" + "="*60)
print("결론:")
print("  - Z(up)에 큰 변화를 주는 joint가 높이(height) 제어")
print("="*60)

env.close()
