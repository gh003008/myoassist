from rl_train.envs.environment_handler import EnvironmentHandler
import rl_train.train.train_configs.config as myoassist_config
import numpy as np

config_path = 'rl_train/train/train_configs/S004_3D_IL_ver1_0_BASE.json'
print('Loading config...')
def load_env():
    default_config = EnvironmentHandler.get_session_config_from_path(config_path, myoassist_config.TrainSessionConfigBase)
    config_type = EnvironmentHandler.get_config_type_from_session_id(default_config.env_params.env_id)
    config = EnvironmentHandler.get_session_config_from_path(config_path, config_type)
    # create non-rendering env
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False, is_evaluate_mode=True)
    return env

env = load_env()
print('Env created')
env.reset()

joint_names = ['pelvis_tx','pelvis_ty','pelvis_tz']
base_pos = env.sim.data.get_body_xpos('pelvis').copy()
print('Base pelvis world pos:', base_pos)

for j in joint_names:
    try:
        joint = env.sim.data.joint(j)
    except Exception as e:
        print('Cannot access joint', j, e)
        continue
    # set a +0.5 offset on this joint
    old = joint.qpos[0]
    joint.qpos[0] = old + 0.5
    env.sim.forward()
    pos = env.sim.data.get_body_xpos('pelvis').copy()
    print(f'After +0.5 on {j}: pelvis world pos =', pos)
    # reset
    joint.qpos[0] = old
    env.sim.forward()

# Also show current joint axes orientation for reference
print('\nJoint axes (from XML semantics):')
for j in joint_names:
    try:
        # axis vector is available in model.jnt_axis via joint id
        jid = env.sim.model.joint_name2id(j)
        axis = env.sim.model.jnt_axis[jid]
        print(j, 'axis=', axis)
    except Exception as e:
        print('axis lookup failed for', j, e)

print('\nDone')
