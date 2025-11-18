#!/usr/bin/env python3
"""
Debug script to check qpos values during reference motion following
"""
import numpy as np
import mujoco

# Load model
model = mujoco.MjModel.from_xml_path('models/26muscle_3D/myoLeg26_BASELINE.xml')
data = mujoco.MjData(model)

# Check stand keyframe
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
stand_qpos = model.key_qpos[key_id]

print("="*80)
print("MuJoCo Model qpos Analysis")
print("="*80)

# Get all joint names
print(f"\nTotal qpos size: {model.nq}")
print(f"Total joints: {model.njnt}")

print(f"\n{'Index':<10} {'Joint Name':<30} {'Stand qpos':<15}")
print("-"*80)

for i in range(model.njnt):
    joint_id = i
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    joint_adr = model.jnt_qposadr[joint_id]
    
    # Get qpos value from stand keyframe
    qpos_value = stand_qpos[joint_adr]
    
    print(f"{joint_adr:<10} {joint_name:<30} {qpos_value:+.6f}")

print("\n" + "="*80)
print("Key joints to check:")
print("="*80)

key_joints = [
    'hip_flexion_r',
    'knee_r_translation1',
    'knee_r_translation2', 
    'knee_angle_r',
    'ankle_angle_r',
    'hip_flexion_l',
    'knee_l_translation1',
    'knee_l_translation2',
    'knee_angle_l',
    'ankle_angle_l',
]

for jname in key_joints:
    try:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        jadr = model.jnt_qposadr[jid]
        val = stand_qpos[jadr]
        print(f"  {jname:<30} qpos[{jadr:2d}] = {val:+.6f}")
    except:
        print(f"  {jname:<30} NOT FOUND")

print("\n" + "="*80)
print("Environment reference_data_keys (what we set):")
print("="*80)

env_keys = [
    'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
    'knee_angle_r', 'knee_angle_l',
    'ankle_angle_r', 'ankle_angle_l',
]

for key in env_keys:
    try:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, key)
        print(f"  ✅ {key}")
    except:
        print(f"  ❌ {key} - NOT IN MODEL")

print("\n" + "="*80)
print("Joints in model but NOT in reference_data_keys:")
print("="*80)

for i in range(model.njnt):
    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if jname not in env_keys:
        jadr = model.jnt_qposadr[i]
        val = stand_qpos[jadr]
        print(f"  ⚠️  {jname:<30} qpos[{jadr:2d}] = {val:+.6f}")
