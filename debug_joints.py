#!/usr/bin/env python3
"""
Check what joints are actually in the model
"""
import mujoco

model_path = 'models/26muscle_3D/myoLeg26_BASELINE.xml'
model = mujoco.MjModel.from_xml_path(model_path)

print("="*80)
print(f"Checking model: {model_path}")
print("="*80)

test_joints = [
    "ankle_angle_l", "ankle_angle_r",
    "hip_flexion_l", "hip_flexion_r",
    "hip_adduction_l", "hip_adduction_r",
    "hip_rotation_l", "hip_rotation_r",
    "knee_angle_l", "knee_angle_r",
    "pelvis_list", "pelvis_tilt", "pelvis_rotation",
    "pelvis_tx", "pelvis_ty", "pelvis_tz"
]

print(f"\nTesting joints from config:")
for jname in test_joints:
    try:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        jadr = model.jnt_qposadr[jid]
        print(f"  ✅ {jname:<20} → joint_id={jid:2d}, qpos[{jadr:2d}]")
    except Exception as e:
        print(f"  ❌ {jname:<20} → NOT FOUND!")

print("\n" + "="*80)
print("All joints in model:")
print("="*80)
for i in range(model.njnt):
    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jadr = model.jnt_qposadr[i]
    print(f"  {i:2d}. {jname:<30} qpos[{jadr:2d}]")
