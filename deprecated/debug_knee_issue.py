"""
Debug why knee/shin is attached to hip
"""
import numpy as np
import mujoco

# Load model
model_path = r"C:\workspace_home\myoassist\models\26muscle_3D\myoLeg26_TUTORIAL.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

print("=" * 80)
print("MUJOCO MODEL ANALYSIS - KNEE JOINT STRUCTURE")
print("=" * 80)

# Find all joints related to knee
print("\nAll joints:")
for i in range(model.njnt):
    jnt_name = model.joint(i).name
    jnt_qposadr = model.jnt_qposadr[i]
    jnt_type = model.jnt_type[i]
    type_names = {0: 'FREE', 1: 'BALL', 2: 'SLIDE', 3: 'HINGE'}
    
    if 'knee' in jnt_name.lower() or 'tibia' in jnt_name.lower():
        print(f"  Joint {i:2d}: {jnt_name:35s} | Type: {type_names[jnt_type]:6s} | qpos[{jnt_qposadr}]")

print("\n" + "=" * 80)
print("BODY HIERARCHY - How bodies connect")
print("=" * 80)

# Get body hierarchy
important_bodies = ['pelvis', 'femur_r', 'tibia_r', 'talus_r', 'femur_l', 'tibia_l', 'talus_l']

for body_name in important_bodies:
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        parent_id = model.body_parentid[body_id]
        
        if parent_id > 0:
            parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
        else:
            parent_name = "world"
        
        print(f"  {body_name:15s} → parent: {parent_name}")
    except:
        pass

print("\n" + "=" * 80)
print("DEFAULT QPOS VALUES (from model)")
print("=" * 80)

# Check default qpos
print("\nDefault qpos (first 30):")
for i in range(min(30, model.nq)):
    jnt_name = "?"
    for j in range(model.njnt):
        if model.jnt_qposadr[j] == i:
            jnt_name = model.joint(j).name
            break
    print(f"  qpos[{i:2d}] = {model.qpos0[i]:8.4f}  ({jnt_name})")

print("\n" + "=" * 80)
print("TESTING: What happens with default qpos?")
print("=" * 80)

# Reset to default
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

print("\nBody positions with DEFAULT qpos:")
for body_name in important_bodies:
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        pos = data.xpos[body_id]
        print(f"  {body_name:15s}: ({pos[0]:>7.3f}, {pos[1]:>7.3f}, {pos[2]:>7.3f})")
    except:
        pass

print("\n" + "=" * 80)
print("TESTING: Set knee_angle_r = -1.2 rad (-70°)")
print("=" * 80)

# Reset
mujoco.mj_resetData(model, data)
data.qpos[11] = -1.2  # knee_angle_r
mujoco.mj_forward(model, data)

print("\nBody positions with knee_angle_r = -1.2:")
for body_name in important_bodies:
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        pos = data.xpos[body_id]
        print(f"  {body_name:15s}: ({pos[0]:>7.3f}, {pos[1]:>7.3f}, {pos[2]:>7.3f})")
    except:
        pass

print("\nKnee translation values:")
print(f"  knee_r_translation1 (qpos[9]):  {data.qpos[9]:8.4f}")
print(f"  knee_r_translation2 (qpos[10]): {data.qpos[10]:8.4f}")

print("\n" + "=" * 80)
print("CHECKING: Are knee translations COUPLED joints?")
print("=" * 80)

# Check if knee_angle drives translations
for i in range(model.neq):
    eq_type = model.eq_type[i]
    if eq_type == 1:  # mjEQ_JOINT (coupled joints)
        eq_data = model.eq_data[i]
        print(f"  Equality constraint {i}: data = {eq_data[:5]}")
        # Try to find which joints
        eq_obj1id = model.eq_obj1id[i]
        eq_obj2id = model.eq_obj2id[i]
        
        if eq_obj1id < model.njnt:
            jnt1_name = model.joint(eq_obj1id).name
        else:
            jnt1_name = f"obj{eq_obj1id}"
        
        if eq_obj2id < model.njnt:
            jnt2_name = model.joint(eq_obj2id).name
        else:
            jnt2_name = f"obj{eq_obj2id}"
        
        print(f"    Couples: {jnt1_name} ↔ {jnt2_name}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
