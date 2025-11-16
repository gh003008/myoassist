"""
Systematic diagnosis of model structure and rendering issues
"""
import numpy as np
import mujoco

def diagnose_baseline_model():
    """Diagnose BASELINE model structure"""
    model_path = "models/26muscle_3D/myoLeg26_BASELINE.xml"
    
    print("=" * 80)
    print("BASELINE MODEL DIAGNOSIS")
    print("=" * 80)
    
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # 1. Check default pose
    print("\n1. DEFAULT POSE (qpos0):")
    print("-" * 80)
    
    important_qpos = {
        0: 'pelvis_tx',
        1: 'pelvis_ty', 
        2: 'pelvis_tz',
        3: 'pelvis_tilt',
        4: 'pelvis_list',
        5: 'pelvis_rotation',
        6: 'hip_flexion_r',
        7: 'hip_adduction_r',
        8: 'hip_rotation_r',
        9: 'knee_r_translation1',
        10: 'knee_r_translation2',
        11: 'knee_angle_r',
        12: 'ankle_angle_r',
        23: 'hip_flexion_l',
        24: 'hip_adduction_l',
        25: 'hip_rotation_l',
        26: 'knee_l_translation1',
        27: 'knee_l_translation2',
        28: 'knee_angle_l',
        29: 'ankle_angle_l',
        40: 'r_shoulder_abd',
        42: 'r_shoulder_flex',
        43: 'r_elbow_flex',
        47: 'l_shoulder_abd',
        49: 'l_shoulder_flex',
        50: 'l_elbow_flex',
    }
    
    for idx, name in important_qpos.items():
        if idx < len(model.qpos0):
            print(f"  qpos0[{idx:2d}] = {model.qpos0[idx]:8.4f}  ({name})")
    
    # 2. Check body positions with default pose
    print("\n2. BODY POSITIONS WITH DEFAULT POSE:")
    print("-" * 80)
    
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    body_names = ['pelvis', 'femur_r', 'tibia_r', 'talus_r', 'calcn_r',
                  'femur_l', 'tibia_l', 'talus_l', 'calcn_l',
                  'humerus_r', 'ulna_r', 'humerus_l', 'ulna_l']
    
    for body_name in body_names:
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            pos = data.xpos[body_id]
            print(f"  {body_name:15s}: ({pos[0]:>7.3f}, {pos[1]:>7.3f}, {pos[2]:>7.3f})")
        except:
            pass
    
    # 3. Test with walking pose (knee bent)
    print("\n3. TEST WALKING POSE (knee_angle_r = -1.0 rad):")
    print("-" * 80)
    
    mujoco.mj_resetData(model, data)
    data.qpos[11] = -1.0  # knee_angle_r
    data.qpos[28] = -1.0  # knee_angle_l
    mujoco.mj_forward(model, data)
    
    for body_name in ['pelvis', 'femur_r', 'tibia_r', 'talus_r', 'femur_l', 'tibia_l', 'talus_l']:
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            pos = data.xpos[body_id]
            print(f"  {body_name:15s}: ({pos[0]:>7.3f}, {pos[1]:>7.3f}, {pos[2]:>7.3f})")
        except:
            pass
    
    print(f"\n  Knee translations:")
    print(f"    qpos[9]  (knee_r_trans1): {data.qpos[9]:8.4f}")
    print(f"    qpos[10] (knee_r_trans2): {data.qpos[10]:8.4f}")
    print(f"    qpos[26] (knee_l_trans1): {data.qpos[26]:8.4f}")
    print(f"    qpos[27] (knee_l_trans2): {data.qpos[27]:8.4f}")
    
    # 4. Check if femur and tibia are at same position
    print("\n4. KINEMATIC CHAIN CHECK:")
    print("-" * 80)
    
    femur_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'femur_r')
    tibia_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'tibia_r')
    
    femur_pos = data.xpos[femur_r_id]
    tibia_pos = data.xpos[tibia_r_id]
    
    distance = np.linalg.norm(femur_pos - tibia_pos)
    print(f"  femur_r → tibia_r distance: {distance:.4f} m")
    
    if distance < 0.01:
        print("  ⚠️  WARNING: femur and tibia are at same position!")
        print("  This means kinematic chain is BROKEN!")
    else:
        print("  ✅ OK: femur and tibia are properly separated")
    
    # 5. Check segment lengths from model
    print("\n5. MODEL SEGMENT LENGTHS:")
    print("-" * 80)
    
    # Get geom sizes for body segments
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and any(part in geom_name.lower() for part in ['femur', 'tibia', 'talus']):
            geom_size = model.geom_size[i]
            geom_type = model.geom_type[i]
            print(f"  {geom_name:30s}: size={geom_size}, type={geom_type}")
    
    # 6. Check if arms exist
    print("\n6. ARM JOINTS CHECK:")
    print("-" * 80)
    
    arm_joint_names = ['r_shoulder_abd', 'r_shoulder_flex', 'r_elbow_flex',
                       'l_shoulder_abd', 'l_shoulder_flex', 'l_elbow_flex']
    
    has_arms = False
    for jnt_name in arm_joint_names:
        try:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            has_arms = True
            print(f"  ✅ Found: {jnt_name}")
        except:
            print(f"  ❌ Not found: {jnt_name}")
    
    if has_arms:
        print("\n  ⚠️  Model HAS arms! This is why you see arms in rendering.")
        print("  Solution: Set arm joints to neutral pose or use model without arms.")
    
    return model, data

def test_reference_data_application():
    """Test applying actual reference data"""
    print("\n\n" + "=" * 80)
    print("TESTING REFERENCE DATA APPLICATION")
    print("=" * 80)
    
    model_path = "models/26muscle_3D/myoLeg26_BASELINE.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Load HDF5 v6 data
    hdf5_data = np.load("rl_train/reference_data/S004_trial01_08mps_3D_HDF5_v6.npz")
    q_ref = hdf5_data['q_ref']
    
    print(f"\nReference data shape: {q_ref.shape}")
    print(f"Frame 1000 values:")
    
    joint_order = [
        (0, 'pelvis_tx'),
        (1, 'pelvis_ty'),
        (2, 'pelvis_tz'),
        (3, 'pelvis_list'),
        (4, 'pelvis_tilt'),
        (5, 'pelvis_rotation'),
        (6, 'hip_flexion_r'),
        (7, 'hip_adduction_r'),
        (8, 'hip_rotation_r'),
        (9, 'hip_flexion_l'),
        (10, 'hip_adduction_l'),
        (11, 'hip_rotation_l'),
        (12, 'knee_angle_r'),
        (13, 'knee_angle_l'),
        (14, 'ankle_angle_r'),
        (15, 'ankle_angle_l'),
    ]
    
    frame_idx = 1000
    for idx, name in joint_order:
        val = q_ref[frame_idx, idx]
        if 'pelvis_t' in name:
            print(f"  q_ref[{idx:2d}] = {val:8.4f} m     ({name})")
        else:
            print(f"  q_ref[{idx:2d}] = {val:8.4f} rad ({np.degrees(val):>7.2f}°)  ({name})")
    
    # Apply to model
    print("\nApplying to model...")
    qpos_mapping = [
        (0, 0), (1, 1), (2, 2), (3, 4), (4, 3), (5, 5),
        (6, 6), (7, 7), (8, 8),
        (9, 23), (10, 24), (11, 25),
        (12, 11), (13, 28),
        (14, 12), (15, 29)
    ]
    
    # Start from qpos0
    data.qpos[:] = model.qpos0[:]
    
    for ref_idx, qpos_idx in qpos_mapping:
        data.qpos[qpos_idx] = q_ref[frame_idx, ref_idx]
    
    # Add height offset
    data.qpos[1] += 0.95
    
    mujoco.mj_forward(model, data)
    
    print("\nBody positions after applying reference:")
    for body_name in ['pelvis', 'femur_r', 'tibia_r', 'talus_r', 'femur_l', 'tibia_l', 'talus_l']:
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            pos = data.xpos[body_id]
            print(f"  {body_name:15s}: ({pos[0]:>7.3f}, {pos[1]:>7.3f}, {pos[2]:>7.3f})")
        except:
            pass

if __name__ == "__main__":
    model, data = diagnose_baseline_model()
    test_reference_data_application()
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
