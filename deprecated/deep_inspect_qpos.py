"""
Deep inspection of qpos values and model structure
"""
import numpy as np
import mujoco
from pathlib import Path

def inspect_model_structure(model_path):
    """Inspect MuJoCo model structure in detail"""
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    print("=" * 80)
    print("MUJOCO MODEL STRUCTURE - DETAILED")
    print("=" * 80)
    
    print(f"\nTotal qpos: {model.nq}")
    print(f"Total qvel: {model.nv}")
    print(f"Total actuators: {model.nu}")
    
    print("\n" + "=" * 80)
    print("ALL JOINTS (IN ORDER):")
    print("=" * 80)
    for i in range(model.njnt):
        jnt_id = i
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        jnt_type = model.jnt_type[jnt_id]
        jnt_qposadr = model.jnt_qposadr[jnt_id]
        
        # Get joint type name
        type_names = {0: 'FREE', 1: 'BALL', 2: 'SLIDE', 3: 'HINGE'}
        type_name = type_names.get(jnt_type, f'UNKNOWN({jnt_type})')
        
        # Get number of qpos for this joint
        if jnt_type == 0:  # FREE
            nqpos = 7
        elif jnt_type == 1:  # BALL
            nqpos = 4
        else:  # SLIDE or HINGE
            nqpos = 1
        
        qpos_range = f"{jnt_qposadr}" if nqpos == 1 else f"{jnt_qposadr}:{jnt_qposadr+nqpos}"
        
        print(f"Joint {i:2d}: {jnt_name:30s} | Type: {type_name:6s} | qpos[{qpos_range}]")
    
    print("\n" + "=" * 80)
    print("QPOS TO JOINT MAPPING:")
    print("=" * 80)
    qpos_to_jnt = {}
    for i in range(model.njnt):
        jnt_qposadr = model.jnt_qposadr[i]
        jnt_type = model.jnt_type[i]
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        
        if jnt_type == 0:  # FREE (7 qpos)
            for offset in range(7):
                qpos_to_jnt[jnt_qposadr + offset] = f"{jnt_name}[{offset}]"
        elif jnt_type == 1:  # BALL (4 qpos)
            for offset in range(4):
                qpos_to_jnt[jnt_qposadr + offset] = f"{jnt_name}[{offset}]"
        else:  # SLIDE/HINGE (1 qpos)
            qpos_to_jnt[jnt_qposadr] = jnt_name
    
    for qpos_idx in range(model.nq):
        if qpos_idx in qpos_to_jnt:
            print(f"qpos[{qpos_idx:2d}] = {qpos_to_jnt[qpos_idx]}")
        else:
            print(f"qpos[{qpos_idx:2d}] = [UNMAPPED]")
    
    return model, data

def compare_npz_vs_hdf5(npz_path, hdf5_npz_path, model):
    """Compare NPZ and HDF5-based reference data"""
    print("\n" + "=" * 80)
    print("COMPARING NPZ vs HDF5 CONVERTED DATA")
    print("=" * 80)
    
    npz_data = np.load(npz_path)
    hdf5_data = np.load(hdf5_npz_path)
    
    q_ref_npz = npz_data['q_ref']
    q_ref_hdf5 = hdf5_data['q_ref']
    
    print(f"\nNPZ q_ref shape: {q_ref_npz.shape}")
    print(f"HDF5 q_ref shape: {q_ref_hdf5.shape}")
    
    # Check frame 100 (middle of gait)
    frame = 100
    print(f"\n--- FRAME {frame} COMPARISON ---")
    
    joint_names = [
        'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
        'pelvis_list', 'pelvis_tilt', 'pelvis_rotation',
        'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
        'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
        'knee_angle_r', 'knee_angle_l',
        'ankle_angle_r', 'ankle_angle_l'
    ]
    
    print("\n{:20s} | {:>12s} | {:>12s} | {:>12s}".format(
        "Joint", "NPZ", "HDF5", "Diff"
    ))
    print("-" * 80)
    
    for i, name in enumerate(joint_names):
        npz_val = q_ref_npz[frame, i]
        hdf5_val = q_ref_hdf5[frame, i]
        diff = hdf5_val - npz_val
        
        # Convert to degrees for angles (not translations)
        if 'pelvis_t' not in name:
            npz_deg = np.degrees(npz_val)
            hdf5_deg = np.degrees(hdf5_val)
            diff_deg = np.degrees(diff)
            print(f"{name:20s} | {npz_deg:>10.2f}° | {hdf5_deg:>10.2f}° | {diff_deg:>10.2f}°")
        else:
            print(f"{name:20s} | {npz_val:>10.4f}m | {hdf5_val:>10.4f}m | {diff:>10.4f}m")
    
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON (ALL FRAMES)")
    print("=" * 80)
    
    for i, name in enumerate(joint_names):
        npz_vals = q_ref_npz[:, i]
        hdf5_vals = q_ref_hdf5[:, i]
        
        if 'pelvis_t' not in name:
            npz_min, npz_max = np.degrees(npz_vals.min()), np.degrees(npz_vals.max())
            hdf5_min, hdf5_max = np.degrees(hdf5_vals.min()), np.degrees(hdf5_vals.max())
            print(f"\n{name}:")
            print(f"  NPZ:  [{npz_min:>8.2f}°, {npz_max:>8.2f}°]")
            print(f"  HDF5: [{hdf5_min:>8.2f}°, {hdf5_max:>8.2f}°]")
        else:
            npz_min, npz_max = npz_vals.min(), npz_vals.max()
            hdf5_min, hdf5_max = hdf5_vals.min(), hdf5_vals.max()
            print(f"\n{name}:")
            print(f"  NPZ:  [{npz_min:>8.4f}m, {npz_max:>8.4f}m]")
            print(f"  HDF5: [{hdf5_min:>8.4f}m, {hdf5_max:>8.4f}m]")

def test_qpos_application(model, q_ref, frame_idx=100):
    """Test applying q_ref to qpos and check body positions"""
    data = mujoco.MjData(model)
    
    print("\n" + "=" * 80)
    print(f"TESTING QPOS APPLICATION (Frame {frame_idx})")
    print("=" * 80)
    
    # Define the mapping (from render_hdf5_reference.py)
    ref_to_qpos = [
        (0, 0, 'pelvis_tx'),
        (1, 1, 'pelvis_ty'),
        (2, 2, 'pelvis_tz'),
        (3, 3, 'pelvis_list'),
        (4, 4, 'pelvis_tilt'),
        (5, 5, 'pelvis_rotation'),
        (6, 6, 'hip_flexion_r'),
        (7, 7, 'hip_adduction_r'),
        (8, 8, 'hip_rotation_r'),
        (9, 9, 'hip_flexion_l'),
        (10, 10, 'hip_adduction_l'),
        (11, 11, 'hip_rotation_l'),
        (12, 12, 'knee_angle_r'),
        (13, 13, 'knee_angle_l'),
        (14, 14, 'ankle_angle_r'),
        (15, 15, 'ankle_angle_l'),
    ]
    
    print("\nApplying q_ref to qpos...")
    for ref_idx, qpos_idx, name in ref_to_qpos:
        data.qpos[qpos_idx] = q_ref[frame_idx, ref_idx]
        val = q_ref[frame_idx, ref_idx]
        if 'pelvis_t' in name:
            print(f"  qpos[{qpos_idx:2d}] = q_ref[{ref_idx:2d}] = {val:8.4f}m  ({name})")
        else:
            print(f"  qpos[{qpos_idx:2d}] = q_ref[{ref_idx:2d}] = {val:8.4f} ({np.degrees(val):>7.2f}°)  ({name})")
    
    # Forward kinematics
    mujoco.mj_forward(model, data)
    
    print("\n" + "=" * 80)
    print("BODY POSITIONS AFTER FORWARD KINEMATICS")
    print("=" * 80)
    
    important_bodies = [
        'pelvis', 'femur_r', 'tibia_r', 'talus_r', 'calcn_r', 'toes_r',
        'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l'
    ]
    
    for body_name in important_bodies:
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            pos = data.xpos[body_id]
            print(f"{body_name:15s}: pos=({pos[0]:>7.3f}, {pos[1]:>7.3f}, {pos[2]:>7.3f})")
        except:
            pass

if __name__ == "__main__":
    model_path = "models/26muscle_3D/myoLeg26_TUTORIAL.xml"
    npz_path = "rl_train/reference_data/S004_trial01_08mps_3D.npz"
    hdf5_npz_path = "rl_train/reference_data/S004_trial01_08mps_3D_HDF5_v5.npz"
    
    print("DEEP INSPECTION OF QPOS AND MODEL")
    print("=" * 80)
    
    # Inspect model
    model, data = inspect_model_structure(model_path)
    
    # Compare NPZ vs HDF5
    compare_npz_vs_hdf5(npz_path, hdf5_npz_path, model)
    
    # Test qpos application with HDF5 data
    hdf5_data = np.load(hdf5_npz_path)
    test_qpos_application(model, hdf5_data['q_ref'], frame_idx=100)
    
    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)
