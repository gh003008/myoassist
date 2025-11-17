"""
Compare NPZ vs HDF5 rendering side by side
"""
import numpy as np
import mujoco
import imageio

def render_with_model(q_data, joint_order, model_path, num_frames=100, height_offset=0.95):
    """Render motion with given model
    
    Args:
        q_data: (n_frames, n_dof) array of joint positions
        joint_order: list of joint names in order
        model_path: path to MuJoCo XML
    """
    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Create joint mapping
    joint_to_qpos = {}
    for i in range(model.njnt):
        jnt_name = model.joint(i).name
        qpos_addr = model.jnt_qposadr[i]
        joint_to_qpos[jnt_name] = qpos_addr
    
    # Map joint order to qpos indices
    qpos_mapping = []
    for i, jnt_name in enumerate(joint_order):
        # Handle different naming conventions
        mujoco_name = jnt_name.replace('q_', '').replace('hip_flexion', 'hip_flexion').replace('knee_angle', 'knee_angle').replace('ankle_angle', 'ankle_angle')
        
        if mujoco_name in joint_to_qpos:
            qpos_idx = joint_to_qpos[mujoco_name]
            qpos_mapping.append((i, qpos_idx, mujoco_name))
            print(f"  {jnt_name:25s} → qpos[{qpos_idx:2d}]")
    
    # Setup renderer
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.azimuth = 135
    camera.elevation = -20
    camera.distance = 5.0
    camera.lookat[:] = [0, 0.5, 0]
    
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    
    frames = []
    frame_skip = max(1, len(q_data) // num_frames)
    
    print(f"Rendering {num_frames} frames (skip={frame_skip})...")
    for i in range(0, min(num_frames * frame_skip, len(q_data)), frame_skip):
        # Use model's default pose
        data.qpos[:] = model.qpos0[:]
        
        # Apply joint values
        for q_idx, qpos_idx, _ in qpos_mapping:
            data.qpos[qpos_idx] = q_data[i, q_idx]
        
        # Add height offset
        data.qpos[1] += height_offset
        
        # Forward kinematics
        mujoco.mj_forward(model, data)
        
        # Render
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)
    
    return frames

if __name__ == "__main__":
    model_path = "models/26muscle_3D/myoLeg26_BASELINE.xml"
    
    print("=" * 80)
    print("LOADING NPZ DATA")
    print("=" * 80)
    npz_data = np.load("rl_train/reference_data/S004_trial01_08mps_3D.npz", allow_pickle=True)
    series_data = npz_data['series_data'].item()
    
    # Extract joint data in order
    npz_joints = [
        'q_pelvis_tx', 'q_pelvis_ty', 'q_pelvis_tz',
        'q_pelvis_list', 'q_pelvis_tilt', 'q_pelvis_rotation',
        'q_hip_flexion_r', 'q_hip_adduction_r', 'q_hip_rotation_r',
        'q_hip_flexion_l', 'q_hip_adduction_l', 'q_hip_rotation_l',
        'q_knee_angle_r', 'q_knee_angle_l',
        'q_ankle_angle_r', 'q_ankle_angle_l'
    ]
    
    npz_q_data = np.column_stack([series_data[jnt] for jnt in npz_joints])
    print(f"NPZ shape: {npz_q_data.shape}")
    print(f"NPZ pelvis_ty range: [{npz_q_data[:, 1].min():.4f}, {npz_q_data[:, 1].max():.4f}]")
    print(f"NPZ knee_r range: [{npz_q_data[:, 12].min():.4f}, {npz_q_data[:, 12].max():.4f}]")
    
    print("\n" + "=" * 80)
    print("LOADING HDF5 DATA")
    print("=" * 80)
    hdf5_data = np.load("rl_train/reference_data/S004_trial01_08mps_3D_HDF5_v6.npz")
    hdf5_q_data = hdf5_data['q_ref']
    hdf5_joints = [str(n) for n in hdf5_data['joint_names']]
    
    print(f"HDF5 shape: {hdf5_q_data.shape}")
    print(f"HDF5 pelvis_ty range: [{hdf5_q_data[:, 1].min():.4f}, {hdf5_q_data[:, 1].max():.4f}]")
    print(f"HDF5 knee_r range: [{hdf5_q_data[:, 12].min():.4f}, {hdf5_q_data[:, 12].max():.4f}]")
    
    print("\n" + "=" * 80)
    print("RENDERING NPZ")
    print("=" * 80)
    npz_frames = render_with_model(npz_q_data, npz_joints, model_path, num_frames=300)
    
    print("\n" + "=" * 80)
    print("RENDERING HDF5")
    print("=" * 80)
    hdf5_frames = render_with_model(hdf5_q_data, hdf5_joints, model_path, num_frames=300)
    
    print("\n" + "=" * 80)
    print("SAVING VIDEOS")
    print("=" * 80)
    imageio.mimsave("compare_NPZ_original.mp4", npz_frames, fps=15)
    print("✅ Saved: compare_NPZ_original.mp4")
    
    imageio.mimsave("compare_HDF5_v6.mp4", hdf5_frames, fps=15)
    print("✅ Saved: compare_HDF5_v6.mp4")
    
    print("\n" + "=" * 80)
    print("DONE - Compare the two videos!")
    print("=" * 80)
