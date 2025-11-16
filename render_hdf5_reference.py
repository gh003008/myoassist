#!/usr/bin/env python3
"""
Visualize HDF5-converted reference motion
"""
import numpy as np
import argparse
import mujoco
import imageio
from pathlib import Path

def render_reference_motion(npz_path, model_path, output_path, num_frames=300, height_offset=0.95):
    """Render reference motion from NPZ file
    
    Args:
        height_offset: Vertical offset to lift model above ground (meters)
                      Default 0.95m is approximately pelvis height for standing
    """
    
    print(f'Loading reference: {npz_path}')
    data = np.load(npz_path)
    q_ref = data['q_ref']
    joint_names = data['joint_names']
    
    print(f'  Frames: {q_ref.shape[0]}')
    print(f'  DOF: {q_ref.shape[1]}')
    print(f'  Joints: {list(joint_names)}')
    print(f'  Height offset: {height_offset:.3f} m')
    
    # Load MuJoCo model
    print(f'Loading model: {model_path}')
    model = mujoco.MjModel.from_xml_path(model_path)
    data_mj = mujoco.MjData(model)
    
    # Create joint name to qpos index mapping
    joint_to_qpos = {}
    for i in range(model.njnt):
        jnt_name = model.joint(i).name
        qpos_addr = model.jnt_qposadr[i]
        joint_to_qpos[jnt_name] = qpos_addr
    
    print(f'\n{"="*80}')
    print(f'QPOS MAPPING EXPLANATION')
    print(f'{"="*80}')
    print(f'Our reference data (q_ref) has {q_ref.shape[1]} DOF in a specific order.')
    print(f'MuJoCo model has {model.nq} qpos values (including auto-calculated wrapping points).')
    print(f'We need to map each q_ref column to the correct qpos index.\n')
    
    print(f'[Joint → qpos mapping]:')
    print(f'{"q_ref index":<15} {"Joint name":<25} {"→":<3} {"qpos index":<15} {"MuJoCo joint"}')
    print(f'{"-"*80}')
    
    # Map our reference data to correct qpos indices
    # Expected joint order in q_ref based on MYOASSIST_JOINTS (3D MODEL):
    # CRITICAL: q_ref order now matches qpos order for pelvis rotations!
    ref_joint_order = [
        ('q_pelvis_tx', 'pelvis_tx'),
        ('q_pelvis_ty', 'pelvis_ty'), 
        ('q_pelvis_tz', 'pelvis_tz'),
        ('q_pelvis_tilt', 'pelvis_tilt'),      # q_ref[3] → qpos[3]
        ('q_pelvis_list', 'pelvis_list'),      # q_ref[4] → qpos[4]
        ('q_pelvis_rotation', 'pelvis_rotation'),
        ('hip_flexion_r', 'hip_flexion_r'),
        ('hip_adduction_r', 'hip_adduction_r'),
        ('hip_rotation_r', 'hip_rotation_r'),
        ('hip_flexion_l', 'hip_flexion_l'),
        ('hip_adduction_l', 'hip_adduction_l'),
        ('hip_rotation_l', 'hip_rotation_l'),
        ('knee_angle_r', 'knee_angle_r'),
        ('knee_angle_l', 'knee_angle_l'),
        ('ankle_angle_r', 'ankle_angle_r'),
        ('ankle_angle_l', 'ankle_angle_l'),
    ]
    
    # Build mapping: ref_data column → qpos index
    ref_to_qpos = []
    for ref_idx, (ref_name, mujoco_name) in enumerate(ref_joint_order):
        if ref_idx < q_ref.shape[1] and mujoco_name and mujoco_name in joint_to_qpos:
            qpos_idx = joint_to_qpos[mujoco_name]
            ref_to_qpos.append((ref_idx, qpos_idx, mujoco_name))
            print(f'  q_ref[{ref_idx:2d}]      {ref_name:<25} →   qpos[{qpos_idx:2d}]        {mujoco_name}')
    
    print(f'{"-"*80}')
    print(f'Note: qpos indices are NOT sequential because muscle wrapping points')
    print(f'      (e.g., knee_r_translation1/2) are interspersed between main joints.')
    print(f'{"="*80}\n')
    
    # Setup renderer with better visualization options
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Configure camera for diagonal view (isometric-like)
    # Default is often straight-on; we want 3/4 view
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    
    # Set camera position for diagonal view
    # azimuth: rotation around vertical axis (degrees)
    # elevation: angle above horizontal (degrees)  
    # distance: how far from the model
    camera.azimuth = 135  # 45 degrees from side (diagonal)
    camera.elevation = -20  # Look down slightly
    camera.distance = 5.0  # Zoom level (increased from 3.5 to see more)
    camera.lookat[:] = [0, 0.5, 0]  # Look at pelvis height
    
    # Enable transparency and better rendering
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True  # Enable transparency
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False  # Hide contact points
    
    # Hide arm geoms (visual only - arms still in simulation)
    arm_body_names = ['humerus_r', 'ulna_r', 'radius_r', 'hand_r',
                      'humerus_l', 'ulna_l', 'radius_l', 'hand_l']
    for body_name in arm_body_names:
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            # Find all geoms attached to this body
            for i in range(model.ngeom):
                if model.geom_bodyid[i] == body_id:
                    # Disable visualization by setting geom group to invisible
                    # We'll disable geom group 0 in scene_option
                    pass
        except:
            pass
    
    # Alternatively, just disable specific geom categories
    # This is cleaner - hide all arm-related geoms by name pattern
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and any(part in geom_name.lower() for part in ['humer', 'ulna', 'radius', 'hand', 'arm']):
            # Set geom rgba to fully transparent
            model.geom_rgba[i, 3] = 0.0  # Alpha = 0 (invisible)
    
    print(f'\nCamera settings:')
    print(f'  View angle: Diagonal (azimuth={camera.azimuth}°, elevation={camera.elevation}°)')
    print(f'  Distance: {camera.distance}m')
    print(f'  Transparency: Enabled (can see through floor)')
    
    # Render frames
    print(f'\nRendering {num_frames} frames...')
    frames = []
    
    frame_skip = max(1, q_ref.shape[0] // num_frames)
    
    for i in range(0, min(num_frames * frame_skip, q_ref.shape[0]), frame_skip):
        # CRITICAL FIX: Use "stand" keyframe as base pose
        # qpos0 is all zeros, which breaks kinematic chain
        # "stand" keyframe has proper knee translations set
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        data_mj.qpos[:] = model.key_qpos[key_id]
        
        # Set qpos using correct mapping
        for ref_idx, qpos_idx, jnt_name in ref_to_qpos:
            data_mj.qpos[qpos_idx] = q_ref[i, ref_idx]
        
        # CRITICAL: Add height offset to pelvis_ty to lift model above ground
        # pelvis_ty is at qpos[1]
        # "stand" keyframe already has pelvis_ty=0.91, so we adjust relative to that
        if 1 < len(data_mj.qpos):
            data_mj.qpos[1] = model.key_qpos[key_id][1] + q_ref[i, 1] + (height_offset - 0.91)
        
        # Fix arms: Set arm joints to neutral pose (avoid weird positions)
        # Arms are qpos[40-53] based on model structure
        arm_joints = {
            40: 0.0,   # r_shoulder_abd
            41: 0.0,   # r_shoulder_rot
            42: 0.5,   # r_shoulder_flex (slightly forward)
            43: 0.8,   # r_elbow_flex (bent)
            47: 0.0,   # l_shoulder_abd
            48: 0.0,   # l_shoulder_rot  
            49: 0.5,   # l_shoulder_flex (slightly forward)
            50: 0.8,   # l_elbow_flex (bent)
        }
        for qpos_idx, angle in arm_joints.items():
            if qpos_idx < len(data_mj.qpos):
                data_mj.qpos[qpos_idx] = angle
        
        # Forward kinematics
        mujoco.mj_forward(model, data_mj)
        
        # Render with custom camera and transparency
        renderer.update_scene(data_mj, camera=camera, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)
        
        if (i // frame_skip) % 30 == 0:
            print(f'  Frame {i // frame_skip}/{num_frames}...')
        
        if (i // frame_skip) % 30 == 0:
            print(f'  Frame {i // frame_skip}/{num_frames}...')
    
    # Save video
    print(f'Saving video: {output_path}')
    # FPS calculation for desired video duration:
    # Original data: 100Hz, frame_skip frames apart
    # To make ~1min video from 2min data: effective_fps = num_frames / 60
    effective_fps = num_frames / 60.0  # Target: 1 minute video
    print(f'  Video FPS: {effective_fps:.1f} (target duration: ~60 seconds)')
    imageio.mimsave(output_path, frames, fps=effective_fps)
    
    # Print joint statistics
    print('\nJoint ranges:')
    for i, name in enumerate(joint_names):
        vals = q_ref[:, i]
        print(f'  {name:20s}: [{vals.min():+.3f}, {vals.max():+.3f}] rad')
    
    print(f'\n✅ Done! Saved: {output_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='S004_trial01_08mps_3D_HDF5_v1',
                        help='NPZ file name or path')
    parser.add_argument('--model', type=str, 
                        default=r'C:\workspace_home\myoassist\models\26muscle_3D\myoLeg26_TUTORIAL.xml',
                        help='MuJoCo model XML path')
    parser.add_argument('--frames', type=int, default=300,
                        help='Number of frames to render')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path')
    parser.add_argument('--height', type=float, default=0.95,
                        help='Height offset to lift model above ground (meters)')
    
    args = parser.parse_args()
    
    # Resolve paths
    if Path(args.data).exists():
        npz_path = args.data
    else:
        npz_path = Path('rl_train/reference_data') / f'{args.data}.npz'
    
    if args.output is None:
        args.output = f'ref_{Path(args.data).stem}.mp4'
    
    render_reference_motion(npz_path, args.model, args.output, args.frames, args.height)

if __name__ == '__main__':
    main()
