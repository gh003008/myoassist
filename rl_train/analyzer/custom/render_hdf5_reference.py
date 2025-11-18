#!/usr/bin/env python3
"""
Visualize HDF5-converted reference motion
"""
import numpy as np
import argparse
import mujoco
import imageio
from pathlib import Path

def render_reference_motion(npz_path, model_path, output_path, num_frames=300, height_offset=0.95, fps=100, multiview=False):
    """Render reference motion from NPZ file
    
    Args:
        height_offset: Vertical offset to lift model above ground (meters)
                      Default 0.95m is approximately pelvis height for standing
        fps: Frames per second for output video (default: 30)
        multiview: If True, render front and side views side-by-side
    """
    
    print(f'Loading reference: {npz_path}')
    data = np.load(npz_path)
    q_ref = data['q_ref'].copy()  # Make a copy for modification
    joint_names = data['joint_names']
    
    print(f'  Frames: {q_ref.shape[0]}')
    print(f'  DOF: {q_ref.shape[1]}')
    print(f'  Joints: {list(joint_names)}')
    print(f'  Height offset: {height_offset:.3f} m')
    print(f'  FPS: {fps}')
    print(f'  Multiview: {"Yes (Front + Side)" if multiview else "No (Diagonal only)"}')
    
    # CRITICAL FIX: Flip sign of left hip adduction (index 10)
    # Left hip ab/adduction coordinate system requires sign flip
    if q_ref.shape[1] > 10:  # Make sure we have enough DOFs
        hip_add_l_idx = None
        for i, name in enumerate(joint_names):
            if 'hip_adduction_l' in str(name):
                hip_add_l_idx = i
                break
        
        if hip_add_l_idx is not None:
            q_ref[:, hip_add_l_idx] = -q_ref[:, hip_add_l_idx]
            print(f'  🔄 Applied sign flip to hip_adduction_l (index {hip_add_l_idx})')
    
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
    
    # ============================================================================
    # SYMMETRY CHECK: Compare left vs right joint values
    # ============================================================================
    print(f'{"="*80}')
    print(f'SYMMETRY CHECK: Left vs Right Joint Comparison')
    print(f'{"="*80}')
    
    # Define symmetric joint pairs (left vs right)
    symmetric_pairs = [
        ('hip_flexion_l', 'hip_flexion_r', 9, 6),
        ('hip_adduction_l', 'hip_adduction_r', 10, 7),
        ('hip_rotation_l', 'hip_rotation_r', 11, 8),
        ('knee_angle_l', 'knee_angle_r', 13, 12),
        ('ankle_angle_l', 'ankle_angle_r', 15, 14),
    ]
    
    print(f'{"Joint Pair":<35} {"Mean Diff":<12} {"Max Diff":<12} {"Symmetric?"}')
    print(f'{"-"*80}')
    
    for left_name, right_name, left_idx, right_idx in symmetric_pairs:
        left_vals = q_ref[:, left_idx]
        right_vals = q_ref[:, right_idx]
        
        # For symmetric gait, left and right should have similar ranges but phase-shifted
        # Check if magnitude ranges are similar
        left_range = left_vals.max() - left_vals.min()
        right_range = right_vals.max() - right_vals.min()
        range_diff = abs(left_range - right_range)
        
        # Mean absolute difference
        mean_diff = np.mean(np.abs(left_vals - right_vals))
        max_diff = np.max(np.abs(left_vals - right_vals))
        
        # Symmetric if range difference is small (< 0.05 rad or 3°)
        is_symmetric = range_diff < 0.05
        status = "✅ Yes" if is_symmetric else "⚠️  CHECK"
        
        print(f'{left_name} vs {right_name:<15} {mean_diff:>8.4f} rad   {max_diff:>8.4f} rad   {status}')
        print(f'  Range: L=[{left_vals.min():+.3f}, {left_vals.max():+.3f}] vs R=[{right_vals.min():+.3f}, {right_vals.max():+.3f}]  (diff={range_diff:.4f} rad)')
    
    print(f'{"-"*80}')
    print(f'Note: Symmetric gait means LEFT and RIGHT have similar motion RANGES,')
    print(f'      but they are PHASE-SHIFTED (when left swings, right supports).')
    print(f'{"="*80}\n')
    
    # Setup renderer with better visualization options
    if multiview:
        # Multiview: side-by-side rendering (front + side)
        # Use smaller resolution to fit within framebuffer limits
        renderer = mujoco.Renderer(model, height=720, width=1920)  # Max framebuffer width
        print(f'\n🎥 Renderer: 1920x720 (Front + Side views side-by-side)')
    else:
        renderer = mujoco.Renderer(model, height=720, width=1280)
        print(f'\n🎥 Renderer: 1280x720 (Diagonal view only)')
    
    # Configure cameras
    if multiview:
        # Front view camera
        camera_front = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera_front)
        camera_front.azimuth = 90      # Front view
        camera_front.elevation = -15   # Slightly above
        camera_front.distance = 4.5
        camera_front.lookat[:] = [0, 0.7, 0]
        
        # Side view camera
        camera_side = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera_side)
        camera_side.azimuth = 180        # Side view (sagittal plane)
        camera_side.elevation = -20
        # camera_side.elevation = -10
        camera_side.distance = 3.0
        camera_side.lookat[:] = [0, 0.4, 0]
        
        print(f'  Front view: azimuth=90° (frontal plane)')
        print(f'  Side view:  azimuth=180° (sagittal plane)')
    else:
        # Diagonal view camera
        camera_diagonal = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera_diagonal)
        camera_diagonal.azimuth = 135   # 45 degrees from side (diagonal)
        camera_diagonal.elevation = -20  # Look down slightly
        camera_diagonal.distance = 5.0
        camera_diagonal.lookat[:] = [0, 0.5, 0]
        
        print(f'  Diagonal view: azimuth=135° (3/4 view)')
    
    # Enable transparency and better rendering
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True  # Enable transparency
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False  # Hide contact points
    
    # Make floor transparent
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and 'floor' in geom_name.lower():
            # Set floor to semi-transparent (alpha = 0.3)
            model.geom_rgba[i, 3] = 0.3
            print(f'  Floor transparency set: {geom_name} (alpha=0.3)')
    
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
    
    print(f'  Transparency: Enabled (floor alpha=0.3, arms hidden)')
    
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
        
        # Render with multiview or single view
        if multiview:
            # Render front view (left half: 960px)
            renderer.update_scene(data_mj, camera=camera_front, scene_option=scene_option)
            pixels_front = renderer.render()
            # front_half = pixels_front[:, :960]  # Left 960 pixels
            front_half = pixels_front[:, 480:1440]  # 중앙 960픽셀

            # Render side view (right half: 960px)
            renderer.update_scene(data_mj, camera=camera_side, scene_option=scene_option)
            pixels_side = renderer.render()
            # side_half = pixels_side[:, :960]  # Left 960 pixels
            side_half = pixels_side[:, 480:1440]    # 중앙 960픽셀
            
            # Concatenate horizontally (960 + 960 = 1920)
            pixels = np.concatenate([front_half, side_half], axis=1)
        else:
            # Single diagonal view
            renderer.update_scene(data_mj, camera=camera_diagonal, scene_option=scene_option)
            pixels = renderer.render()
        
        frames.append(pixels)
        
        if (i // frame_skip) % 30 == 0:
            print(f'  Frame {i // frame_skip}/{num_frames}...')
    
    # Save video
    print(f'Saving video: {output_path}')
    print(f'  Video FPS: {fps} (smooth playback)')
    imageio.mimsave(output_path, frames, fps=fps)
    
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
    parser.add_argument('--frames', type=int, default=600,
                        help='Number of frames to render (default: 600 for smooth motion)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Output video FPS (default: 30)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path')
    parser.add_argument('--height', type=float, default=0.95,
                        help='Height offset to lift model above ground (meters)')
    parser.add_argument('--multiview', action='store_true',
                        help='Render front and side views side-by-side')
    
    args = parser.parse_args()
    
    # Resolve paths
    if Path(args.data).exists():
        npz_path = args.data
    else:
        npz_path = Path('rl_train/reference_data') / f'{args.data}.npz'
    
    if args.output is None:
        suffix = '_multiview' if args.multiview else ''
        args.output = f'ref_{Path(args.data).stem}{suffix}.mp4'
    
    render_reference_motion(npz_path, args.model, args.output, args.frames, args.height, args.fps, args.multiview)

if __name__ == '__main__':
    main()
