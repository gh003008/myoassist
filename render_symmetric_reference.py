#!/usr/bin/env python3
"""
Visualize symmetric reference motion for Ver2_1 training

This is a standalone rendering script that does NOT modify environment code.
Based on render_hdf5_reference.py (proven working approach).

Usage:
    python render_symmetric_reference.py
    python render_symmetric_reference.py --multiview --fps 100
"""
import numpy as np
import argparse
import mujoco
import imageio
from pathlib import Path

def render_symmetric_reference(
    npz_path='rl_train/reference_data/S004_trial01_08mps_3D_HDF5_v7_symmetric.npz',
    model_path=r'models\26muscle_3D\myoLeg26_BASELINE.xml',
    output_path='symmetric_reference.mp4',
    num_frames=600,
    height_offset=0.95,
    fps=100,
    multiview=False
):
    """Render symmetric reference motion from NPZ file
    
    This script uses direct MuJoCo rendering (no environment wrapper).
    It correctly handles the reference data format and kinematic chain.
    
    Args:
        npz_path: Path to symmetric reference NPZ file
        model_path: Path to MuJoCo model XML
        output_path: Output video path
        num_frames: Number of frames to render
        height_offset: Vertical offset to lift model (meters)
        fps: Frames per second for output video
        multiview: If True, render front and side views side-by-side
    """
    
    print(f'='*80)
    print(f'SYMMETRIC REFERENCE MOTION RENDERER')
    print(f'='*80)
    print(f'Loading reference: {npz_path}')
    
    data = np.load(npz_path)
    
    # Detect data format (environment format vs MuJoCo renderer format)
    if 'q_ref' in data:
        # MuJoCo renderer format (with q_ prefix)
        q_ref = data['q_ref']
        joint_names_raw = data['joint_names']
        # Remove 'q_' prefix from joint names for MuJoCo compatibility
        joint_names = np.array([str(name).replace('q_', '') if str(name).startswith('q_') else str(name) 
                                for name in joint_names_raw])
        data_format = 'MuJoCo renderer'
    elif 'series_data' in data:
        # Environment format (no q_ prefix)
        series_data = data['series_data']
        metadata = data['metadata'].item()
        
        # Extract reference joint data
        ref_joints = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz', 
                      'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                      'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                      'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
                      'knee_angle_r', 'knee_angle_l',
                      'ankle_angle_r', 'ankle_angle_l']
        
        # Build q_ref array from series_data
        q_ref = np.column_stack([series_data[jnt] for jnt in ref_joints])
        joint_names = np.array(ref_joints)
        data_format = 'Environment'
    else:
        raise ValueError('Unknown data format! Expected either q_ref or series_data')
    
    print(f'  Data format: {data_format}')
    print(f'  Frames: {q_ref.shape[0]}')
    print(f'  DOF: {q_ref.shape[1]}')
    print(f'  Joints: {list(joint_names)}')
    print(f'  Height offset: {height_offset:.3f} m')
    print(f'  FPS: {fps}')
    print(f'  Multiview: {"Yes (Front + Side)" if multiview else "No (Diagonal only)"}')
    
    # Load MuJoCo model
    print(f'\nLoading model: {model_path}')
    model = mujoco.MjModel.from_xml_path(model_path)
    data_mj = mujoco.MjData(model)
    
    # Create joint name to qpos index mapping
    joint_to_qpos = {}
    for i in range(model.njnt):
        jnt_name = model.joint(i).name
        qpos_addr = model.jnt_qposadr[i]
        joint_to_qpos[jnt_name] = qpos_addr
    
    print(f'\n{"="*80}')
    print(f'QPOS MAPPING')
    print(f'{"="*80}')
    print(f'{"q_ref index":<15} {"Joint name":<25} {"→":<3} {"qpos index"}')
    print(f'{"-"*80}')
    
    # Map reference data to qpos indices
    ref_to_qpos = []
    for ref_idx, jnt_name in enumerate(joint_names):
        if jnt_name in joint_to_qpos:
            qpos_idx = joint_to_qpos[jnt_name]
            ref_to_qpos.append((ref_idx, qpos_idx, jnt_name))
            print(f'  q_ref[{ref_idx:2d}]      {jnt_name:<25} →   qpos[{qpos_idx:2d}]')
    
    print(f'{"-"*80}')
    print(f'Note: Model has {model.nq} total qpos (including muscle wrapping points)')
    print(f'{"="*80}\n')
    
    # Symmetry check
    print(f'{"="*80}')
    print(f'SYMMETRY CHECK: Left vs Right Joint Comparison')
    print(f'{"="*80}')
    
    symmetric_pairs = [
        ('hip_flexion_l', 'hip_flexion_r'),
        ('hip_adduction_l', 'hip_adduction_r'),
        ('hip_rotation_l', 'hip_rotation_r'),
        ('knee_angle_l', 'knee_angle_r'),
        ('ankle_angle_l', 'ankle_angle_r'),
    ]
    
    print(f'{"Joint Pair":<40} {"Range Diff":<12} {"Symmetric?"}')
    print(f'{"-"*80}')
    
    joint_name_to_idx = {name: i for i, name in enumerate(joint_names)}
    
    for left_name, right_name in symmetric_pairs:
        if left_name in joint_name_to_idx and right_name in joint_name_to_idx:
            left_idx = joint_name_to_idx[left_name]
            right_idx = joint_name_to_idx[right_name]
            
            left_vals = q_ref[:, left_idx]
            right_vals = q_ref[:, right_idx]
            
            left_range = left_vals.max() - left_vals.min()
            right_range = right_vals.max() - right_vals.min()
            range_diff = abs(left_range - right_range)
            
            is_symmetric = range_diff < 0.05  # < 3 degrees
            status = "✅ Yes" if is_symmetric else "⚠️  CHECK"
            
            print(f'{left_name} vs {right_name:<20} {range_diff:>8.4f} rad   {status}')
            print(f'  L: [{left_vals.min():+.3f}, {left_vals.max():+.3f}]  R: [{right_vals.min():+.3f}, {right_vals.max():+.3f}]')
    
    print(f'{"-"*80}')
    print(f'{"="*80}\n')
    
    # Setup renderer
    if multiview:
        renderer = mujoco.Renderer(model, height=720, width=1920)
        print(f'🎥 Renderer: 1920x720 (Front + Side views)')
    else:
        renderer = mujoco.Renderer(model, height=720, width=1280)
        print(f'🎥 Renderer: 1280x720 (Diagonal view)')
    
    # Configure cameras
    if multiview:
        camera_front = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera_front)
        camera_front.azimuth = 90
        camera_front.elevation = -15
        camera_front.distance = 4.5
        camera_front.lookat[:] = [0, 0.7, 0]
        
        camera_side = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera_side)
        camera_side.azimuth = 180
        camera_side.elevation = -20
        camera_side.distance = 3.0
        camera_side.lookat[:] = [0, 0.4, 0]
    else:
        camera_diagonal = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera_diagonal)
        camera_diagonal.azimuth = 135
        camera_diagonal.elevation = -20
        camera_diagonal.distance = 5.0
        camera_diagonal.lookat[:] = [0, 0.5, 0]
    
    # Rendering options
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    
    # Semi-transparent floor
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and 'floor' in geom_name.lower():
            model.geom_rgba[i, 3] = 0.3
    
    # Hide arms
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and any(part in geom_name.lower() for part in ['humer', 'ulna', 'radius', 'hand', 'arm']):
            model.geom_rgba[i, 3] = 0.0
    
    # Render frames
    print(f'\nRendering {num_frames} frames...')
    frames = []
    
    frame_skip = max(1, q_ref.shape[0] // num_frames)
    
    for i in range(0, min(num_frames * frame_skip, q_ref.shape[0]), frame_skip):
        # CRITICAL: Use "stand" keyframe as base pose
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        data_mj.qpos[:] = model.key_qpos[key_id]
        
        # Overlay reference motion
        for ref_idx, qpos_idx, jnt_name in ref_to_qpos:
            data_mj.qpos[qpos_idx] = q_ref[i, ref_idx]
        
        # Adjust pelvis height
        if 1 < len(data_mj.qpos):
            data_mj.qpos[1] = model.key_qpos[key_id][1] + q_ref[i, 1] + (height_offset - 0.91)
        
        # Fix arms to neutral pose
        arm_joints = {
            40: 0.0, 41: 0.0, 42: 0.5, 43: 0.8,  # Right arm
            47: 0.0, 48: 0.0, 49: 0.5, 50: 0.8,  # Left arm
        }
        for qpos_idx, angle in arm_joints.items():
            if qpos_idx < len(data_mj.qpos):
                data_mj.qpos[qpos_idx] = angle
        
        # Forward kinematics
        mujoco.mj_forward(model, data_mj)
        
        # Render
        if multiview:
            renderer.update_scene(data_mj, camera=camera_front, scene_option=scene_option)
            pixels_front = renderer.render()
            front_half = pixels_front[:, 480:1440]
            
            renderer.update_scene(data_mj, camera=camera_side, scene_option=scene_option)
            pixels_side = renderer.render()
            side_half = pixels_side[:, 480:1440]
            
            pixels = np.concatenate([front_half, side_half], axis=1)
        else:
            renderer.update_scene(data_mj, camera=camera_diagonal, scene_option=scene_option)
            pixels = renderer.render()
        
        frames.append(pixels)
        
        if (i // frame_skip) % 30 == 0:
            print(f'  Frame {i // frame_skip}/{num_frames}...')
    
    # Save video
    print(f'\nSaving video: {output_path}')
    print(f'  Video FPS: {fps}')
    imageio.mimsave(output_path, frames, fps=fps)
    
    # Joint statistics
    print('\nJoint ranges:')
    for i, name in enumerate(joint_names):
        vals = q_ref[:, i]
        print(f'  {name:20s}: [{vals.min():+.3f}, {vals.max():+.3f}] rad')
    
    print(f'\n✅ Done! Saved: {output_path}')
    print(f'\nNext steps:')
    print(f'  1. Watch the video to verify symmetric motion')
    print(f'  2. Check that kinematic chain is correct (no "shin stuck to hip")')
    print(f'  3. If rendering looks good, proceed with Ver2_1 training')
    print(f'='*80)

def main():
    parser = argparse.ArgumentParser(
        description='Render symmetric reference motion for Ver2_1 training verification'
    )
    parser.add_argument('--data', type=str, 
                        default='rl_train/reference_data/S004_trial01_08mps_3D_HDF5_v7_symmetric.npz',
                        help='Path to symmetric reference NPZ file')
    parser.add_argument('--model', type=str,
                        default=r'models\26muscle_3D\myoLeg26_BASELINE.xml',
                        help='Path to MuJoCo model XML')
    parser.add_argument('--frames', type=int, default=600,
                        help='Number of frames to render')
    parser.add_argument('--fps', type=int, default=100,
                        help='Output video FPS')
    parser.add_argument('--output', type=str, default='symmetric_reference.mp4',
                        help='Output video path')
    parser.add_argument('--height', type=float, default=0.95,
                        help='Height offset in meters')
    parser.add_argument('--multiview', action='store_true',
                        help='Render front and side views side-by-side')
    
    args = parser.parse_args()
    
    render_symmetric_reference(
        npz_path=args.data,
        model_path=args.model,
        output_path=args.output,
        num_frames=args.frames,
        height_offset=args.height,
        fps=args.fps,
        multiview=args.multiview
    )

if __name__ == '__main__':
    main()
