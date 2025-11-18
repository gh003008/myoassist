#!/usr/bin/env python3
"""
View symmetric reference motion in real-time with MuJoCo viewer.
Press SPACE to pause/resume, ESC to quit.
"""
import numpy as np
import mujoco
import mujoco.viewer
import argparse
import time


def view_reference_interactively(npz_path, model_path, playback_speed=0.5, height_offset=0.95):
    """
    View reference motion with interactive MuJoCo viewer
    
    Args:
        npz_path: Path to NPZ reference data
        model_path: Path to MuJoCo model XML
        playback_speed: Speed multiplier (0.5 = half speed, 2.0 = double speed)
        height_offset: Vertical offset for pelvis
    """
    
    print(f"\n{'='*80}")
    print(f"INTERACTIVE REFERENCE MOTION VIEWER")
    print(f"{'='*80}\n")
    
    print(f"Loading: {npz_path}")
    data = np.load(npz_path)
    q_ref = data['q_ref']
    joint_names = data['joint_names']
    
    print(f"  Frames: {q_ref.shape[0]}")
    print(f"  DOF: {q_ref.shape[1]}")
    print(f"  Duration: {q_ref.shape[0]/100:.1f} seconds @ 100 Hz")
    print(f"  Playback speed: {playback_speed}x")
    print(f"\n🎮 Controls:")
    print(f"  SPACE: Pause/Resume")
    print(f"  R: Restart from beginning")
    print(f"  +/-: Speed up/slow down")
    print(f"  ESC: Quit\n")
    
    # Load model
    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data_mj = mujoco.MjData(model)
    
    # Create joint name to qpos index mapping
    joint_to_qpos = {}
    for i in range(model.njnt):
        jnt_name = model.joint(i).name
        qpos_addr = model.jnt_qposadr[i]
        joint_to_qpos[jnt_name] = qpos_addr
    
    # Create reference to qpos mapping
    ref_to_qpos = []
    for ref_idx, jnt_name in enumerate(joint_names):
        jnt_name_str = str(jnt_name)
        
        # Remove 'q_' prefix if present
        if jnt_name_str.startswith('q_'):
            clean_name = jnt_name_str[2:]
        else:
            clean_name = jnt_name_str
        
        if clean_name in joint_to_qpos:
            qpos_idx = joint_to_qpos[clean_name]
            ref_to_qpos.append((ref_idx, qpos_idx, clean_name))
    
    print(f"✅ Mapped {len(ref_to_qpos)} joints\n")
    
    # Set floor transparency
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and 'floor' in geom_name.lower():
            model.geom_rgba[i, 3] = 0.3
    
    # Hide arms
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and any(part in geom_name.lower() for part in ['humer', 'ulna', 'radius', 'hand', 'arm']):
            model.geom_rgba[i, 3] = 0.0
    
    # Animation state
    frame_idx = 0
    paused = False
    last_time = time.time()
    
    # Launch viewer
    print("🚀 Launching interactive viewer...")
    print("=" * 80)
    
    with mujoco.viewer.launch_passive(model, data_mj) as viewer:
        # Set initial camera
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 90  # Front view
        viewer.cam.elevation = -15
        viewer.cam.lookat[:] = [0, 0.7, 0]
        
        # Animation loop
        while viewer.is_running():
            current_time = time.time()
            dt = current_time - last_time
            
            # Control playback speed
            target_dt = (1.0 / 100.0) / playback_speed  # 100 Hz adjusted by speed
            
            if not paused and dt >= target_dt:
                last_time = current_time
                
                # Get "stand" keyframe as base
                key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
                data_mj.qpos[:] = model.key_qpos[key_id]
                
                # Set qpos from reference
                for ref_idx, qpos_idx, jnt_name in ref_to_qpos:
                    data_mj.qpos[qpos_idx] = q_ref[frame_idx, ref_idx]
                
                # Adjust pelvis height
                if 1 < len(data_mj.qpos):
                    data_mj.qpos[1] = model.key_qpos[key_id][1] + q_ref[frame_idx, 1] + (height_offset - 0.91)
                
                # Set arm poses
                arm_joints = {
                    40: 0.0, 41: 0.0, 42: 0.5, 43: 0.8,  # Right arm
                    47: 0.0, 48: 0.0, 49: 0.5, 50: 0.8,  # Left arm
                }
                for qpos_idx, angle in arm_joints.items():
                    if qpos_idx < len(data_mj.qpos):
                        data_mj.qpos[qpos_idx] = angle
                
                # Forward kinematics
                mujoco.mj_forward(model, data_mj)
                
                # Advance frame
                frame_idx += 1
                if frame_idx >= q_ref.shape[0]:
                    frame_idx = 0  # Loop
                
                # Print status in console
                if frame_idx % 100 == 0:
                    progress = (frame_idx / q_ref.shape[0]) * 100
                    status = "PAUSED" if paused else f"Playing {playback_speed}x"
                    print(f"\rFrame {frame_idx}/{q_ref.shape[0]} ({progress:.1f}%) - {status}", end='', flush=True)
            
            # Update viewer
            viewer.sync()
            
            # Small sleep to prevent CPU overload
            time.sleep(0.001)
    
    print("\n✅ Viewer closed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View reference motion interactively')
    parser.add_argument('--data', type=str,
                       default='rl_train/reference_data/S004_trial01_08mps_3D_HDF5_v7_symmetric.npz',
                       help='Path to NPZ reference data')
    parser.add_argument('--model', type=str,
                       default='models/26muscle_3D/myoLeg26_BASELINE.xml',
                       help='Path to MuJoCo model XML')
    parser.add_argument('--speed', type=float, default=0.5,
                       help='Playback speed multiplier (default: 0.5 = half speed)')
    parser.add_argument('--height', type=float, default=0.95,
                       help='Pelvis height offset in meters (default: 0.95)')
    
    args = parser.parse_args()
    
    view_reference_interactively(
        args.data,
        args.model,
        playback_speed=args.speed,
        height_offset=args.height
    )
