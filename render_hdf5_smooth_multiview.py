"""
Smooth multi-view reference motion renderer with interpolation
- Supports multiple camera angles (diagonal + front)
- Smooth interpolation for fluid motion
- Batch processing for multiple motion types
"""

import numpy as np
import mujoco
import argparse
import os
from scipy import interpolate
import imageio

# Configuration
CONFIG = {
    'hdf5_path': r'C:\workspace_home\opensim data\LD\S004.h5',
    'model_path': 'models/26muscle_3D/myoLeg26_BASELINE.xml',
    'output_dir': 'reference_videos',
    'height_offset_m': 0.95,  # Pelvis height in "stand" keyframe
    'interpolation_factor': 4,  # 4x more frames for smooth motion
}

# Joint mapping (q_ref index → qpos index, joint name)
JOINT_MAPPING = [
    (0, 0, 'pelvis_tx'),
    (1, 1, 'pelvis_ty'),
    (2, 2, 'pelvis_tz'),
    (3, 3, 'pelvis_tilt'),
    (4, 4, 'pelvis_list'),
    (5, 5, 'pelvis_rotation'),
    (6, 6, 'hip_flexion_r'),
    (7, 7, 'hip_adduction_r'),
    (8, 8, 'hip_rotation_r'),
    (9, 23, 'hip_flexion_l'),
    (10, 24, 'hip_adduction_l'),
    (11, 25, 'hip_rotation_l'),
    (12, 11, 'knee_angle_r'),
    (13, 28, 'knee_angle_l'),
    (14, 12, 'ankle_angle_r'),
    (15, 29, 'ankle_angle_l'),
]

def load_and_convert_hdf5(hdf5_path, subject, motion_type, trial):
    """Load HDF5 data and convert to MyoAssist format"""
    import h5py
    
    with h5py.File(hdf5_path, 'r') as f:
        # Navigate to kin_q group
        kin_q = f[subject][motion_type][trial]['MoCap']['kin_q']
        
        # Load data (unit-aware)
        data = {}
        translation_keys = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']
        
        for key in kin_q.keys():
            if key == 'infos':
                continue  # Skip metadata
            raw_data = kin_q[key][()]
            if key in translation_keys:
                data[key] = raw_data  # METERS
            else:
                data[key] = np.radians(raw_data)  # DEGREES → radians
    
    # Convert to MyoAssist format
    num_frames = len(data['pelvis_tx'])
    q_ref = np.zeros((num_frames, 16))
    
    # Map to q_ref array
    mapping = {
        0: data['pelvis_tx'],
        1: data['pelvis_ty'],
        2: data['pelvis_tz'],
        3: data['pelvis_tilt'],
        4: data['pelvis_list'],
        5: data['pelvis_rotation'],
        6: data['hip_flexion_r'],
        7: data['hip_adduction_r'],
        8: data['hip_rotation_r'],
        9: data['hip_flexion_l'],
        10: data['hip_adduction_l'],
        11: data['hip_rotation_l'],
        12: data['knee_angle_r'],
        13: data['knee_angle_l'],
        14: data['ankle_angle_r'],
        15: data['ankle_angle_l'],
    }
    
    for i, values in mapping.items():
        q_ref[:, i] = values
    
    # Make relative positions (centered at 0)
    pelvis_ty_mean = np.mean(q_ref[:, 1])
    q_ref[:, 1] -= pelvis_ty_mean
    
    return q_ref

def interpolate_motion(q_ref, factor=4):
    """
    Interpolate motion data for smoother animation
    
    Args:
        q_ref: Original motion data (N, 16)
        factor: Interpolation factor (4 = 4x more frames)
    
    Returns:
        Interpolated motion data (N*factor, 16)
    """
    num_frames = q_ref.shape[0]
    num_joints = q_ref.shape[1]
    
    # Original time points
    t_orig = np.linspace(0, 1, num_frames)
    
    # New time points (much denser)
    t_new = np.linspace(0, 1, num_frames * factor)
    
    # Interpolate each joint independently
    q_interp = np.zeros((len(t_new), num_joints))
    
    for j in range(num_joints):
        # Use cubic spline for smooth interpolation
        cs = interpolate.CubicSpline(t_orig, q_ref[:, j])
        q_interp[:, j] = cs(t_new)
    
    return q_interp

def hide_arms(model):
    """Make arm geometries transparent"""
    arm_parts = ['humer', 'ulna', 'radius', 'hand']
    
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name:
            if any(part in geom_name.lower() for part in arm_parts):
                model.geom_rgba[i, 3] = 0.0  # Transparent

def setup_camera(camera, view_type='diagonal'):
    """Setup camera for different views"""
    if view_type == 'diagonal':
        # Diagonal view (original)
        camera.azimuth = 135
        camera.elevation = -20
        camera.distance = 5.0
        camera.lookat[:] = [0, 0.95, 0]
    elif view_type == 'front':
        # Front view
        camera.azimuth = 90
        camera.elevation = -10
        camera.distance = 4.0
        camera.lookat[:] = [0, 0.95, 0]
    elif view_type == 'side':
        # Side view
        camera.azimuth = 0
        camera.elevation = -10
        camera.distance = 4.0
        camera.lookat[:] = [0, 0.95, 0]

def render_motion(q_ref, model_path, output_path, view_type='diagonal', 
                  height_offset=0.95, target_duration=60):
    """
    Render reference motion with specified camera view
    
    Args:
        q_ref: Motion data (N, 16)
        model_path: Path to MuJoCo model
        output_path: Output video path
        view_type: 'diagonal', 'front', or 'side'
        height_offset: Pelvis height offset
        target_duration: Target video duration in seconds
    """
    print(f"\n[{view_type.upper()} VIEW]")
    print(f"Loading model: {model_path}")
    
    # Load model
    model = mujoco.MjModel.from_xml_path(model_path)
    data_mj = mujoco.MjData(model)
    
    # Hide arms
    hide_arms(model)
    
    # Get "stand" keyframe for proper initialization
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id == -1:
        raise ValueError("'stand' keyframe not found in model!")
    
    stand_qpos = model.key_qpos[key_id].copy()
    
    print(f"Motion frames: {len(q_ref)}")
    print(f"Target duration: {target_duration}s")
    
    # Calculate FPS for target duration
    effective_fps = len(q_ref) / target_duration
    print(f"Video FPS: {effective_fps:.1f}")
    
    # Create renderer
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Setup camera
    camera = mujoco.MjvCamera()
    setup_camera(camera, view_type)
    
    frames = []
    print(f"Rendering {len(q_ref)} frames...")
    
    for i in range(len(q_ref)):
        if (i + 1) % 100 == 0:
            print(f"  Frame {i+1}/{len(q_ref)}")
        
        # Initialize with stand keyframe
        data_mj.qpos[:] = stand_qpos
        
        # Apply reference motion
        for ref_idx, qpos_idx, _ in JOINT_MAPPING:
            data_mj.qpos[qpos_idx] = q_ref[i, ref_idx]
        
        # Adjust pelvis height (relative to keyframe base)
        data_mj.qpos[1] = stand_qpos[1] + q_ref[i, 1] + (height_offset - 0.91)
        
        # Forward kinematics
        mujoco.mj_forward(model, data_mj)
        
        # Render
        renderer.update_scene(data_mj, camera=camera)
        pixels = renderer.render()
        frames.append(pixels)
    
    # Save video
    print(f"Saving video: {output_path}")
    imageio.mimsave(output_path, frames, fps=effective_fps, codec='libx264', quality=8)
    print(f"✅ Done! Duration: {len(q_ref)/effective_fps:.1f}s")

def create_side_by_side(video1, video2, output_path):
    """Create side-by-side video from two videos"""
    print(f"\nCreating side-by-side video...")
    
    reader1 = imageio.get_reader(video1)
    reader2 = imageio.get_reader(video2)
    
    frames = []
    for frame1, frame2 in zip(reader1, reader2):
        # Concatenate horizontally
        combined = np.concatenate([frame1, frame2], axis=1)
        frames.append(combined)
    
    fps = reader1.get_meta_data()['fps']
    reader1.close()
    reader2.close()
    
    print(f"Saving side-by-side: {output_path}")
    imageio.mimsave(output_path, frames, fps=fps, codec='libx264', quality=8)
    print(f"✅ Done!")

def process_motion_type(hdf5_path, subject, motion_type, trial, config):
    """Process a single motion type with multiple views"""
    print(f"\n{'='*60}")
    print(f"Processing: {subject} - {motion_type} - {trial}")
    print(f"{'='*60}")
    
    # Load and convert data
    print("Loading HDF5 data...")
    q_ref_full = load_and_convert_hdf5(hdf5_path, subject, motion_type, trial)
    print(f"Full data frames: {len(q_ref_full)}")
    
    # Downsample to 300 frames first (for reasonable video length)
    target_frames = 300
    indices = np.linspace(0, len(q_ref_full)-1, target_frames, dtype=int)
    q_ref_orig = q_ref_full[indices]
    print(f"Downsampled to: {len(q_ref_orig)} frames")
    
    # Interpolate for smooth motion
    print(f"Interpolating (factor={config['interpolation_factor']})...")
    q_ref = interpolate_motion(q_ref_orig, factor=config['interpolation_factor'])
    print(f"Final interpolated frames: {len(q_ref)}")
    
    # Output paths
    os.makedirs(config['output_dir'], exist_ok=True)
    base_name = f"{subject}_{motion_type}_{trial}"
    
    diagonal_path = os.path.join(config['output_dir'], f"{base_name}_diagonal.mp4")
    front_path = os.path.join(config['output_dir'], f"{base_name}_front.mp4")
    combined_path = os.path.join(config['output_dir'], f"{base_name}_combined.mp4")
    
    # Render diagonal view
    render_motion(q_ref, config['model_path'], diagonal_path, 
                  view_type='diagonal', height_offset=config['height_offset_m'],
                  target_duration=60)
    
    # Render front view
    render_motion(q_ref, config['model_path'], front_path, 
                  view_type='front', height_offset=config['height_offset_m'],
                  target_duration=60)
    
    # Create side-by-side
    create_side_by_side(diagonal_path, front_path, combined_path)
    
    print(f"\n✅ Completed: {motion_type}")
    print(f"   - Diagonal: {diagonal_path}")
    print(f"   - Front: {front_path}")
    print(f"   - Combined: {combined_path}")

def main():
    parser = argparse.ArgumentParser(description='Render smooth multi-view reference motions')
    parser.add_argument('--motion-types', nargs='+', 
                        default=['level_08mps', 'level_12mps', 'level_16mps', 'incline_10deg'],
                        help='Motion types to process')
    parser.add_argument('--subject', default='S004', help='Subject ID')
    parser.add_argument('--trial', default='trial_01', help='Trial name')
    parser.add_argument('--interp-factor', type=int, default=4, 
                        help='Interpolation factor (higher = smoother)')
    
    args = parser.parse_args()
    
    # Update config
    CONFIG['interpolation_factor'] = args.interp_factor
    
    print("="*60)
    print("SMOOTH MULTI-VIEW REFERENCE MOTION RENDERER")
    print("="*60)
    print(f"Subject: {args.subject}")
    print(f"Motion types: {args.motion_types}")
    print(f"Interpolation factor: {args.interp_factor}x")
    print(f"Output directory: {CONFIG['output_dir']}")
    
    # Process each motion type
    for motion_type in args.motion_types:
        try:
            process_motion_type(
                CONFIG['hdf5_path'],
                args.subject,
                motion_type,
                args.trial,
                CONFIG
            )
        except Exception as e:
            print(f"\n❌ Error processing {motion_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("ALL DONE! 🎉")
    print("="*60)
    print(f"Videos saved in: {CONFIG['output_dir']}/")

if __name__ == '__main__':
    main()
