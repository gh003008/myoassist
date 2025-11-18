#!/usr/bin/env python3
"""
Visualize symmetric reference motion in actual training environment.
Creates the real MyoAssist environment and follows the reference motion.
"""
import numpy as np
import argparse
from pathlib import Path
import time
import imageio  # For video saving


def visualize_in_training_env(config_path, reference_npz_path, playback_speed=0.3, num_steps=600):
    """
    Visualize reference motion in actual training environment
    
    Args:
        config_path: Path to training config JSON
        reference_npz_path: Path to symmetric reference NPZ
        playback_speed: Speed multiplier (0.3 = slow motion)
        num_steps: Number of simulation steps to run
    """
    
    print(f"\n{'='*100}")
    print(f"VISUALIZING SYMMETRIC REFERENCE IN ACTUAL TRAINING ENVIRONMENT")
    print(f"{'='*100}\n")
    
    from rl_train.envs.environment_handler import EnvironmentHandler # 환경 핸들러 호출
    import rl_train.train.train_configs.config as myoassist_config # Config 호출
    
    # Load config
    print(f"📋 Loading config: {config_path}")
    default_config = EnvironmentHandler.get_session_config_from_path(
        config_path, 
        myoassist_config.TrainSessionConfigBase
    )
    config_type = EnvironmentHandler.get_config_type_from_session_id(
        default_config.env_params.env_id
    )
    config = EnvironmentHandler.get_session_config_from_path(config_path, config_type)
    
    # Override reference data path with symmetric version
    config.env_params.reference_data_path = reference_npz_path
    
    # CRITICAL: Force control_framerate to 100 Hz to match original data
    # This prevents resampling and time axis mismatch
    config.env_params.control_framerate = 100
    print(f"   ✅ Using reference: {reference_npz_path}")
    print(f"   ✅ Control framerate forced to 100 Hz (no resampling)\n")
    
    # Load reference data to check format
    ref_data = np.load(reference_npz_path, allow_pickle=True)
    
    # Check which format we have
    if 'series_data' in ref_data.files and 'metadata' in ref_data.files:
        # Environment format
        series_data = ref_data['series_data'].item()
        metadata = ref_data['metadata'].item()
        n_frames = metadata['data_length']
        n_dof = metadata['dof']
        print(f"📊 Reference data info (ENVIRONMENT FORMAT):")
        print(f"   Format: series_data + metadata (NO q_ prefix)")
        position_keys = [k for k in series_data.keys() if not k.startswith('d')]
        print(f"   Joints: {sorted(position_keys)[:5]}... (showing first 5)")
    elif 'q_ref' in ref_data.files and 'joint_names' in ref_data.files:
        # MuJoCo renderer format
        q_ref = ref_data['q_ref']
        joint_names = ref_data['joint_names']
        n_frames = q_ref.shape[0]
        n_dof = q_ref.shape[1]
        print(f"📊 Reference data info (MUJOCO RENDERER FORMAT):")
        print(f"   Format: q_ref + joint_names (with q_ prefix)")
        print(f"   Joints: {[str(n) for n in joint_names[:5]]}... (showing first 5)")
    else:
        raise ValueError(f"Unknown NPZ format! Keys: {ref_data.files}")
    
    print(f"   Frames: {n_frames}")
    print(f"   DOF: {n_dof}")
    print(f"   Duration: {n_frames/100:.1f} seconds @ 100 Hz\n")
    
    # Create single environment directly (not vectorized for visualization) 환경 하나만 만들어서 시각화
    print(f"🏗️  Creating training environment...")
    
    # Use EnvironmentHandler but force single environment
    original_num_envs = config.env_params.num_envs # 백업
    config.env_params.num_envs = 1
    
    # Create environment through handler for proper setup
    from rl_train.envs.environment_handler import EnvironmentHandler
    env_vec = EnvironmentHandler.create_environment(
        config,
        is_rendering_on=True,
        is_evaluate_mode=True
    )
    
    # Access the actual environment from vectorized wrapper
    if hasattr(env_vec, 'envs'):
        env = env_vec.envs[0]
    elif hasattr(env_vec, 'env'):
        env = env_vec.env
    else:
        env = env_vec
    
    print(f"   ✅ Environment created: {config.env_params.env_id}\n")
    
    # Check reference data sampling rate
    ref_sample_rate = env._reference_data["metadata"].get("resampled_sample_rate", 100)
    ref_data_length = env._reference_data["metadata"].get("resampled_data_length", len(env._reference_data["series_data"]["pelvis_tx"]))
    print(f"📊 Reference data in environment:")
    print(f"   Sample rate: {ref_sample_rate} Hz")
    print(f"   Data length: {ref_data_length} frames")
    print(f"   Duration: {ref_data_length / ref_sample_rate:.1f} seconds\n")
    
    # Setup rendering
    video_enabled = True
    frames = []
    print(f"📹 Video recording enabled")
    
    print(f"\n🎬 Starting visualization...")
    print(f"   Playback speed: {playback_speed}x")
    print(f"   Steps: {num_steps}")
    print(f"   Target video length: {num_steps * 0.01 / playback_speed:.1f} seconds\n")
    
    # Reset environment
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    # Check initial pelvis position
    try:
        pelvis_pos = env.sim.data.body('pelvis').xpos
        print(f"📍 Initial pelvis position: [{pelvis_pos[0]:.3f}, {pelvis_pos[1]:.3f}, {pelvis_pos[2]:.3f}]")
        print(f"   Height (z): {pelvis_pos[2]:.3f} m\n")
    except:
        pass
    
    # Run simulation following reference
    start_time = time.time()
    for step in range(num_steps):
        # IMPORTANT: Only call imitation_step to follow reference motion
        # Do NOT call env.step() after this, as it would advance imitation_index twice!
        # imitation_step() already calls forward() to update physics
        try:
            env.imitation_step(is_x_follow=False)
        except Exception as e:
            print(f"⚠️  imitation_step failed: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # Capture frame
        if video_enabled:
            try:
                # Use render_offscreen to get pixel array
                frame = env.sim.renderer.render_offscreen(width=1280, height=720, camera_id=-1)
                if frame is not None:
                    frames.append(frame)
            except Exception as e:
                if step == 0:
                    print(f"⚠️  Frame capture failed: {e}")
                    import traceback
                    traceback.print_exc()
                    video_enabled = False
        
        # Progress
        if step % 50 == 0:
            elapsed = time.time() - start_time
            progress = (step / num_steps) * 100
            
            # Check pelvis height every 50 steps
            try:
                pelvis_pos = env.sim.data.body('pelvis').xpos
                pelvis_height = pelvis_pos[2]
            except:
                pelvis_height = 0.0
            
            print(f"   Frame {step}/{num_steps} ({progress:.1f}%) - Pelvis height: {pelvis_height:.3f}m", end='\r')
        
        # Control playback speed
        if playback_speed < 1.0:
            time.sleep(0.01 / playback_speed)  # Slow down playback
    
    print(f"\n\n✅ Visualization complete!")
    
    # Check final pelvis position
    try:
        pelvis_pos = env.sim.data.body('pelvis').xpos
        print(f"\n📍 Final pelvis position: [{pelvis_pos[0]:.3f}, {pelvis_pos[1]:.3f}, {pelvis_pos[2]:.3f}]")
        print(f"   Height (z): {pelvis_pos[2]:.3f} m")
    except:
        pass
    
    # Save video BEFORE closing environment
    if video_enabled and len(frames) > 0:
        from datetime import datetime
        
        output_dir = Path("visualize_in_env")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_path = output_dir / f"{timestamp}_symmetric_in_training_env.mp4"
        
        # Calculate FPS for desired playback speed
        # MuJoCo runs at 100 Hz, we want video to play at playback_speed
        video_fps = int(100 * playback_speed)
        
        print(f"\n💾 Saving video...")
        print(f"   Path: {video_path}")
        print(f"   Frames: {len(frames)}")
        print(f"   FPS: {video_fps} (for {playback_speed}x playback)")
        print(f"   Duration: {len(frames)/video_fps:.1f} seconds")
        
        imageio.mimsave(str(video_path), frames, fps=video_fps)
        print(f"✅ Video saved!")
    else:
        print(f"\n⚠️  No frames captured, video not saved")
        print(f"   video_enabled={video_enabled}, frames={len(frames) if 'frames' in locals() else 0}")
    
    env.close()
    
    print(f"\n{'='*100}")
    print(f"DONE!")
    print(f"{'='*100}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize symmetric reference in actual training environment'
    )
    parser.add_argument('--config', type=str,
                       default='rl_train/train/train_configs/S004_3D_IL_ver2_1_BALANCE.json',
                       help='Training config JSON file')
    parser.add_argument('--data', type=str,
                       default='rl_train/reference_data/S004_trial01_08mps_3D_ENV_symmetric.npz',
                       help='Reference NPZ file (use ENV format for best results)')
    parser.add_argument('--speed', type=float, default=0.3,
                       help='Playback speed multiplier (0.3 = slow motion)')
    parser.add_argument('--steps', type=int, default=600,
                       help='Number of simulation steps (default: 600 = 6 seconds)')
    
    args = parser.parse_args()
    
    visualize_in_training_env(
        args.config,
        args.data,
        playback_speed=args.speed,
        num_steps=args.steps
    )
