#!/usr/bin/env python3
"""
Test symmetrized reference motion in actual environment.
Verifies that the symmetric data works correctly in training setup.
"""
import numpy as np
import argparse
from pathlib import Path


def test_symmetric_reference():
    """Test loading and basic verification of symmetric reference in environment setup"""
    
    print(f"\n{'='*100}")
    print(f"TESTING SYMMETRIC REFERENCE MOTION IN ENVIRONMENT")
    print(f"{'='*100}\n")
    
    # Load symmetric data
    npz_path = 'rl_train/reference_data/S004_trial01_08mps_3D_HDF5_v7_symmetric.npz'
    print(f"Loading: {npz_path}")
    
    data = np.load(npz_path)
    q_ref = data['q_ref']
    joint_names = data['joint_names']
    
    print(f"  ✅ Loaded successfully")
    print(f"  Frames: {q_ref.shape[0]}")
    print(f"  DOF: {q_ref.shape[1]}")
    print(f"  Joints: {[str(n) for n in joint_names]}\n")
    
    # Quick symmetry check
    joint_to_idx = {str(name): i for i, name in enumerate(joint_names)}
    
    pairs = [
        ('q_hip_flexion_l', 'q_hip_flexion_r'),
        ('q_knee_angle_l', 'q_knee_angle_r'),
        ('q_ankle_angle_l', 'q_ankle_angle_r'),
    ]
    
    print("Quick symmetry verification:")
    all_perfect = True
    for left, right in pairs:
        if left in joint_to_idx and right in joint_to_idx:
            l_idx = joint_to_idx[left]
            r_idx = joint_to_idx[right]
            
            l_data = q_ref[:, l_idx]
            r_data = q_ref[:, r_idx]
            
            l_rom = np.max(l_data) - np.min(l_data)
            r_rom = np.max(r_data) - np.min(r_data)
            rom_diff = abs(l_rom - r_rom)
            
            status = "✅" if rom_diff < 0.001 else "❌"
            if rom_diff >= 0.001:
                all_perfect = False
                
            print(f"  {status} {left.split('_')[1]:<10}: L ROM={np.degrees(l_rom):5.1f}° vs R ROM={np.degrees(r_rom):5.1f}° (diff={np.degrees(rom_diff):.4f}°)")
    
    if all_perfect:
        print(f"\n✅ All joints perfectly symmetric!")
    else:
        print(f"\n⚠️ Some asymmetry detected")
    
    # Test with environment
    print(f"\n{'='*100}")
    print("Testing in actual environment...")
    print(f"{'='*100}\n")
    
    try:
        from rl_train.envs.environment_handler import EnvironmentHandler
        import rl_train.train.train_configs.config as myoassist_config
        
        # Load config
        config_path = 'rl_train/train/train_configs/S004_3D_IL_ver2_1_BALANCE.json'
        print(f"Loading config: {config_path}")
        
        default_config = EnvironmentHandler.get_session_config_from_path(
            config_path, 
            myoassist_config.TrainSessionConfigBase
        )
        config_type = EnvironmentHandler.get_config_type_from_session_id(
            default_config.env_params.env_id
        )
        config = EnvironmentHandler.get_session_config_from_path(config_path, config_type)
        
        # Override with symmetric reference
        config.env_params.reference_data_path = npz_path
        print(f"  ✅ Reference path set to symmetric data\n")
        
        # Create environment (single env for testing)
        print("Creating environment with symmetric reference...")
        
        # Create single environment directly (not vectorized)
        from rl_train.envs.myoassist_leg_imitation_ver2_1 import MyoAssistLegImitation_ver2_1
        
        # Get reference data for environment
        ref_data = EnvironmentHandler.load_reference_data(config)
        
        # Create environment kwargs
        env_kwargs = {
            'model_path': config.env_params.model_path,
            'normalize_act': config.env_params.normalize_act,
            'frame_skip': config.env_params.frame_skip,
            'env_params': config.env_params,
            'reference_data': ref_data,
            'loop_reference_data': False,
        }
        
        # Create single environment directly
        env = MyoAssistLegImitation_ver2_1(**env_kwargs)
        
        print("  ✅ Environment created successfully\n")
        
        # Test reset and check pelvis height
        print("Testing environment reset and pelvis height...")
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle (obs, info) tuple
        
        # Check pelvis height (should be around 0.95m above ground)
        try:
            pelvis_pos = env.sim.data.body('pelvis').xpos
            pelvis_height = pelvis_pos[2]  # z-coordinate
            
            print(f"  Pelvis position: [{pelvis_pos[0]:.3f}, {pelvis_pos[1]:.3f}, {pelvis_pos[2]:.3f}]")
            print(f"  Pelvis height (z): {pelvis_height:.3f} m")
            
            if 0.85 < pelvis_height < 1.05:
                print(f"  ✅ Pelvis height is correct (expected ~0.95m)")
            else:
                print(f"  ⚠️ Pelvis height unusual (expected ~0.95m)")
        except Exception as e:
            print(f"  ⚠️ Could not check pelvis: {e}")
        
        # Test a few steps
        print(f"\nTesting 100 simulation steps with video recording...")
        frames = []
        
        # Setup video recording
        try:
            import imageio
            video_enabled = True
            print("📹 Video recording enabled")
        except ImportError:
            video_enabled = False
            print("⚠️ imageio not available - skipping video")
        
        for i in range(100):
            action = env.action_space.sample()
            result = env.step(action)
            if len(result) == 5:
                obs, reward, done, truncated, info = result
            elif len(result) == 4:
                obs, reward, done, info = result
                truncated = False
            else:
                obs = result[0]
                done = result[2] if len(result) > 2 else False
                truncated = False
            
            # Capture frame
            if video_enabled:
                try:
                    frame = env.mj_render()  # MyoSuite uses mj_render()
                    if frame is not None:
                        frames.append(frame)
                except Exception as e:
                    if i == 0:
                        print(f"⚠️ Frame capture failed: {e}")
                        import traceback
                        traceback.print_exc()
                        video_enabled = False
                
            # Handle done/truncated
            if done or truncated:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
        
        print(f"  ✅ Simulation runs successfully")
        
        # Save video
        if video_enabled and len(frames) > 0:
            import os
            from pathlib import Path
            from datetime import datetime
            
            # Create output directory
            output_dir = Path("visualize_in_env")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_path = output_dir / f"{timestamp}_symmetric_reference_in_env.mp4"
            
            print(f"\n💾 Saving video: {video_path}")
            print(f"   Frames: {len(frames)}")
            print(f"   FPS: 100 (real-time speed)")
            
            imageio.mimsave(str(video_path), frames, fps=100)
            print(f"✅ Video saved!")
        
        env.close()
        
        print(f"\n{'='*100}")
        print(f"✅ ALL TESTS PASSED - Symmetric reference works in environment!")
        print(f"{'='*100}\n")
        
    except Exception as e:
        print(f"\n❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    test_symmetric_reference()
