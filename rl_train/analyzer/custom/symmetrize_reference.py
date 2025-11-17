#!/usr/bin/env python3
"""
Symmetrize reference motion by copying right side to left with phase shift.
Automatically detects gait cycle and applies half-cycle phase shift.

Pipeline: NPZ input -> Symmetrized NPZ output (same format)
"""
import numpy as np
import argparse
from pathlib import Path
from scipy import signal


def find_gait_cycle(joint_data, min_cycle_frames=50, max_cycle_frames=500):
    """
    Find gait cycle period using autocorrelation.
    
    Args:
        joint_data: 1D array of joint angles over time
        min_cycle_frames: Minimum expected cycle length (default: 50 frames = 0.5s at 100Hz)
        max_cycle_frames: Maximum expected cycle length (default: 500 frames = 5s at 100Hz)
    
    Returns:
        cycle_length: Number of frames in one gait cycle
    """
    # Normalize data
    data_normalized = (joint_data - np.mean(joint_data)) / np.std(joint_data)
    
    # Compute autocorrelation
    autocorr = np.correlate(data_normalized, data_normalized, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Take only positive lags
    
    # Find peaks in autocorrelation (excluding lag 0)
    peaks, properties = signal.find_peaks(
        autocorr[min_cycle_frames:max_cycle_frames],
        prominence=0.3 * np.max(autocorr[min_cycle_frames:max_cycle_frames])
    )
    
    if len(peaks) == 0:
        print(f"  Warning: No clear peaks found, using default cycle length")
        return 100  # Default: 1 second at 100Hz
    
    # First peak after minimum cycle length is likely the gait cycle
    cycle_length = peaks[0] + min_cycle_frames
    
    return cycle_length


def symmetrize_reference_motion(input_npz, output_npz=None, use_left_as_reference=False):
    """
    Symmetrize reference motion by copying one side to the other with phase shift.
    
    Args:
        input_npz: Path to input NPZ file
        output_npz: Path to output NPZ file (if None, adds '_symmetric' suffix)
        use_left_as_reference: If True, copy left->right. If False (default), copy right->left
    
    The function:
    1. Detects gait cycle length using autocorrelation
    2. Applies half-cycle phase shift to maintain proper gait pattern
    3. Copies reference side to target side with phase shift
    4. Preserves pelvis and other non-paired joints
    """
    
    print(f"\n{'='*100}")
    print(f"SYMMETRIZING REFERENCE MOTION")
    print(f"{'='*100}\n")
    
    # Load data
    print(f"Loading: {input_npz}")
    data = np.load(input_npz)
    q_ref = data['q_ref'].copy()  # Make a copy
    joint_names = data['joint_names']
    
    print(f"  Frames: {q_ref.shape[0]}")
    print(f"  DOF: {q_ref.shape[1]}")
    print(f"  Joints: {list(joint_names)}\n")
    
    # Create joint name to index mapping
    joint_to_idx = {}
    for i, name in enumerate(joint_names):
        joint_to_idx[str(name)] = i
    
    # Define joint pairs (reference_side, target_side)
    if use_left_as_reference:
        print("Strategy: Copy LEFT → RIGHT (with phase shift)\n")
        joint_pairs = [
            ('q_hip_flexion_l', 'q_hip_flexion_r'),
            ('q_hip_adduction_l', 'q_hip_adduction_r'),
            ('q_hip_rotation_l', 'q_hip_rotation_r'),
            ('q_knee_angle_l', 'q_knee_angle_r'),
            ('q_ankle_angle_l', 'q_ankle_angle_r'),
        ]
    else:
        print("Strategy: Copy RIGHT → LEFT (with phase shift)\n")
        joint_pairs = [
            ('q_hip_flexion_r', 'q_hip_flexion_l'),
            ('q_hip_adduction_r', 'q_hip_adduction_l'),
            ('q_hip_rotation_r', 'q_hip_rotation_l'),
            ('q_knee_angle_r', 'q_knee_angle_l'),
            ('q_ankle_angle_r', 'q_ankle_angle_l'),
        ]
    
    # Step 1: Detect gait cycle using hip flexion (most reliable)
    print("Step 1: Detecting gait cycle period...")
    ref_hip_joint = joint_pairs[0][0]  # Use hip_flexion as reference
    ref_idx = joint_to_idx[ref_hip_joint]
    hip_data = q_ref[:, ref_idx]
    
    cycle_length = find_gait_cycle(hip_data)
    half_cycle = cycle_length // 2
    
    print(f"  Detected cycle length: {cycle_length} frames")
    print(f"  At 100 Hz sampling: {cycle_length/100:.2f} seconds per cycle")
    print(f"  Half-cycle shift: {half_cycle} frames\n")
    
    # Step 2: Apply symmetrization with phase shift
    print("Step 2: Applying symmetrization with phase shift...")
    print(f"{'Joint Pair':<50} {'Action':<30}")
    print(f"{'-'*100}")
    
    for ref_joint, target_joint in joint_pairs:
        if ref_joint in joint_to_idx and target_joint in joint_to_idx:
            ref_idx = joint_to_idx[ref_joint]
            target_idx = joint_to_idx[target_joint]
            
            # Get reference data
            ref_data = q_ref[:, ref_idx]
            
            # Apply phase shift: roll by half cycle
            # np.roll with positive shift moves data forward in time
            shifted_data = np.roll(ref_data, half_cycle)
            
            # Handle adduction: need to flip sign for anatomical symmetry
            # Left adduction (+) = outward, Right adduction (+) = outward
            # When copying L->R or R->L, adduction should be negated
            if 'adduction' in ref_joint:
                shifted_data = -shifted_data
                sign_note = "(sign flipped)"
            else:
                sign_note = ""
            
            # Copy to target
            q_ref[:, target_idx] = shifted_data
            
            print(f"  {ref_joint:<25} → {target_joint:<25} (shift: {half_cycle:>4} frames) {sign_note}")
    
    print(f"\n{'='*100}")
    print("Step 3: Verification")
    print(f"{'='*100}\n")
    
    # Verify symmetry
    print("Checking new symmetry (should be near-perfect with phase shift)...\n")
    for ref_joint, target_joint in joint_pairs:
        if ref_joint in joint_to_idx and target_joint in joint_to_idx:
            ref_idx = joint_to_idx[ref_joint]
            target_idx = joint_to_idx[target_joint]
            
            ref_data = q_ref[:, ref_idx]
            target_data = q_ref[:, target_idx]
            
            # Check ROM
            ref_rom = np.max(ref_data) - np.min(ref_data)
            target_rom = np.max(target_data) - np.min(target_data)
            rom_diff = abs(ref_rom - target_rom)
            
            # Check phase-shifted correlation
            shifted_target = np.roll(target_data, half_cycle)
            correlation = np.corrcoef(ref_data, shifted_target)[0, 1]
            
            status = "✅" if rom_diff < 0.001 and correlation > 0.99 else "⚠️"
            print(f"  {status} {ref_joint.split('_')[-2]:<12}: ROM diff = {rom_diff:.6f} rad, Phase-shift corr = {correlation:.4f}")
    
    # Step 4: Save output
    if output_npz is None:
        input_path = Path(input_npz)
        output_npz = str(input_path.parent / f"{input_path.stem}_symmetric{input_path.suffix}")
    
    print(f"\n{'='*100}")
    print(f"Saving symmetrized data to: {output_npz}")
    
    # Save with same format as input (allow_pickle for joint_names which is object array)
    np.savez(
        output_npz,
        q_ref=q_ref,
        joint_names=joint_names
    )
    
    print(f"✅ Done! Symmetrized reference motion saved.")
    print(f"{'='*100}\n")
    
    return output_npz


def visualize_before_after(original_npz, symmetric_npz, joint_name='q_hip_flexion_r'):
    """
    Optional: Visualize comparison between original and symmetrized data.
    """
    import matplotlib.pyplot as plt
    
    data_orig = np.load(original_npz)
    data_symm = np.load(symmetric_npz)
    
    joint_names_orig = data_orig['joint_names']
    joint_idx = None
    for i, name in enumerate(joint_names_orig):
        if str(name) == joint_name:
            joint_idx = i
            break
    
    if joint_idx is None:
        print(f"Joint {joint_name} not found")
        return
    
    q_orig = data_orig['q_ref'][:, joint_idx]
    q_symm = data_symm['q_ref'][:, joint_idx]
    
    frames = min(600, len(q_orig))  # Plot first 600 frames
    
    plt.figure(figsize=(12, 6))
    plt.plot(q_orig[:frames], label='Original', alpha=0.7, linewidth=2)
    plt.plot(q_symm[:frames], label='Symmetrized', alpha=0.7, linewidth=2, linestyle='--')
    plt.xlabel('Frame')
    plt.ylabel('Angle (rad)')
    plt.title(f'{joint_name}: Original vs Symmetrized')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = 'symmetry_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"Comparison plot saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Symmetrize reference motion by copying one side to the other with phase shift'
    )
    parser.add_argument('--input', type=str,
                       default='rl_train/reference_data/S004_trial01_08mps_3D_HDF5_v7.npz',
                       help='Input NPZ file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output NPZ file path (default: adds _symmetric suffix)')
    parser.add_argument('--use_left', action='store_true',
                       help='Copy LEFT to RIGHT (default: copy RIGHT to LEFT)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create before/after visualization plot')
    
    args = parser.parse_args()
    
    output_path = symmetrize_reference_motion(
        args.input,
        args.output,
        use_left_as_reference=args.use_left
    )
    
    if args.visualize:
        print("\nGenerating visualization...")
        visualize_before_after(args.input, output_path)
