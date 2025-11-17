#!/usr/bin/env python3
"""
Detailed symmetry analysis for reference motion
Analyzes ROM (Range of Motion), variance, mean, and asymmetry for L/R joints
"""
import numpy as np
import argparse
from pathlib import Path

def analyze_joint_symmetry(npz_path):
    """Detailed analysis of left/right joint symmetry"""
    
    print(f'Loading reference: {npz_path}')
    data = np.load(npz_path)
    q_ref = data['q_ref']
    joint_names = data['joint_names']
    
    print(f'  Total frames: {q_ref.shape[0]}')
    print(f'  DOF: {q_ref.shape[1]}')
    print(f'\n{"="*100}')
    print(f'DETAILED SYMMETRY ANALYSIS: Left vs Right Joints')
    print(f'{"="*100}\n')
    
    # Define joint pairs to compare (L vs R)
    joint_pairs = [
        ('hip_flexion', 'q_hip_flexion_l', 'q_hip_flexion_r'),
        ('hip_adduction', 'q_hip_adduction_l', 'q_hip_adduction_r'),
        ('hip_rotation', 'q_hip_rotation_l', 'q_hip_rotation_r'),
        ('knee_angle', 'q_knee_angle_l', 'q_knee_angle_r'),
        ('ankle_angle', 'q_ankle_angle_l', 'q_ankle_angle_r'),
    ]
    
    # Create mapping from joint name to index
    joint_to_idx = {}
    for i, name in enumerate(joint_names):
        joint_to_idx[str(name)] = i
    
    results = []
    
    for joint_type, left_name, right_name in joint_pairs:
        if left_name in joint_to_idx and right_name in joint_to_idx:
            left_idx = joint_to_idx[left_name]
            right_idx = joint_to_idx[right_name]
            
            left_data = q_ref[:, left_idx]
            right_data = q_ref[:, right_idx]
            
            # Calculate statistics
            left_mean = np.mean(left_data)
            right_mean = np.mean(right_data)
            left_std = np.std(left_data)
            right_std = np.std(right_data)
            left_var = np.var(left_data)
            right_var = np.var(right_data)
            left_min = np.min(left_data)
            left_max = np.max(left_data)
            right_min = np.min(right_data)
            right_max = np.max(right_data)
            left_rom = left_max - left_min
            right_rom = right_max - right_min
            
            # Calculate differences
            rom_diff = abs(left_rom - right_rom)
            rom_diff_pct = (rom_diff / max(left_rom, right_rom)) * 100
            mean_diff = abs(left_mean - right_mean)
            std_diff = abs(left_std - right_std)
            var_diff = abs(left_var - right_var)
            var_diff_pct = (var_diff / max(left_var, right_var)) * 100
            
            # Calculate correlation (should be high for symmetric gait, but phase-shifted)
            correlation = np.corrcoef(left_data, right_data)[0, 1]
            
            # Calculate phase-shifted correlation (180° shift for gait)
            half_cycle = len(left_data) // 2
            shifted_corr = np.corrcoef(left_data[half_cycle:], right_data[:len(right_data)-half_cycle])[0, 1]
            
            results.append({
                'joint': joint_type,
                'left_name': left_name,
                'right_name': right_name,
                'left_mean': left_mean,
                'right_mean': right_mean,
                'left_std': left_std,
                'right_std': right_std,
                'left_var': left_var,
                'right_var': right_var,
                'left_rom': left_rom,
                'right_rom': right_rom,
                'left_min': left_min,
                'left_max': left_max,
                'right_min': right_min,
                'right_max': right_max,
                'rom_diff': rom_diff,
                'rom_diff_pct': rom_diff_pct,
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'var_diff': var_diff,
                'var_diff_pct': var_diff_pct,
                'correlation': correlation,
                'shifted_corr': shifted_corr,
            })
    
    # Sort by ROM difference percentage (biggest asymmetry first)
    results.sort(key=lambda x: x['rom_diff_pct'], reverse=True)
    
    # Print detailed report
    print(f"{'Joint':<20} {'ROM Diff':<12} {'Var Diff':<12} {'Mean Diff':<12} {'Status':<10}")
    print(f"{'-'*100}")
    
    for r in results:
        status = '⚠️ HIGH' if r['rom_diff_pct'] > 15 else ('⚠️ CHECK' if r['rom_diff_pct'] > 10 else '✅ OK')
        print(f"{r['joint']:<20} {r['rom_diff_pct']:>6.1f}% ({r['rom_diff']:>6.4f}) "
              f"{r['var_diff_pct']:>6.1f}% ({r['var_diff']:>6.4f}) "
              f"{r['mean_diff']:>6.4f} rad      {status}")
    
    print(f"\n{'='*100}")
    print(f"DETAILED BREAKDOWN (sorted by asymmetry)")
    print(f"{'='*100}\n")
    
    for r in results:
        print(f"{'='*100}")
        print(f"🔍 {r['joint'].upper()}")
        print(f"{'='*100}")
        print(f"LEFT  ({r['left_name']}):")
        print(f"  Range:       [{r['left_min']:>7.4f}, {r['left_max']:>7.4f}] rad = {r['left_rom']:.4f} rad ROM ({np.degrees(r['left_rom']):.1f}°)")
        print(f"  Mean ± Std:  {r['left_mean']:>7.4f} ± {r['left_std']:.4f} rad")
        print(f"  Variance:    {r['left_var']:.6f} rad²")
        
        print(f"\nRIGHT ({r['right_name']}):")
        print(f"  Range:       [{r['right_min']:>7.4f}, {r['right_max']:>7.4f}] rad = {r['right_rom']:.4f} rad ROM ({np.degrees(r['right_rom']):.1f}°)")
        print(f"  Mean ± Std:  {r['right_mean']:>7.4f} ± {r['right_std']:.4f} rad")
        print(f"  Variance:    {r['right_var']:.6f} rad²")
        
        print(f"\nASYMMETRY:")
        print(f"  ROM difference:      {r['rom_diff']:.4f} rad ({np.degrees(r['rom_diff']):.2f}°) = {r['rom_diff_pct']:.1f}% asymmetry")
        print(f"  Variance difference: {r['var_diff']:.6f} rad² = {r['var_diff_pct']:.1f}% asymmetry")
        print(f"  Mean difference:     {r['mean_diff']:.4f} rad ({np.degrees(r['mean_diff']):.2f}°)")
        print(f"  Std difference:      {r['std_diff']:.4f} rad")
        print(f"  Direct correlation:  {r['correlation']:.3f} (should be low - phase shifted)")
        print(f"  Phase-shift corr:    {r['shifted_corr']:.3f} (should be high for symmetric gait)")
        
        # Interpretation
        if r['rom_diff_pct'] > 15:
            print(f"\n  ⚠️  HIGH ASYMMETRY DETECTED!")
            print(f"      Left joint has {'LARGER' if r['left_rom'] > r['right_rom'] else 'SMALLER'} range of motion")
            print(f"      This may cause visible gait asymmetry")
        elif r['rom_diff_pct'] > 10:
            print(f"\n  ⚠️  Moderate asymmetry - worth checking")
        else:
            print(f"\n  ✅ Acceptable symmetry")
        
        print()
    
    # Summary
    print(f"\n{'='*100}")
    print(f"SUMMARY")
    print(f"{'='*100}")
    high_asymmetry = [r for r in results if r['rom_diff_pct'] > 15]
    moderate_asymmetry = [r for r in results if 10 < r['rom_diff_pct'] <= 15]
    
    if high_asymmetry:
        print(f"\n⚠️  HIGH ASYMMETRY ({len(high_asymmetry)} joints):")
        for r in high_asymmetry:
            larger_side = "LEFT" if r['left_rom'] > r['right_rom'] else "RIGHT"
            print(f"   - {r['joint']}: {r['rom_diff_pct']:.1f}% difference ({larger_side} larger)")
            print(f"     L ROM: {np.degrees(r['left_rom']):>5.1f}° vs R ROM: {np.degrees(r['right_rom']):>5.1f}°")
    
    if moderate_asymmetry:
        print(f"\n⚠️  MODERATE ASYMMETRY ({len(moderate_asymmetry)} joints):")
        for r in moderate_asymmetry:
            larger_side = "LEFT" if r['left_rom'] > r['right_rom'] else "RIGHT"
            print(f"   - {r['joint']}: {r['rom_diff_pct']:.1f}% difference ({larger_side} larger)")
    
    if not high_asymmetry and not moderate_asymmetry:
        print(f"\n✅ All joints show acceptable symmetry (< 10% ROM difference)")
    
    print(f"\n{'='*100}\n")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze joint symmetry in reference motion')
    parser.add_argument('--data', type=str, 
                       default='rl_train/reference_data/S004_trial01_08mps_3D_HDF5_v7.npz',
                       help='Path to NPZ reference data')
    
    args = parser.parse_args()
    analyze_joint_symmetry(args.data)
