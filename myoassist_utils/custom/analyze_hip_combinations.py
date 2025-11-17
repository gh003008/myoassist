import numpy as np

# Load OpenSim data
opensim = np.load(r'C:\workspace_home\opensim data\LD_gdp\S004\level_08mps\trial_01.npz', allow_pickle=True)
data = opensim['model_states']

print('Hip 6-DOF 값 범위 (degrees 변환):')
print('='*70)
print('Index | Min(deg) | Max(deg) | Mean(deg) | Std(deg) | 보행에 적합?')
print('='*70)

for i in range(6):
    col_idx = 27 + i
    vals = data[:, col_idx]
    
    # Convert to degrees
    min_deg = np.degrees(vals.min())
    max_deg = np.degrees(vals.max())
    mean_deg = np.degrees(vals.mean())
    std_deg = np.degrees(vals.std())
    
    # Check if suitable for gait
    # Hip flexion: -30~30 deg, Hip adduction: -10~10 deg, Hip rotation: -20~20 deg
    is_flexion = (min_deg > -40 and max_deg < 40 and std_deg > 5)
    is_add_rot = (abs(mean_deg) < 30 and std_deg > 2 and std_deg < 10)
    is_fixed = (std_deg < 2)
    
    suitable = ""
    if is_flexion:
        suitable = "✓ FLEXION?"
    elif is_add_rot:
        suitable = "✓ ADD/ROT?"
    elif is_fixed:
        suitable = "✗ Fixed"
    
    print(f'hip_r_{i} | {min_deg:8.1f} | {max_deg:8.1f} | {mean_deg:9.1f} | {std_deg:8.1f} | {suitable}')

print('='*70)
print('\n정상 보행 범위:')
print('  Hip flexion: -30~30 deg (swing phase: 0~30, stance: -10~10)')
print('  Hip adduction: -5~5 deg (minimal lateral movement)')
print('  Hip rotation: -10~10 deg (internal/external rotation)')
print('='*70)
