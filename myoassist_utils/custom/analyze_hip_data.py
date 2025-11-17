import numpy as np

# Load OpenSim data
opensim = np.load(r'C:\workspace_home\opensim data\LD_gdp\S004\level_08mps\trial_01.npz', allow_pickle=True)
data = opensim['model_states']

print('Hip joint 6-DOF 값 범위 분석:')
print('='*60)
for i in range(6):
    col_idx = 27 + i
    vals = data[:, col_idx]
    print(f'hip_r_{i}: min={vals.min():7.3f}, max={vals.max():7.3f}, mean={vals.mean():7.3f}, std={vals.std():.3f}')

print('\n' + '='*60)
print('분석:')
print('  - 변화량이 큰 것(std > 0.05): 주요 관절')
print('  - 거의 고정(std < 0.01): 사용 안 함 또는 constraint')
print('='*60)
