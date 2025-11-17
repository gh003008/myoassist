"""
MyoAssist Reference Trajectory 데이터 구조 분석 스크립트
"""
import numpy as np
import h5py
import os
import sys

def check_myoassist_reference_structure():
    """MyoAssist에서 요구하는 reference trajectory 구조 확인"""
    
    print("=" * 80)
    print("MyoAssist Reference Trajectory 구조 분석")
    print("=" * 80)
    
    # 기존 reference data 로드
    ref_path = "../rl_train/reference_data/short_reference_gait.npz"
    
    if os.path.exists(ref_path):
        print(f"\n✅ 기존 reference data 발견: {ref_path}")
        data = np.load(ref_path, allow_pickle=True)
        
        print("\n[파일 내부 키들]")
        for key in data.files:
            print(f"  - {key}")
        
        print("\n[상세 구조 분석]")
        for key in data.files:
            item = data[key]
            if isinstance(item, np.ndarray):
                if item.dtype == object:
                    # dictionary 형태
                    try:
                        item_dict = item.item()
                        print(f"\n📦 {key}: (dictionary)")
                        if isinstance(item_dict, dict):
                            for sub_key, sub_val in item_dict.items():
                                if isinstance(sub_val, np.ndarray):
                                    print(f"    {sub_key}: shape={sub_val.shape}, dtype={sub_val.dtype}")
                                else:
                                    print(f"    {sub_key}: {type(sub_val)} = {sub_val}")
                    except:
                        print(f"\n📦 {key}: {item.shape}, {item.dtype}")
                else:
                    print(f"\n📦 {key}: shape={item.shape}, dtype={item.dtype}")
                    if len(item.shape) == 1 and item.shape[0] < 20:
                        print(f"    값: {item}")
        
        data.close()
    else:
        print(f"\n❌ reference data를 찾을 수 없습니다: {ref_path}")


def check_opensim_h5_data():
    """OpenSim H5 파일 구조 확인"""
    
    print("\n" + "=" * 80)
    print("OpenSim H5 데이터 구조 분석")
    print("=" * 80)
    
    h5_path = r"C:\workspace\opensim data\LD\S004.h5"
    
    if not os.path.exists(h5_path):
        print(f"\n❌ 파일을 찾을 수 없습니다: {h5_path}")
        return None
    
    print(f"\n✅ H5 파일 발견: {h5_path}")
    
    def print_h5_structure(name, obj):
        """HDF5 구조 재귀적 출력"""
        indent = "  " * name.count('/')
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}📄 {name}: shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"{indent}📁 {name}/")
    
    with h5py.File(h5_path, 'r') as f:
        print("\n[HDF5 파일 전체 구조]")
        f.visititems(print_h5_structure)
        
        # MoCap/ik_data 상세 분석
        if 'MoCap' in f and 'ik_data' in f['MoCap']:
            print("\n" + "=" * 80)
            print("MoCap/ik_data 상세 분석")
            print("=" * 80)
            
            ik_data = f['MoCap/ik_data']
            print(f"\n📊 Shape: {ik_data.shape}")
            print(f"📊 Dtype: {ik_data.dtype}")
            
            # 데이터 일부 로드
            data_sample = ik_data[:10, :]  # 처음 10개 프레임
            print(f"\n📊 Data sample (first 10 frames):")
            print(f"    Min: {np.min(data_sample, axis=0)}")
            print(f"    Max: {np.max(data_sample, axis=0)}")
            print(f"    Mean: {np.mean(data_sample, axis=0)}")
            
            # Column 이름 확인 (있다면)
            if 'columns' in f['MoCap']:
                columns = f['MoCap/columns'][:]
                print(f"\n📋 Column names ({len(columns)} columns):")
                for i, col in enumerate(columns):
                    col_name = col.decode('utf-8') if isinstance(col, bytes) else col
                    print(f"    [{i:2d}] {col_name}")
            
            # Attributes 확인
            print(f"\n📋 Attributes:")
            for attr_name, attr_val in ik_data.attrs.items():
                print(f"    {attr_name}: {attr_val}")
            
            return ik_data
        else:
            print("\n❌ MoCap/ik_data를 찾을 수 없습니다")
            return None


def compare_data_requirements():
    """MyoAssist 요구사항과 OpenSim 데이터 비교"""
    
    print("\n" + "=" * 80)
    print("데이터 호환성 분석")
    print("=" * 80)
    
    print("\n[MyoAssist 26muscle_3D 모델이 요구하는 joint 이름들]")
    required_joints = [
        # Pelvis
        "pelvis_tx", "pelvis_ty", "pelvis_tz",
        "pelvis_tilt", "pelvis_list", "pelvis_rotation",
        
        # Right Leg
        "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
        "knee_angle_r",
        "ankle_angle_r",
        "mtp_angle_r",  # 발가락
        
        # Left Leg
        "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
        "knee_angle_l",
        "ankle_angle_l",
        "mtp_angle_l",
    ]
    
    for i, joint in enumerate(required_joints):
        print(f"  [{i:2d}] {joint}")
    
    print(f"\n총 {len(required_joints)}개 관절 필요")
    
    print("\n[일반적인 OpenSim IK 출력 형식]")
    print("  - OpenSim 모델: gait2392, gait2354 등")
    print("  - Joint 이름: /jointset/<joint_name>/joint_angle")
    print("  - 예: /jointset/hip_r/hip_flexion_r")
    print("  - 단위: radians 또는 degrees")
    
    print("\n[확인 필요 사항]")
    print("  ✓ OpenSim joint 이름이 MyoAssist와 일치하는지")
    print("  ✓ 단위가 radians인지 (MyoAssist는 radians 사용)")
    print("  ✓ 샘플링 주파수 (MyoAssist는 보통 30Hz 또는 100Hz)")
    print("  ✓ 데이터 정규화 여부")


if __name__ == "__main__":
    # 1. MyoAssist reference 구조 확인
    check_myoassist_reference_structure()
    
    # 2. OpenSim H5 데이터 확인
    check_opensim_h5_data()
    
    # 3. 요구사항 비교
    compare_data_requirements()
    
    print("\n" + "=" * 80)
    print("분석 완료!")
    print("=" * 80)
