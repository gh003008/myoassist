"""
H5 파일의 메타데이터, 속성, 오일러 각 convention 확인
"""
import h5py
import numpy as np

h5_path = r'C:\workspace\opensim data\LD\S004.h5'

def print_attributes(obj, name):
    """객체의 모든 속성 출력"""
    if len(obj.attrs) > 0:
        print(f"\n{name} - Attributes:")
        for key, value in obj.attrs.items():
            print(f"  {key}: {value}")

def explore_h5_structure(h5_file, max_depth=3):
    """H5 파일 구조와 속성 탐색"""
    
    def recursive_explore(group, prefix="", depth=0):
        if depth > max_depth:
            return
        
        # 현재 그룹의 속성 출력
        print_attributes(group, prefix if prefix else "ROOT")
        
        # 하위 항목 탐색
        for key in group.keys():
            item = group[key]
            full_path = f"{prefix}/{key}" if prefix else key
            
            if isinstance(item, h5py.Group):
                print(f"\n{'  ' * depth}📁 GROUP: {full_path}")
                print_attributes(item, full_path)
                recursive_explore(item, full_path, depth + 1)
            elif isinstance(item, h5py.Dataset):
                print(f"{'  ' * depth}📄 DATASET: {full_path}")
                print(f"{'  ' * depth}   Shape: {item.shape}, dtype: {item.dtype}")
                print_attributes(item, full_path)
    
    print("="*80)
    print("H5 파일 전체 구조 및 메타데이터 탐색")
    print("="*80)
    recursive_explore(h5_file)

def check_specific_paths(h5_file):
    """특정 경로의 메타데이터 확인"""
    print("\n" + "="*80)
    print("주요 데이터셋 상세 정보")
    print("="*80)
    
    paths_to_check = [
        'S004',
        'S004/level_08mps/trial_01',
        'S004/level_08mps/trial_01/MoCap',
        'S004/level_08mps/trial_01/MoCap/ik_data',
        'S004/level_08mps/trial_01/MoCap/kin_q',
        'S004/level_08mps/trial_01/MoCap/body_pos_global',
        'S004/sub_info',
    ]
    
    for path in paths_to_check:
        if path in h5_file:
            print(f"\n{'='*60}")
            print(f"Path: {path}")
            print(f"{'='*60}")
            item = h5_file[path]
            
            # 속성 출력
            if len(item.attrs) > 0:
                print("Attributes:")
                for key, value in item.attrs.items():
                    print(f"  {key}: {value}")
            else:
                print("  (속성 없음)")
            
            # 데이터셋이면 샘플 데이터 출력
            if isinstance(item, h5py.Dataset):
                print(f"Shape: {item.shape}")
                print(f"Dtype: {item.dtype}")
                if item.size < 100:  # 작은 데이터만 출력
                    print(f"Data: {item[:]}")
            
            # 그룹이면 하위 키 출력
            if isinstance(item, h5py.Group):
                print(f"Keys: {list(item.keys())[:10]}...")  # 처음 10개만

def check_infos_dataset(h5_file):
    """infos 데이터셋 상세 분석"""
    print("\n" + "="*80)
    print("'infos' 데이터셋 상세 분석")
    print("="*80)
    
    infos_paths = [
        'S004/level_08mps/trial_01/MoCap/ik_data/infos',
        'S004/level_08mps/trial_01/MoCap/kin_q/infos',
        'S004/level_08mps/trial_01/MoCap/body_pos_global/infos',
    ]
    
    for path in infos_paths:
        if path in h5_file:
            print(f"\n{path}:")
            dataset = h5_file[path]
            print(f"  Shape: {dataset.shape}")
            print(f"  Dtype: {dataset.dtype}")
            print(f"  Attributes: {dict(dataset.attrs)}")
            
            # Reference 타입인 경우 역참조 시도
            try:
                data = dataset[:]
                print(f"  Data type: {type(data)}")
                print(f"  Raw data:\n{data}")
                
                # Object reference 역참조 시도
                if dataset.dtype == h5py.ref_dtype:
                    print("\n  Dereferencing object references:")
                    for i, ref in enumerate(data.flatten()):
                        try:
                            if ref:
                                obj = h5_file[ref]
                                print(f"    [{i}] -> {obj.name}")
                                if isinstance(obj, h5py.Dataset):
                                    print(f"        Type: Dataset, Shape: {obj.shape}")
                                    if obj.dtype.char == 'U' or obj.dtype.char == 'S':
                                        print(f"        Value: {obj[()]}")
                        except:
                            print(f"    [{i}] -> (null reference)")
            except Exception as e:
                print(f"  Error reading data: {e}")

def check_string_datasets(h5_file):
    """문자열 데이터셋 찾기 (메타정보 가능성)"""
    print("\n" + "="*80)
    print("문자열/메타정보 데이터셋 탐색")
    print("="*80)
    
    def find_strings(group, prefix=""):
        for key in group.keys():
            item = group[key]
            full_path = f"{prefix}/{key}" if prefix else key
            
            if isinstance(item, h5py.Dataset):
                # 문자열 타입이거나 작은 데이터셋
                if (item.dtype.char in ['U', 'S', 'O'] or 
                    ('info' in key.lower() or 'meta' in key.lower() or 
                     'label' in key.lower() or 'name' in key.lower())):
                    print(f"\n{full_path}:")
                    print(f"  Shape: {item.shape}, Dtype: {item.dtype}")
                    try:
                        data = item[()]
                        if isinstance(data, bytes):
                            data = data.decode('utf-8')
                        print(f"  Value: {data}")
                    except:
                        try:
                            print(f"  Value: {item[:]}")
                        except:
                            print(f"  (읽기 실패)")
            elif isinstance(item, h5py.Group):
                find_strings(item, full_path)
    
    find_strings(h5_file)

# 메인 실행
print("H5 파일 오일러 각 convention 및 메타데이터 확인")
print("파일:", h5_path)
print()

with h5py.File(h5_path, 'r') as f:
    # 1. 전체 구조 및 속성 확인 (depth 제한)
    explore_h5_structure(f, max_depth=2)
    
    # 2. 특정 경로 상세 확인
    check_specific_paths(f)
    
    # 3. infos 데이터셋 분석
    check_infos_dataset(f)
    
    # 4. 문자열/메타정보 데이터셋 찾기
    check_string_datasets(f)

print("\n" + "="*80)
print("분석 완료!")
print("="*80)
