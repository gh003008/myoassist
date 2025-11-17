"""
Script to inspect and compare data structures between OpenSim and MyoAssist reference data
"""
import numpy as np
import sys

def inspect_opensim_data(filepath):
    """Inspect OpenSim NPZ file structure"""
    print("=" * 80)
    print("OpenSim Data Structure")
    print("=" * 80)
    
    data = np.load(filepath, allow_pickle=True)
    
    print("\n📁 Available Keys:")
    for key in data.keys():
        print(f"  - {key}")
    
    print("\n📊 Data Shapes and Types:")
    for key in data.keys():
        item = data[key]
        if hasattr(item, 'shape'):
            print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
        else:
            print(f"  {key}: type={type(item)}")
    
    print("\n📋 Model States Columns:")
    if 'model_states_columns' in data:
        cols = data['model_states_columns']
        for i, col in enumerate(cols):
            print(f"  [{i:2d}] {col}")
    
    print("\n📈 Sample Data (first 3 rows, first 10 columns):")
    if 'model_states' in data:
        print(data['model_states'][:3, :10])
    
    print("\n🔢 Metadata:")
    for key in ['height_m', 'weight_kg', 'sampling_rate']:
        if key in data:
            print(f"  {key}: {data[key]}")
    
    return data

def inspect_myoassist_data(filepath):
    """Inspect MyoAssist reference data structure"""
    print("\n" + "=" * 80)
    print("MyoAssist Reference Data Structure")
    print("=" * 80)
    
    data = np.load(filepath, allow_pickle=True)
    
    print("\n📁 Available Keys:")
    for key in data.keys():
        print(f"  - {key}")
    
    print("\n📋 Metadata:")
    if 'metadata' in data:
        metadata = data['metadata'].item()
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    print("\n📊 Series Data Keys and Shapes:")
    if 'series_data' in data:
        series = data['series_data'].item()
        print(f"  Total number of keys: {len(series.keys())}")
        print("\n  Available signals:")
        for key in sorted(series.keys()):
            print(f"    {key}: shape={series[key].shape}")
    
    return data

def main():
    opensim_path = r"C:\workspace_home\opensim data\LD_gdp\S004\level_08mps\trial_01.npz"
    myoassist_path = r"rl_train\reference_data\short_reference_gait.npz"
    
    print("\n🔍 Inspecting Data Structures...\n")
    
    try:
        opensim_data = inspect_opensim_data(opensim_path)
    except Exception as e:
        print(f"\n❌ Error loading OpenSim data: {e}")
        return
    
    try:
        myoassist_data = inspect_myoassist_data(myoassist_path)
    except Exception as e:
        print(f"\n❌ Error loading MyoAssist data: {e}")
        return
    
    print("\n" + "=" * 80)
    print("✅ Inspection Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
