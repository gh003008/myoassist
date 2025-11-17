"""
Quick test script for ver1_1 environment

251115_0028: Verify balancing rewards implementation
"""

def test_import():
    """Test that all ver1_1 modules can be imported"""
    print("🧪 Testing imports...")
    try:
        from rl_train.envs.myoassist_leg_imitation_ver1_1 import (
            MyoAssistLegImitation_ver1_1,
            ImitationCustomLearningCallback_ver1_1
        )
        print("✅ ver1_1 imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_config():
    """Test that ver1_1 config file is valid JSON"""
    print("\n🧪 Testing config file...")
    try:
        import json
        with open("rl_train/train/train_configs/S004_3D_IL_ver1_1_BALANCE.json", 'r') as f:
            config = json.load(f)
        
        # Verify new reward keys exist
        rewards = config['env_params']['reward_keys_and_weights']
        assert 'pelvis_list_penalty' in rewards, "Missing pelvis_list_penalty"
        assert 'pelvis_height_reward' in rewards, "Missing pelvis_height_reward"
        
        # Verify max_rot parameter
        assert 'max_rot' in config['env_params'], "Missing max_rot parameter"
        
        print(f"✅ Config valid")
        print(f"   - pelvis_list_penalty weight: {rewards['pelvis_list_penalty']}")
        print(f"   - pelvis_height_reward weight: {rewards['pelvis_height_reward']}")
        print(f"   - max_rot threshold: {config['env_params']['max_rot']}")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_env_registration():
    """Test that ver1_1 environment is registered"""
    print("\n🧪 Testing environment registration...")
    try:
        from myosuite.utils import gym
        
        # Check if environment is registered
        env_id = 'myoAssistLegImitationExo-v1_1'
        spec = gym.spec(env_id)
        
        print(f"✅ Environment registered: {env_id}")
        print(f"   - Entry point: {spec.entry_point}")
        print(f"   - Max episode steps: {spec.max_episode_steps}")
        return True
    except Exception as e:
        print(f"❌ Registration test failed: {e}")
        return False


def test_balancing_methods():
    """Test that balancing reward methods exist"""
    print("\n🧪 Testing balancing methods...")
    try:
        from rl_train.envs.myoassist_leg_imitation_ver1_1 import MyoAssistLegImitation_ver1_1
        
        # Check methods exist
        assert hasattr(MyoAssistLegImitation_ver1_1, '_calculate_balancing_rewards'), \
            "Missing _calculate_balancing_rewards method"
        assert hasattr(MyoAssistLegImitation_ver1_1, '_check_rotation_termination'), \
            "Missing _check_rotation_termination method"
        
        print("✅ All balancing methods present:")
        print("   - _calculate_balancing_rewards()")
        print("   - _check_rotation_termination()")
        return True
    except Exception as e:
        print(f"❌ Method test failed: {e}")
        return False


def run_all_tests():
    """Run all verification tests"""
    print("=" * 60)
    print("MyoAssist ver1_1 Verification Tests")
    print("251115_0028: Balancing Rewards Implementation")
    print("=" * 60)
    
    results = []
    results.append(("Import test", test_import()))
    results.append(("Config test", test_config()))
    results.append(("Registration test", test_env_registration()))
    results.append(("Methods test", test_balancing_methods()))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        all_passed = all_passed and passed
    
    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed! ver1_1 is ready to use.")
        print("\nNext steps:")
        print("1. Run training:")
        print("   python -m rl_train.run_train \\")
        print("       --config_file_path rl_train/train/train_configs/S004_3D_IL_ver1_1_BALANCE.json \\")
        print("       --use_ver1_1 \\")
        print("       --wandb_project myoassist-3D-balancing")
        print("\n2. Monitor WandB for:")
        print("   - reward/pelvis_list_penalty (should approach 0)")
        print("   - reward/pelvis_height_reward (should stay positive)")
        print("   - episode/mean_length (should increase)")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == '__main__':
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
