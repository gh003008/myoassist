#!/usr/bin/env python3
"""
학습 시작 전 Reference Motion 검증 도구

학습에 사용될 정확한 reference motion을 시각화하여 검증합니다.
환경 핸들러가 로드한 데이터를 그대로 가져와 렌더링합니다.
"""
import numpy as np
import mujoco
import imageio
from pathlib import Path
from datetime import datetime
import argparse


def visualize_training_reference(config_path, output_dir='training_reference_validation'):
    """
    학습에서 실제로 사용될 reference motion을 시각화
    
    Args:
        config_path: 학습 config JSON 파일 경로
        output_dir: 출력 디렉토리
    """
    from rl_train.envs.environment_handler import EnvironmentHandler
    import rl_train.train.train_configs.config as myoassist_config
    
    print(f"\n{'='*100}")
    print(f"학습 Reference Motion 검증 도구")
    print(f"{'='*100}\n")
    
    # 1. Config 로드 (학습과 동일한 방식)
    print(f"📋 Step 1: Config 로드")
    print(f"   파일: {config_path}\n")
    
    default_config = EnvironmentHandler.get_session_config_from_path(
        config_path, 
        myoassist_config.TrainSessionConfigBase
    )
    config_type = EnvironmentHandler.get_config_type_from_session_id(
        default_config.env_params.env_id
    )
    config = EnvironmentHandler.get_session_config_from_path(config_path, config_type)
    
    # 2. Reference Data 로드 (학습과 동일한 방식)
    print(f"\n📊 Step 2: Reference Data 로드 (환경 핸들러 사용)")
    ref_data = EnvironmentHandler.load_reference_data(config)
    
    if ref_data is None:
        print("❌ Reference data가 없습니다!")
        return
    
    # 검증 정보 출력
    series_data = ref_data['series_data']
    metadata = ref_data['metadata']
    
    print(f"\n✅ 로드 완료!")
    print(f"   데이터 길이: {metadata['resampled_data_length']} frames")
    print(f"   샘플링 레이트: {metadata['resampled_sample_rate']} Hz")
    print(f"   DOF: {metadata.get('dof', 'N/A')}")
    
    # 데이터 키 확인
    position_keys = [k for k in series_data.keys() if not k.startswith('d')]
    print(f"   Position 키: {sorted(position_keys)[:5]}... (showing first 5)")
    print(f"   총 {len(position_keys)}개 DOF")
    
    # 3. q_ref 형식으로 변환 (렌더링용)
    print(f"\n🔄 Step 3: 렌더링 형식으로 변환")
    
    # Reference joint 순서 정의 (environment에서 사용하는 순서)
    # Environment format은 q_ prefix가 있을 수 있음
    ref_joints_no_prefix = [
        'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
        'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
        'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
        'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
        'knee_angle_r', 'knee_angle_l',
        'ankle_angle_r', 'ankle_angle_l'
    ]
    
    # Check if data has q_ prefix
    first_key = list(series_data.keys())[0]
    has_q_prefix = first_key.startswith('q_') and not first_key.startswith('dq_')
    
    if has_q_prefix:
        ref_joints = ['q_' + joint for joint in ref_joints_no_prefix]
        print(f"   데이터 형식: q_ prefix 있음")
    else:
        ref_joints = ref_joints_no_prefix
        print(f"   데이터 형식: q_ prefix 없음")
    
    # q_ref 배열 생성
    q_ref = np.column_stack([series_data[joint] for joint in ref_joints])
    print(f"   q_ref shape: {q_ref.shape}")
    
    # 대칭성 검증
    print(f"\n🔍 Step 4: 대칭성 검증")
    symmetric_pairs = [
        ('hip_flexion_l', 'hip_flexion_r', 9, 6),
        ('hip_adduction_l', 'hip_adduction_r', 10, 7),
        ('hip_rotation_l', 'hip_rotation_r', 11, 8),
        ('knee_angle_l', 'knee_angle_r', 13, 12),
        ('ankle_angle_l', 'ankle_angle_r', 15, 14),
    ]
    
    print(f"{'Joint Pair':<40} {'Range Diff':<12} {'Status'}")
    print(f"{'-'*70}")
    
    for left_name, right_name, left_idx, right_idx in symmetric_pairs:
        left_vals = q_ref[:, left_idx]
        right_vals = q_ref[:, right_idx]
        
        left_range = left_vals.max() - left_vals.min()
        right_range = right_vals.max() - right_vals.min()
        range_diff = abs(left_range - right_range)
        
        is_symmetric = range_diff < 0.05
        status = "✅ Symmetric" if is_symmetric else "⚠️  Asymmetric"
        
        print(f"{left_name} vs {right_name:<20} {range_diff:>8.4f} rad   {status}")
    
    # 5. 모델 로드 및 렌더링
    print(f"\n🎬 Step 5: 렌더링 시작")
    model = mujoco.MjModel.from_xml_path(config.env_params.model_path)
    data_mj = mujoco.MjData(model)
    
    # Joint name to qpos index mapping
    joint_to_qpos = {}
    for i in range(model.njnt):
        jnt_name = model.joint(i).name
        qpos_addr = model.jnt_qposadr[i]
        joint_to_qpos[jnt_name] = qpos_addr
    
    # 매핑 생성
    ref_to_qpos = []
    # Use joints without q_ prefix for matching with MuJoCo model
    for ref_idx, jnt_name_with_prefix in enumerate(ref_joints):
        # Remove q_ prefix for MuJoCo joint name matching
        if jnt_name_with_prefix.startswith('q_'):
            jnt_name = jnt_name_with_prefix[2:]
        else:
            jnt_name = jnt_name_with_prefix
            
        if jnt_name in joint_to_qpos:
            qpos_idx = joint_to_qpos[jnt_name]
            ref_to_qpos.append((ref_idx, qpos_idx, jnt_name))
    
    print(f"   Joint 매핑: {len(ref_to_qpos)}개")
    
    # 렌더러 설정
    renderer = mujoco.Renderer(model, height=720, width=1920)
    
    # 카메라 설정 (multiview)
    camera_front = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera_front)
    camera_front.azimuth = 90
    camera_front.elevation = -15
    camera_front.distance = 4.5
    camera_front.lookat[:] = [0, 0.7, 0]
    
    camera_side = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera_side)
    camera_side.azimuth = 180
    camera_side.elevation = -20
    camera_side.distance = 3.0
    camera_side.lookat[:] = [0, 0.4, 0]
    
    # 렌더링 옵션
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    
    # Floor 투명하게
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and 'floor' in geom_name.lower():
            model.geom_rgba[i, 3] = 0.3
    
    # Arms 숨김
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and any(part in geom_name.lower() for part in ['humer', 'ulna', 'radius', 'hand', 'arm']):
            model.geom_rgba[i, 3] = 0.0
    
    # 렌더링
    print(f"   렌더링 중...")
    frames = []
    num_frames = 900  # Increased from 600 for smoother, longer video
    frame_skip = max(1, q_ref.shape[0] // num_frames)
    
    for i in range(0, min(num_frames * frame_skip, q_ref.shape[0]), frame_skip):
        # Stand keyframe으로 초기화
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        data_mj.qpos[:] = model.key_qpos[key_id]
        
        # Reference motion 적용
        for ref_idx, qpos_idx, jnt_name in ref_to_qpos:
            data_mj.qpos[qpos_idx] = q_ref[i, ref_idx]
        
        # Pelvis height 조정 (이미 환경 핸들러에서 +0.91m 적용됨)
        # 추가 조정 없이 그대로 사용
        
        # Arms 중립 자세
        arm_joints = {
            40: 0.0, 41: 0.0, 42: 0.5, 43: 0.8,  # Right
            47: 0.0, 48: 0.0, 49: 0.5, 50: 0.8,  # Left
        }
        for qpos_idx, angle in arm_joints.items():
            if qpos_idx < len(data_mj.qpos):
                data_mj.qpos[qpos_idx] = angle
        
        # Forward kinematics
        mujoco.mj_forward(model, data_mj)
        
        # Multiview 렌더링
        renderer.update_scene(data_mj, camera=camera_front, scene_option=scene_option)
        pixels_front = renderer.render()
        front_half = pixels_front[:, 480:1440]
        
        renderer.update_scene(data_mj, camera=camera_side, scene_option=scene_option)
        pixels_side = renderer.render()
        side_half = pixels_side[:, 480:1440]
        
        pixels = np.concatenate([front_half, side_half], axis=1)
        frames.append(pixels)
        
        if (i // frame_skip) % 50 == 0:
            print(f"   Frame {i // frame_skip}/{num_frames}...", end='\r')
    
    print(f"\n   렌더링 완료!")
    
    # 6. 비디오 저장
    print(f"\n💾 Step 6: 비디오 저장")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_file = output_path / f"{timestamp}_training_reference_validation.mp4"
    
    fps = 30
    imageio.mimsave(str(video_file), frames, fps=fps)
    
    print(f"   저장 완료: {video_file}")
    print(f"   FPS: {fps}")
    print(f"   Duration: {len(frames)/fps:.1f}초")
    
    # 7. 요약 리포트
    print(f"\n{'='*100}")
    print(f"✅ 검증 완료!")
    print(f"{'='*100}")
    print(f"\n📊 Reference Motion 요약:")
    print(f"   파일: {config.env_params.reference_data_path}")
    print(f"   프레임: {metadata['resampled_data_length']}")
    print(f"   샘플링: {metadata['resampled_sample_rate']} Hz")
    print(f"   DOF: {metadata.get('dof', len(ref_joints))}")
    print(f"   Duration: {metadata['resampled_data_length'] / metadata['resampled_sample_rate']:.1f}초")
    
    print(f"\n🎥 출력:")
    print(f"   비디오: {video_file}")
    
    print(f"\n⚠️  중요:")
    print(f"   이 비디오가 학습에서 실제로 사용되는 reference motion입니다!")
    print(f"   학습 전 반드시 확인하세요:")
    print(f"   - Kinematic chain이 정상인가? (정강이가 힙에 안 붙었나?)")
    print(f"   - 대칭성이 올바른가? (좌우 다리가 대칭인가?)")
    print(f"   - Pelvis 높이가 적절한가? (땅에 너무 가깝지 않나?)")
    print(f"{'='*100}\n")
    
    return video_file


def main():
    parser = argparse.ArgumentParser(
        description='학습 시작 전 Reference Motion 검증'
    )
    parser.add_argument('--config', type=str,
                       default='rl_train/train/train_configs/S004_3D_IL_ver2_1_BALANCE.json',
                       help='학습 config JSON 파일')
    parser.add_argument('--output', type=str,
                       default='training_reference_validation',
                       help='출력 디렉토리')
    
    args = parser.parse_args()
    
    visualize_training_reference(args.config, args.output)


if __name__ == '__main__':
    main()
