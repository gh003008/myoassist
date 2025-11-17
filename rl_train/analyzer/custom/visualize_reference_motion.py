"""
Reference Motion Data 시각화 도구
==================================

NPZ 파일의 reference motion을 MuJoCo 환경에서 렌더링합니다.

Usage:
    python visualize_reference_motion.py --data rl_train/reference_data/S004_trial01_08mps_3D.npz --output reference_motion.mp4
"""

import numpy as np
import argparse
import json
from pathlib import Path
from rl_train.envs.environment_handler import EnvironmentHandler
import rl_train.train.train_configs.config as myoassist_config

def visualize_reference_motion(config_path, reference_npz_path, output_path, num_frames=300, fps=30, camera_view='side'):
    """Reference motion을 MuJoCo에서 재생하고 비디오로 저장
    
    Args:
        camera_view: 'front', 'side', 'top', 'diagonal' 중 선택
    """
    
    # Reference 데이터 로드
    print(f"📂 Reference 데이터 로드: {reference_npz_path}")
    data = np.load(reference_npz_path, allow_pickle=True)
    metadata = data['metadata'].item()
    series_data = data['series_data'].item()
    
    print(f"\n📊 데이터 정보:")
    print(f"  - 총 프레임 수: {metadata['data_length']}")
    print(f"  - Sampling rate: {metadata['sample_rate']} Hz")
    print(f"  - Duration: {metadata['data_length'] / metadata['sample_rate']:.2f} 초")
    print(f"  - Model type: {metadata['model_type']}")
    print(f"  - DOF: {metadata['dof']}")
    print(f"\n  - Position keys: {[k for k in series_data.keys() if k.startswith('q_')]}")
    print(f"  - Velocity keys: {[k for k in series_data.keys() if k.startswith('dq_')]}")
    
    # Config 로드
    print(f"\n⚙️ Config 로드: {config_path}")
    default_config = EnvironmentHandler.get_session_config_from_path(
        config_path, 
        myoassist_config.TrainSessionConfigBase
    )
    config_type = EnvironmentHandler.get_config_type_from_session_id(
        default_config.env_params.env_id
    )
    config = EnvironmentHandler.get_session_config_from_path(config_path, config_type)
    
    # ⚠️ CRITICAL: Override reference data path with user-specified file
    config.env_params.reference_data_path = reference_npz_path
    print(f"  ✅ Reference data path overridden: {reference_npz_path}")
    
    # 렌더링 환경 생성
    print("🎥 렌더링 환경 생성 중...")
    env = EnvironmentHandler.create_environment(
        config, 
        is_rendering_on=True, 
        is_evaluate_mode=True
    )
    
    # 비디오 저장 준비
    try:
        import imageio
        video_enabled = True
        frames = []
        print("📹 비디오 녹화 활성화됨")
    except ImportError:
        video_enabled = False
        print("⚠️ imageio 없음 - 비디오 저장 건너뜀 (pip install imageio imageio-ffmpeg)")
    
    # Reference motion 따라하기
    print(f"\n🏃 Reference motion 재생 시작 ({num_frames} 프레임)...")
    env.reset()
    
    # 카메라 설정 (다양한 각도)
    camera_configs = {
        'front': {'distance': 3.0, 'azimuth': 90, 'elevation': -10, 'lookat': [0, 0, 1.0]},
        'side': {'distance': 3.0, 'azimuth': 0, 'elevation': -10, 'lookat': [0, 0, 1.0]},
        'diagonal': {'distance': 3.5, 'azimuth': 45, 'elevation': -20, 'lookat': [0, 0, 1.0]},
        'top': {'distance': 4.0, 'azimuth': 90, 'elevation': -60, 'lookat': [0, 0, 0.5]},
        'back': {'distance': 3.0, 'azimuth': 180, 'elevation': -10, 'lookat': [0, 0, 1.0]},
    }
    
    cam_config = camera_configs.get(camera_view, camera_configs['side'])
    env.viewer_setup(**cam_config)
    print(f"📷 카메라 각도: {camera_view} (azimuth={cam_config['azimuth']}°, elevation={cam_config['elevation']}°)")
    
    # Reference 데이터 키 매핑
    joint_names = config.env_params.reference_data_keys
    
    for frame_idx in range(min(num_frames, metadata['data_length'])):
        # Reference 자세로 설정
        for joint_name in joint_names:
            q_key = f'q_{joint_name}'
            dq_key = f'dq_{joint_name}'
            
            if q_key in series_data and dq_key in series_data:
                try:
                    # Joint position 설정
                    joint = env.sim.data.joint(joint_name)
                    joint.qpos[0] = series_data[q_key][frame_idx]
                    joint.qvel[0] = series_data[dq_key][frame_idx]
                except Exception as e:
                    if frame_idx == 0:
                        print(f"⚠️ Joint '{joint_name}' 설정 실패: {e}")
        
        # Forward kinematics 계산
        env.sim.forward()
        
        # 프레임 캡처
        if video_enabled:
            try:
                # MuJoCo offscreen rendering
                frame = env.sim.renderer.render_offscreen(
                    width=640,
                    height=480,
                    camera_id=-1  # Free camera
                )
                frames.append(frame)
            except Exception as e:
                if frame_idx == 0:
                    print(f"⚠️ 프레임 캡처 실패: {e}")
                    video_enabled = False
        
        # 진행률 표시
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{num_frames} ({frame_idx/num_frames*100:.1f}%)")
        
        # 렌더링 속도 조절
        import time
        time.sleep(1.0 / fps)  # 실시간 재생
    
    env.close()
    
    # 비디오 저장
    if video_enabled and len(frames) > 0:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = Path(output_path)
        print(f"\n💾 비디오 저장 중... ({len(frames)} 프레임)")
        imageio.mimsave(str(video_path), frames, fps=fps)
        print(f"✅ 비디오 저장 완료: {video_path}")
        
        # 통계 출력
        print(f"\n📊 비디오 정보:")
        print(f"  - 프레임 수: {len(frames)}")
        print(f"  - FPS: {fps}")
        print(f"  - 길이: {len(frames)/fps:.2f} 초")
        print(f"  - 해상도: {frames[0].shape if frames else 'N/A'}")
    else:
        print(f"\n⚠️ 비디오 저장 실패 (imageio 필요)")
    
    # 데이터 통계
    print(f"\n📈 Reference 데이터 통계:")
    for joint_name in joint_names[:5]:  # 처음 5개만 표시
        q_key = f'q_{joint_name}'
        if q_key in series_data:
            values = series_data[q_key]
            print(f"  {joint_name:20s}: min={values.min():7.3f}, max={values.max():7.3f}, mean={values.mean():7.3f}")

def main():
    parser = argparse.ArgumentParser(description='Reference motion 시각화')
    parser.add_argument('--config', type=str, 
                        default='rl_train/train/train_configs/S004_3D_IL_ver1_0_BASE.json',
                        help='학습 config JSON 파일')
    parser.add_argument('--data', type=str, required=True,
                        help='Reference NPZ 파일 경로')
    parser.add_argument('--output', type=str, default='reference_motion.mp4',
                        help='출력 비디오 파일')
    parser.add_argument('--frames', type=int, default=300,
                        help='재생할 프레임 수')
    parser.add_argument('--fps', type=int, default=30,
                        help='비디오 FPS (기본: 30)')
    parser.add_argument('--camera', type=str, default='side',
                        choices=['front', 'side', 'diagonal', 'top', 'back'],
                        help='카메라 각도 (front: 정면, side: 측면, diagonal: 대각선, top: 위, back: 뒤)')
    
    args = parser.parse_args()
    
    visualize_reference_motion(
        args.config,
        args.data,
        args.output,
        args.frames,
        args.fps,
        args.camera
    )

if __name__ == '__main__':
    main()
