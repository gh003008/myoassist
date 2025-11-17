"""
현재 학습 중인 모델의 체크포인트를 로드해서 렌더링 비디오 생성
"""
import os
import json
import argparse
import numpy as np
from stable_baselines3 import PPO
from rl_train.envs.environment_handler import EnvironmentHandler
import rl_train.train.train_configs.config as myoassist_config
from pathlib import Path

def render_checkpoint(config_path, checkpoint_path, output_path, num_steps=500):
    """체크포인트를 로드하고 렌더링하여 비디오 저장"""
    
    # Config 로드 (run_train.py와 동일한 방식)
    default_config = EnvironmentHandler.get_session_config_from_path(
        config_path, 
        myoassist_config.TrainSessionConfigBase
    )
    config_type = EnvironmentHandler.get_config_type_from_session_id(
        default_config.env_params.env_id
    )
    config = EnvironmentHandler.get_session_config_from_path(config_path, config_type)
    
    # 시드 설정
    np.random.seed(1234)
    
    # 렌더링 환경 생성
    print("🎥 렌더링 환경 생성 중...")
    env = EnvironmentHandler.create_environment(
        config, 
        is_rendering_on=True, 
        is_evaluate_mode=True
    )
    
    # 모델 로드
    print(f"💾 체크포인트 로드: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=env)
    
    # 렌더링 실행
    print(f"🏃 렌더링 시작 ({num_steps} steps)...")
    obs, info = env.reset()
    
    episode_count = 0
    episode_reward = 0
    all_rewards = []
    
    # 비디오 녹화 준비
    try:
        import imageio
        video_enabled = True
        frames = []
        print("📹 비디오 녹화 활성화됨")
    except ImportError:
        video_enabled = False
        print("⚠️ imageio 없음 - 비디오 저장 건너뜀 (pip install imageio 추천)")
    
    for step in range(num_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        
        # 프레임 캡처 (비디오용, 30Hz → 15fps)
        if video_enabled and step % 2 == 0:
            try:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            except Exception as e:
                if step == 0:
                    print(f"⚠️ 프레임 캡처 실패: {e}")
                    video_enabled = False
        
        if step % 50 == 0:
            print(f"  Step {step}/{num_steps}, Reward: {episode_reward:.2f}")
        
        # 렌더링 속도 조절 (실시간처럼)
        import time
        time.sleep(0.01)  # 10ms 딜레이
        
        if truncated or done:
            all_rewards.append(episode_reward)
            episode_count += 1
            print(f"  ✅ 에피소드 {episode_count} 완료, 보상: {episode_reward:.2f}")
            episode_reward = 0
            obs, info = env.reset()
    
    # 마지막 에피소드 보상 추가
    if episode_reward != 0:
        all_rewards.append(episode_reward)
        episode_count += 1
    
    # 비디오 저장
    if video_enabled and len(frames) > 0:
        video_path = output_dir / "rendering.mp4"
        print(f"💾 비디오 저장 중... ({len(frames)} 프레임)")
        imageio.mimsave(str(video_path), frames, fps=15)
        print(f"🎬 비디오 저장 완료: {video_path}")
    
    env.close()
    
    # 결과 저장
    mean_reward = np.mean(all_rewards) if all_rewards else 0.0
    print(f"\n📊 평가 완료:")
    print(f"  - 총 에피소드: {episode_count}")
    print(f"  - 평균 보상: {mean_reward:.2f}")
    print(f"  - 보상 범위: [{min(all_rewards):.2f}, {max(all_rewards):.2f}]")
    
    results = {
        'checkpoint': checkpoint_path,
        'num_steps': num_steps,
        'num_episodes': episode_count,
        'episode_rewards': all_rewards,
        'mean_reward': float(mean_reward),
    }
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "render_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 결과 저장: {results_path}")
    print(f"\n🎬 비디오는 MuJoCo 렌더 창에서 확인하세요!")
    print(f"   (MuJoCo는 자동으로 창을 띄웁니다)")

def main():
    parser = argparse.ArgumentParser(description='현재 체크포인트 렌더링')
    parser.add_argument('--config', type=str, required=True, help='학습 config JSON 파일')
    parser.add_argument('--checkpoint', type=str, required=True, help='체크포인트 .zip 파일')
    parser.add_argument('--output', type=str, default='rl_train/results/manual_render', 
                        help='출력 디렉토리')
    parser.add_argument('--steps', type=int, default=500, help='렌더링 스텝 수')
    
    args = parser.parse_args()
    
    render_checkpoint(args.config, args.checkpoint, args.output, args.steps)

if __name__ == '__main__':
    main()
