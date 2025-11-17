# MyoAssist 모방학습 훈련 실행 스크립트
# 사용법: .\run_imitation_training.ps1 [config_name]
# 예시: .\run_imitation_training.ps1 partial_obs

param(
    [string]$ConfigType = "partial_obs"  # 기본값: partial_obs
)

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "MyoAssist 모방학습 훈련 시작" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

# Conda 환경 활성화
Write-Host "`n[1/3] myoassist 환경 활성화 중..." -ForegroundColor Yellow
conda activate myoassist

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ 환경 활성화 실패" -ForegroundColor Red
    Write-Host "수동으로 활성화 후 다시 시도하세요: conda activate myoassist" -ForegroundColor White
    exit 1
}
Write-Host "✓ 환경 활성화 완료" -ForegroundColor Green

# Config 파일 선택
Write-Host "`n[2/3] Config 파일 설정 중..." -ForegroundColor Yellow

$configPath = switch ($ConfigType) {
    "partial_obs" { "rl_train/train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json" }
    "full_obs" { "rl_train/train/train_configs/imitation_tutorial_22_separated_net_full_obs.json" }
    "speed_control" { "rl_train/train/train_configs/imitation_tutorial_22_separated_net_speed_control.json" }
    "exo_off" { "rl_train/train/train_configs/imitation_tutorial_22_separated_net_exo_off.json" }
    "basic" { "rl_train/train/train_configs/imitation.json" }
    default { "rl_train/train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json" }
}

if (Test-Path $configPath) {
    Write-Host "✓ Config 파일: $configPath" -ForegroundColor Green
} else {
    Write-Host "✗ Config 파일을 찾을 수 없습니다: $configPath" -ForegroundColor Red
    exit 1
}

# 훈련 시작
Write-Host "`n[3/3] 훈련 시작..." -ForegroundColor Yellow
Write-Host "명령어: python rl_train/run_train.py --config_file_path $configPath" -ForegroundColor White
Write-Host ""

python rl_train/run_train.py --config_file_path $configPath

Write-Host "`n==================================" -ForegroundColor Cyan
Write-Host "훈련 완료 또는 중단됨" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "`n결과 확인:" -ForegroundColor Yellow
Write-Host "ls rl_train/results/" -ForegroundColor White
