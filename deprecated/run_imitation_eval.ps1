# MyoAssist 모방학습 평가 스크립트
# 사용법: .\run_imitation_eval.ps1 [session_folder]
# 예시: .\run_imitation_eval.ps1 train_session_20250112-123456

param(
    [Parameter(Mandatory=$false)]
    [string]$SessionFolder = ""
)

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "MyoAssist 모방학습 평가 시작" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

# Conda 환경 활성화
Write-Host "`n[1/3] myoassist 환경 활성화 중..." -ForegroundColor Yellow
conda activate myoassist

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ 환경 활성화 실패" -ForegroundColor Red
    exit 1
}
Write-Host "✓ 환경 활성화 완료" -ForegroundColor Green

# Session 폴더 선택
Write-Host "`n[2/3] Session 폴더 확인 중..." -ForegroundColor Yellow

if ($SessionFolder -eq "") {
    Write-Host "사용 가능한 훈련 세션:" -ForegroundColor Cyan
    Get-ChildItem -Path "rl_train/results" -Directory | ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor White }
    
    Write-Host "`n예제 pretrained 모델 사용:" -ForegroundColor Yellow
    $SessionFolder = "docs/assets/tutorial_rl_models/train_session_20250728-161129_tutorial_partial_obs"
    Write-Host "  $SessionFolder" -ForegroundColor White
}

$sessionPath = $SessionFolder

if (Test-Path $sessionPath) {
    Write-Host "✓ Session 폴더 확인: $sessionPath" -ForegroundColor Green
} else {
    Write-Host "✗ Session 폴더를 찾을 수 없습니다: $sessionPath" -ForegroundColor Red
    Write-Host "다음 중 하나를 선택하세요:" -ForegroundColor Yellow
    Get-ChildItem -Path "rl_train/results" -Directory | ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor White }
    exit 1
}

# 평가 시작
Write-Host "`n[3/3] 평가 시작..." -ForegroundColor Yellow
Write-Host "명령어: python rl_train/run_policy_eval.py $sessionPath" -ForegroundColor White
Write-Host ""

python rl_train/run_policy_eval.py $sessionPath

Write-Host "`n==================================" -ForegroundColor Cyan
Write-Host "평가 완료!" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "`n결과 확인:" -ForegroundColor Yellow
Write-Host "$sessionPath/analyze_results/" -ForegroundColor White
