# MyoAssist 환경 세팅 스크립트 (PowerShell)
# 사용법: .\setup_myoassist_env.ps1

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "MyoAssist 환경 세팅 시작" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

# 1. Conda 환경 존재 확인
Write-Host "`n[1/4] Conda 환경 확인 중..." -ForegroundColor Yellow
$envExists = conda env list | Select-String "myoassist"

if ($envExists) {
    Write-Host "✓ myoassist 환경이 이미 존재합니다." -ForegroundColor Green
} else {
    Write-Host "✗ myoassist 환경이 없습니다. 생성 중..." -ForegroundColor Red
    Write-Host "다음 명령어를 실행하세요:" -ForegroundColor Yellow
    Write-Host "conda create -n myoassist python=3.11 -y" -ForegroundColor White
    Write-Host "conda activate myoassist" -ForegroundColor White
    Write-Host "pip install -e ." -ForegroundColor White
    exit 1
}

# 2. Conda 환경 활성화
Write-Host "`n[2/4] myoassist 환경 활성화 중..." -ForegroundColor Yellow
conda activate myoassist

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ 환경 활성화 실패. 수동으로 활성화하세요:" -ForegroundColor Red
    Write-Host "conda activate myoassist" -ForegroundColor White
    exit 1
}
Write-Host "✓ myoassist 환경 활성화 완료" -ForegroundColor Green

# 3. 필수 패키지 확인
Write-Host "`n[3/4] 필수 패키지 확인 중..." -ForegroundColor Yellow
$packages = @("stable-baselines3", "mujoco", "gymnasium", "numpy", "mediapy")

foreach ($pkg in $packages) {
    $installed = pip list | Select-String $pkg
    if ($installed) {
        Write-Host "  ✓ $pkg 설치됨" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $pkg 미설치. 설치를 권장합니다." -ForegroundColor Red
    }
}

# 4. MyoAssist 패키지 확인
Write-Host "`n[4/4] MyoAssist 패키지 확인 중..." -ForegroundColor Yellow
$myoassistInstalled = pip list | Select-String "MyoAssist"

if ($myoassistInstalled) {
    Write-Host "✓ MyoAssist 설치됨" -ForegroundColor Green
} else {
    Write-Host "✗ MyoAssist 미설치. 설치 중..." -ForegroundColor Yellow
    pip install -e .
}

Write-Host "`n==================================" -ForegroundColor Cyan
Write-Host "환경 세팅 완료!" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "`n다음 명령어로 테스트하세요:" -ForegroundColor Yellow
Write-Host "python test_setup.py" -ForegroundColor White
