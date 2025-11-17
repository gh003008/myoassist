@echo off
REM MyoAssist S004 Motion Imitation Learning - Setup and Training Script
REM ======================================================================

echo.
echo ================================================================================
echo    MyoAssist S004 Motion Imitation Learning
echo ================================================================================
echo.

REM Activate conda environment
echo [1/4] Activating conda environment: myoassist
call conda activate myoassist
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate conda environment 'myoassist'
    echo Please create it first: conda create -n myoassist python=3.9
    pause
    exit /b 1
)

echo.
echo [2/4] Setting up environment...
python setup_environment.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Environment setup failed
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo    Setup Complete! Ready to train.
echo ================================================================================
echo.
echo Choose an option:
echo   1. Quick test (verify setup)
echo   2. Full training
echo   3. Exit
echo.

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Starting quick test...
    python train_S004_motion.py --quick_test
) else if "%choice%"=="2" (
    echo.
    echo Starting full training...
    python train_S004_motion.py
) else (
    echo.
    echo Exiting...
    exit /b 0
)

echo.
echo ================================================================================
echo    Training Complete!
echo ================================================================================
echo.
echo Check results in: rl_train\results\train_session_[timestamp]
echo.
pause
