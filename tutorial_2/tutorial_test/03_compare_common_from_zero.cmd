@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."
REM Same seed and total steps for paired common-vs-guided experiments.
python -m tutorial_2.tutorial_test.train_ppo_v4 train --stage 2 --reward common --steps 500000 --seed 42
if errorlevel 1 goto fail
python -m tutorial_2.tutorial_test.train_ppo_v4 train --stage 4 --reward common --steps 500000 --seed 42
if errorlevel 1 goto fail
python -m tutorial_2.tutorial_test.result_summary_v4
goto end
:fail
echo TRAINING FAILED - READ ERROR ABOVE
:end
pause
