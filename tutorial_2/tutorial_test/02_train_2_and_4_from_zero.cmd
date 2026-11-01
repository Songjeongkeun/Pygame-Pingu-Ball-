@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."
REM Separate from-scratch policies, but same 1-5 team environment and PPO settings.
REM No --resume and no --init-model: always initialized from zero.
python -m tutorial_2.tutorial_test.train_ppo_v4 train --stage 2 --reward guided --steps 500000 --seed 42
if errorlevel 1 goto fail
python -m tutorial_2.tutorial_test.train_ppo_v4 train --stage 4 --reward guided --steps 500000 --seed 42
if errorlevel 1 goto fail
python -m tutorial_2.tutorial_test.result_summary_v4
goto end
:fail
echo TRAINING FAILED - READ ERROR ABOVE
:end
pause
