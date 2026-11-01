@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."
python -m pip install -r requirements-rl.txt
if errorlevel 1 goto fail
python -m unittest discover -s tutorial_2\tutorial_test\tests -p "test_*_v4.py" -v
if errorlevel 1 goto fail
for %%S in (1 2 3 4 5) do (
    python -m tutorial_2.tutorial_test.train_ppo_v4 parity --stage %%S --actions 200
    if errorlevel 1 goto fail
    python -m tutorial_2.tutorial_test.train_ppo_v4 smoke --stage %%S --reward guided --actions 100
    if errorlevel 1 goto fail
)
echo ALL REAL-GAME CHECKS PASSED
goto end
:fail
echo SOME CHECK FAILED - DO NOT START TRAINING
:end
pause
