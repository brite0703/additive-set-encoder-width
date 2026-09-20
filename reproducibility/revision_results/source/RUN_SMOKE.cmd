@echo off
setlocal
cd /d "%~dp0"

set "PY_CMD=python"

echo Checking the fixed Python, NumPy, PyTorch, and CUDA environment...
%PY_CMD% -c "import sys, numpy, torch; assert sys.version_info[:2] == (3, 11), sys.version; assert numpy.__version__ == '2.3.5', numpy.__version__; assert torch.__version__.split('+')[0] == '2.9.0', torch.__version__; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('Python:', sys.version.split()[0]); print('NumPy:', numpy.__version__); print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0))"
if errorlevel 1 goto environment_error

if exist "smoke_results" (
  echo.
  echo smoke_results already exists. This script will not overwrite it.
  echo Rename that directory or inspect it before trying again.
  goto failed
)

echo.
echo Running the reduced fixed-configuration smoke grid...
%PY_CMD% run_revision_experiments.py --config config_revision.json --output smoke_results --smoke
if errorlevel 1 goto failed
%PY_CMD% aggregate_results.py --input smoke_results --output smoke_results\aggregate
if errorlevel 1 goto failed
%PY_CMD% verify_artifacts.py --root smoke_results --write-manifest
if errorlevel 1 goto failed
%PY_CMD% verify_artifacts.py --root smoke_results
if errorlevel 1 goto failed

echo.
echo SMOKE TEST PASSED. Send a screenshot of this window to Jih-Jeng.
pause
exit /b 0

:environment_error
echo.
echo The fixed environment is not ready. Follow the Clean remote environment section in README.md.
:failed
echo.
echo SMOKE TEST DID NOT COMPLETE. Keep this window open and send its final error message.
pause
exit /b 1
