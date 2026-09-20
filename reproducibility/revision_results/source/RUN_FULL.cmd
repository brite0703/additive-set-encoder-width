@echo off
setlocal
cd /d "%~dp0"

set "PY_CMD=python"

%PY_CMD% -c "import sys, numpy, torch; assert sys.version_info[:2] == (3, 11), sys.version; assert numpy.__version__ == '2.3.5', numpy.__version__; assert torch.__version__.split('+')[0] == '2.9.0', torch.__version__; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0))"
if errorlevel 1 goto environment_error

echo.
echo Enter a short folder on a remote-local NVMe or SSD.
echo Do not enter a Google Drive, OneDrive, Dropbox, or network-synchronized folder.
echo Example: D:\NN_revision_runs
set /p "LOCAL_RUN_PARENT=Fast local folder: "
if not defined LOCAL_RUN_PARENT goto path_error
set "RUN_ROOT=%LOCAL_RUN_PARENT%\revision_results"
if not exist "%LOCAL_RUN_PARENT%" mkdir "%LOCAL_RUN_PARENT%"
if errorlevel 1 goto failed
>"remote_local_run_root.txt" echo %RUN_ROOT%

set "RESUME_ARG="
if exist "%RUN_ROOT%\resolved_config.json" set "RESUME_ARG=--resume"

echo.
echo Full output: %RUN_ROOT%
if defined RESUME_ARG echo Compatible completed records will be checked against the configuration hash and skipped.
echo Starting the fixed 960-run grid. Keep this window open.
%PY_CMD% run_revision_experiments.py --config config_revision.json --output "%RUN_ROOT%" %RESUME_ARG%
if errorlevel 1 goto failed

echo.
echo FULL GRID COMPLETED. Now double-click RUN_FINALIZE_AND_SYNC.cmd.
pause
exit /b 0

:environment_error
echo.
echo The fixed environment is not ready. Run RUN_SMOKE.cmd first.
goto failed
:path_error
echo.
echo A remote-local NVMe or SSD folder is required.
:failed
echo.
echo THE FULL GRID STOPPED. Keep the output directory; rerunning this script will resume compatible completed cells.
pause
exit /b 1
