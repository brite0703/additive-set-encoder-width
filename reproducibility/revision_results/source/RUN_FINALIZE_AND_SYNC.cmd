@echo off
setlocal
cd /d "%~dp0"

set "PY_CMD=python"

set "SYNCED_ROOT=%~dp0revision_results"

if not exist "remote_local_run_root.txt" (
  echo remote_local_run_root.txt is missing. Run RUN_FULL.cmd first.
  goto failed
)
set /p "RUN_ROOT="<"remote_local_run_root.txt"
if not defined RUN_ROOT goto failed

if not exist "%RUN_ROOT%\resolved_config.json" (
  echo No full-run result directory was found at %RUN_ROOT%.
  goto failed
)

echo Aggregating and verifying the remote-local result directory...
%PY_CMD% aggregate_results.py --input "%RUN_ROOT%" --output "%RUN_ROOT%\aggregate"
if errorlevel 1 goto failed
%PY_CMD% verify_artifacts.py --root "%RUN_ROOT%" --write-manifest
if errorlevel 1 goto failed
%PY_CMD% verify_artifacts.py --root "%RUN_ROOT%"
if errorlevel 1 goto failed

if exist "%SYNCED_ROOT%" (
  echo.
  echo %SYNCED_ROOT% already exists. This script will not merge into an older synchronized copy.
  echo Rename the existing directory, then run this script again.
  goto failed
)

echo Copying the complete verified artifact into Google Drive...
robocopy "%RUN_ROOT%" "%SYNCED_ROOT%" /E /COPY:DAT /DCOPY:DAT /R:2 /W:5
if errorlevel 8 goto failed

echo Verifying the synchronized copy...
%PY_CMD% verify_artifacts.py --root "%SYNCED_ROOT%"
if errorlevel 1 goto failed

echo.
echo FINAL VERIFICATION PASSED. Leave Google Drive running until synchronization is complete.
pause
exit /b 0

:failed
echo.
echo FINALIZATION DID NOT COMPLETE. Keep this window open and send its final error message.
pause
exit /b 1
