@echo off
setlocal
cd /d "%~dp0" || exit /b 1
set "PYTHONUTF8=1"
set "HF_HOME=%CD%\.model-cache"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "MODE=%~1"
if not defined MODE set "MODE=both"
if /i "%MODE%"=="pilot" goto :data
if /i "%MODE%"=="full" goto :data
if /i "%MODE%"=="both" goto :data
if /i "%MODE%"=="b42" goto :data
echo Usage: run-member02.cmd [pilot^|full^|both^|b42]
exit /b 2

:data
if exist "data\project_data.zip" goto :setup
if not exist "data" mkdir "data"
for %%F in ("project_data.zip" "%USERPROFILE%\Downloads\project_data.zip" "%USERPROFILE%\Downloads\Telegram Desktop\project_data.zip") do (
    if exist "%%~F" (
        copy /b "%%~F" "data\project_data.zip" >nul
        if errorlevel 1 goto :failed
        goto :setup
    )
)
echo Place the original project_data.zip in the data folder, then run again.
goto :failed

:setup
set "RUNPY=%CD%\.venv\Scripts\python.exe"
if exist "%RUNPY%" goto :install
for %%V in (313 312 311) do (
    if exist "%LocalAppData%\Programs\Python\Python%%V\python.exe" (
        "%LocalAppData%\Programs\Python\Python%%V\python.exe" -m venv .venv
        if errorlevel 1 goto :failed
        goto :install
    )
)
for %%V in (3.13 3.12 3.11) do (
    py -%%V -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
    if not errorlevel 1 (
        py -%%V -m venv .venv
        if errorlevel 1 goto :failed
        goto :install
    )
)
echo Requires 64-bit CPython 3.11, 3.12 or 3.13.
goto :failed

:install
echo Checking inputs and preparing the local CPU environment...
"%RUNPY%" -m unittest discover -s tests -v
if errorlevel 1 goto :failed
"%RUNPY%" -m pip install --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu -r requirements-teacher.txt
if errorlevel 1 goto :failed
if /i "%MODE%"=="b42" goto :b42files
"%RUNPY%" -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
if errorlevel 1 goto :failed
"%RUNPY%" -m pip install --only-binary=:all: -r requirements-a123.txt
if errorlevel 1 goto :failed
if /i "%MODE%"=="full" goto :full
call :stage pilot
if errorlevel 1 goto :failed
if /i "%MODE%"=="pilot" goto :done
:full
call :stage full
if errorlevel 1 goto :failed
:done
echo Finished. Results are in the runs folder.
exit /b 0

:b42files
for %%S in (pilot full) do (
    "%RUNPY%" generate.py --stage %%S --teacher B --teacher A --teacher C
    if errorlevel 1 goto :failed
    "%RUNPY%" -c "from pathlib import Path; from protocol import atomic_text; root = Path('runs'); src = root / '%%S' / 'sft' / 'teacher_B.jsonl'; atomic_text(root / 'B42_to_send' / 'teacher_B_%%S.jsonl', src.read_text(encoding='utf-8'))"
    if errorlevel 1 goto :failed
    echo Ready to send: "%CD%\runs\B42_to_send\teacher_B_%%S.jsonl"
)
exit /b 0

:stage
echo ===== %1 =====
"%RUNPY%" generate.py --stage %1
if errorlevel 1 exit /b 1
"%RUNPY%" run_a123.py --stage %1 --device cpu
if errorlevel 1 exit /b 1
exit /b 0

:failed
echo The run stopped. Read the error above. Completed answers are saved.
if not defined MEMBER02_NONINTERACTIVE pause
exit /b 1
