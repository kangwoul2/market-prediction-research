@echo off
setlocal
cd /d "%~dp0"

echo ================================================
echo Market Prediction Research - Locked Development
 echo ================================================

if exist "reports\outputs\final_development_lock.json" (
    if /I not "%ALLOW_RESEARCH_REBUILD%"=="1" (
        echo.
        echo [LOCKED] 개발 연구와 설정 선택이 완료되어 기본 재학습을 막고 있습니다.
        echo [LOCKED] 새로운 미래 데이터는 기존 설정 평가에만 사용해야 합니다.
        echo.
        echo Final report:
        echo   docs\FINAL_RESEARCH_REPORT.md
        echo Study log:
        echo   docs\PERSONAL_STUDY_LOG.md
        echo Prospective protocol:
        echo   docs\PROSPECTIVE_EVALUATION_PROTOCOL.md
        echo.
        echo 연구 파이프라인을 의도적으로 다시 실행하려면:
        echo   set ALLOW_RESEARCH_REBUILD=1
        echo   RUN_ME.bat
        echo.
        pause
        exit /b 0
    )
)

if not exist ".env" copy ".env.example" ".env" >nul

if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] Python virtual environment creating...
    where py >nul 2>nul
    if %errorlevel%==0 (
        py -3.11 -m venv .venv
        if errorlevel 1 py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Could not create .venv. Install Python 3.11 or later and retry.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import hashlib,pathlib,sys; r=pathlib.Path('requirements.txt'); s=pathlib.Path('.venv/.requirements.sha256'); h=hashlib.sha256(r.read_bytes()).hexdigest(); sys.exit(0 if s.exists() and s.read_text().strip()==h else 1)"
if errorlevel 1 (
    echo [SETUP] requirements.txt changed. Installing dependencies...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 goto :error
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :error
    ".venv\Scripts\python.exe" -c "import hashlib,pathlib; r=pathlib.Path('requirements.txt'); pathlib.Path('.venv/.requirements.sha256').write_text(hashlib.sha256(r.read_bytes()).hexdigest())"
) else (
    echo [SKIP] Dependencies unchanged.
)

echo [RUN] Starting checkpointed research pipeline...
".venv\Scripts\python.exe" run_pipeline.py
if errorlevel 1 goto :error

echo.
echo ================================================
echo DONE
 echo ================================================
pause
exit /b 0

:error
echo.
echo ================================================
echo Pipeline stopped with an error or interruption.
echo Finished experiment combinations remain in reports\cache.
echo ================================================
pause
exit /b 1
