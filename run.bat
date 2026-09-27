@echo off
chcp 65001 >nul
title OmniVoice Lab
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERRO] Ambiente .venv nao encontrado em %CD%
    echo        Ele precisa ser recriado antes de rodar o servidor.
    pause
    exit /b 1
)

echo ==========================================================
echo   OmniVoice Lab
echo   Endereco:  http://127.0.0.1:8765
echo   O navegador abre sozinho quando o servidor estiver no ar.
echo   Para desligar e liberar a GPU: feche esta janela ou Ctrl+C
echo ==========================================================
echo.

.venv\Scripts\python.exe -u server.py

echo.
echo Servidor encerrado. Pode fechar esta janela.
pause
