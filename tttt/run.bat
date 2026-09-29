@echo off
title MazeTrack Desktop - Animal Behavior Tracking
cd /d "%~dp0"
echo ====================================================
echo  MazeTrack Desktop - Animal Behavior Tracking
echo  Alternatif Sederhana ANY-maze (Local Client)
echo ====================================================
echo.
echo Sedang memulai aplikasi...
python main.py
if errorlevel 1 (
    echo.
    echo Terjadi kesalahan saat menjalankan aplikasi.
    pause
)
