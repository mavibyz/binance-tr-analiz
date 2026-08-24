@echo off
title Binance TR Canli Sinyal ve Uyari v7
setlocal
cd /d "%~dp0"
echo Gerekli paketler kontrol ediliyor...
py -m pip install streamlit pandas numpy requests plotly winotify
if errorlevel 1 (
 echo Paket kurulumu basarisiz.
 pause
 exit /b 1
)
echo Program baslatiliyor...
py -m streamlit run "%~dp0app.py"
pause
