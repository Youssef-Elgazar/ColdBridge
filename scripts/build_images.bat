@echo off
REM =========================================================
REM  build_images.bat — Build all ColdBridge worker images
REM  Run from the project root: scripts\build_images.bat
REM =========================================================

echo === Building ColdBridge worker images ===
echo.

echo [1/3] Building Python function image...
docker build -t coldbridge/python-fn:latest functions/python_fn/
if %ERRORLEVEL% neq 0 ( echo ERROR: Python image failed & exit /b 1 )
echo   OK  coldbridge/python-fn:latest

echo.
echo [2/3] Building Node.js function image...
docker build -t coldbridge/node-fn:latest functions/node_fn/
if %ERRORLEVEL% neq 0 ( echo ERROR: Node image failed & exit /b 1 )
echo   OK  coldbridge/node-fn:latest

echo.
echo [3/3] Building Java function image...
docker build -t coldbridge/java-fn:latest functions/java_fn/
if %ERRORLEVEL% neq 0 ( echo ERROR: Java image failed & exit /b 1 )
echo   OK  coldbridge/java-fn:latest

echo.
echo === All images built successfully ===
docker images --filter "reference=coldbridge/*"
