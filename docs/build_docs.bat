@echo off
echo Building PADO documentation...

REM Clean build directory
echo Cleaning build directory...
if exist build rmdir /s /q build

REM Build HTML documentation
echo Building HTML documentation...
python -m sphinx -b html source build/html

if errorlevel 1 (
    echo Error building documentation. Check the error messages above.
    exit /b 1
)

echo Documentation build complete!
echo You can find the built documentation in docs/build/html/index.html 