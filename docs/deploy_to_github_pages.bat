@echo off
REM Deploy Sphinx documentation to GitHub Pages

echo Building documentation...
call build_docs.bat

echo Creating .nojekyll file...
echo. > build\html\.nojekyll

echo Deploying to GitHub Pages...
cd build\html

REM Initialize git repository in the build directory
git init
git add .
git commit -m "Deploy documentation to GitHub Pages"

REM Push to gh-pages branch
git push -f https://github.com/shwbaek/pado.git main:gh-pages

echo Deployment complete! Your documentation should be available at:
echo https://shwbaek.github.io/pado/ 