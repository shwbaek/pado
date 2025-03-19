#!/bin/bash
# Deploy Sphinx documentation to GitHub Pages

echo "Building documentation..."
# Run the build command
cd "$(dirname "$0")"
make html

echo "Creating .nojekyll file..."
touch build/html/.nojekyll

echo "Deploying to GitHub Pages..."
cd build/html

# Initialize git repository in the build directory
git init
git add .
git commit -m "Deploy documentation to GitHub Pages"

# Push to gh-pages branch
git push -f https://github.com/shwbaek/pado.git main:gh-pages

echo "Deployment complete! Your documentation should be available at:"
echo "https://shwbaek.github.io/pado/" 