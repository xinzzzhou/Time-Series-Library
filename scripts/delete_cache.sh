#!/bin/bash
find . -name "*.pyc" | xargs rm -rf
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".DS_Store" -exec rm -rf {} +
git add .
git commit -m "sort the code"
git push