#!/bin/bash

# delete unnecessary files
find . -name "*.pyc" | xargs rm -rf
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".DS_Store" -exec rm -rf {} +

# all changes are added to the git
git add .
git commit -m "update data in script"
git config --global push.default simple
git push