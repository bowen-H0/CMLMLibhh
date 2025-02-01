#!/bin/bash

# Ensure the script stops on errors
set -e

# Check if a commit message was provided
if [ -z "$1" ]; then
  echo "Please provide a commit message."
  exit 1
fi

# Display the current Git status
echo "Current Git status:"
git status

# Add all modified files to Git
git add .

# Commit the changes with the provided commit message
git commit -m "$1"

# Push the changes to the remote repository's main branch
git push origin main

# Display a success message after pushing
echo "Changes pushed to GitHub successfully!"
