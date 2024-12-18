#!/bin/bash

# Print message
echo "Setting up the project..."

# Check for Python version
PYTHON_VERSION=$(python3 --version 2>&1)
echo "Using $PYTHON_VERSION"

# Install Poetry dependencies
if command -v poetry >/dev/null 2>&1; then
    echo "Poetry detected, installing dependencies..."
    poetry install
else
    echo "Error: Poetry is not installed. Please install Poetry first."
    exit 1
fi

# Confirm environment setup
echo "Environment setup complete!"
