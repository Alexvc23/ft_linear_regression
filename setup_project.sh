#!/bin/bash

# Print message
echo "Setting up the project..."

# Check for Python version
PYTHON_VERSION=$(python3 --version 2>&1)
echo "Using $PYTHON_VERSION"

# Check if Poetry is installed, otherwise install it
if command -v poetry >/dev/null 2>&1; then
    echo "Poetry detected, installing dependencies..."
else
    echo "Poetry is not installed. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
    echo "Poetry installed successfully."
fi

# Install Poetry dependencies
echo "Installing project dependencies..."
poetry install

# Confirm environment setup
echo "Environment setup complete!"