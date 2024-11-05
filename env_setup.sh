#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
echo "Script directory: $SCRIPT_DIR"

# Initialize conda in the script's environment
eval "$(conda shell.bash hook)"

# Check if "pokerenv" environment already exists
if conda env list | grep -qw 'pokerenv'; then
    echo "Conda environment 'pokerenv' already exists. Skipping creation. Activating pokerenv."
else
    echo "Creating conda environment 'pokerenv'."
    conda create --name pokerenv python=3.10 -y
fi

conda activate pokerenv

function install_if_missing {
    package=$1
    if ! conda list | grep -qw "$package"; then
        echo "Installing $package..."
        shift
        conda install "$@" -y
    else
        echo "$package is already installed."
    fi
}

# Check and install required packages
install_if_missing "pytorch" "pytorch torchvision torchaudio pytorch-cuda=12.2 -c pytorch -c nvidia"
install_if_missing "cudatoolkit-dev" "-c nvidia cudatoolkit-dev=12.2"
install_if_missing "cudnn" "-c conda-forge cudnn=8.1"
install_if_missing "transformers" "transformers pandas numpy=1.26.4 tqdm scikit-learn"

# Install additional Python packages from requirements.txt
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "Installing pip packages from requirements.txt"
    pip install -r "$SCRIPT_DIR/requirements.txt"
else
    echo "No requirements.txt found. Skipping pip installations."
fi

echo "Environment setup complete."