#!/bin/bash

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if a command succeeded
check_error() {
    if [ $? -ne 0 ]; then
        log "ERROR: $1"
        exit 1
    fi
}

# Function to compare version numbers
version_gt() {
    test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"
}

# Function to check if a package is installed in the current environment
check_package() {
    conda list | grep -q "^$1[[:space:]]"
    return $?
}

# Initialize conda for the current shell
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    log "Initializing conda for Linux..."
    module load miniconda3
elif [[ "$OSTYPE" == "darwin"* ]]; then
    log "Initializing conda for macOS..."
    eval "$(conda shell.bash hook)"
else
    log "ERROR: Unsupported operating system: $OSTYPE"
    exit 1
fi
check_error "Failed to initialize conda"

# Check if conda environment exists
if conda env list | grep -q "dl-env"; then
    log "Found existing dl-env environment"
    log "Activating existing environment..."
    conda activate dl-env
    check_error "Failed to activate dl-env environment"
    
    log "Checking installed packages..."
    conda list
    check_error "Failed to list conda packages"
    
    log "Verifying Python version..."
    PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    log "Current Python version: $PYTHON_VERSION"
    
    if ! version_gt "$PYTHON_VERSION" "3.7"; then
        log "WARNING: Python version is below 3.8, updating to Python 3.10..."
        conda install python=3.10 -y
        check_error "Failed to update Python version"
    fi

    # Check for required packages
    log "Checking for required packages..."
    PACKAGES_TO_INSTALL=()
    
    if ! check_package "torch"; then
        log "PyTorch not found"
        PACKAGES_TO_INSTALL+=("torch")
    fi
    
    if ! check_package "torchvision"; then
        log "torchvision not found"
        PACKAGES_TO_INSTALL+=("torchvision")
    fi
    
    if ! check_package "torchaudio"; then
        log "torchaudio not found"
        PACKAGES_TO_INSTALL+=("torchaudio")
    fi
    
    if ! check_package "torch-geometric"; then
        log "PyTorch Geometric not found"
        PACKAGES_TO_INSTALL+=("torch-geometric")
    fi

    if [ ${#PACKAGES_TO_INSTALL[@]} -eq 0 ]; then
        log "All required packages are already installed. Exiting."
        exit 0
    else
        log "Missing packages detected: ${PACKAGES_TO_INSTALL[*]}. Proceeding with installation..."
    fi
else
    log "Creating new dl-env environment..."
    conda create --name dl-env -y
    check_error "Failed to create conda environment"
    
    log "Activating new environment..."
    conda activate dl-env
    check_error "Failed to activate dl-env environment"
    
    log "Installing Python 3.10..."
    conda install python=3.10 -y
    check_error "Failed to install Python 3.10"
    
    log "New environment setup completed"
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    log "ERROR: Python 3 is not installed"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    log "ERROR: pip3 is not installed"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
log "Detected Python version: $PYTHON_VERSION"

if ! version_gt "$PYTHON_VERSION" "3.7"; then
    log "ERROR: Python 3.8 or higher is required"
    exit 1
fi

log "Installing base requirements..."
pip install -r requirements_wo_torch.txt
check_error "Failed to install base requirements"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    log "Installing PyTorch with CUDA 12.7 support for Linux..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [[ "$OSTYPE" == "darwin"* ]]; then
    log "Installing PyTorch for macOS..."
    pip3 install torch torchvision torchaudio
else
    log "ERROR: Unsupported operating system: $OSTYPE"
    exit 1
fi
check_error "Failed to install PyTorch"

log "Uninstalling existing PyTorch Geometric packages..."
pip uninstall torch-scatter torch-sparse torch-geometric torch-cluster -y
check_error "Failed to uninstall existing packages"

# Get PyTorch version
log "Detecting PyTorch version..."
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
check_error "Failed to detect PyTorch version"
log "Detected PyTorch version: $TORCH_VERSION"

# Install dependencies
log "Installing PyTorch Geometric dependencies..."
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}.html
check_error "Failed to install torch-scatter"

pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_VERSION}.html
check_error "Failed to install torch-sparse"

pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH_VERSION}.html
check_error "Failed to install torch-cluster"

# Install PyTorch Geometric
log "Installing PyTorch Geometric..."
pip install git+https://github.com/pyg-team/pytorch_geometric.git
check_error "Failed to install PyTorch Geometric"

# Verify installations
log "Verifying installations..."
python3 -c "import torch; import torch_geometric; print('PyTorch version:', torch.__version__); print('PyTorch Geometric version:', torch_geometric.__version__)"
check_error "Failed to verify installations"

log "Installation completed successfully!"


        
