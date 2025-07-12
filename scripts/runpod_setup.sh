
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

# check if uv is installed
if ! command -v uv &> /dev/null; then
    log "uv could not be found"
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    check_error "Failed to install uv"
    # add uv to path
    export PATH="$HOME/.local/bin:$PATH"
    check_error "Failed to add uv to path"
fi

# install new python version in venv 3.10
uv python install 3.10
check_error "Failed to install python 3.10"

# activate venv
uv venv
check_error "Failed to activate venv"

log "Installing requirements..."
uv pip install -r requirements_wo_torch.txt
check_error "Failed to install requirements"

# install torch 2.1
uv pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html


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


        
