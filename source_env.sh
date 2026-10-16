#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# 1. Activate Python virtual environment
if [ -f "$SCRIPT_DIR/env-setup/bin/activate" ]; then
    source "$SCRIPT_DIR/env-setup/bin/activate"
fi

# 2. Configure CUDA path
export CUDA_INSTALL_PATH="${CUDA_INSTALL_PATH:-/usr/local/cuda-12.9}"
if [ -d "$CUDA_INSTALL_PATH/bin" ]; then
    export PATH="$CUDA_INSTALL_PATH/bin:$PATH"
fi
if [ -d "$CUDA_INSTALL_PATH/lib64" ]; then
    export LD_LIBRARY_PATH="$CUDA_INSTALL_PATH/lib64:$LD_LIBRARY_PATH"
fi

# 3. Source Accel-Sim simulator environment
if [ -f "$SCRIPT_DIR/gpu-simulator/setup_environment.sh" ]; then
    source "$SCRIPT_DIR/gpu-simulator/setup_environment.sh" "$@"
fi

# 4. Source GPU App Collection environment if available
if [ -f "$SCRIPT_DIR/gpu-app-collection/src/setup_environment" ]; then
    source "$SCRIPT_DIR/gpu-app-collection/src/setup_environment"
fi
