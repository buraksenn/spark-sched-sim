#!/bin/bash

# Bulk evaluation script for all seed-model combinations
# This script runs get_bulk_results.py for each model in each seed folder

set -e  # Exit on any error

# Configuration
ARTIFACTS_BASE="/Users/buraksen/d/thesis/artifacts"
OUTPUTS_BASE="outputs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Bulk Evaluation Script ===${NC}"
echo "Artifacts directory: ${ARTIFACTS_BASE}"
echo "Outputs directory: ${OUTPUTS_BASE}"
echo "Script directory: ${SCRIPT_DIR}"
echo ""

# Check if artifacts directory exists
if [ ! -d "${ARTIFACTS_BASE}" ]; then
    echo -e "${RED}Error: Artifacts directory not found: ${ARTIFACTS_BASE}${NC}"
    exit 1
fi

# Create outputs directory
mkdir -p "${OUTPUTS_BASE}"

# Find all seed directories
seed_dirs=($(find "${ARTIFACTS_BASE}" -maxdepth 1 -type d -name "seed*" | sort))

if [ ${#seed_dirs[@]} -eq 0 ]; then
    echo -e "${RED}Error: No seed directories found in ${ARTIFACTS_BASE}${NC}"
    exit 1
fi

echo -e "${GREEN}Found ${#seed_dirs[@]} seed directories:${NC}"
for seed_dir in "${seed_dirs[@]}"; do
    echo "  - $(basename "${seed_dir}")"
done
echo ""

# Process each seed directory
total_combinations=0
completed_combinations=0

for seed_dir in "${seed_dirs[@]}"; do
    seed_name=$(basename "${seed_dir}")
    echo -e "${YELLOW}Processing ${seed_name}...${NC}"
    
    # Create output directory for this seed
    seed_output_dir="${OUTPUTS_BASE}/${seed_name}"
    mkdir -p "${seed_output_dir}"
    
    # Find all model directories in this seed
    model_dirs=($(find "${seed_dir}" -maxdepth 1 -type d -name "*" | grep -v "^${seed_dir}$" | sort))
    
    if [ ${#model_dirs[@]} -eq 0 ]; then
        echo -e "${YELLOW}  Warning: No model directories found in ${seed_dir}${NC}"
        continue
    fi
    
    echo "  Found ${#model_dirs[@]} model directories:"
    for model_dir in "${model_dirs[@]}"; do
        echo "    - $(basename "${model_dir}")"
    done
    
    # Process each model in this seed
    for model_dir in "${model_dirs[@]}"; do
        model_name=$(basename "${model_dir}")
        output_path="${seed_output_dir}/${model_name}.json"
        
        # Check if checkpoints directory exists
        checkpoints_dir="${model_dir}/checkpoints"
        if [ ! -d "${checkpoints_dir}" ]; then
            echo -e "${YELLOW}    Warning: No checkpoints directory found in ${model_dir}${NC}"
            continue
        fi
        
        # Check if there are any numeric checkpoint directories
        checkpoint_count=$(find "${checkpoints_dir}" -maxdepth 1 -type d -name "[0-9]*" | wc -l)
        if [ ${checkpoint_count} -eq 0 ]; then
            echo -e "${YELLOW}    Warning: No numeric checkpoint directories found in ${checkpoints_dir}${NC}"
            continue
        fi
        
        total_combinations=$((total_combinations + 1))
        
        echo -e "${BLUE}    Processing: ${seed_name}/${model_name}${NC}"
        echo "      Base path: ${model_dir}"
        echo "      Output path: ${output_path}"
        echo "      Checkpoints found: ${checkpoint_count}"
        
        # Run the evaluation
        cd "${SCRIPT_DIR}"
        if python get_bulk_results.py --base_path "${model_dir}" --output_path "${output_path}"; then
            completed_combinations=$((completed_combinations + 1))
            echo -e "${GREEN}    ✓ Completed: ${seed_name}/${model_name}${NC}"
        else
            echo -e "${RED}    ✗ Failed: ${seed_name}/${model_name}${NC}"
        fi
        echo ""
    done
    echo ""
done

# Summary
echo -e "${BLUE}=== Summary ===${NC}"
echo "Total combinations found: ${total_combinations}"
echo "Successfully completed: ${completed_combinations}"
echo "Failed: $((total_combinations - completed_combinations))"

if [ ${completed_combinations} -eq ${total_combinations} ]; then
    echo -e "${GREEN}All evaluations completed successfully!${NC}"
else
    echo -e "${YELLOW}Some evaluations failed. Check the output above for details.${NC}"
fi

echo ""
echo -e "${BLUE}Results saved in: ${OUTPUTS_BASE}/${NC}"
echo "Use plot_bulk_results.py to visualize the results." 