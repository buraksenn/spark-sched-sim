#!/bin/bash

# Script to run enhanced plotting for bulk evaluation results

set -e

# Configuration
OUTPUTS_DIR="outputs"
PLOTS_DIR="plots"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Enhanced Plotting Script ===${NC}"
echo "Outputs directory: ${OUTPUTS_DIR}"
echo "Plots directory: ${PLOTS_DIR}"
echo "Script directory: ${SCRIPT_DIR}"
echo ""

# Check if outputs directory exists
if [ ! -d "${OUTPUTS_DIR}" ]; then
    echo -e "${RED}Error: Outputs directory not found: ${OUTPUTS_DIR}${NC}"
    echo "Please run the bulk evaluation first using run_bulk_evaluation.sh"
    exit 1
fi

# Check if there are any JSON files
json_files=$(find "${OUTPUTS_DIR}" -name "*.json" | wc -l)
if [ ${json_files} -eq 0 ]; then
    echo -e "${RED}Error: No JSON result files found in ${OUTPUTS_DIR}${NC}"
    echo "Please run the bulk evaluation first using run_bulk_evaluation.sh"
    exit 1
fi

echo -e "${GREEN}Found ${json_files} JSON result files${NC}"

# Create plots directory
mkdir -p "${PLOTS_DIR}"

# Run the enhanced plotting script
echo -e "${YELLOW}Generating comprehensive plots...${NC}"
cd "${SCRIPT_DIR}"

if python plot_bulk_results_enhanced.py "${OUTPUTS_DIR}" --output_dir "${PLOTS_DIR}"; then
    echo -e "${GREEN}✓ Plotting completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}Generated plots in: ${PLOTS_DIR}${NC}"
    echo "Available plots:"
    ls -la "${PLOTS_DIR}"/*.png "${PLOTS_DIR}"/*.csv 2>/dev/null || true
else
    echo -e "${RED}✗ Plotting failed!${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}=== Plot Descriptions ===${NC}"
echo "1. learning_curves.png: Shows learning progress for each model across all seeds"
echo "2. convergence_analysis.png: Final performance and improvement comparison"
echo "3. seed_comparison.png: Seed-to-seed comparison for each model"
echo "4. performance_heatmap.png: Visual performance matrix across seeds and models"
echo "5. summary_statistics.csv: Detailed numerical statistics for thesis analysis"
echo ""
echo "These plots provide comprehensive analysis for your thesis evaluation!" 