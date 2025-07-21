#!/bin/bash

# Script to run comprehensive thesis analysis for RL-based Job Scheduling

set -e

# Configuration
OUTPUTS_DIR="outputs"
THESIS_PLOTS_DIR="thesis_plots"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Comprehensive Thesis Analysis Script ===${NC}"
echo "Outputs directory: ${OUTPUTS_DIR}"
echo "Thesis plots directory: ${THESIS_PLOTS_DIR}"
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

# Create thesis plots directory
mkdir -p "${THESIS_PLOTS_DIR}"

# Run the thesis analysis script
echo -e "${YELLOW}Generating comprehensive thesis analysis...${NC}"
cd "${SCRIPT_DIR}"

if python plot_thesis_analysis.py "${OUTPUTS_DIR}" --output_dir "${THESIS_PLOTS_DIR}"; then
    echo -e "${GREEN}✓ Thesis analysis completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}Generated plots in: ${THESIS_PLOTS_DIR}${NC}"
    echo "Available files:"
    ls -la "${THESIS_PLOTS_DIR}"/ 2>/dev/null || true
else
    echo -e "${RED}✗ Thesis analysis failed!${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}=== THESIS VISUALIZATIONS SUMMARY ===${NC}"
echo ""
echo -e "${YELLOW}Individual Model Analysis:${NC}"
echo "• learning_curve_[model].png - Detailed learning curves for each model"
echo "• Shows individual seed performance + mean with confidence bands"
echo "• Perfect for demonstrating model-specific learning dynamics"
echo ""
echo -e "${YELLOW}Comparative Analysis:${NC}"
echo "• convergence_analysis_detailed.png - Convergence speed, stability, efficiency"
echo "• performance_distributions.png - Statistical distributions with violin plots"
echo "• improvement_analysis.png - Learning effectiveness and efficiency analysis"
echo "• statistical_comparison.png - Hypothesis testing and significance analysis"
echo "• learning_dynamics.png - Normalized learning curves comparison"
echo "• thesis_summary_analysis.png - Overall model ranking and comparison"
echo ""
echo -e "${YELLOW}Statistical Data for Thesis:${NC}"
echo "• thesis_summary_statistics.csv - Comprehensive metrics for each model-seed"
echo "• thesis_model_comparison.csv - Aggregated statistics by model type"
echo ""
echo -e "${GREEN}Key Thesis Contributions Highlighted:${NC}"
echo "✓ Learning convergence analysis"
echo "✓ Statistical significance testing"
echo "✓ Performance improvement quantification"
echo "✓ Model robustness across different seeds"
echo "✓ Learning efficiency analysis"
echo "✓ Publication-ready visualizations (300 DPI)"
echo ""
echo "These visualizations provide comprehensive evidence for your RL-based"
echo "job scheduling improvements and are ready for thesis inclusion!" 