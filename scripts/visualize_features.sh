#!/bin/bash
# =============================================================================
# visualize_features.sh - Raster Feature Visualization Script
# =============================================================================
# 
# Description:
#   This script creates visualizations for raster features that have been
#   extracted using the raster_features package. It works with the category-based
#   feature extraction approach and directly leverages the existing visualization
#   module in the package.
#
# Author: Marco Accardi
# Date: April 13, 2025
# =============================================================================

# Set bash to exit on error and prevent unset variable usage
set -e
set -o nounset

# Default values
RASTER_FILE=""
FEATURE_DIR="results"
OUTPUT_DIR="visualizations"
CATEGORIES=()
ALL_CATEGORIES=false
VISUALIZE_3D=false
SHOW_PLOTS=false
INTERACTIVE=false
COMPARE_FEATURES=false
SAMPLE_RATE=0.1
VERBOSE=false

# Text formatting for pretty output
BOLD="\033[1m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
RESET="\033[0m"

# ----------------------------------------------------------------------------
# Function definitions
# ----------------------------------------------------------------------------

# Display detailed usage information
show_usage() {
    echo -e "${BOLD}NAME${RESET}"
    echo "    visualize_features.sh - Create visualizations for extracted raster features"
    echo
    echo -e "${BOLD}SYNOPSIS${RESET}"
    echo "    $0 -r <raster_file> [OPTIONS]"
    echo
    echo -e "${BOLD}DESCRIPTION${RESET}"
    echo "    This script creates visualizations for features extracted from raster data."
    echo "    It works with the output from the extract_by_category.sh script and visualizes"
    echo "    each feature category separately."
    echo
    echo -e "${BOLD}OPTIONS${RESET}"
    echo "    -r <file>       Path to the original raster file (.asc or .tif) ${RED}[REQUIRED]${RESET}"
    echo "    -f <directory>  Directory containing feature CSV files (default: results)"
    echo "    -o <directory>  Output directory for visualizations (default: visualizations)"
    echo "    -c <categories> Comma-separated list of categories to visualize (e.g., terrain,stats)"
    echo "    -a              Visualize all available feature categories (detects from files)"
    echo "    -3              Include 3D visualizations (requires matplotlib)"
    echo "    -i              Create interactive 3D visualizations (requires plotly)"
    echo "    -p              Compare features across categories (for selected categories only)"
    echo "    -s              Show plots interactively (default: save to files only)"
    echo "    -S <rate>       Sample rate for 3D visualizations (0.0-1.0, default: 0.1)"
    echo "    -v              Verbose output"
    echo "    -h              Display this help message and exit"
    echo
    echo -e "${BOLD}EXAMPLES${RESET}"
    echo "    # Visualize terrain features"
    echo "    $0 -r data/dem.asc -c terrain"
    echo
    echo "    # Visualize multiple categories with verbose output"
    echo "    $0 -r data/dem.asc -c terrain,stats -o output/viz -v"
    echo
    echo "    # Visualize all available categories with 3D plots"
    echo "    $0 -r data/dem.asc -a -3"
    echo
    echo "    # Create interactive 3D visualizations with 5% sampling"
    echo "    $0 -r data/dem.asc -c terrain -i -S 0.05"
    echo
    echo "    # Compare features across categories"
    echo "    $0 -r data/dem.asc -c terrain,stats -p"
    echo
    exit 1
}

# Display error message and exit
show_error() {
    echo -e "${RED}ERROR: $1${RESET}" >&2
    exit 1
}

# Display informational message
info() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}INFO: $1${RESET}"
    fi
}

# Display warning message
warning() {
    echo -e "${YELLOW}WARNING: $1${RESET}" >&2
}

# Display success message
success() {
    echo -e "${GREEN}SUCCESS: $1${RESET}"
}

# Check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Check prerequisites
check_prerequisites() {
    # Check for Python
    if ! command_exists python; then
        show_error "Python is required but not found in PATH"
    fi
    
    # Check for raster_features package
    if ! python -c "import raster_features" &> /dev/null; then
        show_error "raster_features package is not installed or not in PYTHONPATH"
    fi
}

# Parse command line arguments
parse_arguments() {
    while getopts "r:f:o:c:a3ipsS:vh" opt; do
        case $opt in
            r) RASTER_FILE="$OPTARG" ;;
            f) FEATURE_DIR="$OPTARG" ;;
            o) OUTPUT_DIR="$OPTARG" ;;
            c) IFS=',' read -ra CATEGORIES <<< "$OPTARG" ;;
            a) ALL_CATEGORIES=true ;;
            3) VISUALIZE_3D=true ;;
            i) INTERACTIVE=true ;;
            p) COMPARE_FEATURES=true ;;
            s) SHOW_PLOTS=true ;;
            S) SAMPLE_RATE="$OPTARG" ;;
            v) VERBOSE=true ;;
            h) show_usage ;;
            *) show_usage ;;
        esac
    done
}

# Validate input parameters
validate_inputs() {
    # Check if raster file is provided
    if [ -z "$RASTER_FILE" ]; then
        show_error "Raster file is required (-r option)"
    fi

    # Check if raster file exists
    if [ ! -f "$RASTER_FILE" ]; then
        show_error "Raster file not found: $RASTER_FILE"
    fi
    
    # If feature directory doesn't exist, create a warning
    if [ ! -d "$FEATURE_DIR" ]; then
        warning "Feature directory does not exist: $FEATURE_DIR"
        warning "No feature files will be found unless the directory exists"
    fi
    
    # Check for interactive visualization dependencies
    if [ "$INTERACTIVE" = true ]; then
        if ! python -c "import plotly" &> /dev/null; then
            warning "Plotly is required for interactive visualizations but not found"
            warning "Install it with: pip install plotly"
            warning "Continuing without interactive visualizations"
            INTERACTIVE=false
        fi
    fi
    
    # Validate sample rate
    if (( $(echo "$SAMPLE_RATE <= 0.0" | bc -l) )) || (( $(echo "$SAMPLE_RATE > 1.0" | bc -l) )); then
        warning "Sample rate must be between 0.0 and 1.0, setting to default (0.1)"
        SAMPLE_RATE=0.1
    fi
}

# Detect all available feature categories for a given raster file
detect_categories() {
    local base_name="$1"
    local count=0
    
    info "Detecting available feature categories..."
    
    # Find all feature files matching the pattern
    for FILE in "$FEATURE_DIR"/"$base_name"_*_features.csv; do
        if [ -f "$FILE" ]; then
            # Extract category name from filename
            local category
            category=$(basename "$FILE" | sed -E "s/${base_name}_(.+)_features.csv/\1/")
            CATEGORIES+=("$category")
            ((count++))
        fi
    done
    
    if [ $count -eq 0 ]; then
        warning "No feature files found matching pattern: ${base_name}_*_features.csv in $FEATURE_DIR"
        return 1
    else
        info "Found $count feature categories: ${CATEGORIES[*]}"
        return 0
    fi
}

# Process a single feature category
process_category() {
    local category="$1"
    local raster_file="$2"
    local base_name="$3"
    local feature_dir="$4"
    local output_dir="$5"
    
    local feature_csv="${feature_dir}/${base_name}_${category}_features.csv"
    local feature_json="${feature_dir}/${base_name}_${category}_features.json"
    local category_output_dir="${output_dir}/${category}"
    
    # Check if feature CSV exists
    if [ ! -f "$feature_csv" ]; then
        warning "Feature file not found: $feature_csv"
        return 1
    fi
    
    # Create category output directory
    mkdir -p "$category_output_dir"
    
    info "Processing $category features from $feature_csv"
    
    # Handle interactive visualization separately since it uses a different script
    if [ "$INTERACTIVE" = true ]; then
        info "Creating interactive visualizations for $category"
        
        # Check if plotly is installed
        if ! python -c "import plotly" &>/dev/null; then
            warning "Plotly is required for interactive visualizations but not found"
            warning "Install it with: pip install plotly"
            warning "Continuing without interactive visualizations"
        else
            # Create interactive directory
            local interactive_dir="${category_output_dir}/interactive"
            mkdir -p "$interactive_dir"
            
            # Build interactive visualization command
            local interactive_cmd=(
                "python" "-m" "raster_features.utils.visualization_scripts.visualize_interactive"
                "-r" "$raster_file"
                "-f" "$feature_csv"
                "-o" "$interactive_dir"
                "-s" "$SAMPLE_RATE"
                "-z" "1.5"  # Default z-exaggeration factor
            )
            
            # Add open flag if showing plots
            if [ "$SHOW_PLOTS" = true ]; then
                interactive_cmd+=("--open")
            fi
            
            # Run the interactive visualization
            info "Running: ${interactive_cmd[*]}"
            if "${interactive_cmd[@]}"; then
                success "Created interactive visualizations for $category"
            else
                warning "Failed to create interactive visualizations for $category"
            fi
        fi
    fi
    
    # Build static visualization command with appropriate options
    local cmd_args=()
    
    # Use the new visualize-csv command from the integrated CLI
    cmd_args+=("python" "-m" "raster_features.cli" "visualize-csv")
    cmd_args+=("--csv" "$feature_csv")
    cmd_args+=("--output" "$category_output_dir")
    
    # Add 3D visualization if requested
    if [ "$VISUALIZE_3D" = true ]; then
        cmd_args+=("--create-3d")
        cmd_args+=("--sample-rate" "$SAMPLE_RATE")
    fi
    
    # Add show plots option if requested
    if [ "$SHOW_PLOTS" = true ]; then
        cmd_args+=("--show-plots")
    fi
    
    # Add verbose output if requested
    if [ "$VERBOSE" = true ]; then
        cmd_args+=("--verbose")
    fi
    
    # Run the visualization command
    info "Running: ${cmd_args[*]}"
    if "${cmd_args[@]}"; then
        success "Visualized $category features"
        return 0
    else
        warning "Failed to visualize $category features"
        return 1
    fi
}

# Compare features across categories
compare_features() {
    local raster_file="$1"
    local base_name="$2"
    local feature_dir="$3"
    local output_dir="$4"
    local categories=("${@:5}")
    
    if [ ${#categories[@]} -lt 2 ]; then
        warning "Feature comparison requires at least 2 categories"
        return 1
    fi
    
    echo -e "${BOLD}Comparing features across categories...${RESET}"
    
    # Create comparison output directory
    local comparison_output="${output_dir}/comparisons"
    mkdir -p "$comparison_output"
    
    # Prepare feature files and columns for comparison
    local feature_files=()
    local feature_columns=()
    
    for category in "${categories[@]}"; do
        # Construct feature CSV path
        local feature_csv="${feature_dir}/${base_name}_${category}_features.csv"
        
        # Check if feature file exists
        if [ ! -f "$feature_csv" ]; then
            warning "Feature file not found: $feature_csv"
            continue
        fi
        
        feature_files+=("$feature_csv")
        
        # Get a representative feature from each category
        # This is a simple approach - in a real implementation you might want to
        # let the user specify which features to compare
        case "$category" in
            "terrain")
                feature_columns+=("${base_name}_${category}_features:slope")
                feature_columns+=("${base_name}_${category}_features:roughness")
                ;;
            "stats")
                feature_columns+=("${base_name}_${category}_features:mean")
                feature_columns+=("${base_name}_${category}_features:stddev")
                ;;
            "spatial")
                feature_columns+=("${base_name}_${category}_features:moran_i")
                ;;
            "texture")
                feature_columns+=("${base_name}_${category}_features:glcm_contrast")
                ;;
            "spectral")
                feature_columns+=("${base_name}_${category}_features:fft_peak")
                ;;
            "hydrology")
                feature_columns+=("${base_name}_${category}_features:flow_direction")
                ;;
            *)
                # For other categories, try to get the first non-id/x/y/elevation column
                local first_feature=$(python -c "
import pandas as pd
df = pd.read_csv('$feature_csv')
cols = [c for c in df.columns if c not in ['id', 'x', 'y', 'elevation']]
print(cols[0] if cols else '')
")
                if [ -n "$first_feature" ]; then
                    feature_columns+=("${base_name}_${category}_features:$first_feature")
                fi
                ;;
        esac
    done
    
    # Set show_plots flag
    local show_flag=""
    if [ "$SHOW_PLOTS" = true ]; then
        show_flag="--show"
    fi
    
    # Run the comparison script
    if [ ${#feature_files[@]} -ge 2 ] && [ ${#feature_columns[@]} -ge 2 ]; then
        info "Running feature comparison"
        info "Feature files: ${feature_files[*]}"
        info "Feature columns: ${feature_columns[*]}"
        
        python -m raster_features.utils.visualization_scripts.compare_features \
            -r "$raster_file" \
            -f "${feature_files[@]}" \
            -c "${feature_columns[@]}" \
            -o "$comparison_output" \
            $show_flag
        
        if [ $? -eq 0 ]; then
            success "Created feature comparison in $comparison_output"
            return 0
        else
            warning "Failed to create feature comparison"
            return 1
        fi
    else
        warning "Not enough valid features found for comparison"
        return 1
    fi
}

# ----------------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------------

# Parse and validate arguments
parse_arguments "$@"
validate_inputs
check_prerequisites

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get base filename without extension
FILENAME=$(basename "$RASTER_FILE")
BASE_NAME="${FILENAME%.*}"

# If all categories option is set, find all available feature categories
if [ "$ALL_CATEGORIES" = true ]; then
    if ! detect_categories "$BASE_NAME"; then
        show_error "No feature categories found. Make sure feature files exist in $FEATURE_DIR"
    fi
fi

# Check if any categories are specified
if [ ${#CATEGORIES[@]} -eq 0 ]; then
    show_error "No feature categories specified. Use -c option or -a to detect automatically."
fi

# Display configuration
echo -e "${BOLD}=== Raster Feature Visualization ===${RESET}"
echo "Raster file:       $RASTER_FILE"
echo "Feature directory: $FEATURE_DIR"
echo "Output directory:  $OUTPUT_DIR"
echo "Feature categories: ${CATEGORIES[*]}"
echo "3D visualization: $VISUALIZE_3D"
echo "Interactive visualization: $INTERACTIVE"
echo "Feature comparison: $COMPARE_FEATURES"
echo "Show plots: $SHOW_PLOTS"
echo "Sample rate: $SAMPLE_RATE"
echo "Verbose: $VERBOSE"
echo -e "${BOLD}===================================${RESET}"

# Process each category
SUCCESS_COUNT=0
TOTAL_CATEGORIES=${#CATEGORIES[@]}

for CATEGORY in "${CATEGORIES[@]}"; do
    if process_category "$CATEGORY" "$RASTER_FILE" "$BASE_NAME" "$FEATURE_DIR" "$OUTPUT_DIR"; then
        ((SUCCESS_COUNT++))
    fi
done

# Run feature comparison if requested
if [ "$COMPARE_FEATURES" = true ] && [ $TOTAL_CATEGORIES -ge 2 ]; then
    compare_features "$RASTER_FILE" "$BASE_NAME" "$FEATURE_DIR" "$OUTPUT_DIR" "${CATEGORIES[@]}"
fi

# Display completion message
echo
if [ $SUCCESS_COUNT -eq $TOTAL_CATEGORIES ]; then
    success "All $SUCCESS_COUNT feature categories visualized successfully"
else
    warning "$SUCCESS_COUNT of $TOTAL_CATEGORIES feature categories visualized successfully"
fi
echo "Results saved to: $OUTPUT_DIR"