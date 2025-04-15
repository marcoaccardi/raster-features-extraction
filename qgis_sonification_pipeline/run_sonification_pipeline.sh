#!/bin/bash
#
# QGIS Sonification Pipeline Runner
# ---------------------------------
# This script runs the complete QGIS sonification pipeline on ASC DEM files.
# It automatically derives output names from input ASC files and handles errors.
#
# Usage:
#   ./run_sonification_pipeline.sh [options] <input_asc_file>
#
# Options:
#   -h, --help             Show this help message
#   -o, --output-dir DIR   Output directory (default: ./output)
#   -e, --epsg EPSG        EPSG code (default: EPSG:32616)
#   -d, --direction DIR    Direction for temporal simulation (left_to_right, top_to_bottom, diagonal)
#   -p, --points NUM       Number of points for temporal simulation (default: 100)
#   -w, --window NUM       Window size for rolling statistics (default: 5)
#   -c, --clean            Clean output directory before processing
#   -v, --verbose          Enable verbose output
#   -s, --skip-stage NUM   Skip specified stage (0-5)
#   -q, --qgis             Use QGIS functionality when available

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="${SCRIPT_DIR}"

# Default values
OUTPUT_DIR="${SCRIPT_DIR}/output"
EPSG="EPSG:32616"
DIRECTION="left_to_right"
NUM_POINTS=100
WINDOW_SIZE=5
VERBOSE=0
CLEAN=0
SKIP_STAGES=()
CONDA_ENV="raster_features"
USE_QGIS=0
QGIS_PATH="/Applications/QGIS-LTR.app/Contents/Resources"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print usage
print_usage() {
    echo "Usage: $0 [options] <input_asc_file>"
    echo ""
    echo "Options:"
    echo "  -h, --help             Show this help message"
    echo "  -o, --output-dir DIR   Output directory (default: ${OUTPUT_DIR})"
    echo "  -e, --epsg EPSG        EPSG code (default: ${EPSG})"
    echo "  -d, --direction DIR    Direction for temporal simulation (${DIRECTION}, top_to_bottom, diagonal)"
    echo "  -p, --points NUM       Number of points for temporal simulation (default: ${NUM_POINTS})"
    echo "  -w, --window NUM       Window size for rolling statistics (default: ${WINDOW_SIZE})"
    echo "  -c, --clean            Clean output directory before processing"
    echo "  -v, --verbose          Enable verbose output"
    echo "  -s, --skip-stage NUM   Skip specified stage (0-5)"
    echo "  -q, --qgis             Use QGIS functionality when available"
    echo ""
    echo "Example:"
    echo "  $0 -o ./my_results -d diagonal dataset/S0603-M3-Rose_Garden-UTM16N-1m.asc"
}

# Function to log messages
log() {
    local level=$1
    local message=$2
    local color=$NC
    
    case $level in
        "INFO")
            color=$GREEN
            ;;
        "WARNING")
            color=$YELLOW
            ;;
        "ERROR")
            color=$RED
            ;;
        "DEBUG")
            color=$BLUE
            ;;
    esac
    
    if [[ $level == "DEBUG" && $VERBOSE -eq 0 ]]; then
        return
    fi
    
    echo -e "${color}[$(date +'%Y-%m-%d %H:%M:%S')] [${level}] ${message}${NC}"
}

# Function to check dependencies
check_dependencies() {
    log "INFO" "Checking dependencies..."
    
    # Check Python
    if ! command -v python &> /dev/null; then
        log "ERROR" "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Check GDAL
    if ! python -c "from osgeo import gdal" &> /dev/null; then
        log "WARNING" "GDAL Python bindings not found. Some features may not work."
    fi
    
    # Check if pipeline directory exists
    if [ ! -d "$PIPELINE_DIR" ]; then
        log "ERROR" "Pipeline directory not found: $PIPELINE_DIR"
        exit 1
    fi
    
    # Check if scripts exist
    for i in {0..5}; do
        script_path="${PIPELINE_DIR}/scripts/0${i}_*.py"
        if [ ! -f $(ls $script_path 2>/dev/null | head -1) ] && [[ ! " ${SKIP_STAGES[@]} " =~ " ${i} " ]]; then
            log "ERROR" "Script not found for stage ${i}"
            exit 1
        fi
    done
    
    log "INFO" "All dependencies checked."
}

# Function to clean output directory
clean_output_dir() {
    if [ -d "$1" ]; then
        log "INFO" "Cleaning output directory: $1"
        rm -rf "$1"/*
    fi
    mkdir -p "$1"
}

# Function to run a pipeline stage
run_stage() {
    local stage=$1
    local input_file=$2
    local output_dir=$3
    local base_name=$4
    local stage_name=""
    local cmd=""
    local script_path=""
    
    # Check if stage should be skipped
    if [[ " ${SKIP_STAGES[@]} " =~ " ${stage} " ]]; then
        log "INFO" "Skipping stage ${stage}"
        return 0
    fi
    
    # Create wrapper script to handle environment when using QGIS
    if [ $USE_QGIS -eq 1 ] && [ -d "$QGIS_PATH" ]; then
        TMP_SCRIPT=$(mktemp)
        cat > "$TMP_SCRIPT" << 'EOL'
#!/bin/bash
# Get QGIS path from first argument
QGIS_PATH="$1"
shift

# Set QGIS environment variables properly
export PYTHONPATH="${QGIS_PATH}/python:${PYTHONPATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${QGIS_PATH}/plugins"
export QGIS_PREFIX_PATH="${QGIS_PATH}"
export USE_QGIS=1

# Debug information
echo "QGIS environment setup:"
echo "PYTHONPATH: $PYTHONPATH"
echo "QGIS_PREFIX_PATH: $QGIS_PREFIX_PATH"

# Run the actual Python script with remaining arguments
python "$@"
EOL
        chmod +x "$TMP_SCRIPT"
        script_path="$TMP_SCRIPT ${QGIS_PATH}"
    else
        script_path="python"
    fi
    
    case $stage in
        0)
            stage_name="Input Preparation"
            output_file="${output_dir}/${base_name}.tif"
            if [ $USE_QGIS -eq 1 ] && [ -d "$QGIS_PATH" ]; then
                cmd="conda run -n ${CONDA_ENV} bash $script_path ${PIPELINE_DIR}/scripts/00_project_input.py --input \"${input_file}\" --output \"${output_file}\" --epsg \"${EPSG}\""
            else
                cmd="conda run -n ${CONDA_ENV} python ${PIPELINE_DIR}/scripts/00_project_input.py --input \"${input_file}\" --output \"${output_file}\" --epsg \"${EPSG}\""
            fi
            ;;
        1)
            stage_name="Feature Extraction"
            if [ $USE_QGIS -eq 1 ] && [ -d "$QGIS_PATH" ]; then
                cmd="conda run -n ${CONDA_ENV} bash $script_path ${PIPELINE_DIR}/scripts/01_extract_features.py --input \"${output_dir}/${base_name}.tif\" --output_dir \"${output_dir}\""
            else
                cmd="conda run -n ${CONDA_ENV} python ${PIPELINE_DIR}/scripts/01_extract_features.py --input \"${output_dir}/${base_name}.tif\" --output_dir \"${output_dir}\""
            fi
            ;;
        2)
            stage_name="Zonal Statistics"
            if [ $USE_QGIS -eq 1 ] && [ -d "$QGIS_PATH" ]; then
                cmd="conda run -n ${CONDA_ENV} bash $script_path ${PIPELINE_DIR}/scripts/02_zonal_statistics.py --input_dir \"${output_dir}\" --output_dir \"${output_dir}\""
            else
                cmd="conda run -n ${CONDA_ENV} python ${PIPELINE_DIR}/scripts/02_zonal_statistics.py --input_dir \"${output_dir}\" --output_dir \"${output_dir}\""
            fi
            ;;
        3)
            stage_name="Feature Masking"
            if [ $USE_QGIS -eq 1 ] && [ -d "$QGIS_PATH" ]; then
                cmd="conda run -n ${CONDA_ENV} bash $script_path ${PIPELINE_DIR}/scripts/03_feature_masking.py --input_dir \"${output_dir}\" --output_dir \"${output_dir}\""
            else
                cmd="conda run -n ${CONDA_ENV} python ${PIPELINE_DIR}/scripts/03_feature_masking.py --input_dir \"${output_dir}\" --output_dir \"${output_dir}\""
            fi
            ;;
        4)
            stage_name="Polygonize Masks"
            if [ $USE_QGIS -eq 1 ] && [ -d "$QGIS_PATH" ]; then
                cmd="conda run -n ${CONDA_ENV} bash $script_path ${PIPELINE_DIR}/scripts/04_polygonize_masks.py --input_dir \"${output_dir}\" --output_dir \"${output_dir}\" --extract_centroids"
            else
                cmd="conda run -n ${CONDA_ENV} python ${PIPELINE_DIR}/scripts/04_polygonize_masks.py --input_dir \"${output_dir}\" --output_dir \"${output_dir}\" --extract_centroids"
            fi
            ;;
        5)
            stage_name="Temporal Simulation"
            if [ $USE_QGIS -eq 1 ] && [ -d "$QGIS_PATH" ]; then
                cmd="conda run -n ${CONDA_ENV} bash $script_path ${PIPELINE_DIR}/scripts/05_temporal_simulation.py --input_dir \"${output_dir}\" --output_dir \"${output_dir}\" --direction \"${DIRECTION}\" --num_points ${NUM_POINTS} --window_size ${WINDOW_SIZE}"
            else
                cmd="conda run -n ${CONDA_ENV} python ${PIPELINE_DIR}/scripts/05_temporal_simulation.py --input_dir \"${output_dir}\" --output_dir \"${output_dir}\" --direction \"${DIRECTION}\" --num_points ${NUM_POINTS} --window_size ${WINDOW_SIZE}"
            fi
            ;;
        *)
            log "ERROR" "Unknown stage: ${stage}"
            return 1
            ;;
    esac
    
    log "INFO" "Running Stage ${stage}: ${stage_name}"
    log "DEBUG" "Command: ${cmd}"
    
    # Run the command
    if [ $VERBOSE -eq 1 ]; then
        eval $cmd
    else
        eval $cmd > /dev/null 2>&1 || {
            log "ERROR" "Stage ${stage} failed. Run with -v for details."
            return 1
        }
    fi
    
    log "INFO" "Stage ${stage} completed successfully."
    return 0
}

# Function to setup QGIS environment
setup_qgis_env() {
    if [ $USE_QGIS -eq 1 ]; then
        if [ -d "$QGIS_PATH" ]; then
            log "INFO" "Setting up QGIS environment from: $QGIS_PATH"
            export PYTHONPATH="$QGIS_PATH/python:$PYTHONPATH"
            export QT_QPA_PLATFORM_PLUGIN_PATH="$QGIS_PATH/plugins"
            export QGIS_PREFIX_PATH="$QGIS_PATH"
            return 0
        else
            log "WARNING" "QGIS installation not found at: $QGIS_PATH"
            return 1
        fi
    fi
    return 0
}

# Parse command line arguments
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            print_usage
            exit 0
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        -e|--epsg)
            EPSG="$2"
            shift
            shift
            ;;
        -d|--direction)
            DIRECTION="$2"
            shift
            shift
            ;;
        -p|--points)
            NUM_POINTS="$2"
            shift
            shift
            ;;
        -w|--window)
            WINDOW_SIZE="$2"
            shift
            shift
            ;;
        -c|--clean)
            CLEAN=1
            shift
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -s|--skip-stage)
            SKIP_STAGES+=("$2")
            shift
            shift
            ;;
        -q|--qgis)
            USE_QGIS=1
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

# Check if input file is provided
if [ $# -eq 0 ]; then
    log "ERROR" "No input ASC file provided"
    print_usage
    exit 1
fi

INPUT_FILE="$1"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    log "ERROR" "Input file not found: $INPUT_FILE"
    exit 1
fi

# Check if input file is an ASC file
if [[ ! "$INPUT_FILE" =~ \.asc$ ]]; then
    log "ERROR" "Input file is not an ASC file: $INPUT_FILE"
    exit 1
fi

# Get base name for output files
BASE_NAME=$(basename "$INPUT_FILE" .asc)
OUTPUT_SUBDIR="${OUTPUT_DIR}/${BASE_NAME}"

# Create output directory
mkdir -p "$OUTPUT_SUBDIR"

# Clean output directory if requested
if [ $CLEAN -eq 1 ]; then
    clean_output_dir "$OUTPUT_SUBDIR"
fi

# Setup QGIS environment
setup_qgis_env

# Check dependencies
check_dependencies

# Start processing
log "INFO" "Starting QGIS sonification pipeline for $INPUT_FILE"
log "INFO" "Output directory: $OUTPUT_SUBDIR"

# Create a timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_SUBDIR}/pipeline_${TIMESTAMP}.log"

# Log configuration
{
    log "INFO" "Pipeline configuration:"
    log "INFO" "  Input file: $INPUT_FILE"
    log "INFO" "  Output directory: $OUTPUT_SUBDIR"
    log "INFO" "  EPSG code: $EPSG"
    log "INFO" "  Direction: $DIRECTION"
    log "INFO" "  Number of points: $NUM_POINTS"
    log "INFO" "  Window size: $WINDOW_SIZE"
    log "INFO" "  Skip stages: ${SKIP_STAGES[*]:-None}"
    log "INFO" "  Use QGIS: $USE_QGIS"
} | tee -a "$LOG_FILE"

# Run each stage
start_time=$(date +%s)

for stage in {0..5}; do
    stage_start=$(date +%s)
    run_stage $stage "$INPUT_FILE" "$OUTPUT_SUBDIR" "$BASE_NAME" 2>&1 | tee -a "$LOG_FILE"
    stage_status=${PIPESTATUS[0]}
    stage_end=$(date +%s)
    stage_duration=$((stage_end - stage_start))
    
    if [ $stage_status -ne 0 ]; then
        log "ERROR" "Pipeline failed at stage $stage" | tee -a "$LOG_FILE"
        exit 1
    fi
    
    log "INFO" "Stage $stage completed in ${stage_duration}s" | tee -a "$LOG_FILE"
done

end_time=$(date +%s)
duration=$((end_time - start_time))

log "INFO" "Pipeline completed successfully in ${duration}s" | tee -a "$LOG_FILE"
log "INFO" "Results saved to: $OUTPUT_SUBDIR"
log "INFO" "Log file: $LOG_FILE"

# Create a summary file
SUMMARY_FILE="${OUTPUT_SUBDIR}/summary.txt"
{
    echo "QGIS Sonification Pipeline Summary"
    echo "=================================="
    echo ""
    echo "Input file: $INPUT_FILE"
    echo "Processing date: $(date)"
    echo "Processing time: ${duration}s"
    echo ""
    echo "Output directories:"
    find "$OUTPUT_SUBDIR" -type d -mindepth 1 | sort | sed 's/^/- /'
    echo ""
    echo "Generated files:"
    find "$OUTPUT_SUBDIR" -type f -name "*.tif" -o -name "*.shp" -o -name "*.csv" -o -name "*.json" -o -name "*.geojson" | sort | sed 's/^/- /'
} > "$SUMMARY_FILE"

log "INFO" "Summary saved to: $SUMMARY_FILE"
exit 0
