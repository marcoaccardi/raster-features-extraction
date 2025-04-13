#!/bin/bash
# batch_visualize.sh - Visualize features from multiple raster files
# 
# This script processes multiple ASC files and generates visualizations
# for specified feature categories for each one.
#
# Author: Cascade AI
# Date: April 13, 2025

# Set bash to exit on error and prevent unset variable usage
set -e
set -o nounset

# Default values
RASTER_DIR="dataset/"
OUTPUT_DIR="visualizations/"
FEATURE_DIR="results/"
CATEGORIES="terrain,stats"
VERBOSE=false
SHOW_3D=false
INTERACTIVE=false

# Text formatting
BOLD="\033[1m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
RESET="\033[0m"

# Display usage
show_usage() {
    echo -e "${BOLD}NAME${RESET}"
    echo "    batch_visualize.sh - Process multiple raster files for visualization"
    echo
    echo -e "${BOLD}SYNOPSIS${RESET}"
    echo "    $0 [OPTIONS]"
    echo
    echo -e "${BOLD}DESCRIPTION${RESET}"
    echo "    This script processes all ASC files in a specified directory and"
    echo "    generates feature visualizations for each one."
    echo
    echo -e "${BOLD}OPTIONS${RESET}"
    echo "    -d <directory>  Directory containing raster files (default: dataset/)"
    echo "    -f <directory>  Directory containing feature files (default: results/)"
    echo "    -o <directory>  Output directory for visualizations (default: visualizations/)"
    echo "    -c <categories> Comma-separated list of categories to visualize (default: terrain,stats)"
    echo "    -3              Include 3D visualization tips"
    echo "    -s              Show plots interactively (default: save to files only)"
    echo "    -v              Verbose output"
    echo "    -h              Display this help message"
    echo
    exit 1
}

# Parse command line arguments
while getopts "d:f:o:c:3svh" opt; do
    case $opt in
        d) RASTER_DIR="$OPTARG" ;;
        f) FEATURE_DIR="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        c) CATEGORIES="$OPTARG" ;;
        3) SHOW_3D=true ;;
        s) INTERACTIVE=true ;;
        v) VERBOSE=true ;;
        h) show_usage ;;
        *) show_usage ;;
    esac
done

# Validate inputs
if [ ! -d "$RASTER_DIR" ]; then
    echo -e "${RED}Error: Raster directory '$RASTER_DIR' not found${RESET}" >&2
    exit 1
fi

if [ ! -d "$FEATURE_DIR" ]; then
    echo -e "${YELLOW}Warning: Feature directory '$FEATURE_DIR' not found${RESET}" >&2
    echo "No feature files will be found unless the directory exists"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set up visualization options
VIZ_OPTIONS=""
if [ "$SHOW_3D" = true ]; then
    VIZ_OPTIONS="$VIZ_OPTIONS -3"
fi
if [ "$INTERACTIVE" = true ]; then
    VIZ_OPTIONS="$VIZ_OPTIONS -s"
fi
if [ "$VERBOSE" = true ]; then
    VIZ_OPTIONS="$VIZ_OPTIONS -v"
fi

# Count how many files we'll process
FILE_COUNT=$(find "$RASTER_DIR" -name "*.asc" -type f | wc -l)
if [ "$FILE_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: No ASC files found in '$RASTER_DIR'${RESET}" >&2
    exit 1
fi

echo -e "${BOLD}=== Batch Visualization ===${RESET}"
echo "Raster directory: $RASTER_DIR"
echo "Feature directory: $FEATURE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Feature categories: $CATEGORIES"
echo "Files to process: $FILE_COUNT"
echo -e "${BOLD}==========================${RESET}"

# Process each ASC file
PROCESSED=0
SUCCESS=0

for RASTER_FILE in "$RASTER_DIR"/*.asc; do
    if [ -f "$RASTER_FILE" ]; then
        ((PROCESSED++))
        echo -e "${BLUE}[$PROCESSED/$FILE_COUNT] Processing:${RESET} $RASTER_FILE"
        
        # Create output directory based on raster filename
        BASENAME=$(basename "$RASTER_FILE" .asc)
        RASTER_OUTPUT="$OUTPUT_DIR/$BASENAME"
        mkdir -p "$RASTER_OUTPUT"
        
        # Run visualization
        if ./scripts/visualize_features.sh -r "$RASTER_FILE" -f "$FEATURE_DIR" -o "$RASTER_OUTPUT" -c "$CATEGORIES" $VIZ_OPTIONS; then
            echo -e "${GREEN}Successfully processed:${RESET} $BASENAME"
            ((SUCCESS++))
        else
            echo -e "${RED}Failed to process:${RESET} $BASENAME"
        fi
        
        echo "----------------------------------------"
    fi
done

# Display summary
echo -e "\n${BOLD}=== Batch Visualization Summary ===${RESET}"
echo "Total files processed: $PROCESSED"
echo "Successfully completed: $SUCCESS"
echo "Failed: $((PROCESSED - SUCCESS))"
echo "Results saved to: $OUTPUT_DIR"
echo -e "${BOLD}==================================${RESET}"

if [ "$SUCCESS" -eq "$PROCESSED" ]; then
    echo -e "${GREEN}All files processed successfully.${RESET}"
    exit 0
else
    echo -e "${YELLOW}Some files failed processing. Check the output for details.${RESET}"
    exit 1
fi
