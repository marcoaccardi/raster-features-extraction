#!/bin/bash
# =============================================================================
# run_batch_folder.sh - Process all raster files in a folder
# =============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

# Default values
INPUT_DIR=""
OUTPUT_DIR="batch_output"
FEATURES="all"
WINDOW_SIZE=5
PARALLEL=false
METADATA=false
VISUALIZE=false
LOG_LEVEL="INFO"
PATTERN="*.asc"
MAX_PARALLEL=4
TIMEOUT=0  # No timeout by default

# Help message
function show_help {
    echo "Usage: $0 [options] -i <input_directory>"
    echo ""
    echo "Description:"
    echo "  Process all raster files in a directory and save results to specified output."
    echo "  This script can be used for batch processing from cron jobs or SLURM."
    echo ""
    echo "Options:"
    echo "  -i, --input DIR        Input directory containing raster files [required]"
    echo "  -o, --output DIR       Output directory [default: batch_output]"
    echo "  -f, --features LIST    Comma-separated list of feature groups to extract:"
    echo "                         terrain,stats,spatial,texture,spectral,hydro,ml or 'all'"
    echo "                         [default: all]"
    echo "  -w, --window-size N    Window size for windowed features [default: 5]"
    echo "  -p, --parallel         Enable parallel processing for each feature extraction"
    echo "  -m, --metadata         Generate metadata"
    echo "  -v, --visualize        Generate visualizations"
    echo "  -l, --log-level LEVEL  Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    echo "                         [default: INFO]"
    echo "  -g, --glob PATTERN     File pattern to match [default: *.asc]"
    echo "  -j, --jobs N           Maximum number of parallel jobs [default: 4]"
    echo "  -t, --timeout N        Timeout in seconds for each file (0 for no timeout) [default: 0]"
    echo "  -h, --help             Show this help message and exit"
    echo ""
    echo "Example:"
    echo "  $0 -i raster_folder -o results -f terrain,hydro -j 8"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -f|--features)
            FEATURES="$2"
            shift 2
            ;;
        -w|--window-size)
            WINDOW_SIZE="$2"
            shift 2
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -m|--metadata)
            METADATA=true
            shift
            ;;
        -v|--visualize)
            VISUALIZE=true
            shift
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -g|--glob)
            PATTERN="$2"
            shift 2
            ;;
        -j|--jobs)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Check if input directory is provided
if [ -z "$INPUT_DIR" ]; then
    echo "Error: Input directory is required."
    show_help
fi

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create a summary log file
SUMMARY_LOG="$OUTPUT_DIR/batch_summary_$(date +%Y%m%d_%H%M%S).log"
echo "Batch processing started at $(date)" > "$SUMMARY_LOG"
echo "Input directory: $INPUT_DIR" >> "$SUMMARY_LOG"
echo "Pattern: $PATTERN" >> "$SUMMARY_LOG"
echo "Features: $FEATURES" >> "$SUMMARY_LOG"
echo "Window size: $WINDOW_SIZE" >> "$SUMMARY_LOG"
echo "Max parallel jobs: $MAX_PARALLEL" >> "$SUMMARY_LOG"
echo "----------------------------------------" >> "$SUMMARY_LOG"

# Function to process a single file
process_file() {
    local input_file="$1"
    local basename=$(basename "$input_file" .asc)
    local output_file="$OUTPUT_DIR/${basename}_features.csv"
    local log_file="$OUTPUT_DIR/${basename}_extraction.log"
    
    echo "[$(date +%H:%M:%S)] Processing: $input_file"
    
    # Build command
    local cmd="python -m raster_features.cli --input \"$input_file\" --output \"$output_file\" --features \"$FEATURES\" --window-size $WINDOW_SIZE --log-level $LOG_LEVEL --log-file \"$log_file\""
    
    # Add optional flags
    if [ "$PARALLEL" = true ]; then
        cmd="$cmd --parallel"
    fi
    
    if [ "$METADATA" = true ]; then
        cmd="$cmd --metadata"
    fi
    
    if [ "$VISUALIZE" = true ]; then
        cmd="$cmd --visualize"
    fi
    
    # Execute command with optional timeout
    local start_time=$(date +%s)
    if [ "$TIMEOUT" -gt 0 ]; then
        timeout "$TIMEOUT" bash -c "$cmd" > /dev/null 2>&1
        local status=$?
        if [ $status -eq 124 ]; then
            echo "[$(date +%H:%M:%S)] TIMEOUT: $input_file" | tee -a "$SUMMARY_LOG"
            return 1
        elif [ $status -ne 0 ]; then
            echo "[$(date +%H:%M:%S)] FAILED: $input_file" | tee -a "$SUMMARY_LOG"
            return 1
        fi
    else
        eval "$cmd" > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "[$(date +%H:%M:%S)] FAILED: $input_file" | tee -a "$SUMMARY_LOG"
            return 1
        fi
    fi
    
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    echo "[$(date +%H:%M:%S)] COMPLETED: $input_file (${elapsed}s)" | tee -a "$SUMMARY_LOG"
    return 0
}

# Export the function so it can be used by xargs
export -f process_file
export INPUT_DIR OUTPUT_DIR FEATURES WINDOW_SIZE PARALLEL METADATA VISUALIZE LOG_LEVEL SUMMARY_LOG TIMEOUT

# Find all raster files and process them
echo "Finding raster files matching pattern: $PATTERN"
file_count=$(find "$INPUT_DIR" -name "$PATTERN" | wc -l)
echo "Found $file_count files to process."

if [ "$file_count" -eq 0 ]; then
    echo "No files found matching pattern: $PATTERN"
    exit 1
fi

# Process files in parallel using xargs
find "$INPUT_DIR" -name "$PATTERN" | xargs -P "$MAX_PARALLEL" -I{} bash -c 'process_file "{}"'

# Generate final summary
success_count=$(grep "COMPLETED" "$SUMMARY_LOG" | wc -l)
failed_count=$(grep "FAILED" "$SUMMARY_LOG" | wc -l)
timeout_count=$(grep "TIMEOUT" "$SUMMARY_LOG" | wc -l)

echo "----------------------------------------" >> "$SUMMARY_LOG"
echo "Batch processing completed at $(date)" >> "$SUMMARY_LOG"
echo "Total files: $file_count" >> "$SUMMARY_LOG"
echo "Successful: $success_count" >> "$SUMMARY_LOG"
echo "Failed: $failed_count" >> "$SUMMARY_LOG"
echo "Timeout: $timeout_count" >> "$SUMMARY_LOG"

# Print summary to console
echo ""
echo "Batch processing completed."
echo "Summary: $SUMMARY_LOG"
echo "Total files: $file_count"
echo "Successful: $success_count"
echo "Failed: $failed_count"
echo "Timeout: $timeout_count"
