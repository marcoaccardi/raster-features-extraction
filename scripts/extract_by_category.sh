#!/bin/bash
# Script to extract features by category, creating separate files for each feature type
# This helps manage memory usage and provides more organized results

# Default values
INPUT_FILE=""
OUTPUT_DIR="results"
WINDOW_SIZE=31
LOG_LEVEL="INFO"
SAVE_METADATA=true
SELECTED_CATEGORIES=""
MAX_MEMORY="8G"
USE_FALLBACK=false
SKIP_STATS=false
USE_OPTIMIZED=true

# Feature categories
FEATURE_CATEGORIES=("terrain" "stats" "spatial" "texture" "spectral" "hydrology" "ml")

# Get conda environment
CONDA_ENV="raster_features"
# Get python executable from conda environment
CONDA_PYTHON=$(conda run -n $CONDA_ENV which python)

if [ -z "$CONDA_PYTHON" ]; then
    echo "Could not find Python executable in conda environment: $CONDA_ENV"
    echo "Make sure the environment exists and has Python installed"
    exit 1
fi

echo "Using Python from conda environment $CONDA_ENV: $CONDA_PYTHON"

# Add project root to PYTHONPATH
export PYTHONPATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd):$PYTHONPATH

# Function to display usage information
usage() {
    echo "Usage: $0 -i <input_file> [-o <output_dir>] [-w <window_size>] [-l <log_level>] [-c <categories>] [-m <max_memory>] [-f] [-k] [-p]"
    echo ""
    echo "Options:"
    echo "  -i  Input raster file (.asc)"
    echo "  -o  Output directory (default: results)"
    echo "  -w  Window size for neighborhood operations (default: 31)"
    echo "  -l  Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)"
    echo "  -c  Specific categories to extract, comma-separated (default: all)"
    echo "      Available categories: terrain,stats,spatial,texture,spectral,hydrology,ml"
    echo "  -m  Maximum memory to use in extraction (default: 8G)"
    echo "  -f  Use fallback method when GDAL fails (bypasses GDAL dependency)"
    echo "  -k  Skip adding feature statistics to metadata"
    echo "  -p  Use optimized implementation for spectral feature extraction (faster)"
    echo "  -h  Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 -i dataset/dem.asc -o results -w 7 -c terrain,stats,hydrology -m 4G"
    exit 1
}

# Parse command line arguments
while getopts "i:o:w:l:c:m:fkph" opt; do
    case $opt in
        i) INPUT_FILE="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        w) WINDOW_SIZE="$OPTARG" ;;
        l) LOG_LEVEL="$OPTARG" ;;
        c) SELECTED_CATEGORIES="$OPTARG" ;;
        m) MAX_MEMORY="$OPTARG" ;;
        f) USE_FALLBACK=true ;;
        k) SKIP_STATS=true ;;
        p) USE_OPTIMIZED=true ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Check if input file is provided
if [ -z "$INPUT_FILE" ]; then
    echo "Error: Input file is required"
    usage
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get base filename without extension
FILENAME=$(basename "$INPUT_FILE")
BASE_NAME="${FILENAME%.*}"

# Get file size in MB
FILE_SIZE_KB=$(du -k "$INPUT_FILE" | cut -f1)
FILE_SIZE_MB=$(echo "scale=1; $FILE_SIZE_KB / 1024" | bc)

# Determine if this is a large file (>50MB)
IS_LARGE_FILE=false
if (( $(echo "$FILE_SIZE_MB > 50" | bc -l) )); then
    IS_LARGE_FILE=true
    echo "Large file detected ($FILE_SIZE_MB MB). Using memory-efficient settings."
fi

# If specific categories are requested, use them instead of the default list
if [ ! -z "$SELECTED_CATEGORIES" ]; then
    # Convert comma-separated string to array
    IFS=',' read -ra FEATURE_CATEGORIES <<< "$SELECTED_CATEGORIES"
    echo "Extracting selected categories: ${FEATURE_CATEGORIES[*]}"
fi

# For large files with high resolution (50cm), adjust window size if not explicitly set
if [ "$IS_LARGE_FILE" = true ] && [[ "$FILENAME" == *"50cm"* ]] && [ "$WINDOW_SIZE" -eq 31 ]; then
    WINDOW_SIZE=9
    echo "Automatically adjusting window size to $WINDOW_SIZE for 50cm resolution data"
fi

# Extract spectral features
extract_spectral_features() {
    echo "===== Extracting spectral features ====="
    
    if [ "$USE_OPTIMIZED" = true ]; then
        echo "Using optimized implementation for spectral feature extraction"
        
        # Get absolute path to the optimized spectral script in the features directory
        PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
        SCRIPT_PATH="${PROJECT_ROOT}/raster_features/features/extract_optimized_spectral.py"
        echo "Running optimized spectral extraction: $SCRIPT_PATH"
        
        # Check if GDAL is available using our dedicated script
        echo "Checking for GDAL availability..."
        SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
        GDAL_CHECK_SCRIPT="${SCRIPT_DIR}/check_gdal.py"
        
        # Make the script executable
        chmod +x "$GDAL_CHECK_SCRIPT"
        
        # Directory for spectral results
        output_dir="${OUTPUT_DIR}/spectral"
        mkdir -p "$output_dir"
        
        # Define output file
        output_file="${output_dir}/spectral_features.csv"
        
        # Run the GDAL check script
        $CONDA_PYTHON "$GDAL_CHECK_SCRIPT"
        GDAL_EXIT_CODE=$?
        
        # GDAL available if exit code is 0
        if [ $GDAL_EXIT_CODE -eq 0 ]; then
            GDAL_AVAILABLE=1
            echo "GDAL is available, using optimized implementation with GDAL"
            $CONDA_PYTHON "$SCRIPT_PATH" "$INPUT_FILE" "$output_file" --use-gdal
        else
            GDAL_AVAILABLE=0
            echo "GDAL is not available, using optimized implementation without GDAL"
            $CONDA_PYTHON "$SCRIPT_PATH" "$INPUT_FILE" "$output_file"
        fi
        
        # Check if extraction was successful
        if [ $? -ne 0 ] || [ ! -f "$output_file" ]; then
            echo "✗ Failed to extract spectral features with optimized implementation"
            echo "Falling back to standard spectral extraction"
            # Fall back to standard extraction using the CLI script
            $CONDA_PYTHON -m raster_features.cli --input "$INPUT_FILE" --output "$OUTPUT_DIR" --features spectral --window-size $WINDOW_SIZE --log-level $LOG_LEVEL
        else
            echo "✓ Successfully extracted spectral features with optimized implementation"
            
            # Create a symlink to the spectral features in the expected location for the visualization script
            expected_file="${OUTPUT_DIR}/${BASE_NAME}_spectral_features.csv"
            if [ ! -f "$expected_file" ]; then
                echo "Creating symlink for spectral features at: $expected_file"
                # Use relative paths for symlinks to avoid issues with absolute paths
                cd "$OUTPUT_DIR"
                ln -sf "spectral/spectral_features.csv" "${BASE_NAME}_spectral_features.csv"
                
                # If we have a JSON metadata file, also create a symlink for it
                if [ -f "spectral/spectral_features.json" ]; then
                    ln -sf "spectral/spectral_features.json" "${BASE_NAME}_spectral_features.json"
                    echo "Creating symlink for spectral features JSON at: ${OUTPUT_DIR}/${BASE_NAME}_spectral_features.json"
                fi
                # Go back to original directory
                cd - > /dev/null
            fi
        fi
    else
        # Use standard implementation with the correct command structure
        $CONDA_PYTHON -m raster_features.cli --input "$INPUT_FILE" --output "$OUTPUT_DIR" --features spectral --window-size $WINDOW_SIZE --log-level $LOG_LEVEL
    fi
}

# Process each feature category
for category in "${FEATURE_CATEGORIES[@]}"; do
    if [ "$category" = "spectral" ]; then
        extract_spectral_features
    else
        echo "===== Extracting $category features ====="
        
        # Set output filename for this category
        OUTPUT_FILE="$OUTPUT_DIR/${BASE_NAME}_${category}_features.csv"
        
        # Skip if output file already exists
        if [ -f "$OUTPUT_FILE" ]; then
            echo "Output file already exists. Skipping $category."
            continue
        fi
        
        # Apply category-specific optimizations for large files
        EXTRA_ARGS=""
        if [ "$IS_LARGE_FILE" = true ]; then
            case $category in
                "spatial")
                    # For spatial features, use rook weights instead of queen to save memory
                    EXTRA_ARGS="--config spatial.weights_type=rook spatial.calculate_local=false"
                    ;;
                "texture")
                    # For texture features, reduce entropy calculation which uses more memory
                    EXTRA_ARGS="--config texture.calculate_entropy=false"
                    ;;
                "hydrology")
                    # For hydrology features, calculate only basic flow metrics
                    EXTRA_ARGS="--config hydrology.calculate_advanced_flow=false"
                    ;;
                "ml")
                    # For machine learning features, don't calculate autoencoder features (uses PyTorch)
                    EXTRA_ARGS="--config ml.calculate_autoencoder=false"
                    ;;
            esac
        fi
        
        # Build command
        CMD="$CONDA_PYTHON -m raster_features.cli --input \"$INPUT_FILE\" --output \"$OUTPUT_FILE\" --features \"$category\" --window-size $WINDOW_SIZE --log-level $LOG_LEVEL $EXTRA_ARGS"
        
        # Add metadata flag if enabled
        if [ "$SAVE_METADATA" = true ]; then
            CMD="$CMD --save-metadata"
        fi
        
        # Display and execute command with memory limit
        echo "Running: $CMD"
        
        # Capture output to check for GDAL errors
        if [ "$IS_LARGE_FILE" = true ]; then
            # Run with memory limit for large files
            OUTPUT=$(/bin/bash -c "ulimit -v $(echo $MAX_MEMORY | sed 's/G/*1024*1024/;s/M/*1024/;s/K//') && $CMD" 2>&1)
            EXIT_CODE=$?
        else
            # Run normally for small files
            OUTPUT=$(eval $CMD 2>&1)
            EXIT_CODE=$?
        fi
        
        # Check if extraction was successful
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✓ Successfully extracted $category features"
            echo "  Output: $OUTPUT_FILE"
        else
            # Check for GDAL errors in the output
            if echo "$OUTPUT" | grep -q "No module named '_gdal\|osgeo"; then
                echo "⚠ GDAL error detected when extracting $category features"
                
                if [ "$USE_FALLBACK" = true ]; then
                    echo "  Attempting fallback extraction method..."
                    # Try the main extraction with fallback flag first
                    FALLBACK_CMD="$CONDA_PYTHON -m raster_features.cli --input \"$INPUT_FILE\" --output \"$OUTPUT_DIR\" --features \"$category\" --window-size $WINDOW_SIZE --log-level $LOG_LEVEL"
                    
                    if [ "$SAVE_METADATA" = true ]; then
                        FALLBACK_CMD="$FALLBACK_CMD --save-metadata"
                    fi
                    
                    echo "  Running: $FALLBACK_CMD"
                    FALLBACK_OUTPUT=$(eval $FALLBACK_CMD 2>&1)
                    FALLBACK_EXIT=$?
                    
                    # If that also fails, use our standalone extraction script
                    if [ $FALLBACK_EXIT -ne 0 ]; then
                        echo "  Main fallback also failed, using standalone extraction script..."
                        
                        # Build standalone extraction command
                        STANDALONE_CMD="$CONDA_PYTHON scripts/extract_basic_features.py --input \"$INPUT_FILE\" --output \"$OUTPUT_FILE\" --window-size $WINDOW_SIZE --log-level $LOG_LEVEL --feature-type \"$category\""
                        
                        if [ "$SAVE_METADATA" = true ]; then
                            STANDALONE_CMD="$STANDALONE_CMD --save-metadata"
                        fi
                        
                        echo "  Running: $STANDALONE_CMD"
                        eval $STANDALONE_CMD
                        
                        if [ $? -eq 0 ]; then
                            echo "✓ Successfully extracted basic features for $category using standalone script"
                            echo "  Output: $OUTPUT_FILE"
                        else
                            echo "✗ All fallback extraction methods failed for $category"
                        fi
                    else
                        echo "✓ Successfully extracted $category features using fallback method"
                        echo "  Output: $OUTPUT_FILE"
                    fi
                else
                    echo "✗ Failed to extract $category features due to GDAL dependency issues"
                    echo "  To use a fallback method that doesn't require GDAL, run with the -f flag"
                fi
            else
                echo "✗ Failed to extract $category features"
                
                # Provide helpful error message based on category
                case $category in
                    "spatial")
                        echo "  Spatial features often require more memory. Try again with:"
                        echo "  $0 -i \"$INPUT_FILE\" -o \"$OUTPUT_DIR\" -w $WINDOW_SIZE -c spatial -m 16G"
                        ;;
                    "texture")
                        echo "  Texture features may require additional dependencies."
                        echo "  Make sure scikit-image is properly installed with: pip install scikit-image>=0.19.0"
                        ;;
                    "ml")
                        echo "  ML features require PyTorch for autoencoder functionality."
                        echo "  Install PyTorch or try again with: --config ml.calculate_autoencoder=false"
                        ;;
                esac
            fi
        fi
        
        echo ""
        
        # Sleep briefly to allow system to recover
        sleep 2
    fi
done

echo "Feature extraction by category completed."
echo "Results saved in: $OUTPUT_DIR"

# Add feature statistics to metadata files if not disabled
if [ "$SKIP_STATS" = false ]; then
    echo "Adding feature statistics to metadata files..."
    # Process all JSON files in the output directory
    JSON_FILES=$(find "$OUTPUT_DIR" -name "*.json")
    if [ -z "$JSON_FILES" ]; then
        echo "No JSON metadata files found in $OUTPUT_DIR"
    else
        FOUND=0
        PROCESSED=0
        for json_file in $JSON_FILES; do
            FOUND=$((FOUND+1))
            # Check if feature_stats already exists in the JSON file
            if ! grep -q "feature_stats" "$json_file"; then
                echo "Adding statistics to $json_file"
                # Get the corresponding CSV file
                csv_file="${json_file%.json}.csv"
                if [ -f "$csv_file" ]; then
                    # Use the add_feature_stats module from the utils package
                    $CONDA_PYTHON -m raster_features.utils.add_feature_stats "$csv_file" "$json_file"
                    PROCESSED=$((PROCESSED+1))
                else
                    echo "Warning: CSV file $csv_file not found"
                fi
            else
                echo "Statistics already present in $json_file, skipping"
            fi
        done
        echo "Feature statistics added to $PROCESSED/$FOUND JSON metadata files."
    fi
fi
