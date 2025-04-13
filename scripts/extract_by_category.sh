#!/bin/bash
# Script to extract features by category, creating separate files for each feature type
# This helps manage memory usage and provides more organized results

# Default values
INPUT_FILE=""
OUTPUT_DIR="results"
WINDOW_SIZE=5
LOG_LEVEL="INFO"
SAVE_METADATA=true
SELECTED_CATEGORIES=""
MAX_MEMORY="8G"

# Feature categories
FEATURE_CATEGORIES=("terrain" "stats" "spatial" "texture" "spectral" "hydrology" "ml")

# Function to display usage information
usage() {
    echo "Usage: $0 -i <input_file> [-o <output_dir>] [-w <window_size>] [-l <log_level>] [-c <categories>] [-m <max_memory>]"
    echo ""
    echo "Options:"
    echo "  -i  Input raster file (.asc)"
    echo "  -o  Output directory (default: results)"
    echo "  -w  Window size for neighborhood operations (default: 5)"
    echo "  -l  Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)"
    echo "  -c  Specific categories to extract, comma-separated (default: all)"
    echo "      Available categories: terrain,stats,spatial,texture,spectral,hydrology,ml"
    echo "  -m  Maximum memory to use in extraction (default: 8G)"
    echo "  -h  Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 -i dataset/dem.asc -o results -w 7 -c terrain,stats,hydrology -m 4G"
    exit 1
}

# Parse command line arguments
while getopts "i:o:w:l:c:m:h" opt; do
    case $opt in
        i) INPUT_FILE="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        w) WINDOW_SIZE="$OPTARG" ;;
        l) LOG_LEVEL="$OPTARG" ;;
        c) SELECTED_CATEGORIES="$OPTARG" ;;
        m) MAX_MEMORY="$OPTARG" ;;
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
if [ "$IS_LARGE_FILE" = true ] && [[ "$FILENAME" == *"50cm"* ]] && [ "$WINDOW_SIZE" -eq 5 ]; then
    WINDOW_SIZE=9
    echo "Automatically adjusting window size to $WINDOW_SIZE for 50cm resolution data"
fi

# Process each feature category
for category in "${FEATURE_CATEGORIES[@]}"; do
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
                # For texture features, reduce complexity
                EXTRA_ARGS="--config texture.glcm_distances=[1] texture.glcm_angles=[0,1.5708] texture.calculate_lbp=false texture.calculate_keypoints=false"
                ;;
            "ml")
                # For ML features, disable autoencoder
                EXTRA_ARGS="--config ml.calculate_autoencoder=false"
                ;;
        esac
    fi
    
    # Build command with memory limit
    CMD="python -m raster_features.cli --input \"$INPUT_FILE\" --output \"$OUTPUT_FILE\" --features \"$category\" --window-size $WINDOW_SIZE --log-level $LOG_LEVEL $EXTRA_ARGS"
    
    # Add metadata flag if enabled
    if [ "$SAVE_METADATA" = true ]; then
        CMD="$CMD --save-metadata"
    fi
    
    # Display and execute command with memory limit
    echo "Running: $CMD"
    
    # Use ulimit to prevent memory issues
    if [ "$IS_LARGE_FILE" = true ]; then
        # Run with memory limit for large files
        /bin/bash -c "ulimit -v $(echo $MAX_MEMORY | sed 's/G/*1024*1024/;s/M/*1024/;s/K//') && $CMD"
    else
        # Run normally for small files
        eval $CMD
    fi
    
    # Check if extraction was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully extracted $category features"
        echo "  Output: $OUTPUT_FILE"
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
    
    echo ""
    
    # Sleep briefly to allow system to recover
    sleep 2
done

echo "Feature extraction by category completed."
echo "Results saved in: $OUTPUT_DIR"
