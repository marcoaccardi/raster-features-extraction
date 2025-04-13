#!/bin/bash
# =============================================================================
# run_tests.sh - Run test suite for raster features extraction
# =============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

# Default values
TEST_DIR="tests"
VERBOSE=false
COVERAGE=false
PYTEST_ARGS=""
TEST_SYNTHETIC=false
OUTPUT_DIR="test_results"
TEST_PATTERN="test_*.py"

# Help message
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Description:"
    echo "  Run the test suite for the raster features extraction package."
    echo ""
    echo "Options:"
    echo "  -d, --dir DIR          Test directory [default: tests]"
    echo "  -v, --verbose          Run tests in verbose mode"
    echo "  -c, --coverage         Generate coverage report"
    echo "  -s, --synthetic        Run tests with synthetic data generation"
    echo "  -o, --output DIR       Output directory for test results [default: test_results]"
    echo "  -p, --pattern PATTERN  Test file pattern [default: test_*.py]"
    echo "  -a, --args ARGS        Additional arguments to pass to pytest"
    echo "  -h, --help             Show this help message and exit"
    echo ""
    echo "Example:"
    echo "  $0 -v -c -s -o test_results_$(date +%Y%m%d)"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -d|--dir)
            TEST_DIR="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -s|--synthetic)
            TEST_SYNTHETIC=true
            shift
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -p|--pattern)
            TEST_PATTERN="$2"
            shift 2
            ;;
        -a|--args)
            PYTEST_ARGS="$2"
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

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Setup environment variables for tests
export RASTER_FEATURES_TEST_DIR="$TEST_DIR"
export RASTER_FEATURES_TEST_OUTPUT="$OUTPUT_DIR"
export RASTER_FEATURES_TEST_SYNTHETIC="$TEST_SYNTHETIC"

# Build command
CMD="python -m pytest"

# Add options
if [ "$VERBOSE" = true ]; then
    CMD="$CMD -v"
fi

if [ "$COVERAGE" = true ]; then
    CMD="$CMD --cov=raster_features --cov-report=term --cov-report=html:$OUTPUT_DIR/coverage"
fi

# Add test directory and pattern
CMD="$CMD $TEST_DIR -k \"$TEST_PATTERN\""

# Add additional arguments
if [ -n "$PYTEST_ARGS" ]; then
    CMD="$CMD $PYTEST_ARGS"
fi

# Add output logging
CMD="$CMD | tee $OUTPUT_DIR/test_results_$(date +%Y%m%d_%H%M%S).log"

# Print command
echo "Running: $CMD"

# Execute command
eval "$CMD"

# Check exit status
if [ $? -eq 0 ]; then
    echo "All tests passed."
    # Generate test summary
    if [ "$COVERAGE" = true ]; then
        echo "Coverage report available at: $OUTPUT_DIR/coverage/index.html"
    fi
else
    echo "Some tests failed. See log for details."
    exit 1
fi

# If synthetic data testing was enabled, run additional validation
if [ "$TEST_SYNTHETIC" = true ]; then
    echo "Running synthetic data validation..."
    python -m raster_features.tests.test_synthetic_validation --output "$OUTPUT_DIR/synthetic_validation"
    if [ $? -eq 0 ]; then
        echo "Synthetic data validation successful."
    else
        echo "Synthetic data validation failed."
        exit 1
    fi
fi

echo "Test run completed successfully."
