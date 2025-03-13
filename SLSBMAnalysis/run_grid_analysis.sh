#!/bin/bash

# Default values
NUM_HOMOPHILY=10
NUM_INFO=10
FEATURE_DIM=100
COMMUNITY_SIZE=250
DEGREE=10
NOISE_LEVEL=0.1
MODEL_TYPE="gcn"
NUM_LAYERS=3
HIDDEN_DIM=32
EPOCHS=100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_homophily) NUM_HOMOPHILY="$2"; shift ;;
        --num_info) NUM_INFO="$2"; shift ;;
        --feature_dim) FEATURE_DIM="$2"; shift ;;
        --community_size) COMMUNITY_SIZE="$2"; shift ;;
        --degree) DEGREE="$2"; shift ;;
        --noise_level) NOISE_LEVEL="$2"; shift ;;
        --model_type) MODEL_TYPE="$2"; shift ;;
        --num_layers) NUM_LAYERS="$2"; shift ;;
        --hidden_dim) HIDDEN_DIM="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="results/grid_analysis_${TIMESTAMP}"
mkdir -p "$BASE_DIR"

# Log configuration
echo "=== Configuration ===" | tee "$BASE_DIR/config.log"
echo "Timestamp: $TIMESTAMP" | tee -a "$BASE_DIR/config.log"
echo "Number of homophily levels: $NUM_HOMOPHILY" | tee -a "$BASE_DIR/config.log"
echo "Number of informativeness levels: $NUM_INFO" | tee -a "$BASE_DIR/config.log"
echo "Feature dimension: $FEATURE_DIM" | tee -a "$BASE_DIR/config.log"
echo "Community size: $COMMUNITY_SIZE" | tee -a "$BASE_DIR/config.log"
echo "Average degree: $DEGREE" | tee -a "$BASE_DIR/config.log"
echo "Noise level: $NOISE_LEVEL" | tee -a "$BASE_DIR/config.log"
echo "Model type: $MODEL_TYPE" | tee -a "$BASE_DIR/config.log"
echo "Number of layers: $NUM_LAYERS" | tee -a "$BASE_DIR/config.log"
echo "Hidden dimension: $HIDDEN_DIM" | tee -a "$BASE_DIR/config.log"
echo "Number of epochs: $EPOCHS" | tee -a "$BASE_DIR/config.log"

# Function to run analysis for a feature type
run_analysis() {
    local feature_type=$1
    local output_dir="$BASE_DIR/${feature_type}_features"
    
    echo "=== Generating graphs with ${feature_type} features ===" | tee -a "$BASE_DIR/run.log"
    
    # Generate graphs
    python generate_sbm_grid.py \
        --num_homophily "$NUM_HOMOPHILY" \
        --num_info "$NUM_INFO" \
        --feature_dim "$FEATURE_DIM" \
        --community_size "$COMMUNITY_SIZE" \
        --degree "$DEGREE" \
        --noise_level "$NOISE_LEVEL" \
        --output_dir "$output_dir" \
        $([ "$feature_type" = "random" ] && echo "--random_features") \
        2>&1 | tee -a "$BASE_DIR/run.log"
    
    echo "=== Running analysis for ${feature_type} features ===" | tee -a "$BASE_DIR/run.log"
    
    # Run analysis
    python main_sbm_grid.py \
        --data_dir "$output_dir/${feature_type}" \
        --model_type "$MODEL_TYPE" \
        --hidden_dim "$HIDDEN_DIM" \
        --num_layers "$NUM_LAYERS" \
        --epochs "$EPOCHS" \
        --output_dir "$output_dir/analysis" \
        2>&1 | tee -a "$BASE_DIR/run.log"
}

# Run for both feature types
echo "Starting analysis..." | tee "$BASE_DIR/run.log"
echo "Results will be saved in: $BASE_DIR" | tee -a "$BASE_DIR/run.log"

run_analysis "correlated"
run_analysis "random"

echo "=== Analysis complete ===" | tee -a "$BASE_DIR/run.log"
echo "Results are saved in: $BASE_DIR" | tee -a "$BASE_DIR/run.log"
