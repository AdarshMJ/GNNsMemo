#!/bin/bash

# This script generates SBM graphs with fixed informativeness and varying homophily
# and then runs the analysis to study memorization

# Configuration
INFORMATIVENESS=0.1    # Fixed informativeness level
NUM_GRAPHS=10          # Number of graphs to generate with varying homophily
FEATURE_DIM=100        # Node feature dimension
NOISE_LEVEL=0.1        # Noise level for node features
COMMUNITY_SIZE=250     # Size of each community
MODEL_TYPE="gcn"       # Model type (gcn, gat, graphconv)
HIDDEN_DIM=32          # Hidden dimension for GNN
NUM_LAYERS=2           # Number of layers in GNN
EPOCHS=100             # Training epochs
RANDOM_FEATURES=false  # Whether to use random features (true/false)
DATASET_PATH="data/fixed_info_sbm" # Path for generated datasets
MIN_HOMOPHILY=-0.1     # Minimum homophily value
MAX_HOMOPHILY=0.2     # Maximum homophily value

# Create necessary directories
mkdir -p "$DATASET_PATH"
mkdir -p "results"

echo "=============================================="
echo "Step 1: Generating SBM graphs with fixed informativeness ($INFORMATIVENESS)"
echo "and varying homophily levels"
echo "=============================================="

# Build the feature type parameter
FEATURE_PARAM=""
if [ "$RANDOM_FEATURES" = "true" ]; then
    FEATURE_PARAM="--random_features"
    DATASET_PATH="${DATASET_PATH}_random"
    echo "Using random node features"
else
    echo "Using community-correlated node features"
fi

python generate_fixed_info.py \
    --informativeness $INFORMATIVENESS \
    --num_graphs $NUM_GRAPHS \
    --feature_dim $FEATURE_DIM \
    --noise_level $NOISE_LEVEL \
    --community_size $COMMUNITY_SIZE \
    --min_homophily $MIN_HOMOPHILY \
    --max_homophily $MAX_HOMOPHILY \
    --output_dir $DATASET_PATH \
    $FEATURE_PARAM

echo "=============================================="
echo "Step 2: Running memorization analysis on generated graphs"
echo "=============================================="

python main_sbm.py \
    --model_type $MODEL_TYPE \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --epochs $EPOCHS \
    --fixed_informativeness $INFORMATIVENESS \
    --n_graphs $NUM_GRAPHS \
    --dataset_path $DATASET_PATH

echo "=============================================="
echo "Analysis complete!"
echo "=============================================="