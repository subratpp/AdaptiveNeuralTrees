#!/bin/bash

# Script to run ANT training 5 times with different seeds and collect statistics
# Usage: ./train.sh <dataset>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset>"
    echo "Example: $0 protein"
    exit 1
fi

DATASET=$1
NUM_RUNS=5
SEEDS=("seed1" "seed2" "seed3" "seed4" "seed5")

echo "============================================================"
echo "Training Adaptive Neural Trees on $DATASET"
echo "Running $NUM_RUNS different seeds..."
echo "============================================================"
echo ""

# Arrays to store accuracies and training times from each run
soft_train_accs=()
soft_test_accs=()
hard_train_accs=()
hard_test_accs=()
train_times=()

# Run training 5 times
for i in $(seq 0 $((NUM_RUNS - 1))); do
    SEED=${SEEDS[$i]}
    
    echo "[Run $((i+1))/$NUM_RUNS] Training with experiment=$SEED"
    
    # Run training
    python tree.py --dataset "$DATASET" --experiment "$SEED"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed for seed $SEED"
        exit 1
    fi
    
    # Parse performance.txt
    PERF_FILE="./experiments/$DATASET/$SEED/checkpoints/performance.txt"
    
    if [ ! -f "$PERF_FILE" ]; then
        echo "ERROR: Performance file not found: $PERF_FILE"
        exit 1
    fi
    
    # Extract accuracies using grep and sed
    soft_train=$(grep "Train accuracy:" "$PERF_FILE" | head -1 | grep -oP '\d+\.\d+' | head -1)
    soft_test=$(grep "Test  accuracy:" "$PERF_FILE" | head -1 | grep -oP '\d+\.\d+' | head -1)
    hard_train=$(grep "Train accuracy:" "$PERF_FILE" | tail -1 | grep -oP '\d+\.\d+' | head -1)
    hard_test=$(grep "Test  accuracy:" "$PERF_FILE" | tail -1 | grep -oP '\d+\.\d+' | head -1)
    
    # Extract training time (seconds)
    train_time=$(grep "Training Time (seconds):" "$PERF_FILE" | grep -oP '\d+\.\d+')
    
    soft_train_accs+=($soft_train)
    soft_test_accs+=($soft_test)
    hard_train_accs+=($hard_train)
    hard_test_accs+=($hard_test)
    train_times+=($train_time)
    
    echo "  Soft: Train=${soft_train}% Test=${soft_test}%"
    echo "  Hard: Train=${hard_train}% Test=${hard_test}%"
    echo "  Time: ${train_time}s"
    echo ""
done

echo "============================================================"
echo "Results Summary: $DATASET (5 Seeds)"
echo "============================================================"
echo ""

# Function to calculate mean
calc_mean() {
    local sum=0
    local count=0
    for val in "$@"; do
        sum=$(echo "$sum + $val" | bc)
        ((count++))
    done
    if [ $count -gt 0 ]; then
        echo "scale=2; $sum / $count" | bc
    fi
}

# Function to calculate standard deviation
calc_std() {
    local mean=$1
    shift
    local sum_sq=0
    local count=0
    for val in "$@"; do
        diff=$(echo "$val - $mean" | bc)
        sum_sq=$(echo "$sum_sq + ($diff * $diff)" | bc)
        ((count++))
    done
    if [ $count -gt 1 ]; then
        variance=$(echo "scale=4; $sum_sq / ($count - 1)" | bc)
        echo "scale=2; sqrt($variance)" | bc -l
    fi
}

# Calculate statistics for each metric
soft_train_mean=$(calc_mean "${soft_train_accs[@]}")
soft_train_std=$(calc_std "$soft_train_mean" "${soft_train_accs[@]}")

soft_test_mean=$(calc_mean "${soft_test_accs[@]}")
soft_test_std=$(calc_std "$soft_test_mean" "${soft_test_accs[@]}")

hard_train_mean=$(calc_mean "${hard_train_accs[@]}")
hard_train_std=$(calc_std "$hard_train_mean" "${hard_train_accs[@]}")

hard_test_mean=$(calc_mean "${hard_test_accs[@]}")
hard_test_std=$(calc_std "$hard_test_mean" "${hard_test_accs[@]}")

# Calculate average training time
avg_train_time=$(calc_mean "${train_times[@]}")
avg_train_hours=$(echo "scale=0; $avg_train_time / 3600" | bc)
avg_train_mins=$(echo "scale=0; ($avg_train_time % 3600) / 60" | bc)
avg_train_secs=$(echo "scale=0; $avg_train_time % 60" | bc)

# Print results
echo "SOFT INFERENCE (Multi-path):"
echo "  Train: $soft_train_mean ± $soft_train_std %"
echo "  Test:  $soft_test_mean ± $soft_test_std %"
echo ""
echo "HARD INFERENCE (Single-path greedy):"
echo "  Train: $hard_train_mean ± $hard_train_std %"
echo "  Test:  $hard_test_mean ± $hard_test_std %"
echo ""
echo "Training Time (Average across 5 seeds):"
echo "  ${avg_train_hours}h ${avg_train_mins}m ${avg_train_secs}s (${avg_train_time}s)"
echo ""

# Show individual runs for reference
echo "Individual Run Results:"
echo "Run | Soft Train | Soft Test | Hard Train | Hard Test | Time (s)"
echo "----|------------|-----------|-----------|-----------|----------"
for i in $(seq 0 $((NUM_RUNS - 1))); do
    printf "%3d | %9.2f%% | %8.2f%% | %9.2f%% | %8.2f%% | %8.1f\n" \
        $((i+1)) "${soft_train_accs[$i]}" "${soft_test_accs[$i]}" \
        "${hard_train_accs[$i]}" "${hard_test_accs[$i]}" "${train_times[$i]}"
done

echo ""
echo "============================================================"

# Create summary results file
SUMMARY_FILE="./experiments/$DATASET/results_summary.txt"
mkdir -p "./experiments/$DATASET"

cat > "$SUMMARY_FILE" << EOF
============================================================
Final Results Summary: $DATASET
============================================================
Dataset: $DATASET
Number of Seeds: 5
Date: $(date)

============================================================
SOFT INFERENCE (Multi-path)
============================================================
Train Accuracy: $soft_train_mean ± $soft_train_std %
Test Accuracy:  $soft_test_mean ± $soft_test_std %

============================================================
HARD INFERENCE (Single-path greedy)
============================================================
Train Accuracy: $hard_train_mean ± $hard_train_std %
Test Accuracy:  $hard_test_mean ± $hard_test_std %

============================================================
TRAINING TIME (Average across 5 seeds)
============================================================
Average Time: ${avg_train_hours}h ${avg_train_mins}m ${avg_train_secs}s
Average Time (seconds): $avg_train_time

============================================================
Individual Run Details
============================================================
Run | Soft Train | Soft Test | Hard Train | Hard Test | Time (s)
----|------------|-----------|-----------|-----------|----------
EOF

for i in $(seq 0 $((NUM_RUNS - 1))); do
    printf "%3d | %9.2f%% | %8.2f%% | %9.2f%% | %8.2f%% | %8.1f\n" \
        $((i+1)) "${soft_train_accs[$i]}" "${soft_test_accs[$i]}" \
        "${hard_train_accs[$i]}" "${hard_test_accs[$i]}" "${train_times[$i]}" >> "$SUMMARY_FILE"
done

echo "" >> "$SUMMARY_FILE"
echo "============================================================" >> "$SUMMARY_FILE"
echo "Results saved in: ./experiments/$DATASET/seed{1..5}/checkpoints/" >> "$SUMMARY_FILE"
echo "============================================================" >> "$SUMMARY_FILE"

echo "Summary results saved to: $SUMMARY_FILE"
echo "All results saved in: ./experiments/$DATASET/seed{1..5}/checkpoints/"
echo "============================================================"
