#!/bin/bash

# Default parameters
PROFILE_OUT="./profile_out"
LOG_FILE="./profile_run.log"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --profile_out)
            PROFILE_OUT="$2"
            shift 2
            ;;
        --log)
            LOG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check and handle existing PROFILE_OUT directory
if [ -d "$PROFILE_OUT" ]; then
    echo "Warning: PROFILE_OUT directory '$PROFILE_OUT' already exists."
fi

# Check and handle existing LOG_FILE
if [ -f "$LOG_FILE" ]; then
    echo "Warning: LOG_FILE '$LOG_FILE' already exists. Overwriting..."
fi

# Create output directory
mkdir -p "$PROFILE_OUT"

# Separator function
print_separator() {
    echo "========================================" >> "$LOG_FILE"
    echo "========================================"
}

# Function to run tests
run_test() {
    local test_name=$1
    local script_path=$2
    local csv_path=$3

    echo "Running $test_name test..." | tee -a "$LOG_FILE"
    print_separator

    local output_csv="$PROFILE_OUT/${test_name}_results.csv"

    python3 ./benchmarks/profile/profile_run.py \
        --script "$script_path" \
        --input_csv "$csv_path" \
        --output_csv "$output_csv" \
        2>&1 | tee -a "$LOG_FILE"

    echo "Results saved to: $output_csv" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# Main execution flow
{


echo "Starting profile run at $(date)"
print_separator

# Run GEMM test
run_test "gemm" "./tests/ops/test_gemm.py" "./benchmarks/input_params/gemm.csv"

# Run MHA test
run_test "mha" "./benchmarks/profile/profile_mha.py" "./benchmarks/input_params/mha.csv"

# Run GQA test
run_test "gqa" "./benchmarks/profile/profile_gqa.py" "./benchmarks/input_params/gqa.csv"

# Run MHA Decode test
run_test "mha_decode" "./tests/ops/test_mha_decode.py" "./benchmarks/input_params/mha_decode.csv"

# Run GQA Decode test
run_test "gqa_decode" "./tests/ops/test_gqa_decode.py" "./benchmarks/input_params/gqa_decode.csv"

# Run MLA Decode test
run_test "deepseek_mla_decode" "./tests/ops/test_deepseek_mla_decode.py" "./benchmarks/input_params/deepseek_mla_decode.csv"

# Run Sparse MLA test
run_test "deepseek_dsa_decode" "./tests/ops/test_deepseek_dsa_decode.py" "./benchmarks/input_params/deepseek_dsa_decode.csv"

print_separator
echo "All tests completed at $(date)"

} | tee -a "$LOG_FILE"
