#!/bin/bash

# Accept log file name as input parameter, default to tileops_test.log
LOG_FILE="${1:-tileops_test.log}"

# Run all Python test files in tests directory
echo -e "\033[0;34mRunning all Python test files...\033[0m"

# Store test results for summary
declare -a test_names
declare -a test_results

# Initialize counters
passed_count=0
failed_count=0

# Find all .py files in current directory where script is located
script_dir=$(dirname -- "${BASH_SOURCE[0]}")
test_files=$(find "$script_dir" -name "test*.py" -type f | sort)

if [ -z "$test_files" ]; then
  echo "No test files found in $script_dir" | tee -a "$LOG_FILE"
  exit 1
fi

# Table header alignment, assuming filename max length of 50 characters
printf "| %-50s | %-8s |\n" "Test File" "Status"
printf "|%s|\n" "--------------------------------------------------|----------"

# Run each test file
for test_file in $test_files; do
  file_name=$(basename "$test_file")
  echo -e "\033[0;36mRunning test: $test_file\033[0m"
  echo "----------------------------------------" >> "$LOG_FILE"

  # Extract the module name from the path for pytest
  relative_path=${test_file#$script_dir/}

  # Run pytest on the specific test file
  if python -m pytest "$test_file" -v -r fE >> "$LOG_FILE" 2>&1; then
    echo -e "\033[0;32m[PASS] $test_file\033[0m"
    printf "| %-50s | ✅ Pass  |\n" "$file_name"
    test_names+=("$file_name")
    test_results+=("✅ Pass")
    passed_count=$((passed_count + 1))
  else
    echo -e "\033[0;31m[FAIL] $test_file\033[0m"
    printf "| %-50s | ❌ Fail  |\n" "$file_name"
    test_names+=("$file_name")
    test_results+=("❌ Fail")
    failed_count=$((failed_count + 1))
  fi

  echo "----------------------------------------" >> "$LOG_FILE"
done

# Add statistics summary to log file
echo "" | tee -a "$LOG_FILE"
echo "Summary:" | tee -a "$LOG_FILE"
echo "- Passed: $passed_count" | tee -a "$LOG_FILE"
echo "- Failed: $failed_count" | tee -a "$LOG_FILE"
echo "- Total:  $((passed_count + failed_count))" | tee -a "$LOG_FILE"

# Print test results summary table
echo "" | tee -a "$LOG_FILE"
echo -e "\033[0;34mTest Results Summary:\033[0m" | tee -a "$LOG_FILE"
echo -e "\033[0;34m====================\033[0m" | tee -a "$LOG_FILE"
printf "| %-50s | %-8s |\n" "Test File" "Status" | tee -a "$LOG_FILE"
printf "|%s|\n" "--------------------------------------------------|----------" | tee -a "$LOG_FILE"

# Print final summary table from stored results
for i in "${!test_names[@]}"; do
    printf "| %-50s | %-8s |\n" "${test_names[$i]}" "${test_results[$i]}" | tee -a "$LOG_FILE"
done

# If there are failed tests, CI fails
if [ $failed_count -gt 0 ]; then
  echo -e "\033[0;31mError: $failed_count test(s) failed, stopping pipeline.\033[0m" | tee -a "$LOG_FILE"
  exit 1
else
  echo -e "\033[0;32mAll tests passed!\033[0m" | tee -a "$LOG_FILE"
fi
