    #!/bin/bash

    # Accept log file name as input parameter, default to tileops_test.log
    LOG_FILE="${1:-tileops_test.log}"

    # Run all Python test files in tests directory
    echo -e "\033[0;34mRunning all Python test files...\033[0m" | tee -a "$LOG_FILE"

    # Create test result summary file
    echo "Test Results Summary" > test_summary.txt
    echo "===================" >> test_summary.txt
    echo "" >> test_summary.txt

    # Initialize counters
    passed_count=0
    failed_count=0

    # Enter tests directory
    cd tests || { echo "Failed to enter tests directory"; exit 1; }

    # Find all .py files in current directory (tests)
    test_files=$(find . -name "*.py" -type f | sort)

    if [ -z "$test_files" ]; then
      echo "No test files found in tests directory" | tee -a "../$LOG_FILE"
      exit 1
    fi

    # Table header alignment, assuming filename max length of 50 characters
    printf "| %-50s | %-8s |\n" "Test File" "Status" >> test_summary.txt
    printf "|%s|\n" "--------------------------------------------------|----------" >> test_summary.txt

    # Run each test file
    for test_file in $test_files; do
      file_name=$(basename "$test_file")
      echo -e "\033[0;36mRunning test: $test_file\033[0m" | tee -a "$LOG_FILE"
      echo "----------------------------------------" >> "$LOG_FILE"

      if python "$test_file" >> "../$LOG_FILE" 2>&1; then
        echo -e "\033[0;32m[PASS] $test_file\033[0m" | tee -a "../$LOG_FILE"
        printf "| %-50s | ✅ Pass  |\n" "$file_name" >> ../test_summary.txt
        passed_count=$((passed_count + 1))
      else
        echo -e "\033[0;31m[FAIL] $test_file\033[0m" | tee -a "../$LOG_FILE"
        printf "| %-50s | ❌ Fail  |\n" "$file_name" >> ../test_summary.txt
        failed_count=$((failed_count + 1))
      fi
      
      echo "----------------------------------------" >> "$LOG_FILE"
    done

    # Return to parent directory
    cd ..

    # Add statistics summary
    echo "" >> test_summary.txt
    echo "Summary:" >> test_summary.txt
    echo "- Passed: $passed_count" >> test_summary.txt
    echo "- Failed: $failed_count" >> test_summary.txt
    echo "- Total:  $((passed_count + failed_count))" >> test_summary.txt

    # Print test results summary table
    echo "" | tee -a "$LOG_FILE"
    echo -e "\033[0;34mTest Results Summary:\033[0m" | tee -a "$LOG_FILE"
    echo -e "\033[0;34m====================\033[0m" | tee -a "$LOG_FILE"
    cat test_summary.txt | tee -a "$LOG_FILE"

    # If there are failed tests, CI fails
    if [ $failed_count -gt 0 ]; then
      echo -e "\033[0;31mError: $failed_count test(s) failed, stopping pipeline.\033[0m" | tee -a "$LOG_FILE"
      exit 1
    else
      echo -e "\033[0;32mAll tests passed!\033[0m" | tee -a "$LOG_FILE"
    fi