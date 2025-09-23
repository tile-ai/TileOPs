    # Run all Python test files in tests directory
    echo -e "\033[0;34mRunning all Python test files...\033[0m" | tee -a tileops_test.log
    
    # 创建测试结果统计文件
    echo "Test Results Summary" > test_summary.txt
    echo "===================" >> test_summary.txt
    echo "" >> test_summary.txt

    # 初始化计数器
    passed_count=0
    failed_count=0

    # 查找tests目录下的所有.py文件
    test_files=$(find tests -name "*.py" -type f | sort)

    if [ -z "$test_files" ]; then
      echo "No test files found in tests directory" | tee -a tileops_test.log
      exit 1
    fi

    # 表头对齐，假设文件名最长50字符
    printf "| %-50s | %-8s |\n" "Test File" "Status" >> test_summary.txt
    printf "|%s|\n" "--------------------------------------------------|----------" >> test_summary.txt

    # 运行每个测试文件
    for test_file in $test_files; do
      file_name=$(basename "$test_file")
      echo -e "\033[0;36mRunning test: $test_file\033[0m" | tee -a tileops_test.log
      echo "----------------------------------------" >> tileops_test.log

      if python "$test_file" >> tileops_test.log 2>&1; then
        echo -e "\033[0;32m[PASS] $test_file\033[0m" | tee -a tileops_test.log
        printf "| %-50s | ✅ Pass  |\n" "$file_name" >> test_summary.txt
        passed_count=$((passed_count + 1))
      else
        echo -e "\033[0;31m[FAIL] $test_file\033[0m" | tee -a tileops_test.log
        printf "| %-50s | ❌ Fail  |\n" "$file_name" >> test_summary.txt
        failed_count=$((failed_count + 1))
      fi

      echo "----------------------------------------" >> tileops_test.log
    done

    # 添加统计摘要
    echo "" >> test_summary.txt
    echo "Summary:" >> test_summary.txt
    echo "- Passed: $passed_count" >> test_summary.txt
    echo "- Failed: $failed_count" >> test_summary.txt
    echo "- Total:  $((passed_count + failed_count))" >> test_summary.txt

    # 打印测试结果摘要表
    echo "" | tee -a tileops_test.log
    echo -e "\033[0;34mTest Results Summary:\033[0m" | tee -a tileops_test.log
    echo -e "\033[0;34m====================\033[0m" | tee -a tileops_test.log
    cat test_summary.txt | tee -a tileops_test.log

    # 如果有失败的测试，CI失败
    if [ $failed_count -gt 0 ]; then
      echo -e "\033[0;31mError: $failed_count test(s) failed, stopping pipeline.\033[0m" | tee -a tileops_test.log
      exit 1
    else
      echo -e "\033[0;32mAll tests passed!\033[0m" | tee -a tileops_test.log
    fi