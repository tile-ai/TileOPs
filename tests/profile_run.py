#!/usr/bin/env python3
# run_benchmark.py

import argparse
import csv
import subprocess
import sys
import re
from pathlib import Path

def parse_output(output_lines):
    """
    解析脚本输出，提取latency, TFlops, Bandwidth信息
    """
    results = {}
    for line in output_lines:
        # 提取latency
        latency_match = re.search(r'latency:\s*([0-9.]+)\s*ms', line)
        if latency_match:
            results['latency(ms)'] = float(latency_match.group(1))
            
        # 提取TFlops
        tflops_match = re.search(r'TFlops:\s*([0-9.]+)', line)
        if tflops_match:
            results['TFlops'] = float(tflops_match.group(1))
            
        # 提取Bandwidth
        bandwidth_match = re.search(r'Bandwidth:\s*([0-9.]+)\s*GB/s', line)
        if bandwidth_match:
            results['Bandwidth(GB/s)'] = float(bandwidth_match.group(1))
            
    return results

def run_test_script(script_path, args_dict):
    """
    运行指定的测试脚本并返回输出
    """
    # 构建命令行参数
    cmd = [sys.executable, str(script_path)]
    
    # 根据脚本类型添加参数
    if 'gemm' in script_path.name.lower():
        cmd.extend([
            '--M', str(args_dict['M']),
            '--N', str(args_dict['N']),
            '--K', str(args_dict['K']),
            '--dtype', str(args_dict['dtype'])
        ])
    elif 'mha' in script_path.name.lower():
        cmd.extend([
            '--batch', str(args_dict['batch']),
            '--seq_len', str(args_dict['seq_len']),
            '--heads', str(args_dict['heads']),
            '--dim', str(args_dict['dim']),
            '--dtype', str(args_dict['dtype'])
        ])
        if args_dict.get('causal', 'False').lower() == 'true':
            cmd.append('--causal')
    else:
        raise ValueError(f"Unsupported script type: {script_path}")
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # 运行脚本并捕获输出
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"Error running script: {result.stderr}")
            return None
        return result.stdout.splitlines()
    except subprocess.TimeoutExpired:
        print("Script execution timed out")
        return None
    except Exception as e:
        print(f"Error running script: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Batch run test scripts with CSV parameters')
    parser.add_argument('--script', required=True, help='Path to the test script (.py file)')
    parser.add_argument('--input_csv', required=True, help='Path to input CSV file with parameters')
    parser.add_argument('--output_csv', required=True, help='Path to output CSV file for results')
    
    args = parser.parse_args()
    
    # 验证脚本路径
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"Error: Script {script_path} does not exist")
        return 1
    
    # 验证输入CSV文件
    input_csv_path = Path(args.input_csv)
    if not input_csv_path.exists():
        print(f"Error: Input CSV {input_csv_path} does not exist")
        return 1
    
    # 读取CSV文件
    try:
        with open(input_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            input_params = list(reader)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return 1
    
    if not input_params:
        print("No parameters found in CSV file")
        return 1
    
    # 获取表头作为输出CSV的字段
    fieldnames = list(input_params[0].keys()) + ['latency', 'TFlops', 'Bandwidth']
    
    # 准备输出文件
    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # 运行每个参数组合
    for i, params in enumerate(input_params):
        print(f"\nRunning test {i+1}/{len(input_params)} with parameters: {params}")
        
        # 运行测试脚本
        output_lines = run_test_script(script_path, params)
        if output_lines is None:
            print("Skipping this test due to execution error")
            continue
            
        # 解析输出结果
        parsed_results = parse_output(output_lines)
        
        # 合并输入参数和结果
        combined_result = {**params, **parsed_results}
        results.append(combined_result)
        
        print(f"Results: {parsed_results}")
    
    # 写入结果到CSV文件
    try:
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    except Exception as e:
        print(f"Error writing output CSV file: {e}")
        return 1
    
    # 打印结果表格到屏幕
    if results:
        # 计算每列的最大宽度
        col_widths = {}
        for field in fieldnames:
            col_widths[field] = len(field)
            for result in results:
                value = result.get(field, '')
                col_widths[field] = max(col_widths[field], len(str(value)))
        
        # 打印表头
        header = " | ".join(field.ljust(col_widths[field]) for field in fieldnames)
        print("\n" + "="*len(header))
        print("FINAL RESULTS")
        print("="*len(header))
        print(header)
        print("-"*len(header))
        
        # 打印数据行
        for result in results:
            row = " | ".join(str(result.get(field, '')).ljust(col_widths[field]) for field in fieldnames)
            print(row)
        print("="*len(header))
        
        print(f"\nResults saved to: {output_csv_path}")
    else:
        print("No results to display")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())