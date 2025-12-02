#!/usr/bin/env python3
# run_benchmark.py

import argparse
import csv
import subprocess
import sys
import re
from pathlib import Path


def build_gemm_cmd(args_dict):
    """
    Build command arguments for GEMM test script
    """
    cmd_args = [
        '--M',
        str(args_dict['M']), '--N',
        str(args_dict['N']), '--K',
        str(args_dict['K']), '--dtype',
        str(args_dict['dtype'])
    ]
    return cmd_args


def build_mha_cmd(args_dict):
    """
    Build command arguments for MHA test script
    """
    cmd_args = [
        '--batch',
        str(args_dict['batch']), '--seq_len',
        str(args_dict['seq_len']), '--heads',
        str(args_dict['heads']), '--dim',
        str(args_dict['dim']), '--dtype',
        str(args_dict['dtype'])
    ]
    if args_dict.get('causal', 'False').lower() == 'true':
        cmd_args.append('--causal')
    return cmd_args


def build_gqa_cmd(args_dict):
    """
    Build command arguments for GQA test script
    """
    cmd_args = [
        '--batch',
        str(args_dict['batch']), '--seq_len',
        str(args_dict['seq_len']), '--heads',
        str(args_dict['heads']), '--heads_kv',
        str(args_dict['heads_kv']), '--dim',
        str(args_dict['dim']), '--dtype',
        str(args_dict['dtype'])
    ]
    if args_dict.get('causal', 'False').lower() == 'true':
        cmd_args.append('--causal')
    return cmd_args


def build_mha_decode_cmd(args_dict):
    """
    Build command arguments for MHA decode test script
    """
    cmd_args = [
        '--batch',
        str(args_dict['batch']), '--seq_len_q',
        str(args_dict['seq_len_q']), '--seq_len_kv',
        str(args_dict['seq_len_kv']), '--heads',
        str(args_dict['heads']), '--dim',
        str(args_dict['dim']), '--dtype',
        str(args_dict['dtype'])
    ]
    return cmd_args


def build_gqa_decode_cmd(args_dict):
    """
    Build command arguments for GQA decode test script
    """
    cmd_args = [
        '--batch',
        str(args_dict['batch']), '--seq_len_kv',
        str(args_dict['seq_len_kv']), '--heads',
        str(args_dict['heads']), '--groups',
        str(args_dict['groups']), '--dim',
        str(args_dict['dim']), '--dtype',
        str(args_dict['dtype'])
    ]
    return cmd_args


def build_mla_decode_cmd(args_dict):
    """
    Build command arguments for MLA decode test script
    """
    cmd_args = [
        '--batch',
        str(args_dict['batch']), '--seq_len_kv',
        str(args_dict['seq_len_kv']), '--heads',
        str(args_dict['heads']), '--kv_head_num',
        str(args_dict['kv_head_num']), '--dim',
        str(args_dict['dim']), '--pe_dim',
        str(args_dict['pe_dim']), '--dtype',
        str(args_dict['dtype'])
    ]
    return cmd_args


def build_sparse_mla_cmd(args_dict):
    """
    Build command arguments for Sparse MLA test script
    """
    cmd_args = [
        '--batch',
        str(args_dict['batch']), '--seq_len',
        str(args_dict['seq_len']), '--seq_len_kv',
        str(args_dict['seq_len_kv']), '--heads',
        str(args_dict['heads']), '--dim',
        str(args_dict['dim']), '--tail_dim',
        str(args_dict['tail_dim']), '--topk',
        str(args_dict['topk']), '--kv_stride',
        str(args_dict['kv_stride']), '--kv_group',
        str(args_dict['kv_group']), '--q_start_index_s',
        str(args_dict.get('q_start_index_s', 1024)), '--dtype',
        str(args_dict['dtype'])
    ]

    # Handle optional sm_scale parameter
    if 'sm_scale' in args_dict and args_dict['sm_scale'] != 'None':
        cmd_args.extend(['--sm_scale', str(args_dict['sm_scale'])])

    return cmd_args


def parse_output(output_lines):
    """
    Parse script output to extract separate forward and backward latency, TFlops, and Bandwidth information
    """
    results = {}
    current_section = 'fwd'  # 'fwd' or 'bwd'

    for line in output_lines:
        # Detect section markers (you'll need to add these to your test scripts)
        if 'Backward Results:' in line:
            current_section = 'bwd'
            continue

        # Extract latency
        latency_match = re.search(r'tl-latency:\s*([0-9.]+)\s*ms', line)
        if latency_match:
            results[f'{current_section}-tl-latency(ms)'] = float(latency_match.group(1))

        # Extract TFlops
        tflops_match = re.search(r'tl-TFlops:\s*([0-9.]+)', line)
        if tflops_match:
            results[f'{current_section}-tl-TFlops'] = float(tflops_match.group(1))

        # Extract Bandwidth
        bandwidth_match = re.search(r'tl-Bandwidth:\s*([0-9.]+)\s*GB/s', line)
        if bandwidth_match:
            results[f'{current_section}-tl-Bandwidth(GB/s)'] = float(bandwidth_match.group(1))

        # Extract baseline metrics
        baseline_latency_match = re.search(r'Baseline-latency:\s*([0-9.]+)\s*ms', line)
        if baseline_latency_match:
            results[f'{current_section}-Baseline-latency(ms)'] = float(
                baseline_latency_match.group(1))

        baseline_tflops_match = re.search(r'Baseline-TFlops:\s*([0-9.]+)', line)
        if baseline_tflops_match:
            results[f'{current_section}-Baseline-TFlops'] = float(baseline_tflops_match.group(1))

        baseline_bandwidth_match = re.search(r'Baseline-Bandwidth:\s*([0-9.]+)\s*GB/s', line)
        if baseline_bandwidth_match:
            results[f'{current_section}-Baseline-Bandwidth(GB/s)'] = float(
                baseline_bandwidth_match.group(1))

    return results


def run_test_script(script_path, args_dict):
    """
    Run the specified test script and return output
    """
    # Build command line arguments based on script type
    script_name = script_path.name.lower()

    if 'gemm' in script_name:
        cmd_args = build_gemm_cmd(args_dict)
    elif 'mha_decode' in script_name:
        cmd_args = build_mha_decode_cmd(args_dict)
    elif 'gqa_decode' in script_name:
        cmd_args = build_gqa_decode_cmd(args_dict)
    elif 'mla_decode' in script_name:
        cmd_args = build_mla_decode_cmd(args_dict)
    elif 'sparse_mla' in script_name:
        cmd_args = build_sparse_mla_cmd(args_dict)
    elif 'mha' in script_name:
        cmd_args = build_mha_cmd(args_dict)
    elif 'gqa' in script_name:
        cmd_args = build_gqa_cmd(args_dict)
    else:
        raise ValueError(f"Unsupported script type: {script_path}")

    # Build full command with executable
    cmd = [sys.executable, str(script_path)] + cmd_args

    print(f"Running command: {' '.join(cmd)}")

    try:
        # Run script and capture output
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

    # Validate script path
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"Error: Script {script_path} does not exist")
        return 1

    # Validate input CSV file
    input_csv_path = Path(args.input_csv)
    if not input_csv_path.exists():
        print(f"Error: Input CSV {input_csv_path} does not exist")
        return 1

    # Read CSV file
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

    # Get headers as output CSV fields
    fieldnames = list(input_params[0].keys()) + [
        'fwd-tl-latency(ms)', 'fwd-tl-TFlops', 'fwd-tl-Bandwidth(GB/s)', 'fwd-Baseline-latency(ms)',
        'fwd-Baseline-TFlops', 'fwd-Baseline-Bandwidth(GB/s)', 'bwd-tl-latency(ms)',
        'bwd-tl-TFlops', 'bwd-tl-Bandwidth(GB/s)', 'bwd-Baseline-latency(ms)',
        'bwd-Baseline-TFlops', 'bwd-Baseline-Bandwidth(GB/s)'
    ]

    # Prepare output file
    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    # Run each parameter combination
    for i, params in enumerate(input_params):
        print(f"\nRunning test {i+1}/{len(input_params)} with parameters: {params}")

        # Run test script
        output_lines = run_test_script(script_path, params)
        if output_lines is None:
            print("Skipping this test due to execution error")
            output_fields = [f for f in fieldnames if f not in params]
            error_result = {**params, **{f: None for f in output_fields}}
            results.append(error_result)
            continue

        # Parse output results
        parsed_results = parse_output(output_lines)

        # Merge input parameters and results
        combined_result = {**params, **parsed_results}
        results.append(combined_result)

        print(f"Results: {parsed_results}")

    # Write results to CSV file
    try:
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    except Exception as e:
        print(f"Error writing output CSV file: {e}")
        return 1

    # Print results table to screen
    if results:
        # Calculate column widths and filter out empty columns
        col_widths = {}
        visible_fields = []  # Store columns to be displayed

        for field in fieldnames:
            col_widths[field] = len(field)
            has_non_empty_value = False  # Flag to track if column has non-empty values

            # Iterate through all results to check if column contains non-empty values and calculate max width
            for result in results:
                value = result.get(field, '')
                display_value = str(value) if value is not None else ''
                col_widths[field] = max(col_widths[field], len(display_value))

                # If a non-empty value is found, mark this column as needing to be displayed
                if display_value.strip():  # Use strip() to check for whitespace-only strings
                    has_non_empty_value = True

            # Only add column to visible fields list if it has non-empty values
            if has_non_empty_value or field in input_params[0].keys():
                visible_fields.append(field)

        # If visible_fields is empty, display all fields
        if not visible_fields:
            visible_fields = fieldnames

        # Update col_widths to only include visible fields
        filtered_col_widths = {field: col_widths[field] for field in visible_fields}

        # Print header
        header = " | ".join(field.ljust(filtered_col_widths[field]) for field in visible_fields)
        print("\n" + "=" * len(header))
        print("FINAL RESULTS")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        # Print data rows
        for result in results:
            row = " | ".join(
                str(result.get(field, '')).ljust(filtered_col_widths[field])
                for field in visible_fields)
            print(row)
        print("=" * len(header))

        print(f"\nResults saved to: {output_csv_path}")
    else:
        print("No results to display")

    return 0


if __name__ == "__main__":
    sys.exit(main())
