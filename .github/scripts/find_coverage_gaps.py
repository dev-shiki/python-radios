#!/usr/bin/env python3
"""Mencari modul dengan coverage rendah"""
import json
import sys

def find_low_coverage(threshold=80):
    try:
        with open('coverage-initial.json', 'r') as f:
            coverage_data = json.load(f)
    except Exception as e:
        print(f"Error membaca data coverage: {e}", file=sys.stderr)
        print("[]")
        return
    
    low_coverage_files = []
    
    for file_path, file_data in coverage_data.get('files', {}).items():
        # Skip test files and __init__ files
        if '/test_' not in file_path and '/__init__.py' not in file_path:
            coverage = file_data.get('summary', {}).get('percent_covered', 0)
            if coverage < threshold:
                low_coverage_files.append({
                    'path': file_path,
                    'coverage': coverage,
                    'missing_lines': file_data.get('summary', {}).get('missing_lines', 0)
                })
    
    # Sort by coverage (lowest first)
    low_coverage_files.sort(key=lambda x: x['coverage'])
    
    # Output just the paths for the workflow
    paths = [f['path'] for f in low_coverage_files]
    print(json.dumps(paths))

if __name__ == "__main__":
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 80
    find_low_coverage(threshold)