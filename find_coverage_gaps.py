import json
import sys

def find_low_coverage(threshold=80):
    try:
        with open('coverage-initial.json', 'r') as f:
            coverage_data = json.load(f)
    except:
        print("[]")
        return
    
    low_coverage_files = []
    for file_path, file_data in coverage_data.get('files', {}).items():
        if '/test_' not in file_path and '/__init__.py' not in file_path:
            coverage = file_data.get('summary', {}).get('percent_covered', 0)
            if coverage < threshold:
                low_coverage_files.append({
                    'path': file_path,
                    'coverage': coverage
                })
    
    low_coverage_files.sort(key=lambda x: x['coverage'])
    print(json.dumps([f['path'] for f in low_coverage_files]))

if __name__ == "__main__":
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 80
    find_low_coverage(threshold)
