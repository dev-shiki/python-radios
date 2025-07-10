#!/usr/bin/env python3
"""Membuat ringkasan coverage markdown dengan penemuan file yang ditingkatkan dan penanganan error"""
import json
import os
from pathlib import Path


def find_coverage_file(filename: str, search_locations: list) -> tuple:
    """
    Mencari file coverage di berbagai lokasi yang mungkin.
    
    Args:
        filename: Nama file dasar yang dicari
        search_locations: List path/pattern yang akan dicoba
        
    Returns:
        Tuple dari (found_path, data) atau (None, None)
    """
    for location in search_locations:
        path = Path(location)
        if path.exists() and path.is_file():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                print(f"Found {filename} at: {path}")
                return str(path), data
            except Exception as e:
                print(f"Failed to read {path}: {e}")
                continue
    
    print(f"Could not find {filename} in any location:")
    for loc in search_locations:
        print(f"  - {loc}")
    return None, None


def get_coverage_value(data: dict) -> float:
    """Mengekstrak persentase coverage dari data coverage."""
    if not data:
        return 0.0
    
    # Try different possible data structures
    if 'totals' in data and 'percent_covered' in data['totals']:
        return data['totals']['percent_covered']
    elif 'percent_covered' in data:
        return data['percent_covered']
    else:
        # Try to calculate from covered/total lines
        if 'totals' in data:
            totals = data['totals']
            if 'num_statements' in totals and totals['num_statements'] > 0:
                covered = totals.get('covered_lines', 0)
                return (covered / totals['num_statements']) * 100
    return 0.0


def create_summary():
    """Membuat ringkasan coverage dengan penemuan file yang robust."""
    
    # Get environment variables for file names
    initial_filename = os.getenv('COVERAGE_INITIAL', 'coverage-initial.json')
    final_filename = os.getenv('COVERAGE_FINAL', 'coverage-final.json')
    
    print(f"Looking for coverage files...")
    print(f"Initial file: {initial_filename}")
    print(f"Final file: {final_filename}")
    
    # Define search locations for initial coverage
    initial_locations = [
        initial_filename,
        f"./{initial_filename}",
        f"coverage-initial/{initial_filename}",
        "coverage-initial.json",
        "./coverage-initial.json"
    ]
    
    # Define search locations for final coverage
    final_locations = [
        f"coverage-final/{final_filename}",
        final_filename,
        f"./{final_filename}",
        "coverage-final/coverage-final.json",
        "coverage-final/coverage.json",
        "coverage-post-generation.json",
        "coverage-after-generation.json",
        "./coverage-final.json",
        "./coverage.json"
    ]
    
    # Add environment variable from CI
    coverage_after_gen = os.getenv('COVERAGE_AFTER_GEN')
    if coverage_after_gen:
        final_locations.insert(0, coverage_after_gen)
        final_locations.insert(1, f"./{coverage_after_gen}")
    
    # Find and load initial coverage
    initial_path, initial_data = find_coverage_file("initial coverage", initial_locations)
    initial_total = get_coverage_value(initial_data)
    
    # Find and load final coverage
    final_path, final_data = find_coverage_file("final coverage", final_locations)
    final_total = get_coverage_value(final_data)
    
    # Debug current directory
    print("\nCurrent directory contents:")
    for item in Path(".").iterdir():
        print(f"  - {item}")
    
    if Path("coverage-final").exists():
        print("\ncoverage-final directory contents:")
        for item in Path("coverage-final").iterdir():
            print(f"  - {item}")
    
    # Create summary
    with open('COVERAGE_SUMMARY.md', 'w') as f:
        f.write("# Coverage Summary\n\n")
        
        if initial_path:
            f.write(f"**Initial Coverage**: {initial_total:.1f}%\n")
        else:
            f.write("**Initial Coverage**: Not found\n")
        
        if final_path:
            f.write(f"**Final Coverage**: {final_total:.1f}%\n")
        else:
            f.write("**Final Coverage**: Not found\n")
        
        if initial_path and final_path:
            improvement = final_total - initial_total
            f.write(f"**Improvement**: {improvement:+.1f}%\n\n")
            
            # Module-level improvements
            if initial_data and final_data and 'files' in initial_data and 'files' in final_data:
                f.write("## Module Improvements\n\n")
                improvements = []
                
                for file in initial_data['files']:
                    if file in final_data['files']:
                        initial_file_data = initial_data['files'][file]
                        final_file_data = final_data['files'][file]
                        
                        # Handle different data structures
                        initial_cov = 0
                        final_cov = 0
                        
                        if 'summary' in initial_file_data:
                            initial_cov = initial_file_data['summary'].get('percent_covered', 0)
                        elif 'percent_covered' in initial_file_data:
                            initial_cov = initial_file_data['percent_covered']
                        
                        if 'summary' in final_file_data:
                            final_cov = final_file_data['summary'].get('percent_covered', 0)
                        elif 'percent_covered' in final_file_data:
                            final_cov = final_file_data['percent_covered']
                        
                        if final_cov > initial_cov:
                            improvements.append({
                                'file': file,
                                'initial': initial_cov,
                                'final': final_cov,
                                'delta': final_cov - initial_cov
                            })
                
                # Sort by improvement
                improvements.sort(key=lambda x: x['delta'], reverse=True)
                
                # Show top 10 improvements
                for imp in improvements[:10]:
                    f.write(f"- `{imp['file']}`: {imp['initial']:.1f}% → {imp['final']:.1f}% ")
                    f.write(f"(+{imp['delta']:.1f}%)\n")
        
        # Add file paths for debugging
        f.write("\n## Debug Information\n\n")
        if initial_path:
            f.write(f"- Initial coverage file: `{initial_path}`\n")
        else:
            f.write("- Initial coverage file: Not found\n")
        
        if final_path:
            f.write(f"- Final coverage file: `{final_path}`\n")
        else:
            f.write("- Final coverage file: Not found\n")
            f.write("\nSearched locations for final coverage:\n")
            for loc in final_locations:
                f.write(f"  - `{loc}`\n")
    
    print("\nSummary created: COVERAGE_SUMMARY.md")
    
    # Exit with error if no final coverage found
    if not final_path:
        raise Exception("No final coverage data found")


if __name__ == "__main__":
    try:
        create_summary()
    except Exception as e:
        print(f"Error creating summary: {e}")
        # Ensure error is visible in CI
        import sys
        sys.exit(1)