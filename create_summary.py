import json
import os

try:
    with open('coverage-initial.json', 'r') as f:
        initial = json.load(f)
    with open('coverage-final/coverage.json', 'r') as f:
        final = json.load(f)
    
    initial_total = initial.get('totals', {}).get('percent_covered', 0)
    final_total = final.get('totals', {}).get('percent_covered', 0)
    
    with open('COVERAGE_SUMMARY.md', 'w') as f:
        f.write("# Coverage Summary\n\n")
        f.write(f"**Initial Coverage**: {initial_total:.1f}%\n")
        f.write(f"**Final Coverage**: {final_total:.1f}%\n")
        f.write(f"**Improvement**: {final_total - initial_total:.1f}%\n\n")
        
        # Add module-level improvements
        f.write("## Module Improvements\n\n")
        if 'files' in initial and 'files' in final:
            for file in initial['files']:
                if file in final['files']:
                    initial_cov = initial['files'][file]['summary']['percent_covered']
                    final_cov = final['files'][file]['summary']['percent_covered']
                    if final_cov > initial_cov:
                        f.write(f"- `{file}`: {initial_cov:.1f}% → {final_cov:.1f}% (+{final_cov-initial_cov:.1f}%)\n")
except Exception as e:
    print(f"Error creating summary: {e}")
    with open('COVERAGE_SUMMARY.md', 'w') as f:
        f.write("# Coverage Summary\nNo summary available\n")
