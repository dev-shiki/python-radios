#!/usr/bin/env python3
"""Create coverage summary markdown"""
import json

def create_summary():
    try:
        with open('coverage-initial.json', 'r') as f:
            initial = json.load(f)
        
        # Try multiple locations for final coverage
        final = None
        for path in ['coverage-final/coverage.json', 'coverage-post-generation.json']:
            try:
                with open(path, 'r') as f:
                    final = json.load(f)
                break
            except:
                continue
        
        if not final:
            raise Exception("No final coverage data found")
        
        initial_total = initial.get('totals', {}).get('percent_covered', 0)
        final_total = final.get('totals', {}).get('percent_covered', 0)
        
        with open('COVERAGE_SUMMARY.md', 'w') as f:
            f.write("# Coverage Summary\n\n")
            f.write(f"**Initial Coverage**: {initial_total:.1f}%\n")
            f.write(f"**Final Coverage**: {final_total:.1f}%\n")
            f.write(f"**Improvement**: {final_total - initial_total:.1f}%\n\n")
            
            # Add module-level improvements
            f.write("## Module Improvements\n\n")
            improvements = []
            
            if 'files' in initial and 'files' in final:
                for file in initial['files']:
                    if file in final['files']:
                        initial_cov = initial['files'][file]['summary']['percent_covered']
                        final_cov = final['files'][file]['summary']['percent_covered']
                        if final_cov > initial_cov:
                            improvements.append({
                                'file': file,
                                'initial': initial_cov,
                                'final': final_cov,
                                'delta': final_cov - initial_cov
                            })
            
            # Sort by improvement
            improvements.sort(key=lambda x: x['delta'], reverse=True)
            
            for imp in improvements[:10]:  # Top 10
                f.write(f"- `{imp['file']}`: {imp['initial']:.1f}% → {imp['final']:.1f}% ")
                f.write(f"(+{imp['delta']:.1f}%)\n")
            
    except Exception as e:
        print(f"Error creating summary: {e}")
        with open('COVERAGE_SUMMARY.md', 'w') as f:
            f.write("# Coverage Summary\n")
            f.write("Error: Could not generate summary\n")
            f.write(f"Details: {str(e)}\n")

if __name__ == "__main__":
    create_summary()