#!/usr/bin/env python
"""
AI-powered test refiner that improves tests based on coverage feedback.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openai


class TestRefiner:
    """Refine generated tests based on coverage feedback."""
    
    def __init__(self, api_key: str):
        """Initialize the refiner."""
        self.api_key = api_key
        self.openai_client = self._setup_openai()
    
    def _setup_openai(self):
        """Configure OpenAI client."""
        if "openrouter" in self.api_key.lower() or os.getenv("OPENROUTER_API_KEY"):
            return openai.OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        return openai.OpenAI(api_key=self.api_key)
    
    def analyze_coverage_changes(self) -> Dict:
        """
        Compare initial and post-generation coverage to identify improvements needed.
        
        Returns:
            Dictionary of coverage analysis and recommendations
        """
        # Load coverage reports
        initial_coverage = self._load_coverage_report("coverage-initial.json")
        post_coverage = self._load_coverage_report("coverage-post.json")
        
        if not initial_coverage or not post_coverage:
            return {}
        
        analysis = {
            "improvements": [],
            "remaining_gaps": [],
            "failed_tests": []
        }
        
        # Compare coverage for each file
        for file_path, initial_data in initial_coverage.get('files', {}).items():
            post_data = post_coverage.get('files', {}).get(file_path, {})
            
            initial_coverage_pct = initial_data.get('summary', {}).get('percent_covered', 0)
            post_coverage_pct = post_data.get('summary', {}).get('percent_covered', 0)
            
            # Identify improvements
            if post_coverage_pct > initial_coverage_pct:
                analysis["improvements"].append({
                    "file": file_path,
                    "improvement": post_coverage_pct - initial_coverage_pct,
                    "still_missing": self._find_uncovered_lines(post_data)
                })
            
            # Identify remaining gaps
            if post_coverage_pct < 90:  # Adjust threshold as needed
                analysis["remaining_gaps"].append({
                    "file": file_path,
                    "coverage": post_coverage_pct,
                    "uncovered_lines": self._find_uncovered_lines(post_data)
                })
        
        # Check for test failures
        test_output = self._run_tests()
        if test_output:
            analysis["failed_tests"] = self._parse_test_failures(test_output)
        
        return analysis
    
    def _load_coverage_report(self, filename: str) -> Optional[Dict]:
        """Load coverage report from JSON file."""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    
    def _find_uncovered_lines(self, coverage_data: Dict) -> List[int]:
        """Extract uncovered line numbers from coverage data."""
        missing_lines = []
        for line_num, count in coverage_data.get('executed_lines', {}).items():
            if count == 0:
                missing_lines.append(int(line_num))
        return missing_lines
    
    def _run_tests(self) -> str:
        """Run tests to capture failures."""
        try:
            result = subprocess.run(
                ["pytest", "-v", "--tb=short"],
                capture_output=True,
                text=True
            )
            return result.stdout + result.stderr
        except Exception as e:
            print(f"Error running tests: {e}")
            return ""
    
    def _parse_test_failures(self, test_output: str) -> List[Dict]:
        """Parse test failures from pytest output."""
        failures = []
        current_failure = None
        
        for line in test_output.split('\n'):
            if "FAILED" in line:
                if current_failure:
                    failures.append(current_failure)
                current_failure = {
                    "test": line.split()[0],
                    "error": []
                }
            elif current_failure and not line.startswith('==='):
                current_failure["error"].append(line)
        
        if current_failure:
            failures.append(current_failure)
        
        return failures
    
    def refine_test_file(self, test_file: Path, analysis_data: Dict) -> str:
        """
        Refine a test file based on coverage and error feedback.
        
        Args:
            test_file: Path to the test file
            analysis_data: Coverage and error analysis data
            
        Returns:
            Refined test code
        """
        # Read current test code
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                current_code = f.read()
        except Exception as e:
            print(f"Error reading {test_file}: {e}")
            return ""
        
        # Create refinement prompt
        prompt = self._create_refinement_prompt(test_file, current_code, analysis_data)
        
        # Generate refined tests
        try:
            response = self.openai_client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[
                    {"role": "system", "content": self._get_refinement_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error refining tests: {e}")
            return ""
    
    def _get_refinement_system_prompt(self) -> str:
        """Return the system prompt for test refinement."""
        return """You are an expert Python test refiner. Your task is to:
1. Fix failing tests based on error messages
2. Add missing test coverage for uncovered lines
3. Improve test quality and completeness
4. Maintain existing working tests
5. Follow pytest best practices
6. Return only the refined test code without explanations"""
    
    def _create_refinement_prompt(self, test_file: Path, current_code: str, analysis_data: Dict) -> str:
        """Create a prompt for refining tests."""
        # Extract relevant analysis data for this file
        file_str = str(test_file)
        source_file = file_str.replace('test_', '').replace('tests/', '')
        
        uncovered_lines = []
        test_failures = []
        
        for item in analysis_data.get('remaining_gaps', []):
            if item['file'].endswith(Path(source_file).name):
                uncovered_lines = item.get('uncovered_lines', [])
                break
        
        for failure in analysis_data.get('failed_tests', []):
            if test_file.name in failure.get('test', ''):
                test_failures.append(failure)
        
        prompt = f"""Refine these pytest test cases:

CURRENT TEST FILE: {test_file}

CURRENT CODE:
```python
{current_code}
```

ISSUES TO ADDRESS:
"""
        
        if uncovered_lines:
            prompt += f"\n1. Uncovered lines: {uncovered_lines}"
            prompt += "\n   Add tests to cover these lines."
        
        if test_failures:
            prompt += "\n2. Test failures:"
            for i, failure in enumerate(test_failures):
                prompt += f"\n   Test: {failure['test']}"
                prompt += f"\n   Error: {' '.join(failure['error'][:3])}"
        
        prompt += """

REQUIREMENTS:
1. Fix all failing tests
2. Add tests for uncovered lines
3. Keep existing working tests unchanged
4. Improve test completeness
5. Return only the refined test code

Return the complete refined test file."""
        
        return prompt
    
    def save_refined_test(self, test_file: Path, refined_code: str) -> bool:
        """
        Save the refined test file.
        
        Args:
            test_file: Path to the test file
            refined_code: Refined test code
            
        Returns:
            True if saved successfully
        """
        try:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(refined_code)
            print(f"Refined test file: {test_file}")
            return True
        except Exception as e:
            print(f"Error saving refined test: {e}")
            return False


def main():
    """Main entry point for test refinement."""
    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: No API key found")
        sys.exit(1)
    
    # Initialize refiner
    refiner = TestRefiner(api_key)
    
    # Analyze coverage changes
    analysis = refiner.analyze_coverage_changes()
    
    if not analysis:
        print("No coverage analysis available for refinement")
        return
    
    # Save analysis for review
    with open("coverage-analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    # Find all test files that need refinement
    test_files = list(Path("tests").rglob("test_*.py"))
    
    if not test_files:
        print("No test files found to refine")
        return
    
    print(f"Refining {len(test_files)} test files...")
    
    # Refine each test file
    for test_file in test_files:
        print(f"Refining: {test_file}")
        refined_code = refiner.refine_test_file(test_file, analysis)
        
        if refined_code:
            refiner.save_refined_test(test_file, refined_code)
    
    print("Test refinement complete!")


if __name__ == "__main__":
    main()