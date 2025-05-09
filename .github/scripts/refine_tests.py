#!/usr/bin/env python
"""
AI-powered test refiner that improves tests based on raw test failure output.
"""

import os
import subprocess
import sys
from pathlib import Path

import openai


class TestRefiner:
    """Refine generated tests by passing raw test failure output to AI."""
    
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
    
    def run_tests(self, test_path: str = None) -> str:
        """
        Run tests and capture raw output.
        
        Args:
            test_path: Optional specific test file or directory to run
            
        Returns:
            Test output as string
        """
        cmd = ["pytest", "-v"]
        if test_path:
            cmd.append(test_path)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            return result.stdout + result.stderr
        except Exception as e:
            print(f"Error running tests: {e}")
            return ""
    
    def get_failing_test_files(self, test_output: str) -> list:
        """
        Extract failing test file paths from test output.
        
        Args:
            test_output: Output from pytest run
            
        Returns:
            List of failing test file paths
        """
        failing_files = set()
        
        for line in test_output.split('\n'):
            if "FAILED" in line and "::" in line:
                # Extract file path from test identifier (e.g., tests/test_file.py::TestClass::test_method)
                file_path = line.split("FAILED ")[1].split("::")[0] if "FAILED " in line else ""
                if file_path and file_path.endswith('.py'):
                    failing_files.add(file_path)
        
        return list(failing_files)
    
    def create_refinement_prompt(self, test_file: Path, current_code: str, test_output: str) -> str:
        """
        Create prompt for test refinement using raw test output.
        
        Args:
            test_file: Path to test file
            current_code: Current test code
            test_output: Raw test failure output
            
        Returns:
            Prompt for test refinement
        """
        # Filter test output to only include failures for this specific test file
        file_name = test_file.name
        relevant_failures = []
        capture = False
        
        for line in test_output.split('\n'):
            # Start capturing on failures for this file
            if file_name in line and "FAILED" in line:
                capture = True
                relevant_failures.append(line)
            # Continue capturing until next test or end of failures
            elif capture and line.strip() and not line.startswith("="):
                relevant_failures.append(line)
            # Stop capturing at test separator
            elif capture and line.startswith("="):
                capture = False
        
        filtered_output = "\n".join(relevant_failures)
        
        prompt = f"""Fix this Python test code based on the failure output:

TEST FILE PATH: {test_file}

CURRENT TEST CODE:
```python
{current_code}
```

TEST FAILURE OUTPUT:
```
{filtered_output}
```

Return only the fixed code, with no explanations.
"""
        print(prompt)
        return prompt
    
    def refine_test_file(self, test_file: Path, test_output: str) -> str:
        """
        Refine a test file based on raw test output.
        
        Args:
            test_file: Path to the test file
            test_output: Raw test output containing failures
            
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
        prompt = self.create_refinement_prompt(test_file, current_code, test_output)
        
        # Generate refined tests
        try:
            print(f"Generating refinements for {test_file}...")
            response = self.openai_client.chat.completions.create(
                model="anthropic/claude-3.7-sonnet",  # High-quality model for test fixing
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            refined_code = response.choices[0].message.content
            
            # Extract code block if present
            if "```python" in refined_code and "```" in refined_code:
                refined_code = refined_code.split("```python")[1].split("```")[0].strip()
            
            return refined_code
        except Exception as e:
            print(f"Error refining tests: {e}")
            return ""
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for test refinement."""
        return """You are a test repair specialist who fixes Python test code based on failure output."""
    
    def save_refined_test(self, test_file: Path, refined_code: str) -> bool:
        """
        Save the refined test file.
        
        Args:
            test_file: Path to the test file
            refined_code: Refined test code
            
        Returns:
            True if saved successfully
        """
        if not refined_code.strip():
            print(f"No refinements generated for {test_file}")
            return False
            
        try:
            # Create backup
            backup_file = test_file.with_suffix(f"{test_file.suffix}.bak")
            with open(backup_file, 'w', encoding='utf-8') as f:
                with open(test_file, 'r', encoding='utf-8') as src:
                    f.write(src.read())
            
            # Save new code
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(refined_code)
            
            print(f"✅ Refined test file: {test_file} (backup saved to {backup_file})")
            return True
        except Exception as e:
            print(f"❌ Error saving refined test: {e}")
            return False


def main():
    """Main entry point for test refinement."""
    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY") 
    if not api_key:
        print("Error: No API key found. Set OPENAI_API_KEY or OPENROUTER_API_KEY environment variable.")
        sys.exit(1)
    
    # Initialize refiner
    refiner = TestRefiner(api_key)
    
    # Run all tests to get failures
    print("Running tests to identify failures...")
    all_test_output = refiner.run_tests()
    
    # Get failing test files
    failing_files = refiner.get_failing_test_files(all_test_output)
    
    if not failing_files:
        print("No failing test files found.")
        return
    
    print(f"Found {len(failing_files)} failing test files to refine:")
    for i, file_path in enumerate(failing_files, 1):
        test_file = Path(file_path)
        if not test_file.exists():
            print(f"⚠️ Test file not found: {test_file}")
            continue
            
        print(f"[{i}/{len(failing_files)}] Refining: {test_file}")
        refined_code = refiner.refine_test_file(test_file, all_test_output)
        
        if refined_code:
            refiner.save_refined_test(test_file, refined_code)
    
    print("\nTest refinement complete! Run tests again to verify fixes.")


if __name__ == "__main__":
    main()