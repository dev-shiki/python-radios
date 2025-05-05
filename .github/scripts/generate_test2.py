#!/usr/bin/env python
"""
AI-powered test generator that works with any Python project structure.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import openai
import pytest


class UniversalTestGenerator:
    """Simple, universal test generator for any Python project."""
    
    def __init__(self, 
                 api_key: str,
                 coverage_threshold: float = 80.0):
        """Initialize with minimal configuration."""
        self.api_key = api_key
        self.coverage_threshold = coverage_threshold
        self.openai_client = self._setup_openai()
    
    def _setup_openai(self):
        """Configure OpenAI client for various providers."""
        # Support multiple providers by checking API key pattern
        if "openrouter" in self.api_key.lower() or os.getenv("OPENROUTER_API_KEY"):
            return openai.OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        # Default to OpenAI
        return openai.OpenAI(api_key=self.api_key)
    
    def find_files_needing_tests(self, 
                               coverage_data: Dict = None,
                               target_files: List[str] = None) -> List[Path]:
        """
        Find Python files that need tests.
        
        Args:
            coverage_data: Coverage data from pytest-cov
            target_files: Specific files to target
        
        Returns:
            List of Python file paths that need tests
        """
        files_to_test = []
        
        # If specific files are requested
        if target_files:
            for file_path in target_files:
                path = Path(file_path)
                if path.exists() and path.suffix == '.py':
                    files_to_test.append(path)
            return files_to_test
        
        # Otherwise, find files based on coverage data
        if coverage_data:
            # Extract files with low coverage
            for file_path, data in coverage_data.get('files', {}).items():
                coverage_pct = data.get('summary', {}).get('percent_covered', 0)
                if coverage_pct < self.coverage_threshold:
                    files_to_test.append(Path(file_path))
        else:
            # Fallback: find all Python files
            files_to_test = list(Path('.').rglob('*.py'))
            # Exclude common directories
            files_to_test = [f for f in files_to_test 
                           if not any(part.startswith('.') or part == '__pycache__' 
                                    or 'test' in part for part in f.parts)]
        
        return files_to_test
    
    def generate_test_for_file(self, file_path: Path) -> str:
        """
        Generate tests for a single Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Generated test code
        """
        # Read the source code
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
        
        # Create prompt
        prompt = self._create_prompt(file_path, source_code)
        
        # Generate tests
        try:
            response = self.openai_client.chat.completions.create(
                model="google/gemini-2.0-flash-001",  # Or your preferred model
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating tests for {file_path}: {e}")
            return ""
    
    def _get_system_prompt(self) -> str:
        """Return the system prompt for the AI model."""
        return """You are an expert Python test generator. Your task is to:
1. Generate comprehensive pytest test cases for the provided code
2. Include imports, fixtures, and test functions
3. Test both happy paths and edge cases
4. Use proper mocking for external dependencies
5. Follow pytest best practices
6. Return only the test code without explanations"""
    
    def _create_prompt(self, file_path: Path, source_code: str) -> str:
        """Create a prompt for generating tests."""
        return f"""Generate pytest test cases for this Python file:

FILE: {file_path}

SOURCE CODE:
```python
{source_code}
```

Generate a complete test file with:
1. All necessary imports
2. Test fixtures where needed
3. Test functions for each function/method
4. Edge cases and error conditions
5. Proper mocking for external dependencies

Return only the test code without explanations."""
    
    def save_test_file(self, source_file: Path, test_code: str) -> Path:
        """
        Save the generated test file.
        
        Args:
            source_file: Original Python file
            test_code: Generated test code
            
        Returns:
            Path to the saved test file
        """
        # Determine test file path
        test_path = self._get_test_path(source_file)
        
        # Create directories if needed
        test_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the test file
        try:
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
            print(f"Saved test file: {test_path}")
            return test_path
        except Exception as e:
            print(f"Error saving test file: {e}")
            return None
    
    def _get_test_path(self, source_file: Path) -> Path:
        """Determine where to save the test file."""
        # Try to find tests directory
        current_dir = Path.cwd()
        tests_dir = current_dir / 'tests'
        
        if not tests_dir.exists():
            tests_dir = current_dir / 'test'
        
        if not tests_dir.exists():
            tests_dir = current_dir
        
        # Create test filename
        test_filename = f"test_{source_file.stem}.py"
        
        # Try to match directory structure
        try:
            rel_path = source_file.relative_to(current_dir)
            if len(rel_path.parts) > 1:
                # Create directory structure in tests
                subdir = tests_dir / Path(*rel_path.parts[:-1])
                subdir.mkdir(parents=True, exist_ok=True)
                return subdir / test_filename
        except ValueError:
            pass
        
        return tests_dir / test_filename


def main():
    """Main entry point."""
    # Get configuration from environment
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY")
        sys.exit(1)
    
    coverage_threshold = float(os.getenv("COVERAGE_THRESHOLD", "80"))
    target_files_str = os.getenv("TARGET_FILES", "")
    target_files = [f.strip() for f in target_files_str.split(",") if f.strip()]
    
    # Initialize generator
    generator = UniversalTestGenerator(api_key, coverage_threshold)
    
    # Load coverage data if available
    coverage_data = None
    if Path("coverage-initial.json").exists():
        try:
            with open("coverage-initial.json", "r") as f:
                coverage_data = json.load(f)
        except Exception as e:
            print(f"Error loading coverage data: {e}")
    
    # Find files to test
    files_to_test = generator.find_files_needing_tests(coverage_data, target_files)
    
    if not files_to_test:
        print("No files found to generate tests for")
        return
    
    print(f"Generating tests for {len(files_to_test)} files...")
    
    # Generate tests for each file
    for file_path in files_to_test:
        print(f"Processing: {file_path}")
        test_code = generator.generate_test_for_file(file_path)
        
        if test_code:
            generator.save_test_file(file_path, test_code)
    
    print("Test generation complete!")


if __name__ == "__main__":
    main()