#!/usr/bin/env python
"""
AI-powered test refiner that improves tests based on raw test failure output.
"""

import json
import time
import os
import subprocess
import sys
import re  # Added missing import
from pathlib import Path

import openai


class TestRefiner:
    """Refine generated tests by passing raw test failure output to AI."""
    
    def __init__(self, api_key: str):
        """Initialize the refiner."""
        self.api_key = api_key
        self.openai_client = self._setup_openai()
        self.refinement_log = {
            "start_time": time.time(),
            "refinements": [],
            "total_attempts": 0,
            "successful_refinements": 0
        }
    
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
        cmd = ["pytest", "-v", "-s", "--tb=short"]  # Added -s and --tb=short for better output
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
    
    def extract_source_imports(self, test_code: str) -> list:
        """
        Extract imported modules from test code.
        
        Args:
            test_code: Test file content
            
        Returns:
            List of imported module paths
        """
        imports = []
        
        # Pattern for from X import Y
        from_import_pattern = r'from\s+([a-zA-Z0-9_.]+)\s+import'
        # Pattern for import X
        import_pattern = r'^import\s+([a-zA-Z0-9_.]+)'
        
        for line in test_code.split('\n'):
            from_match = re.match(from_import_pattern, line)
            if from_match:
                module = from_match.group(1)
                if not module.startswith('.'):  # Skip relative imports
                    imports.append(module)
            
            import_match = re.match(import_pattern, line)
            if import_match:
                module = import_match.group(1)
                imports.append(module)
        
        return imports
    
    def get_source_file_content(self, module_path: str) -> str:
        """
        Get source file content for a module.
        
        Args:
            module_path: Module path (e.g., 'radios.radio_browser')
            
        Returns:
            Source file content or empty string
        """
        # Try different source locations
        possible_paths = [
            f"src/{module_path.replace('.', '/')}.py",
            f"{module_path.replace('.', '/')}.py",
            f"lib/{module_path.replace('.', '/')}.py"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        return f"# Source: {path}\n{f.read()}"
                except Exception as e:
                    print(f"Error reading {path}: {e}")
        
        return ""
    
    def extract_full_stack_trace(self, test_output: str, test_file_name: str) -> str:
        """
        Extract full stack trace for a specific test file.
        
        Args:
            test_output: Full test output
            test_file_name: Name of the test file
            
        Returns:
            Full stack trace including all context
        """
        lines = test_output.split('\n')
        relevant_lines = []
        capture = False
        stack_trace_started = False
        
        for i, line in enumerate(lines):
            # Start capturing on test failure
            if test_file_name in line and "FAILED" in line:
                capture = True
                relevant_lines.append(line)
                continue
            
            if capture:
                # Include all lines until next test or separator
                if line.startswith("=") and len(line) > 20:
                    if stack_trace_started:
                        capture = False
                else:
                    relevant_lines.append(line)
                    if not stack_trace_started and (
                        "Traceback" in line or 
                        ".py:" in line or 
                        "E   " in line or
                        ">" in line
                    ):
                        stack_trace_started = True
        
        return "\n".join(relevant_lines)
    
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
        # Get full stack trace with context
        file_name = test_file.name
        full_stack_trace = self.extract_full_stack_trace(test_output, file_name)
        
        # Extract source files being tested
        imported_modules = self.extract_source_imports(current_code)
        source_contents = []
        
        for module in imported_modules:
            content = self.get_source_file_content(module)
            if content:
                source_contents.append(content)
        
        # Combine source contents
        source_context = "\n\n".join(source_contents) if source_contents else "# No source files found"
        
        prompt = f"""Fix the failing Python test based on the error output below.

TEST FILE PATH: {test_file}

CURRENT TEST CODE:
```python
{current_code}
```

TEST FAILURE OUTPUT:
```
{full_stack_trace}
```

SOURCE CODE CONTEXT:
```python
{source_context}
```

IMPORTANT GUIDELINES:
1. Fix ALL errors shown in the output
2. Maintain the original test structure and intent
3. Consider issues with:
   - Mocking (correct return values and assertions)
   - Data structures (field requirements, ordering)
   - Asynchronous code (proper awaiting and async patterns)
   - URL handling (string vs object comparisons)
   - Type compatibility (expected vs actual types)
4. Return ONLY the complete fixed test code with no explanations
5. Make minimal changes necessary to fix the failing tests

The fixed code should pass when executed with pytest.
"""
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
        refinement_start = time.time()
        # Read current test code
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                current_code = f.read()
        except Exception as e:
            print(f"Error reading {test_file}: {e}")
            return ""
        
        # Create refinement prompt
        prompt = self.create_refinement_prompt(test_file, current_code, test_output)
        
        refinement_record = {
            "file": str(test_file),
            "timestamp": refinement_start,
            "test_failures": len(re.findall(r'FAILED', test_output)),
            "prompt_length": len(prompt)
        }
        
        # Generate refined tests
        try:
            response = self.openai_client.chat.completions.create(
                model="anthropic/claude-3.7-sonnet",
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            refined_code = response.choices[0].message.content
            
            # Clean the refined code similar to generate_test2.py
            refined_code = self._clean_generated_code(refined_code)
            
            # Update refinement record
            refinement_record.update({
                "success": True,
                "response_length": len(refined_code),
                "refinement_time": time.time() - refinement_start
            })
            
            # Add token usage if available
            if hasattr(response, 'usage'):
                refinement_record["token_usage"] = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            self.refinement_log["successful_refinements"] += 1
            
        except Exception as e:
            refinement_record.update({
                "success": False,
                "error": str(e),
                "refinement_time": time.time() - refinement_start
            })
            refined_code = ""
            print(f"Error during refinement: {e}")
        
        self.refinement_log["refinements"].append(refinement_record)
        self.refinement_log["total_attempts"] += 1
        
        return refined_code
    
    def _clean_generated_code(self, code: str) -> str:
        """
        Clean generated code by removing Markdown artifacts and other non-Python syntax.
        
        Args:
            code: The code generated by the AI
            
        Returns:
            Cleaned Python code
        """
        # Remove markdown code block delimiters if present
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'```\s*', '', code)
        
        # Remove any leading/trailing whitespace
        code = code.strip()
        
        # Check if first non-empty line looks like an import statement or function/class definition
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if non_empty_lines and not (
            non_empty_lines[0].startswith('import ') or 
            non_empty_lines[0].startswith('from ') or
            non_empty_lines[0].startswith('def ') or
            non_empty_lines[0].startswith('class ') or
            non_empty_lines[0].startswith('#') or
            non_empty_lines[0].startswith('@')
        ):
            # Try to find the start of actual Python code
            for i, line in enumerate(non_empty_lines):
                if (line.startswith('import ') or 
                    line.startswith('from ') or 
                    line.startswith('def ') or
                    line.startswith('class ')):
                    # Found the start of code, remove everything before it
                    code = '\n'.join(lines[lines.index(line):])
                    break
        
        # Remove any "Output:" or similar text at the end
        code = re.sub(r'\nOutput:.*$', '', code, flags=re.DOTALL)
        
        return code
    
    def save_refinement_log(self):
        """Save refinement log to file"""
        self.refinement_log["end_time"] = time.time()
        self.refinement_log["total_time"] = self.refinement_log["end_time"] - self.refinement_log["start_time"]
        
        with open('refinement_log.json', 'w') as f:
            json.dump(self.refinement_log, f, indent=2)

    def _get_system_prompt(self) -> str:
        """Get system prompt for test refinement."""
        return """You are an expert Python test engineer who specializes in fixing failing pytest tests. Your task is to analyze test failure outputs and produce corrected test code that passes successfully. You understand common testing patterns, mocking techniques, and framework-specific behaviors.

Provide complete, working code that addresses all test failures while maintaining the original test's intent and coverage. Return only the fixed code without explanations unless specifically requested.
"""
    
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
        print("Error: No API key found. Set OPENROUTER_API_KEY environment variable.")
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
    
    refiner.save_refinement_log()

    print("\nTest refinement complete! Run tests again to verify fixes.")


if __name__ == "__main__":
    main()