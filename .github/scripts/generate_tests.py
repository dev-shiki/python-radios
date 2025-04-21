#!/usr/bin/env python
"""
Generate test files using generative AI for Python modules to improve code coverage.

This script:
1. Analyzes the current code coverage
2. Extracts uncovered code sections
3. Generates tests using the SambaNova API with Meta-Llama model
4. Validates the generated tests by running them
5. Revises any tests with errors using AI
6. Writes the final validated tests to the appropriate test files
"""

import argparse
import ast
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import openai
import pytest
from coverage import Coverage

# Configure OpenAI client for SambaNova
openai.api_key = os.environ.get("SAMBANOVA_API_KEY")
openai.base_url = "https://api.sambanova.ai/v1"

# Configure the AI model to use
AI_MODEL = "Meta-Llama-3.1-8B-Instruct"

class TestGenerator:
    """Generate tests for Python modules using AI."""

    def __init__(self, module_path: str, coverage_threshold: int = 80):
        """Initialize the test generator.
        
        Args:
            module_path: Path to the module to generate tests for
            coverage_threshold: Target coverage percentage to achieve
        """
        self.module_path = Path(module_path)
        self.coverage_threshold = coverage_threshold
        self.project_root = self._find_project_root()
        
        # Derive module information
        self.module_name = self._get_module_name()
        self.test_file_path = self._get_test_file_path()
        
        # Load module and analyze
        self.module = self._load_module()
        self.module_ast = self._parse_module()
        self.function_signatures = self._extract_function_signatures()
    
    def _find_project_root(self) -> Path:
        """Find the project root directory (containing pyproject.toml or setup.py)."""
        # Start with the absolute path of the module
        current_dir = self.module_path.absolute().parent
        
        # Check if the module path is already absolute and contains project markers
        while current_dir != current_dir.parent:
            if (current_dir / "pyproject.toml").exists() or (current_dir / "setup.py").exists():
                return current_dir
            current_dir = current_dir.parent
        
        # If not found with absolute path, try from the current working directory
        current_dir = Path.cwd()
        while current_dir != current_dir.parent:
            if (current_dir / "pyproject.toml").exists() or (current_dir / "setup.py").exists():
                return current_dir
            current_dir = current_dir.parent
        
        # If no project root is found, use the current directory
        return Path.cwd()
    
    def _get_module_name(self) -> str:
        """Convert file path to importable module name."""
        # Make sure both paths are absolute
        module_abs_path = self.module_path.absolute()
        project_root_abs = self.project_root.absolute()
        
        try:
            # Try to get relative path
            rel_path = module_abs_path.relative_to(project_root_abs)
        except ValueError:
            # If that fails, try to handle relative paths
            try:
                # Try treating module_path as relative to project_root
                potential_abs_path = project_root_abs / self.module_path
                if potential_abs_path.exists():
                    rel_path = self.module_path
                else:
                    # If we can't figure it out, just use the module path name parts
                    rel_path = Path(*self.module_path.parts)
            except Exception:
                # Last resort - just use the file name
                rel_path = Path(self.module_path.name)
        
        module_parts = list(rel_path.parts)
        
        # Handle src directory if present
        if module_parts and module_parts[0] == "src":
            module_parts.pop(0)
        
        # Remove .py extension
        if module_parts and module_parts[-1].endswith('.py'):
            module_parts[-1] = module_parts[-1].replace(".py", "")
        
        return ".".join(module_parts)
    
    def _get_test_file_path(self) -> Path:
        """Determine the location for the test file."""
        module_dir = self.module_path.parent
        filename = f"test_{self.module_path.stem}.py"
        
        # Check if tests directory exists at project root
        tests_dir = self.project_root / "tests"
        if tests_dir.exists() and tests_dir.is_dir():
            # Create matching path structure under tests directory
            rel_path = module_dir.relative_to(self.project_root / "src" if (self.project_root / "src").exists() else self.project_root)
            test_dir = tests_dir / rel_path
            test_dir.mkdir(parents=True, exist_ok=True)
            return test_dir / filename
        
        # Fall back to creating test in the same directory as the module
        return module_dir / filename
    
    def _load_module(self):
        """Import the module dynamically."""
        try:
            # Add project root to system path if not already there
            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))
            
            # If there's a src directory, add it too
            src_dir = self.project_root / "src"
            if src_dir.exists() and str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            
            return importlib.import_module(self.module_name)
        except ImportError as e:
            print(f"Error importing module {self.module_name}: {e}")
            sys.exit(1)
    
    def _parse_module(self) -> ast.Module:
        """Parse the module into an AST."""
        try:
            with open(self.module_path, "r", encoding="utf-8") as f:
                return ast.parse(f.read())
        except Exception as e:
            print(f"Error parsing module {self.module_path}: {e}")
            sys.exit(1)
    
    def _extract_function_signatures(self) -> Dict[str, Dict]:
        """Extract function and class method signatures from the module."""
        signatures = {}
        
        for node in ast.walk(self.module_ast):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                
                # Skip private methods and dunder methods
                if func_name.startswith("_") and not (func_name.startswith("__") and func_name.endswith("__")):
                    continue
                
                # Get function signature
                signatures[func_name] = {
                    "args": self._get_function_args(node),
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "docstring": ast.get_docstring(node) or "",
                }
            
            elif isinstance(node, ast.ClassDef):
                class_name = node.name
                
                for method in node.body:
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_name = method.name
                        
                        # Skip private methods and dunder methods
                        if method_name.startswith("_") and not (method_name.startswith("__") and method_name.endswith("__")):
                            continue
                        
                        # Get method signature
                        full_name = f"{class_name}.{method_name}"
                        signatures[full_name] = {
                            "args": self._get_function_args(method),
                            "is_async": isinstance(method, ast.AsyncFunctionDef),
                            "docstring": ast.get_docstring(method) or "",
                            "class_name": class_name,
                        }
        
        return signatures
    
    def _get_function_args(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
        """Extract function arguments."""
        args = []
        
        for arg in node.args.args:
            args.append(arg.arg)
        
        return args
    
    def get_uncovered_functions(self) -> Dict[str, Dict]:
        """Identify functions with insufficient test coverage."""
        cov = Coverage()
        
        # Load coverage data if it exists
        cov.load()
        
        # Get coverage data for the module
        module_data = cov.get_data().get_file_data(str(self.module_path.resolve()))
        
        if not module_data:
            # If no coverage data, assume all functions need tests
            return self.function_signatures
        
        # Extract uncovered line numbers
        uncovered_lines = set(line for line in module_data['lines'] if line not in module_data['executed_lines'])
        
        # Identify functions with uncovered lines
        uncovered_functions = {}
        for node in ast.walk(self.module_ast):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                
                # Skip private methods
                if func_name.startswith("_") and not (func_name.startswith("__") and func_name.endswith("__")):
                    continue
                
                function_lines = set(range(node.lineno, node.end_lineno + 1))
                
                # If the function has uncovered lines, add it to the list
                if function_lines.intersection(uncovered_lines):
                    if func_name in self.function_signatures:
                        uncovered_functions[func_name] = self.function_signatures[func_name]
            
            elif isinstance(node, ast.ClassDef):
                class_name = node.name
                
                for method in node.body:
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_name = method.name
                        
                        # Skip private methods
                        if method_name.startswith("_") and not (method_name.startswith("__") and method_name.endswith("__")):
                            continue
                        
                        full_name = f"{class_name}.{method_name}"
                        method_lines = set(range(method.lineno, method.end_lineno + 1))
                        
                        # If the method has uncovered lines, add it to the list
                        if method_lines.intersection(uncovered_lines):
                            if full_name in self.function_signatures:
                                uncovered_functions[full_name] = self.function_signatures[full_name]
        
        return uncovered_functions
    
    def generate_tests(self) -> str:
        """Generate tests for uncovered functions using AI."""
        # Get module source code
        with open(self.module_path, "r", encoding="utf-8") as f:
            module_source = f.read()
        
        # Get existing test file content if it exists
        existing_tests = ""
        if self.test_file_path.exists():
            with open(self.test_file_path, "r", encoding="utf-8") as f:
                existing_tests = f.read()
        
        # Get uncovered functions
        uncovered_functions = self.get_uncovered_functions()
        
        if not uncovered_functions:
            print(f"No uncovered functions found in {self.module_path}")
            return existing_tests
        
        print(f"Generating tests for {len(uncovered_functions)} uncovered functions in {self.module_path}")
        
        # Create prompt for the AI
        prompt = self._create_ai_prompt(module_source, existing_tests, uncovered_functions)
        
        # Generate tests using the AI
        generated_tests = self._call_ai(prompt)
        
        # If tests already exist, integrate the new tests
        if existing_tests:
            return self._integrate_new_tests(existing_tests, generated_tests)
        
        return generated_tests
    
    def _create_ai_prompt(self, module_source: str, existing_tests: str, uncovered_functions: Dict[str, Dict]) -> str:
        """Create a prompt for the AI to generate tests."""
        prompt = f"""As an expert Python testing specialist, generate pytest test functions for a Python module. 

MODULE SOURCE CODE:
```python
{module_source}
```

UNCOVERED FUNCTIONS NEEDING TESTS:
"""
        
        for func_name, details in uncovered_functions.items():
            is_async = details.get("is_async", False)
            args = details.get("args", [])
            docstring = details.get("docstring", "")
            class_name = details.get("class_name", None)
            
            prompt += f"\n{'async ' if is_async else ''}function {func_name}({', '.join(args)})"
            if docstring:
                prompt += f"\nDocstring: {docstring}"
            if class_name:
                prompt += f"\nPart of class: {class_name}"
        
        # Add project structure context
        prompt += f"\n\nPROJECT STRUCTURE CONTEXT:"
        prompt += f"\n- Module path: {self.module_path}"
        prompt += f"\n- Test file path: {self.test_file_path}"
        prompt += f"\n- Module name for imports: {self.module_name}"
        
        prompt += "\n\nEXISTING TESTS (if any):\n"
        if existing_tests:
            prompt += f"```python\n{existing_tests}\n```"
        else:
            prompt += "None"
        
        # Include information about project dependencies
        dependencies = []
        try:
            import toml
            pyproject_path = self.project_root / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, "r", encoding="utf-8") as f:
                    pyproject = toml.load(f)
                    if "dependencies" in pyproject.get("tool", {}).get("poetry", {}):
                        dependencies = list(pyproject["tool"]["poetry"]["dependencies"].keys())
                    if "dev-dependencies" in pyproject.get("tool", {}).get("poetry", {}):
                        dependencies.extend(list(pyproject["tool"]["poetry"]["dev-dependencies"].keys()))
        except (ImportError, Exception) as e:
            pass
        
        if dependencies:
            prompt += "\n\nPROJECT DEPENDENCIES:\n" + ", ".join(dependencies)
        
        prompt += """

REQUIREMENTS:
1. Generate pytest test functions with descriptive names that explain what they're testing
2. Use pytest.mark.asyncio for async functions to properly test them
3. For class methods, create appropriate test fixtures that properly mock dependencies
4. Include ALL necessary import statements at the top
5. Create realistic mocks for external dependencies like aiohttp, requests, filesystem, etc.
6. Write comprehensive assertions that verify both happy paths and error cases
7. If HTTP requests are involved, use aresponses or unittest.mock.patch to mock them
8. For database or I/O operations, use appropriate mocking strategies
9. Add type hints to fixture functions for better maintainability
10. Follow the project's existing pattern and style for tests
11. For each test function, test one specific behavior or scenario
12. Ensure all tests are isolated and don't depend on other tests
13. Write tests that specifically target the uncovered functions, lines and branches

CODE FORMAT RULES:
1. First import statements (stdlib, then third-party, then local)
2. Then fixtures (with clear docstrings explaining their purpose)
3. Then test functions (with clear docstrings explaining what they test)
4. Use descriptive variable names and avoid magic numbers
5. Include proper type hints where appropriate
6. Add appropriate parametrize decorators for testing multiple scenarios

EXAMPLE TEST STRUCTURE:
```python
# Imports section
import pytest
from unittest.mock import MagicMock, patch
from module import function_to_test

# Fixtures section
@pytest.fixture
def mock_dependency() -> MagicMock:
    #Create a mocked version of the dependency.
    return MagicMock()

# Test functions section
def test_function_success_scenario(mock_dependency):
    #Test that function_to_test succeeds under normal conditions.
    # Arrange
    mock_dependency.return_value = expected_value
    
    # Act
    result = function_to_test(mock_dependency)
    
    # Assert
    assert result == expected_value
    mock_dependency.assert_called_once()
```

RESULT FORMAT (just the code, no explanations):
```python
# Complete test file with imports, fixtures, and test functions
```
"""
        
        return prompt
    
    def _call_ai(self, prompt: str) -> str:
        """Call the SambaNova API with the prompt and return the generated code."""
        try:
            client = openai.OpenAI(
                api_key=os.environ.get("SAMBANOVA_API_KEY"),
                base_url="https://api.sambanova.ai/v1",
            )
            
            response = client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are an expert Python testing specialist with deep knowledge of pytest, "
                            "unittest.mock, and testing best practices. Your specialty is writing "
                            "comprehensive, effective test suites that achieve high code coverage "
                            "while testing edge cases and maintaining proper isolation."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more deterministic code generation
                top_p=0.1,
                max_tokens=4000,
            )
            
            # Extract code from the response
            content = response.choices[0].message.content
            
            # Extract code block from Markdown
            code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            
            return content.strip()
        
        except Exception as e:
            print(f"Error calling AI API: {e}")
            sys.exit(1)
    
    def _integrate_new_tests(self, existing_tests: str, generated_tests: str) -> str:
        """Integrate newly generated tests with existing tests."""
        # Parse the existing and generated test files
        try:
            existing_ast = ast.parse(existing_tests)
            generated_ast = ast.parse(generated_tests)
        except SyntaxError as e:
            print(f"Error parsing test code: {e}")
            # If we can't parse, just append the generated tests
            return f"{existing_tests}\n\n# AI-generated tests\n{generated_tests}"
        
        # Get existing test function names
        existing_test_names = set()
        for node in ast.walk(existing_ast):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                existing_test_names.add(node.name)
        
        # Extract imports from generated tests that don't exist in existing tests
        existing_imports = set()
        for node in ast.walk(existing_ast):
            if isinstance(node, ast.Import):
                for name in node.names:
                    existing_imports.add(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for name in node.names:
                        existing_imports.add(f"{node.module}.{name.name}")
        
        new_imports = []
        for node in ast.walk(generated_ast):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name not in existing_imports:
                        new_imports.append(ast.unparse(node))
                        break
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    import_needed = False
                    for name in node.names:
                        if f"{node.module}.{name.name}" not in existing_imports:
                            import_needed = True
                            break
                    if import_needed:
                        new_imports.append(ast.unparse(node))
        
        # Extract new test functions from generated tests
        new_functions = []
        for node in ast.walk(generated_ast):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_") and node.name not in existing_test_names:
                new_functions.append(ast.unparse(node))
        
        # Check if there are fixture functions in the generated tests
        fixtures = []
        fixture_names = set()
        for node in ast.walk(generated_ast):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_fixture = False
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == "pytest":
                        is_fixture = True
                        break
                    elif isinstance(decorator, ast.Attribute) and decorator.attr == "fixture":
                        is_fixture = True
                        break
                
                if is_fixture and node.name not in existing_test_names and node.name not in fixture_names:
                    fixtures.append(ast.unparse(node))
                    fixture_names.add(node.name)
        
        # Combine everything
        result = existing_tests.rstrip()
        
        if new_imports:
            result += "\n\n# Additional imports for AI-generated tests\n" + "\n".join(new_imports)
        
        if fixtures:
            result += "\n\n# AI-generated fixtures\n" + "\n\n".join(fixtures)
        
        if new_functions:
            result += "\n\n# AI-generated tests\n" + "\n\n".join(new_functions)
        
        return result
    
    def validate_tests(self, test_code: str) -> Tuple[bool, str, str]:
        """Validate the generated tests by running them.
        
        Args:
            test_code: The test code to validate
            
        Returns:
            A tuple containing (success, error_message, validated_code)
        """
        # Create a temporary directory to run the tests in
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Create a copy of the module in the temp directory
            module_dir = temp_dir_path / self.module_path.parent.name
            module_dir.mkdir(parents=True, exist_ok=True)
            
            temp_module_path = module_dir / self.module_path.name
            with open(self.module_path, "r", encoding="utf-8") as src, open(temp_module_path, "w", encoding="utf-8") as dst:
                dst.write(src.read())
            
            # Create an __init__.py file in the module directory
            init_path = module_dir / "__init__.py"
            init_path.touch()
            
            # Create the test file in the temp directory
            test_dir = temp_dir_path / "tests"
            test_dir.mkdir(parents=True, exist_ok=True)
            
            # Create an __init__.py file in the tests directory
            test_init_path = test_dir / "__init__.py"
            test_init_path.touch()
            
            temp_test_path = test_dir / self.test_file_path.name
            with open(temp_test_path, "w", encoding="utf-8") as f:
                f.write(test_code)
            
            # Add the temp directory to the Python path
            sys.path.insert(0, str(temp_dir_path))
            
            # Run pytest in a subprocess to catch syntax errors and import errors
            try:
                result = subprocess.run(
                    ["pytest", "-xvs", str(temp_test_path)],
                    capture_output=True,
                    text=True,
                    cwd=str(temp_dir_path),
                    timeout=60,  # Timeout after 60 seconds
                )
                
                if result.returncode != 0:
                    return False, result.stderr or result.stdout, test_code
                
                return True, "", test_code
            except subprocess.TimeoutExpired:
                return False, "Tests timed out after 60 seconds", test_code
            except Exception as e:
                return False, str(e), test_code
            finally:
                # Remove the temp directory from the Python path
                sys.path.remove(str(temp_dir_path))
    
    def revise_tests(self, test_code: str, error_message: str) -> str:
        """Revise the tests using AI based on error messages.
        
        Args:
            test_code: The test code to revise
            error_message: The error message from the validation
            
        Returns:
            The revised test code
        """
        prompt = f"""I generated the following pytest test code but it has errors. 
Please fix the issues and provide a corrected version.

ORIGINAL TEST CODE:
```python
{test_code}
```

ERROR MESSAGE:
```
{error_message}
```

Please analyze the error message and fix the issues in the test code. Common problems include:
1. Import errors (missing or incorrect imports)
2. Syntax errors
3. Incorrect test fixtures
4. Issues with mocking
5. Async function handling issues

Provide only the corrected code without explanations.

CORRECTED CODE:
"""
        
        return self._call_ai(prompt)
    
    def write_tests(self) -> None:
        """Generate tests, validate them, revise if needed, and write to the test file."""
        generated_code = self.generate_tests()
        
        # Validate the generated tests
        print("Validating generated tests...")
        success, error_message, validated_code = self.validate_tests(generated_code)
        
        # If validation failed, try to revise the tests
        if not success:
            print(f"Test validation failed with error:\n{error_message}\n\nAttempting to revise tests...")
            revised_code = self.revise_tests(generated_code, error_message)
            
            # Validate the revised tests
            print("Validating revised tests...")
            success, error_message, validated_code = self.validate_tests(revised_code)
            
            if success:
                print("Revised tests validated successfully.")
                generated_code = revised_code
            else:
                print(f"Revised tests still have errors:\n{error_message}\n\nProceeding with best version...")
                # Choose the version that's most likely to be correct
                # This could be enhanced with more sophisticated logic
                generated_code = revised_code  # Usually the revised version is better
        else:
            print("Tests validated successfully.")
        
        # Ensure the parent directory exists
        self.test_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the generated tests to the test file
        with open(self.test_file_path, "w", encoding="utf-8") as f:
            f.write(generated_code)
        
        print(f"Tests written to {self.test_file_path}")


def main():
    """Main function to parse arguments and generate tests."""
    parser = argparse.ArgumentParser(description="Generate tests for Python modules using AI")
    parser.add_argument("--module", "-m", required=True, help="Path to the module to generate tests for")
    parser.add_argument(
        "--coverage-threshold", "-c", type=int, default=80, 
        help="Target coverage percentage to achieve (default: 80)"
    )
    parser.add_argument(
        "--output", "-o", 
        help="Path to write the test file (default: auto-detected based on project structure)"
    )
    
    args = parser.parse_args()
    
    # Validate that the module exists
    module_path = Path(args.module)
    if not module_path.exists():
        print(f"Error: Module {args.module} does not exist")
        sys.exit(1)
    
    # Initialize and run the test generator
    generator = TestGenerator(module_path, args.coverage_threshold)
    
    if args.output:
        generator.test_file_path = Path(args.output)
    
    generator.write_tests()


if __name__ == "__main__":
    main()