#!/usr/bin/env python
"""
Generate test files using generative AI for Python modules to improve code coverage.

This script:
1. Analyzes the module to extract class and function signatures
2. Generates appropriate test fixtures that properly handle constructor requirements
3. Creates test cases for each function and method
4. Saves the generated tests to the appropriate test file
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
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import openai

# Configure OpenAI client for SambaNova
openai.api_key = os.environ.get("SAMBANOVA_API_KEY")
openai.base_url = "https://api.sambanova.ai/v1"

# Configure the AI model to use
AI_MODEL = "Meta-Llama-3.3-70B-Instruct"

class TestGenerator:
    """Generate tests for Python modules using AI."""

    def __init__(self, module_path: str, coverage_threshold: int = 80):
        """Initialize the test generator.
        
        Args:
            module_path: Path to the module to generate tests for
            coverage_threshold: Target coverage percentage to achieve
        """
        # Convert to Path and resolve to absolute path
        self.module_path = Path(module_path).resolve()
        self.coverage_threshold = coverage_threshold
        self.project_root = self._find_project_root()
        
        # Derive module information
        self.module_name = self._get_module_name()
        self.test_file_path = self._get_test_file_path()
        
        # Parse module
        self.module_ast = self._parse_module()
        
        # Extract module details
        self.imports = self._extract_imports()
        self.class_signatures = self._extract_class_signatures()
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
        # Start with the module path relative to the project root
        try:
            rel_path = self.module_path.relative_to(self.project_root)
        except ValueError:
            # Fallback to just using the module name
            rel_path = Path(self.module_path.name)
        
        parts = list(rel_path.parts)
        
        # Handle src directory
        if parts and parts[0] == "src":
            parts.pop(0)
        
        # Remove .py extension
        if parts and parts[-1].endswith('.py'):
            parts[-1] = parts[-1][:-3]  # Remove .py extension
        
        return ".".join(parts)
    
    def _get_test_file_path(self) -> Path:
        """Determine the location for the test file."""
        filename = f"test_{self.module_path.stem}.py"
        
        # Look for tests directory
        tests_dir = self.project_root / "tests"
        if tests_dir.exists() and tests_dir.is_dir():
            # Check if the module is in a src directory
            src_path = self.project_root / "src"
            if src_path.exists() and str(self.module_path).startswith(str(src_path)):
                # Create matching structure under tests
                try:
                    rel_path = self.module_path.parent.relative_to(src_path)
                    test_dir = tests_dir / rel_path
                    test_dir.mkdir(parents=True, exist_ok=True)
                    return test_dir / filename
                except Exception:
                    pass
            
            # Fallback - put the test directly in the tests directory
            return tests_dir / filename
        
        # If no tests directory is found, create the test in the same directory as the module
        return self.module_path.parent / filename
        
    def _parse_module(self) -> ast.Module:
        """Parse the module into an AST."""
        try:
            with open(self.module_path, "r", encoding="utf-8") as f:
                return ast.parse(f.read())
        except Exception as e:
            print(f"Error parsing module {self.module_path}: {e}")
            sys.exit(1)
    
    def _extract_imports(self) -> List[str]:
        """Extract import statements from the module."""
        imports = []
        
        for node in ast.walk(self.module_ast):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(f"import {name.name}")
            elif isinstance(node, ast.ImportFrom):
                names_str = ", ".join(n.name for n in node.names)
                imports.append(f"from {node.module} import {names_str}")
        
        return imports
    
    def _extract_class_signatures(self) -> Dict[str, Dict[str, Any]]:
        """Extract class definitions and their constructor signatures."""
        classes = {}
        
        for node in ast.walk(self.module_ast):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                init_method = None
                
                # Find the __init__ method
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        init_method = item
                        break
                
                if init_method:
                    required_args = []
                    optional_args = {}
                    
                    # Skip 'self'
                    for arg_idx, arg in enumerate(init_method.args.args[1:], start=0):
                        # Calculate default index - offset by the number of args without defaults
                        default_idx = arg_idx - (len(init_method.args.args) - 1 - len(init_method.args.defaults))
                        
                        if default_idx >= 0:  # Has default
                            # Try to extract default value
                            default_value = None
                            try:
                                default_node = init_method.args.defaults[default_idx]
                                if isinstance(default_node, ast.Constant):
                                    default_value = default_node.value
                                elif isinstance(default_node, ast.Name):
                                    default_value = default_node.id
                                # Add more types as needed
                            except:
                                default_value = None
                            
                            optional_args[arg.arg] = default_value
                        else:
                            required_args.append(arg.arg)
                    
                    # Extract base classes
                    base_classes = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            base_classes.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            base_classes.append(f"{base.value.id}.{base.attr}")
                    
                    # Get class docstring
                    docstring = ast.get_docstring(node) or ""
                    
                    classes[class_name] = {
                        "required_args": required_args,
                        "optional_args": optional_args,
                        "base_classes": base_classes,
                        "docstring": docstring
                    }
                else:
                    # Class without an __init__ method
                    classes[class_name] = {
                        "required_args": [],
                        "optional_args": {},
                        "base_classes": [],
                        "docstring": ast.get_docstring(node) or ""
                    }
        
        return classes
    
    def _extract_function_signatures(self) -> Dict[str, Dict[str, Any]]:
        """Extract function and method signatures from the module."""
        functions = {}
        
        # First, build a lookup to identify nodes inside class definitions
        class_nodes = {}
        for node in ast.walk(self.module_ast):
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    class_nodes[child] = node
        
        # Now find top-level functions (not inside classes)
        for node in ast.walk(self.module_ast):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node not in class_nodes:
                func_name = node.name
                
                # Skip private functions and dunder methods that aren't meant to be directly called
                if func_name.startswith("_") and not (func_name.startswith("__") and func_name.endswith("__")):
                    continue
                
                # Extract function parameters
                params = []
                for arg in node.args.args:
                    params.append(arg.arg)
                
                functions[func_name] = {
                    "params": params,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "docstring": ast.get_docstring(node) or ""
                }
        
        # Now extract class methods
        for class_name, class_info in self.class_signatures.items():
            for node in ast.walk(self.module_ast):
                if (isinstance(node, ast.ClassDef) and node.name == class_name):
                    for method in node.body:
                        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_name = method.name
                            
                            # Skip private methods and __init__
                            if (method_name.startswith("_") and 
                                not (method_name.startswith("__") and method_name.endswith("__"))) or method_name == "__init__":
                                continue
                            
                            # Extract method parameters
                            params = []
                            for arg in method.args.args[1:]:  # Skip self
                                params.append(arg.arg)
                            
                            full_name = f"{class_name}.{method_name}"
                            functions[full_name] = {
                                "params": params,
                                "is_async": isinstance(method, ast.AsyncFunctionDef),
                                "docstring": ast.get_docstring(method) or "",
                                "class_name": class_name
                            }
        
        return functions
    
    def load_existing_tests(self) -> str:
        """Load existing test file if it exists."""
        if self.test_file_path.exists():
            with open(self.test_file_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""
    
    def generate_tests(self) -> str:
        """Generate tests for the module using AI."""
        # Get module source code
        with open(self.module_path, "r", encoding="utf-8") as f:
            module_source = f.read()
        
        # Get existing test file content if it exists
        existing_tests = self.load_existing_tests()
        
        # Create prompt for the AI
        prompt = self._create_ai_prompt(module_source, existing_tests)
        
        # Generate tests using the AI
        generated_tests = self._call_ai(prompt)
        
        # If tests already exist, integrate the new tests
        if existing_tests:
            return self._integrate_new_tests(existing_tests, generated_tests)
        
        return generated_tests
    
    def _create_ai_prompt(self, module_source: str, existing_tests: str) -> str:
        """Create a prompt for the AI to generate tests with detailed guidance."""
        # Format class signatures for the prompt
        class_signatures_str = json.dumps(self.class_signatures, indent=2)
        
        # Format function signatures for the prompt
        function_signatures_str = json.dumps(self.function_signatures, indent=2)
        
        prompt = f"""As an expert Python testing specialist, generate pytest test functions for a Python module.

MODULE SOURCE CODE:
```python
{module_source}
```

CLASS SIGNATURES:
```json
{class_signatures_str}
```

FUNCTION SIGNATURES:
```json
{function_signatures_str}
```

MODULE NAME: {self.module_name}
MODULE PATH: {self.module_path}
TEST FILE PATH: {self.test_file_path}

EXISTING TESTS (if any):
"""
        
        if existing_tests:
            prompt += f"```python\n{existing_tests}\n```"
        else:
            prompt += "None"
        
        prompt += """

TEST GENERATION REQUIREMENTS:

1. REQUIRED IMPORTS:
   - Always include these imports for testing async code:
     ```python
     import pytest
     from unittest.mock import AsyncMock, MagicMock, patch
     ```
   - Never use `asyncio.Mock` or similar - always use `unittest.mock.AsyncMock`

2. FIXTURE CREATION:
   - Create fixtures for all dependencies needed in tests
   - For classes, create fixtures that properly initialize all required arguments
   - For RadioBrowser and similar classes, ALWAYS include user_agent and other required parameters
   - Example for RadioBrowser:
     ```python
     @pytest.fixture
     def mock_session():
         # Create a mock session.
         return MagicMock()
         
     @pytest.fixture
     def radio_browser(mock_session):
         # Return a RadioBrowser instance.
         return RadioBrowser(user_agent="TestAgent", session=mock_session)
     ```

3. ASYNC TESTING:
   - Use @pytest.mark.asyncio for async test functions
   - Use AsyncMock() from unittest.mock for mocking async functions and methods
   - For async HTTP clients, make sure response methods are properly mocked
   - Example:
     ```python
     @pytest.mark.asyncio
     async def test_async_method(radio_browser, mock_session):
         # Setup
         mock_response = AsyncMock()
         mock_response.text.return_value = '{"result": "success"}'
         mock_session.request.return_value = mock_response
         
         # Test
         result = await radio_browser.method()
         
         # Assert
         assert result is not None
     ```

4. MOCKING STRATEGY:
   - Use mock_session.request.return_value = AsyncMock() for HTTP responses
   - Always set all required attributes on mocked responses (status, headers, text, etc.)
   - When mocking JSON responses, use proper string values: '{"key": "value"}'
   - For binary responses: `mock_response.return_value = b'{"key": "value"}'`

5. TEST COVERAGE:
   - Include tests for success and error cases
   - Test edge cases like empty responses or error handling
   - Verify correct parameters are passed to dependencies

STRUCTURE OF THE TEST FILE:
1. Import statements (ALWAYS include unittest.mock import, NEVER use asyncio.Mock)
2. Fixtures (setup, mocks, etc.)
3. Tests for module functions
4. Tests for each class and its methods

RESULT FORMAT:
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
                            "comprehensive, effective test suites that achieve high code coverage. "
                            "IMPORTANT: You know to ALWAYS use unittest.mock.AsyncMock for mocking "
                            "async functions, never asyncio.Mock which doesn't exist."
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
                # Return ONLY the code inside the code block, without the ```python and ``` markers
                return code_match.group(1).strip()
            
            # If no code block is found, return the content as is, but ensure it doesn't have markdown code markers
            cleaned_content = re.sub(r"```python|```", "", content)
            return cleaned_content.strip()
        
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
    
    def validate_tests(self, test_code: str) -> Tuple[bool, str]:
        """Validate the generated tests by running a syntax check."""
        try:
            ast.parse(test_code)
            return True, ""
        except SyntaxError as e:
            return False, str(e)
    
    def revise_tests(self, test_code: str, error_message: str) -> str:
        """Revise the tests using AI based on error messages."""
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

Please analyze the error message and fix the issues in the test code. Provide only the corrected code without explanations.

IMPORTANT NOTES:
1. Always use unittest.mock.AsyncMock for mocking async functions, NEVER use asyncio.Mock which doesn't exist.
2. Make sure all imports are correct, especially importing AsyncMock and MagicMock from unittest.mock.
3. Fix any other syntax or import errors found in the code.

CORRECTED CODE:
"""
        
        return self._call_ai(prompt)
    
    def check_for_mock_errors(self, test_code: str) -> Tuple[bool, str]:
        """Check for common mocking errors in the test code."""
        errors = []
        
        # Check for asyncio.Mock which doesn't exist
        if "asyncio.Mock" in test_code:
            errors.append("Found 'asyncio.Mock' in the code, which doesn't exist. Use unittest.mock.AsyncMock instead.")
        
        # Check for incorrect imports of AsyncMock
        if "AsyncMock" in test_code and "from unittest.mock import" not in test_code and "import unittest.mock" not in test_code:
            errors.append("Using AsyncMock without importing it from unittest.mock.")
        
        if errors:
            return False, "\n".join(errors)
        return True, ""
    
    def write_tests(self) -> None:
        """Generate tests, validate them, revise if needed, and write to the test file."""
        print(f"Generating tests for {self.module_path}")
        generated_code = self.generate_tests()
        
        # Check for common mock errors
        mock_success, mock_errors = self.check_for_mock_errors(generated_code)
        if not mock_success:
            print(f"Mock errors detected:\n{mock_errors}\n\nAttempting to revise tests...")
            generated_code = self.revise_tests(generated_code, mock_errors)
            # Check again
            mock_success, mock_errors = self.check_for_mock_errors(generated_code)
            if not mock_success:
                print(f"Warning: Mock errors still present after revision: {mock_errors}")
        
        # Validate the generated tests
        success, error_message = self.validate_tests(generated_code)
        
        # If validation failed, try to revise the tests
        if not success:
            print(f"Test validation failed with error:\n{error_message}\n\nAttempting to revise tests...")
            revised_code = self.revise_tests(generated_code, error_message)
            
            # Validate the revised tests
            success, error_message = self.validate_tests(revised_code)
            
            if success:
                print("Revised tests validated successfully.")
                generated_code = revised_code
            else:
                print(f"Revised tests still have errors:\n{error_message}\n\nProceeding with best version...")
                # Choose the version that's most likely to be correct
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