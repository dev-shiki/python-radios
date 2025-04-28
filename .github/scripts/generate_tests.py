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
        try:
            # Initialize coverage
            cov = Coverage()
            
            # Try to load existing coverage data
            try:
                cov.load()
            except:
                # If loading fails, return all functions as uncovered
                print("No coverage data found, assuming all functions need tests")
                return self.function_signatures
            
            # Get the absolute path to the module
            module_path_str = str(self.module_path.resolve())
            
            # Get coverage data for the module
            data = cov.get_data()
            
            # Check if the module is in the coverage data
            if module_path_str not in data.measured_files():
                # Module not in coverage data, assume all functions need tests
                print(f"Module {module_path_str} not found in coverage data, assuming all functions need tests")
                return self.function_signatures
            
            # Get line execution information
            line_data = data.lines(module_path_str)
            executed_lines = set(data.lines(module_path_str) or [])
            
            # Get all lines in the file
            with open(module_path_str, 'r') as f:
                total_lines = set(range(1, len(f.readlines()) + 1))
            
            # Calculate uncovered lines
            uncovered_lines = total_lines - executed_lines
            
            # Identify functions with uncovered lines
            uncovered_functions = {}
            for node in ast.walk(self.module_ast):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_name = node.name
                    
                    # Skip private methods that aren't dunder methods
                    if func_name.startswith("_") and not (func_name.startswith("__") and func_name.endswith("__")):
                        continue
                    
                    # Get all lines in the function
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
        except Exception as e:
            print(f"Error getting uncovered functions: {e}")
            # If something goes wrong, return all functions as uncovered
            return self.function_signatures
    
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
        """Create a prompt for the AI to generate tests with improved guidance for common issues."""
        # Extract function information and module context for the prompt
        function_details = self._extract_detailed_function_info(uncovered_functions)
        used_libraries = self._identify_used_libraries(module_source)
        
        prompt = f"""As an expert Python testing specialist, generate pytest test functions for a Python module.

MODULE SOURCE CODE:
```python
{module_source}
```

FUNCTIONS REQUIRING TESTS:
{function_details}

PROJECT CONTEXT:
- Module path: {self.module_path}
- Test file path: {self.test_file_path}
- Module name for imports: {self.module_name}
- Libraries used in module: {', '.join(used_libraries)}

EXISTING TESTS (if any):
"""
    
        if existing_tests:
            prompt += f"```python\n{existing_tests}\n```"
        else:
            prompt += "None"
            

        prompt += """

TEST GENERATION REQUIREMENTS:

CRITICAL CONCEPTS TO UNDERSTAND:
1. JSON HANDLING:
   - When mocking JSON responses, always return valid JSON strings/bytes
   - For orjson, mock responses must be strings or bytes, not None or other types
   - Example: `mock_response.return_value = '{"key": "value"}'`
   - For binary responses: `mock_response.return_value = b'{"key": "value"}'`

2. ASYNC FUNCTIONS:
   - For mocking async functions, use AsyncMock from unittest.mock
   - All mocked async functions must return awaitable objects
   - Return values for async functions must be set using AsyncMock or Future objects
   - Example: `mock_async_func = AsyncMock(return_value=expected_result)`

3. HTTP CLIENT MOCKING:
   - Mock session creation AND request methods
   - For libraries like aiohttp, mock both ClientSession and response methods
   - Set appropriate status codes and content types for mock responses
   - Example for aiohttp:
     ```python
     mock_response = AsyncMock()
     mock_response.status = 200
     mock_response.headers = {"Content-Type": "application/json"}
     mock_response.text.return_value = '{"data": "value"}'
     mock_session.request.return_value = mock_response
     ```

4. FIXTURE USAGE:
   - NEVER call fixtures directly in test code
   - ALWAYS pass fixtures as parameters to test functions
   - Example: `def test_function(mock_dependency):`
   - Fixtures should be defined separately with clear purpose

5. PATCHING:
   - Always patch at the exact import location used by the module under test
   - Incorrect: `@patch('module')`
   - Correct: `@patch('module.submodule.Class.method')`
   - For class methods, patch the class attribute: `@patch.object(Class, 'method')`

6. OBJECT INITIALIZATION:
   - Ensure all objects are properly initialized before testing their methods
   - Set required attributes even if not directly related to the test
   - For nullable attributes that are accessed, provide mock objects

STRUCTURE:
1. First: Import statements (stdlib first, then third-party, then local)
2. Second: Fixture definitions with clear docstrings
3. Third: Test functions with descriptive names and docstrings

TEST PATTERNS TO FOLLOW:
1. For API clients:
   ```python
   @pytest.mark.asyncio
   async def test_api_method(mock_session):
       # Arrange
       client = Client(session=mock_session)
       mock_response = AsyncMock()
       mock_response.text.return_value = '{"result": "success"}'
       mock_response.status = 200
       mock_session.request.return_value = mock_response
       
       # Act
       result = await client.method()
       
       # Assert
       mock_session.request.assert_called_once_with(
           "GET", "https://api.example.com/endpoint",
           headers={"User-Agent": "TestClient"}
       )
       assert result == {"result": "success"}
   ```

2. For classes with dependencies:
   ```python
   @pytest.fixture
   def mock_dependency():
       ##Create a mock dependency for testing.
       return MagicMock()
   
   def test_class_method(mock_dependency):
       # Arrange
       instance = ClassUnderTest(dependency=mock_dependency)
       mock_dependency.method.return_value = "expected"
       
       # Act
       result = instance.method_to_test()
       
       # Assert
       assert result == "expected"
       mock_dependency.method.assert_called_once()
   ```

3. For error handling:
   ```python
   @pytest.mark.asyncio
   async def test_method_error_handling(mock_session):
       # Arrange
       client = Client(session=mock_session)
       mock_session.request.side_effect = Exception("Test error")
       
       # Act & Assert
       with pytest.raises(CustomError) as excinfo:
           await client.method()
       
       assert "Test error" in str(excinfo.value)
   ```

RESULT FORMAT (just the code, no explanations):
```python
# Complete test file with imports, fixtures, and test functions
```
"""
    
        return prompt
    
    def _extract_detailed_function_info(self, uncovered_functions: Dict[str, Dict]) -> str:
        """Extract detailed information about functions to test, including parameters, return types and usage patterns."""
        result = ""
        
        for func_name, details in uncovered_functions.items():
            is_async = details.get("is_async", False)
            args = details.get("args", [])
            docstring = details.get("docstring", "")
            class_name = details.get("class_name", None)
            
            # Analyze function signature
            signature = f"{'async ' if is_async else ''}{func_name}({', '.join(args)})"
            
            # Extract return type and parameter types if available
            return_type = self._extract_return_type(func_name) or "Unknown"
            
            # Get function usage patterns
            usage_patterns = self._analyze_function_usage(func_name)
            
            # Format the function information
            result += f"\n\nFunction: {signature}\n"
            result += f"Return Type: {return_type}\n"
            
            if class_name:
                result += f"Class: {class_name}\n"
            
            if docstring:
                result += f"Description: {docstring}\n"
            
            if usage_patterns:
                result += f"Usage Patterns: {usage_patterns}\n"
            
            # Extract external dependencies
            dependencies = self._extract_dependencies(func_name)
            if dependencies:
                result += f"Dependencies: {', '.join(dependencies)}\n"
        
        return result

    def _extract_return_type(self, func_name: str) -> str:
        """Extract the return type of a function from its signature or annotations."""
        # This is a stub - in a real implementation, this would analyze the AST or use introspection
        return "Unknown"

    def _analyze_function_usage(self, func_name: str) -> str:
        """Analyze how the function is typically used based on its body and other module references."""
        # This is a stub - in a real implementation, this would analyze function call patterns
        return ""

    def _extract_dependencies(self, func_name: str) -> List[str]:
        """Extract external dependencies used by the function."""
        # This is a stub - in a real implementation, this would identify imports and library calls
        return []

    def _identify_used_libraries(self, module_source: str) -> List[str]:
        """Identify the libraries used in the module to help with mocking."""
        libraries = []
        
        # Basic regex pattern to identify imports
        import_pattern = r'import\s+([a-zA-Z0-9_.]+)|from\s+([a-zA-Z0-9_.]+)\s+import'
        
        import_matches = re.findall(import_pattern, module_source)
        for match in import_matches:
            # Get the library name (could be either group)
            lib = match[0] if match[0] else match[1]
            
            # Get the top-level package
            top_level = lib.split('.')[0]
            
            if top_level and top_level not in libraries and not top_level.startswith('_'):
                libraries.append(top_level)
        
        return libraries

    def _generate_test_template(self, func_name: str, details: Dict) -> str:
        """Generate a test template based on function type."""
        is_async = details.get("is_async", False)
        args = details.get("args", [])
        class_name = details.get("class_name", None)
        
        template = ""
        
        # Different templates for different function types
        if is_async:
            if class_name:
                template = self._async_class_method_template(class_name, func_name, args)
            else:
                template = self._async_function_template(func_name, args)
        else:
            if class_name:
                template = self._class_method_template(class_name, func_name, args)
            else:
                template = self._function_template(func_name, args)
        
        return template



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
        """Revise the tests using AI based on error messages with improved error analysis."""
        # Categorize the error to provide better guidance
        error_category = self._categorize_error(error_message)
        guidance = self._get_error_guidance(error_category)

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

ERROR CATEGORY: {error_category}

GUIDANCE TO FIX THIS TYPE OF ERROR:
{guidance}

Please analyze the error message and fix the issues in the test code. Provide only the corrected code without explanations.

CORRECTED CODE:
"""
        
        return self._call_ai(prompt)

    def _categorize_error(self, error_message: str) -> str:
        """Categorize the error type based on the error message to provide targeted guidance."""
        if "JSONDecodeError" in error_message:
            return "JSON_DECODE_ERROR"
        elif "expected call not found" in error_message:
            return "MOCK_ASSERTION_ERROR"
        elif "Need a valid target to patch" in error_message:
            return "PATCH_PATH_ERROR"
        elif "'NoneType' object has no attribute" in error_message:
            return "ATTRIBUTE_ERROR"
        elif "can't be used in 'await' expression" in error_message:
            return "ASYNC_MOCK_ERROR"
        elif "Fixture" in error_message and "called directly" in error_message:
            return "FIXTURE_USAGE_ERROR"
        else:
            return "GENERAL_ERROR"

    def _get_error_guidance(self, error_category: str) -> str:
        """Get targeted guidance for fixing specific error categories."""
        guidance = {
            "JSON_DECODE_ERROR": """
    When mocking JSON responses:
    1. Ensure the mock returns a valid JSON string or bytes
    2. For orjson library, the input must be bytes, bytearray, memoryview, or str
    3. Replace: mock_response.return_value = None
    With: mock_response.return_value = '{"key": "value"}'
    4. Make sure the JSON string is properly formatted
            """,
            
            "MOCK_ASSERTION_ERROR": """
    For mock assertion issues:
    1. Ensure the mock is called with exactly the expected arguments
    2. Check the method signature to verify all required parameters
    3. For positional vs. keyword args, ensure they match the expected call pattern
    4. Use mock.assert_called_once_with() with the exact expected arguments
            """,
            
            "PATCH_PATH_ERROR": """
    For patching errors:
    1. Patch at the exact location where the module is imported, not where it's defined
    2. Use the full import path as used in the module under test
    3. Replace: @patch('module')
    With: @patch('package.module.Class.method')
    4. For class methods, consider using @patch.object(Class, 'method')
            """,
            
            "ATTRIBUTE_ERROR": """
    For NoneType attribute errors:
    1. Ensure objects are properly initialized before methods are called
    2. For nullable attributes, check they are set before use
    3. Provide mock objects for all dependencies accessed in the code
    4. Initialize client sessions or connections before testing methods that use them
            """,
            
            "ASYNC_MOCK_ERROR": """
    For async mocking issues:
    1. Use AsyncMock for mocking async functions
    2. Always ensure mocked async functions return awaitable objects
    3. Replace: mock_func = MagicMock()
    With: mock_func = AsyncMock(return_value=expected_result)
    4. For side effects: mock_func.side_effect = AsyncMock(side_effect=Exception())
            """,
            
            "FIXTURE_USAGE_ERROR": """
    For fixture usage errors:
    1. NEVER call fixtures directly in test code
    2. ALWAYS pass fixtures as parameters to test functions
    3. Replace: result = test_function(fixture())
    With: result = test_function(fixture)
    4. Ensure all fixtures used in a test are included in the test function parameters
            """,
            
            "GENERAL_ERROR": """
    General troubleshooting:
    1. Check import statements to ensure all required modules are imported
    2. Verify type hints and function signatures match the expected usage
    3. Ensure all variables are defined before they are used
    4. Review the module structure and dependencies to ensure correct mocking
            """
        }
    
        return guidance.get(error_category, guidance["GENERAL_ERROR"])
    
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