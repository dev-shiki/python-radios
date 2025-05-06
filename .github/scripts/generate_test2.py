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
        return """You are a master Python test generator. Your expertise:
- Write concise yet comprehensive pytest tests
- Mock external dependencies perfectly  
- Test edge cases and error paths
- Use correct async/await patterns
- Name tests clearly and descriptively

Always deliver production-ready, minimal test code that achieves maximum coverage."""

    def _get_function_args(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[Dict]:
        """Extract function arguments with their types and defaults."""
        args = []
        
        # Regular arguments
        for i, arg in enumerate(node.args.args):
            arg_info = {
                "name": arg.arg,
                "type": ast.unparse(arg.annotation) if arg.annotation else None,
                "default": None
            }
            
            # Check for default values
            defaults_start = len(node.args.args) - len(node.args.defaults)
            if i >= defaults_start:
                default_idx = i - defaults_start
                arg_info["default"] = ast.unparse(node.args.defaults[default_idx])
            
            args.append(arg_info)
        
        # *args
        if node.args.vararg:
            args.append({
                "name": f"*{node.args.vararg.arg}",
                "type": "*args",
                "default": None
            })
        
        # **kwargs
        if node.args.kwarg:
            args.append({
                "name": f"**{node.args.kwarg.arg}",
                "type": "**kwargs",
                "default": None
            })
        
        return args

    def _extract_return_type(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[str]:
        """Extract return type annotation from function definition."""
        if node.returns:
            return ast.unparse(node.returns)
        return None

    def _extract_uncovered_functions(source_code: str) -> Dict[str, Dict]:
        """Extract functions that need testing from source code."""
        uncovered_functions = {}
        
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_name = node.name
                    
                    # Skip private functions (but keep dunder methods)
                    if func_name.startswith("_") and not (func_name.startswith("__") and func_name.endswith("__")):
                        continue
                    
                    uncovered_functions[func_name] = {
                        "args": _get_function_args(node),
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                        "docstring": ast.get_docstring(node) or "",
                        "return_type": _extract_return_type(node),
                        "line_no": node.lineno
                    }
                
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    for method in node.body:
                        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_name = method.name
                            
                            # Skip private methods
                            if method_name.startswith("_") and not (method_name.startswith("__") and method_name.endswith("__")):
                                continue
                            
                            full_name = f"{class_name}.{method_name}"
                            uncovered_functions[full_name] = {
                                "args": _get_function_args(method),
                                "is_async": isinstance(method, ast.AsyncFunctionDef),
                                "docstring": ast.get_docstring(method) or "",
                                "class_name": class_name,
                                "return_type": _extract_return_type(method),
                                "line_no": method.lineno
                            }
        except Exception as e:
            print(f"Error extracting functions: {e}")
        
        return uncovered_functions

    def _extract_model_structure(source_code: str) -> Dict[str, Dict[str, Dict]]:
        """Extract dataclass/model structures from the code."""
        models = {}
        
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it's a dataclass or similar model
                    is_dataclass = any(
                        (isinstance(d, ast.Name) and d.id in ["dataclass", "dataclasses", "pydantic"]) or
                        (isinstance(d, ast.Call) and 
                        isinstance(d.func, (ast.Name, ast.Attribute)) and 
                        any(name in ast.unparse(d.func) for name in ["dataclass", "BaseModel"]))
                        for d in node.decorator_list
                    )
                    
                    if is_dataclass:
                        fields = {}
                        for item in node.body:
                            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                                field_name = item.target.id
                                field_type = ast.unparse(item.annotation)
                                
                                default_value = None
                                required = True
                                
                                if item.value:
                                    if isinstance(item.value, ast.Call) and isinstance(item.value.func, ast.Name):
                                        if item.value.func.id == 'field':
                                            # Parse field options
                                            for keyword in item.value.keywords:
                                                if keyword.arg == 'default':
                                                    default_value = ast.unparse(keyword.value)
                                                    required = False
                                                elif keyword.arg == 'default_factory':
                                                    default_value = f"factory: {ast.unparse(keyword.value)}"
                                                    required = False
                                    else:
                                        default_value = ast.unparse(item.value)
                                        required = False
                                
                                fields[field_name] = {
                                    "type": field_type,
                                    "required": required,
                                    "default": default_value
                                }
                        
                        models[node.name] = fields
        except Exception as e:
            print(f"Error extracting models: {e}")
        
        return models

    def _extract_detailed_function_info(uncovered_functions: Dict[str, Dict]) -> str:
        """Create detailed text description of functions needing tests."""
        details = []
        
        for func_name, info in uncovered_functions.items():
            desc = f"\nFunction: {func_name}"
            
            if info.get("is_async"):
                desc += " (async)"
            
            desc += f"\nLine: {info.get('line_no', 'unknown')}"
            
            # Arguments
            args = info.get("args", [])
            if args:
                desc += "\nArguments:"
                for arg in args:
                    arg_desc = f"  - {arg['name']}"
                    if arg.get('type'):
                        arg_desc += f": {arg['type']}"
                    if arg.get('default'):
                        arg_desc += f" = {arg['default']}"
                    desc += f"\n{arg_desc}"
            
            # Return type
            if info.get("return_type"):
                desc += f"\nReturns: {info['return_type']}"
            
            # Docstring
            if info.get("docstring"):
                desc += f"\nDocstring: {info['docstring']}"
            
            details.append(desc)
        
        return "\n".join(details)

    def _generate_mock_value_for_type(field_type: str) -> str:
        """Generate appropriate mock values based on type hints."""
        field_type = field_type.lower()
        
        # Handle basic types
        type_mapping = {
            'str': '"mock_string"',
            'int': '42',
            'float': '42.0',
            'bool': 'true',
            'dict': '{"key": "value"}',
            'list': '["item1", "item2"]',
            'set': '["item1", "item2"]',
            'tuple': '["item1", "item2"]',
            'datetime': '"2024-01-01T00:00:00"',
            'uuid': '"123e4567-e89b-12d3-a456-426614174000"',
            'path': '"/path/to/file"',
            'pathlib.path': '"/path/to/file"',
            'httpurl': '"https://example.com"',
            'emailstr': '"test@example.com"'
        }
        
        # Handle composite types
        if 'optional' in field_type or '|' in field_type:
            # Extract inner type from Optional[Type] or Type1 | Type2
            inner_type = field_type.replace('optional[', '').replace(']', '')
            if '|' in inner_type:
                inner_type = inner_type.split('|')[0].strip()
            elif ',' in inner_type:
                inner_type = inner_type.split(',')[0].strip()
            inner_type = inner_type.replace('none', '').strip()
            
            return _generate_mock_value_for_type(inner_type)
        
        elif 'list[' in field_type:
            # Extract inner type from List[Type]
            inner_type = re.search(r'list\[(.*?)\]', field_type)
            if inner_type:
                item_value = _generate_mock_value_for_type(inner_type.group(1))
                return f'[{item_value}]'
            return '["item1", "item2"]'
        
        elif 'dict[' in field_type:
            # Extract key/value types from Dict[Key, Value]
            types = re.search(r'dict\[(.*?)\]', field_type)
            if types:
                key_type, value_type = types.group(1).split(',')
                key_value = _generate_mock_value_for_type(key_type.strip())
                value_value = _generate_mock_value_for_type(value_type.strip())
                return f'{{{key_value}: {value_value}}}'
            return '{"key": "value"}'
        
        # Return specific type value or default
        for type_key, mock_value in type_mapping.items():
            if type_key in field_type:
                return mock_value
        
        return '"mock_value"'

    def _identify_used_libraries(source_code: str) -> List[str]:
        """Identify external libraries used in the module."""
        libraries = set()
        
        # Standard library modules that shouldn't be considered as external
        stdlib_modules = {'os', 'sys', 'json', 're', 'pathlib', 'datetime', 'collections', 'typing', 
                        'functools', 'itertools', 'asyncio', 'logging', 'unittest', 'dataclasses'}
        
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        base_module = name.name.split('.')[0]
                        if base_module not in stdlib_modules:
                            libraries.add(base_module)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        base_module = node.module.split('.')[0]
                        if base_module not in stdlib_modules:
                            libraries.add(base_module)
        except Exception as e:
            print(f"Error identifying libraries: {e}")
        
        return sorted(list(libraries))

    def _create_prompt(self, source_file: Path, 
                            source_code: str, 
                            uncovered_functions: Dict[str, Dict] = None,
                            models: Dict[str, Dict] = None,
                            used_libraries: List[str] = None) -> str:
        """Create a prompt for generating tests."""
        if uncovered_functions is None:
            uncovered_functions = self._extract_uncovered_functions(source_code)
        
        # If models not provided, extract them
        if models is None:
            models = self._extract_model_structure(source_code)
        
        # If libraries not provided, extract them
        if used_libraries is None:
            used_libraries = self._identify_used_libraries(source_code)
        
        # Extract function details
        function_details = self._extract_detailed_function_info(uncovered_functions)
        
        # Generate model examples
        model_examples = ""
        if models:
            model_examples = "\nMODEL MOCK EXAMPLES:\n"
            for model_name, fields in models.items():
                mock_data = "{\n"
                for field_name, field_info in fields.items():
                    field_type = field_info.get('type', 'str')
                    mock_value = self._generate_mock_value_for_type(field_type)
                    mock_data += f'    "{field_name}": {mock_value},\n'
                mock_data += "}"
                model_examples += f"\n{model_name} mock example:\n```json\n{mock_data}\n```\n"
        
        prompt = f"""You are an expert Python testing specialist. Generate comprehensive pytest test cases for the given code.

MODULE INFORMATION:
- File: {source_file}
- Module path: {self.module_name}
- Test file path: {self.test_file_path}

SOURCE CODE:
```python
{source_code}
```

FUNCTIONS REQUIRING TESTS:
{function_details}

LIBRARY CONTEXT:
- Used libraries: {', '.join(used_libraries)}
{model_examples}

CRITICAL TEST GENERATION RULES:

1. IMPORTS & STRUCTURE
   - Import pytest, unittest.mock.AsyncMock/MagicMock as needed
   - Use the EXACT import paths as used in the source module
   - Order imports: stdlib → third-party → local project imports
   - Import only what's needed for testing

2. ASYNC TESTING RULES (MANDATORY):
   - ALL async functions MUST have @pytest.mark.asyncio decorator
   - Use AsyncMock exclusively for async functions/methods
   - For async context managers:
     ```python
     mock_session = AsyncMock()
     mock_response = AsyncMock()
     mock_session.request.return_value.__aenter__.return_value = mock_response
     ```
   - ALWAYS await async calls in tests:
     ```python
     result = await radio_browser.stats()  # CORRECT
     result = radio_browser.stats()       # WRONG - returns coroutine
     ```
   
3. MOCKING GUIDELINES:
   - Mock at the import location, not definition location
   - Use proper mock types: AsyncMock for async, MagicMock for sync
   - Set return_value or side_effect appropriately
   - For sequential calls, use side_effect with a list
   - Mock ALL external dependencies (APIs, databases, file systems)

4. FIXTURES:
   - Create fixtures for complex test setup
   - Use pytest.fixture decorator
   - Never call fixtures directly - pass as parameters
   - Consider fixture scope (function, class, module)
   - Name fixtures descriptively

5. ERROR HANDLING:
   - Test both success and error cases
   - Use pytest.raises for expected exceptions
   - Mock timeouts, connection errors, and API failures
   - Verify error messages and types

6. DATACLASS/MODEL TESTING:
   - Include ALL required fields in mock data
   - Match field types exactly
   - Handle Optional/Union types properly
   - Test serialization/deserialization if relevant

7. TEST COVERAGE REQUIREMENTS:
   - Test each branch of conditional logic
   - Test edge cases (empty inputs, None values, errors)
   - Test boundary conditions
   - Verify all side effects (calls, state changes)

8. HTTP/API TESTING:
   - Mock API responses completely
   - Test different status codes
   - Mock network errors
   - Verify request parameters

9. TEST NAMING:
   - Use descriptive test names: test_function_succeeds_when_condition
   - Follow pattern: test_[what]_[when]_[expected]
   - Be specific about what's being tested

10. ASSERTIONS:
    - Use specific assertions (assert_called_once_with)
    - Verify return values and types
    - Check state changes
    - Validate side effects

EXAMPLE TEST PATTERNS:

1. Async function testing:
```python
@pytest.mark.asyncio
async def test_async_function():
    # Setup
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_client.get.return_value = mock_response
    
    # Test
    result = await async_function(mock_client)
    
    # Verify
    assert result == expected
    mock_client.get.assert_called_once_with(url)
```

2. Exception testing:
```python
@pytest.mark.asyncio
async def test_handles_timeout():
    mock_client = AsyncMock()
    mock_client.get.side_effect = asyncio.TimeoutError()
    
    with pytest.raises(TimeoutError):
        await async_function(mock_client)
```

3. Context manager testing:
```python
def test_context_manager():
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = "resource"
    
    with mock_ctx as resource:
        assert resource == "resource"
    
    mock_ctx.__enter__.assert_called_once()
    mock_ctx.__exit__.assert_called_once()
```

OUTPUT FORMAT:
- Start with necessary imports
- Define fixtures if needed
- Write test functions
- Include docstrings explaining what each test does
- Return only the test code, no explanations

COMMON PITFALLS TO AVOID:
- Never use regular Mock for async functions
- Don't call fixtures directly
- Avoid magic numbers - use constants
- Don't test implementation details
- Don't skip error cases
- Don't forget to await async calls

Generate complete, production-ready test code following all these guidelines.
"""
    
        return prompt
    
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