#!/usr/bin/env python
"""
AI-powered test generator that works with any Python project structure.
"""

import argparse
import time
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import ast
import re
import openai
import pytest


class UniversalTestGenerator:
    """Simple, universal test generator for any Python project."""
    
    def __init__(self, 
                 api_key: str,
                 coverage_threshold: float = 80.0,
                 model: str = "anthropic/claude-3.7-sonnet"):
        """Initialize with minimal configuration."""
        self.api_key = api_key
        self.coverage_threshold = coverage_threshold
        self.model = model
        self.openai_client = self._setup_openai()
        self.module_name = ""
        self.test_file_path = None
        self.api_logs = []
        self.start_time = time.time()
    
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
        generation_start = time.time()
        # Set module name and test file path
        self.module_name = self._get_module_name(file_path)
        self.test_file_path = self._get_test_path(file_path)
        
        # Read the source code
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
        
        # Create prompt with enhanced analysis
        prompt = self._create_prompt(file_path, source_code)
        
        # Generate tests
        try:
            # Log API request
            request_log = {
                "timestamp": time.time(),
                "file": str(file_path),
                "prompt_length": len(prompt),
                "model": self.model
            }
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            generated_code = response.choices[0].message.content
            clean_code = self._clean_generated_code(generated_code)
            
            # Log API response
            response_log = {
                **request_log,
                "response_length": len(generated_code),
                "clean_code_length": len(clean_code),
                "generation_time": time.time() - generation_start,
                "success": True
            }
            
            # Add token usage if available
            if hasattr(response, 'usage'):
                response_log["token_usage"] = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            # Save to file
            with open('api_interaction_logs.json', 'a') as f:
                json.dump(response_log, f)
                f.write('\n')
            
            return clean_code
            
        except Exception as e:
            # Log error
            error_log = {
                **request_log,
                "error": str(e),
                "generation_time": time.time() - generation_start,
                "success": False
            }
            
            with open('api_interaction_logs.json', 'a') as f:
                json.dump(error_log, f)
                f.write('\n')
            
            print(f"Error generating tests for {file_path}: {e}")
            return ""
    
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
        # If not, we might have additional text before the actual code
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
    
    def _get_system_prompt(self) -> str:
        """Return the system prompt for the AI model."""
        return """You are an expert Python test engineer specializing in pytest. You generate production-quality test code that follows best practices. Your tests are comprehensive, maintainable, and correct. You excel at testing complex systems including data models, async code, and external dependencies."""

    def _get_module_name(self, file_path: Path) -> str:
        """Convert file path to importable module name."""
        current_dir = Path.cwd()
        
        try:
            rel_path = file_path.relative_to(current_dir)
        except ValueError:
            rel_path = file_path
        
        parts = list(rel_path.parts)
        
        # Handle src directory
        if parts and parts[0] == "src":
            parts.pop(0)
        
        # Remove .py extension
        if parts and parts[-1].endswith('.py'):
            parts[-1] = parts[-1][:-3]  # Remove .py extension
        
        return ".".join(parts)

    def _get_function_args(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[Dict]:
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

    def _extract_return_type(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[str]:
        """Extract return type annotation from function definition."""
        if node.returns:
            return ast.unparse(node.returns)
        return None

    def _extract_uncovered_functions(self, source_code: str) -> Dict[str, Dict]:
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
                        "args": self._get_function_args(node),
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                        "docstring": ast.get_docstring(node) or "",
                        "return_type": self._extract_return_type(node),
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
                                "args": self._get_function_args(method),
                                "is_async": isinstance(method, ast.AsyncFunctionDef),
                                "docstring": ast.get_docstring(method) or "",
                                "class_name": class_name,
                                "return_type": self._extract_return_type(method),
                                "line_no": method.lineno
                            }
        except Exception as e:
            print(f"Error extracting functions: {e}")
        
        return uncovered_functions

    def _identify_used_libraries(self, source_code: str) -> List[str]:
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

    def _extract_detailed_function_info(self, uncovered_functions: Dict[str, Dict]) -> str:
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

    def find_model_references(self, source_file: Path) -> List[Path]:
        """
        Find model definition files referenced by the source file.
        
        Args:
            source_file: Path to the source file being tested
            
        Returns:
            List of paths to model definition files
        """
        model_files = set()
        source_dir = source_file.parent
        project_root = self._find_project_root(source_file)
        
        # Read the source file to extract imports
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except Exception as e:
            print(f"Error reading {source_file}: {e}")
            return []
        
        # Extract imports from the source code
        imports = self._extract_imports(source_code)
        
        # Check common model file locations
        potential_paths = []
        
        # 1. Directly imported modules
        for module_path in imports:
            # Convert module path to file path
            parts = module_path.split('.')
            
            # Try absolute path from project root
            file_path = project_root.joinpath(*parts).with_suffix('.py')
            if file_path.exists():
                potential_paths.append(file_path)
            
            # Try relative path from source directory
            file_path = source_dir.joinpath(*parts).with_suffix('.py')
            if file_path.exists():
                potential_paths.append(file_path)
        
        # 2. Common model files in the same package
        common_model_files = ['const.py','models.py', 'schemas.py', 'entities.py', 'types.py', 'dataclasses.py']
        for model_file in common_model_files:
            file_path = source_dir / model_file
            if file_path.exists():
                potential_paths.append(file_path)
        
        # 3. Check if there's a models directory
        model_dirs = [source_dir / 'models', project_root / 'models', 
                      source_dir / 'schemas', project_root / 'schemas']
        
        for model_dir in model_dirs:
            if model_dir.exists() and model_dir.is_dir():
                for file_path in model_dir.glob('**/*.py'):
                    potential_paths.append(file_path)
        
        # Analyze each potential file to check if it contains model definitions
        for file_path in potential_paths:
            if self._contains_model_definitions(file_path):
                model_files.add(file_path)
        
        return list(model_files)

    def _find_project_root(self, file_path: Path) -> Path:
        """Find the project root directory."""
        current_path = file_path.parent.absolute()
        
        # Look for common project root indicators
        indicators = ['pyproject.toml', 'setup.py', '.git', 'requirements.txt']
        
        while current_path != current_path.parent:
            for indicator in indicators:
                if (current_path / indicator).exists():
                    return current_path
            current_path = current_path.parent
        
        # If no root found, return the current directory
        return Path.cwd()

    def _extract_imports(self, source_code: str) -> List[str]:
        """Extract imported modules from source code."""
        imports = []
        
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except Exception as e:
            print(f"Error extracting imports: {e}")
        
        return imports

    def _contains_model_definitions(self, file_path: Path) -> bool:
        """
        Check if a file contains model class definitions or enums.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Quick check for common model or enum indicators
            indicators = [
                'class', '@dataclass', 'mashumaro', 'BaseModel', 
                'DataClassJSONMixin', 'SerializationMixin', 'field',
                'Enum', 'enum', 'IntEnum', 'StrEnum', '(str, Enum)'
            ]
            
            if not any(indicator in content for indicator in indicators):
                return False
            
            # More detailed AST analysis
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check decorators
                    for decorator in node.decorator_list:
                        decorator_str = ast.unparse(decorator)
                        if any(model_dec in decorator_str for model_dec in ['dataclass', 'BaseModel']):
                            return True
                    
                    # Check base classes for model and enum patterns
                    for base in node.bases:
                        base_str = ast.unparse(base)
                        if any(model_base in base_str for model_base in 
                            ['DataClassJSONMixin', 'BaseModel', 'SerializationMixin']):
                            return True
                        # Add explicit Enum detection
                        if any(enum_base in base_str for enum_base in 
                            ['Enum', 'IntEnum', 'StrEnum', 'str, Enum']):
                            return True
                    
                    # Check for model-like structure (many typed attributes)
                    typed_attrs = sum(1 for item in node.body 
                                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name))
                    if typed_attrs >= 3:  # If class has several typed attributes, it's likely a model
                        return True
                    
                    # Check for enum-like structure (many class variables with values)
                    enum_attrs = sum(1 for item in node.body 
                                if isinstance(item, ast.Assign) and 
                                any(isinstance(target, ast.Name) for target in item.targets))
                    if enum_attrs >= 3:  # If class has several constant assignments, it might be an enum
                        return True
            
            return False
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return False

    def extract_model_definitions(self, file_path: Path) -> str:
        """
        Extract model class and enum definitions from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            String containing model class and enum definitions
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            model_definitions = []
            
            # Get imports that might be needed for the models
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.unparse(node))
            
            # Extract top-level variables that might be needed for enums
            top_level_vars = []
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Assign):
                    top_level_vars.append(ast.unparse(node))
            
            # Extract class definitions that appear to be models or enums
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it's a model class or enum
                    is_model_or_enum = False
                    
                    # Check decorators
                    for decorator in node.decorator_list:
                        decorator_str = ast.unparse(decorator)
                        if any(dec in decorator_str for dec in ['enum', 'dataclass', 'BaseModel']):
                            is_model_or_enum = True
                            break
                    
                    # Check base classes
                    if not is_model_or_enum:
                        for base in node.bases:
                            base_str = ast.unparse(base)
                            # Check for model base classes
                            if any(model_base in base_str for model_base in 
                                ['DataClassJSONMixin', 'BaseModel', 'SerializationMixin']):
                                is_model_or_enum = True
                                break
                            # Check for enum base classes
                            if any(enum_base in base_str for enum_base in 
                                ['Enum', 'IntEnum', 'StrEnum']) or 'str, Enum' in base_str:
                                is_model_or_enum = True
                                break
                    
                    # Check for model-like structure
                    if not is_model_or_enum:
                        typed_attrs = sum(1 for item in node.body 
                                        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name))
                        if typed_attrs >= 3:  # If class has several typed attributes, it's likely a model
                            is_model_or_enum = True
                    
                    # Check for enum-like structure
                    if not is_model_or_enum:
                        enum_attrs = sum(1 for item in node.body 
                                    if isinstance(item, ast.Assign) and 
                                    any(isinstance(target, ast.Name) for target in item.targets))
                        if enum_attrs >= 3:  # If class has several constant assignments, it might be an enum
                            is_model_or_enum = True
                    
                    if is_model_or_enum:
                        model_definitions.append(ast.unparse(node))
            
            # Combine imports, top-level vars, and model definitions
            unique_imports = list(dict.fromkeys(imports))  # Remove duplicates while preserving order
            result_parts = unique_imports + [""] + top_level_vars + [""] + model_definitions
            model_code = "\n".join(filter(None, result_parts))  # filter removes empty strings
            
            return model_code
        except Exception as e:
            print(f"Error extracting definitions from {file_path}: {e}")
            return ""

    def find_enum_references(self, source_file: Path) -> List[Path]:
        """
        Find enum definition files referenced by the source file.
        
        Args:
            source_file: Path to the source file being tested
            
        Returns:
            List of paths to enum definition files
        """
        enum_files = set()
        source_dir = source_file.parent
        project_root = self._find_project_root(source_file)
        
        # First, check if there's a const.py in the same directory
        const_file = source_dir / 'const.py'
        if const_file.exists():
            enum_files.add(const_file)
        
        # Read the source file to extract imports
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except Exception as e:
            print(f"Error reading {source_file}: {e}")
            return []
        
        # Extract imports from the source code
        imports = self._extract_imports(source_code)
        
        # Check for potential enum imports
        for module_path in imports:
            # Check if this import might be an enum module
            if 'const' in module_path.lower() or 'enum' in module_path.lower():
                # Convert module path to file path
                parts = module_path.split('.')
                
                # Try absolute path from project root
                file_path = project_root.joinpath(*parts).with_suffix('.py')
                if file_path.exists():
                    enum_files.add(file_path)
                
                # Try relative path from source directory
                file_path = source_dir.joinpath(*parts).with_suffix('.py')
                if file_path.exists():
                    enum_files.add(file_path)
        
        # Check common enum file names
        common_enum_files = ['enums.py', 'constants.py', 'consts.py', 'types.py']
        for enum_file in common_enum_files:
            file_path = source_dir / enum_file
            if file_path.exists():
                enum_files.add(file_path)
        
        return list(enum_files)

    def _create_prompt(self, source_file: Path, source_code: str) -> str:
        """Create a prompt for generating tests with model reference files."""
        # Extract information
        uncovered_functions = self._extract_uncovered_functions(source_code)
        used_libraries = self._identify_used_libraries(source_code)
        
        # Find and extract model reference files
        model_reference_files = self.find_model_references(source_file)
        
        # Add enum references
        enum_reference_files = self.find_enum_references(source_file)
        # Combine model and enum files, removing duplicates
        all_reference_files = list(set(model_reference_files + enum_reference_files))
        
        model_reference_code = ""
        
        if all_reference_files:
            model_reference_code = "\nMODELS AND TYPES:\n"
            for file_path in all_reference_files:
                model_definitions = self.extract_model_definitions(file_path)
                if model_definitions:
                    # Use str(file_path) instead of trying to make it relative
                    # This avoids the ValueError when paths can't be made relative
                    model_reference_code += f"\n# From {file_path}\n```python\n{model_definitions}\n```\n"
        
        # Extract function details
        function_details = self._extract_detailed_function_info(uncovered_functions)
        
        prompt = f"""Generate complete pytest tests for the code below.

MODULE INFORMATION:
- File: {source_file}
- Module path: {self.module_name}
- Test file path: {self.test_file_path}

SOURCE CODE:
```python
{source_code}
```

{model_reference_code}

FUNCTIONS REQUIRING TESTS:
{function_details}

LIBRARY CONTEXT:
- Used libraries: {', '.join(used_libraries)}

REQUIREMENTS:

## MODEL SETUP
- Include EVERY field when initializing models (required and optional)
- Verify field names exist in actual model definition (case-sensitive)
- Ensure correct model class is used when similar names exist
- Match exact field types and constraints from specifications

## MOCKING STRATEGY
- Import Mock/AsyncMock from unittest.mock
- Set return_value/side_effect BEFORE using mocks
- Use side_effect with exception instances (not classes)
- For DNS testing:
  - Always mock DNSResolver.query with proper SRV records
  - Ensure DNS mocks are set up BEFORE HTTP client creation
  - Use consistent hostnames between DNS and HTTP mocks

## ASYNC HANDLING
- Use pytest.mark.asyncio for tests with await expressions
- Always await async calls (including mocked functions)
- Properly configure AsyncMock with awaitable return values
- Handle async context managers and iterators correctly

## VALIDATION APPROACH
- For URLs: Use unittest.mock.ANY for parameters or str(url) for exact matching
- For collections: Avoid index-based assertions, find by key/property instead
- For API responses: Test structure and types rather than exact values
- Use set comparisons for unordered collections
- Test pagination with both single and multi-page scenarios

## PYTEST PRACTICES
- Create appropriately scoped fixtures
- Use parametrize for testing variations
- Test exceptions with pytest.raises contextmanager
- Group related tests in classes when logical

Return only runnable pytest code with no explanations or markdown. The code must be immediately usable without any modifications.
"""
        return prompt
    
    def save_test_file(self, file_path: Path, test_code: str) -> Path:
        """
        Save the generated test file.
        
        Args:
            file_path: Original Python file
            test_code: Generated test code
            
        Returns:
            Path to the saved test file
        """
        # Determine test file path if not already set
        if not self.test_file_path:
            self.test_file_path = self._get_test_path(file_path)
        
        # Create directories if needed
        self.test_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the test file
        try:
            with open(self.test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
            print(f"Saved test file: {self.test_file_path}")
            return self.test_file_path
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


class MSPPTestGenerator:
    """Multi-Stage Prompt Processing Test Generator implementing single-call MSPP concept."""
    
    def __init__(self, 
                 api_key: str,
                 coverage_threshold: float = 80.0,
                 model: str = "anthropic/claude-3.7-sonnet"):
        """Initialize with configuration for single-call MSPP."""
        if not api_key:
            raise ValueError("API key is required")
        
        self.api_key = api_key
        self.coverage_threshold = coverage_threshold
        self.model = model
        self.openai_client = self._setup_openai()
        self.module_name = ""
        self.test_file_path = None
        self.api_logs = []
        self.start_time = time.time()
    
    def _setup_openai(self):
        """Configure OpenAI client for various providers."""
        try:
            if "openrouter" in self.api_key.lower() or os.getenv("OPENROUTER_API_KEY"):
                return openai.OpenAI(
                    api_key=self.api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
            return openai.OpenAI(api_key=self.api_key)
        except Exception as e:
            print(f"Error setting up OpenAI client: {e}")
            raise
    
    def generate_test_with_msp(self, file_path: Path) -> str:
        """Generate tests using single-call MSPP approach."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            # Establish context (same as before)
            context = self._establish_context(file_path)
            
            # Single API call with multi-stage prompt
            prompt = self._create_mspp_prompt(context)
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                top_p=0.95
            )
            
            generated_code = response.choices[0].message.content
            return self._clean_generated_code(generated_code)
            
        except Exception as e:
            print(f"Error generating tests for {file_path}: {e}")
            return ""
    
    def _create_mspp_prompt(self, context: dict) -> str:
        """Create comprehensive MSPP prompt in single call."""
        return f"""Generate complete pytest tests using Multi-Stage Prompt Processing approach.

STAGE 1 - CONTEXT ESTABLISHMENT:
MODULE INFORMATION:
- File: {context['file_path']}
- Module path: {context['module_name']}
- Test file path: {context['test_file_path']}

SOURCE CODE:
```python
{context['source_code']}
```

{context['model_reference_code']}

FUNCTIONS REQUIRING TESTS:
{context['function_details']}

LIBRARY CONTEXT:
- Used libraries: {context['libraries']}

STAGE 2 - TEST STRUCTURE DESIGN:
Design the test structure including:
✓ Framework selection (pytest)
✓ Test skeleton with appropriate fixtures
✓ Mock object initialization patterns  
✓ Test class organization
✓ Import statements and dependencies

STAGE 3 - TEST LOGIC IMPLEMENTATION:
Implement comprehensive test logic:
✓ Assertion strategies for all scenarios
✓ Edge case handling and error conditions
✓ Async pattern implementation (if needed)
✓ Mock behavior definition and setup
✓ Parameter validation testing

STAGE 4 - REFINEMENT & OPTIMIZATION:
Apply refinement techniques:
✓ Error detection and correction
✓ Test optimization for maintainability
✓ Coverage maximization strategies
✓ Code quality improvements
✓ Performance considerations

STAGE 5 - VALIDATION & INTEGRATION:
Final validation steps:
✓ Syntax validation
✓ Quality checks
✓ Integration readiness
✓ Proper documentation
✓ Best practices compliance

REQUIREMENTS:

## MODEL SETUP
- Include EVERY field when initializing models (required and optional)
- Verify field names exist in actual model definition (case-sensitive)
- Ensure correct model class is used when similar names exist
- Match exact field types and constraints from specifications

## MOCKING STRATEGY
- Import Mock/AsyncMock from unittest.mock
- Set return_value/side_effect BEFORE using mocks
- Use side_effect with exception instances (not classes)
- For DNS testing:
  - Always mock DNSResolver.query with proper SRV records
  - Ensure DNS mocks are set up BEFORE HTTP client creation
  - Use consistent hostnames between DNS and HTTP mocks

## ASYNC HANDLING
- Use pytest.mark.asyncio for tests with await expressions
- Always await async calls (including mocked functions)
- Properly configure AsyncMock with awaitable return values
- Handle async context managers and iterators correctly

## VALIDATION APPROACH
- For URLs: Use unittest.mock.ANY for parameters or str(url) for exact matching
- For collections: Avoid index-based assertions, find by key/property instead
- For API responses: Test structure and types rather than exact values
- Use set comparisons for unordered collections
- Test pagination with both single and multi-page scenarios

## PYTEST PRACTICES
- Create appropriately scoped fixtures
- Use parametrize for testing variations
- Test exceptions with pytest.raises contextmanager
- Group related tests in classes when logical

Return ONLY the final, production-ready pytest code. No explanations, no markdown blocks, just pure Python code that can be immediately executed."""
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the AI model."""
        return """You are an expert Python test engineer specializing in pytest. You generate production-quality test code that follows best practices. Your tests are comprehensive, maintainable, and correct. You excel at testing complex systems including data models, async code, and external dependencies."""
    
    def _get_module_name(self, file_path: Path) -> str:
        """Convert file path to importable module name."""
        current_dir = Path.cwd()
        
        try:
            rel_path = file_path.relative_to(current_dir)
        except ValueError:
            rel_path = file_path
        
        parts = list(rel_path.parts)
        
        # Handle src directory
        if parts and parts[0] == "src":
            parts.pop(0)
        
        # Remove .py extension
        if parts and parts[-1].endswith('.py'):
            parts[-1] = parts[-1][:-3]  # Remove .py extension
        
        return ".".join(parts)
    
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
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean generated code by removing Markdown artifacts and other non-Python syntax."""
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
    
    def save_test_file(self, file_path: Path, test_code: str) -> Path:
        """Save the generated test file."""
        # Determine test file path if not already set
        if not self.test_file_path:
            self.test_file_path = self._get_test_path(file_path)
        
        # Create directories if needed
        self.test_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the test file
        try:
            with open(self.test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
            print(f"Saved test file: {self.test_file_path}")
            return self.test_file_path
        except Exception as e:
            print(f"Error saving test file: {e}")
            return None
    
    def find_files_needing_tests(self, 
                               coverage_data: Dict = None,
                               target_files: List[str] = None) -> List[Path]:
        """Find Python files that need tests."""
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

    def _establish_context(self, file_path: Path) -> dict:
        """Establish the initial context for test generation."""
        # Read source code
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Set module name and test file path
        self.module_name = self._get_module_name(file_path)
        self.test_file_path = self._get_test_path(file_path)
        
        # Extract information
        uncovered_functions = self._extract_uncovered_functions(source_code)
        used_libraries = self._identify_used_libraries(source_code)
        
        # Find and extract model reference files
        model_reference_files = self.find_model_references(file_path)
        enum_reference_files = self.find_enum_references(file_path)
        all_reference_files = list(set(model_reference_files + enum_reference_files))
        
        model_reference_code = ""
        if all_reference_files:
            model_reference_code = "\nMODELS AND TYPES:\n"
            for ref_file in all_reference_files:
                model_definitions = self.extract_model_definitions(ref_file)
                if model_definitions:
                    model_reference_code += f"\n# From {ref_file}\n```python\n{model_definitions}\n```\n"
        
        # Extract function details
        function_details = self._extract_detailed_function_info(uncovered_functions)
        
        return {
            "file_path": str(file_path),
            "module_name": self.module_name,
            "test_file_path": str(self.test_file_path),
            "source_code": source_code,
            "model_reference_code": model_reference_code,
            "function_details": function_details,
            "libraries": ", ".join(used_libraries)
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate tests for Python modules')
    parser.add_argument('--module', required=False, help='Specific module path to generate tests for')
    parser.add_argument('--msp', action='store_true', help='Use Multi-Stage Prompt Processing')
    args = parser.parse_args()
    
    # Get configuration from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY")
        sys.exit(1)
    
    # Get optional model
    model = os.getenv("OPENAI_MODEL", "anthropic/claude-3.7-sonnet")
    
    coverage_threshold = float(os.getenv("COVERAGE_THRESHOLD", "80"))
    
    # If module is provided via argument, use it
    if args.module:
        target_files = [args.module]
    else:
        target_files_str = os.getenv("TARGET_FILES", "")
        target_files = [f.strip() for f in target_files_str.split(",") if f.strip()]
    
    # Initialize appropriate generator
    if args.msp:
        generator = MSPPTestGenerator(api_key, coverage_threshold, model)
    else:
        generator = UniversalTestGenerator(api_key, coverage_threshold, model)
    
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
        if args.msp:
            test_code = generator.generate_test_with_msp(file_path)
        else:
            test_code = generator.generate_test_for_file(file_path)
        
        if test_code:
            # Validate if the code is syntactically correct Python
            try:
                ast.parse(test_code)
                print(f"✅ Generated code is syntactically correct")
            except SyntaxError as e:
                print(f"⚠️ Warning: Generated code has syntax error: {e}")
                print("Attempting additional cleanup...")
                # Try more aggressive cleanup if normal cleanup didn't work
                test_code = test_code.replace('```', '')
                try:
                    ast.parse(test_code)
                    print(f"✅ Fixed syntax issues")
                except SyntaxError as e:
                    print(f"❌ Could not fix all syntax issues: {e}")
                    # Continue anyway, but warn the user
            
            # Save the test file
            generator.save_test_file(file_path, test_code)
    
    print("Test generation complete!")


if __name__ == "__main__":
    main()