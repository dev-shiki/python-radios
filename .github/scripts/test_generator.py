"""
Enhanced Test Generation Workflow

This module improves the test generation workflow with more robust error handling,
pattern detection, and post-processing to ensure generated tests work correctly
across a variety of Python projects.
"""

import os
import re
import ast
import logging
import threading
from collections import deque
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import time
import json
from pathlib import Path

logger = logging.getLogger("enhanced-test-generator")

class TestPatternDetector:
    """
    Detects common patterns and requirements in Python code that affect test generation
    """
    
    # Common Python frameworks and their test requirements
    FRAMEWORK_PATTERNS = {
        # Async frameworks
        'asyncio': {'async': True, 'needs_pytest_asyncio': True},
        'aiohttp': {'async': True, 'needs_aiohttp_client': True},
        'fastapi': {'async': True, 'needs_httpx': True},
        
        # ORM/Database
        'sqlalchemy': {'needs_db_mocking': True},
        'django.db': {'needs_django_test_case': True},
        'peewee': {'needs_db_mocking': True},
        
        # API clients
        'requests': {'needs_requests_mock': True},
        'httpx': {'needs_httpx_mock': True},
        
        # Data serialization
        'pydantic': {'needs_model_validation': True, 'strict_typing': True},
        'marshmallow': {'needs_schema_validation': True},
        'dataclass': {'needs_dataclass_validation': True},
        'mashumaro': {'needs_strict_field_validation': True, 'strict_typing': True},
        
        # Testing frameworks
        'pytest': {'framework': 'pytest'},
        'unittest': {'framework': 'unittest'},
    }
    
    # Common model libraries and their field requirements
    MODEL_PATTERNS = {
        # Pydantic patterns
        'BaseModel': {'lib': 'pydantic', 'required_validation': True},
        'Field(': {'lib': 'pydantic', 'strict_fields': True},
        
        # Dataclasses
        '@dataclass': {'lib': 'dataclasses', 'complete_init': True},
        
        # SQLAlchemy
        'Column(': {'lib': 'sqlalchemy', 'db_fields': True},
        
        # Marshmallow
        'Schema': {'lib': 'marshmallow', 'schema_fields': True},
        
        # Mashumaro
        'DataClassJSONMixin': {'lib': 'mashumaro', 'strict_fields': True},
        'from_dict': {'lib': 'mashumaro', 'dict_conversion': True},
        'to_dict': {'lib': 'mashumaro', 'dict_conversion': True},
    }
    
    @staticmethod
    def detect_requirements(code: str) -> Dict[str, Any]:
        """
        Detect test requirements based on code patterns
        """
        requirements = {
            # Framework needs
            'framework': 'pytest',  # Default to pytest
            'async': False,
            'needs_pytest_asyncio': False,
            'needs_mocking': False,
            'needs_db_mocking': False,
            
            # Typing/validation needs
            'strict_typing': False,
            'needs_model_validation': False,
            'model_libraries': [],
            
            # Common libraries detected
            'libraries': [],
        }
        
        # Check framework patterns
        for pattern, pattern_reqs in TestPatternDetector.FRAMEWORK_PATTERNS.items():
            if pattern in code:
                requirements['libraries'].append(pattern)
                # Update requirements based on pattern
                for key, value in pattern_reqs.items():
                    requirements[key] = value
        
        # Check model patterns
        for pattern, pattern_reqs in TestPatternDetector.MODEL_PATTERNS.items():
            if pattern in code:
                lib = pattern_reqs.get('lib')
                if lib and lib not in requirements['model_libraries']:
                    requirements['model_libraries'].append(lib)
                    
                # Update requirements based on pattern
                for key, value in pattern_reqs.items():
                    if key != 'lib':
                        requirements[key] = value
        
        # Detect async code specifically
        if 'async ' in code or 'await ' in code or 'asyncio' in code:
            requirements['async'] = True
            requirements['needs_pytest_asyncio'] = True
        
        # Detect mocking needs
        if 'mock' in code.lower() or 'patch' in code.lower() or len(requirements['libraries']) > 1:
            requirements['needs_mocking'] = True
        
        return requirements

    @staticmethod
    def extract_model_fields(code: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract model field definitions from code to ensure complete test data
        """
        models = {}
        
        try:
            # Parse code into AST
            tree = ast.parse(code)
            
            # Look for class definitions
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    fields = []
                    
                    # Check class bases to detect model types
                    is_model_class = False
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id in [
                            'BaseModel', 'Model', 'DataClassJSONMixin'
                        ]:
                            is_model_class = True
                            break
                    
                    if not is_model_class:
                        # Check for dataclass decorator
                        for decorator in node.decorator_list:
                            if isinstance(decorator, ast.Name) and decorator.id == 'dataclass':
                                is_model_class = True
                                break
                    
                    if is_model_class:
                        # Extract field definitions
                        for item in node.body:
                            # Check for variable annotations (standard in modern Python)
                            if isinstance(item, ast.AnnAssign):
                                field_name = item.target.id if isinstance(item.target, ast.Name) else None
                                field_type = ast.unparse(item.annotation) if hasattr(ast, 'unparse') else None
                                
                                if field_name:
                                    field_info = {
                                        'name': field_name,
                                        'type': field_type,
                                        'required': True,  # Default to required
                                        'default': None,
                                    }
                                    
                                    # Check if there's a default value
                                    if item.value:
                                        field_info['default'] = ast.unparse(item.value) if hasattr(ast, 'unparse') else 'default_value'
                                        field_info['required'] = False
                                    
                                    # Check if field type indicates optional
                                    if field_type and ('Optional' in field_type or 'None' in field_type):
                                        field_info['required'] = False
                                    
                                    fields.append(field_info)
                        
                        models[class_name] = fields
        except Exception as e:
            logger.warning(f"Error extracting model fields: {e}")
        
        # Fallback to regex pattern matching if AST approach fails or misses fields
        if not models:
            # Look for class definitions with regex
            class_pattern = re.compile(r'class\s+(\w+)(?:\(.*?\))?\s*:', re.DOTALL)
            for match in class_pattern.finditer(code):
                class_name = match.group(1)
                
                # Find the class body
                start_pos = match.end()
                level = 0
                end_pos = start_pos
                
                # Basic parsing to find end of class definition
                for i in range(start_pos, len(code)):
                    if code[i] == '{':
                        level += 1
                    elif code[i] == '}':
                        level -= 1
                    
                    if level < 0 and code[i] == '\n' and code[i-1] != '\\':
                        # Found a new line with same indentation as class definition
                        end_pos = i
                        break
                
                class_body = code[start_pos:end_pos]
                
                # Look for field definitions in the class body
                field_pattern = re.compile(r'(\w+)\s*(?::\s*([^=\n]+))?(?:\s*=\s*([^,\n]+))?')
                fields = []
                
                for field_match in field_pattern.finditer(class_body):
                    field_name = field_match.group(1)
                    field_type = field_match.group(2)
                    default_value = field_match.group(3)
                    
                    if field_name and not field_name.startswith('_'):
                        field_info = {
                            'name': field_name,
                            'type': field_type.strip() if field_type else None,
                            'required': True,
                            'default': default_value.strip() if default_value else None,
                        }
                        
                        if default_value or (field_type and ('Optional' in field_type or 'None' in field_type)):
                            field_info['required'] = False
                        
                        fields.append(field_info)
                
                if fields:
                    models[class_name] = fields
        
        return models

class MockDataGenerator:
    """
    Generates appropriate mock data for tests based on code analysis
    """
    
    # Type mapping for generating appropriate mock values
    TYPE_MAPPING = {
        'str': lambda field_name: f"test-{field_name}",
        'int': lambda field_name: 1 if field_name == 'version' else 100,
        'float': lambda field_name: 1.0,
        'bool': lambda field_name: True,
        'list': lambda field_name: [],
        'dict': lambda field_name: {},
        'List': lambda field_name: [],
        'Dict': lambda field_name: {},
        'Optional': lambda field_name: None,
        'datetime': lambda field_name: "2023-01-01T00:00:00Z",
        'date': lambda field_name: "2023-01-01",
        'uuid': lambda field_name: "00000000-0000-0000-0000-000000000000",
        'UUID': lambda field_name: "00000000-0000-0000-0000-000000000000",
    }
    
    # Field-specific special values
    SPECIAL_FIELD_VALUES = {
        'id': "test-id-12345",
        'uuid': "00000000-0000-0000-0000-000000000000",
        'name': "Test Name",
        'email': "test@example.com",
        'url': "https://example.com",
        'created_at': "2023-01-01T00:00:00Z",
        'updated_at': "2023-01-01T00:00:00Z",
        'version': 1,  # Common version fields are integers, not strings
        'code': "test-code",
        'status': "active",
        'type': "test",
        'supported_version': 1,  # Integer, not string
        'change_uuid': "00000000-0000-0000-0000-000000000000",
    }
    
    @staticmethod
    def generate_mock_data(model_name: str, fields: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate appropriate mock data for a model
        """
        mock_data = {}
        
        for field in fields:
            field_name = field['name']
            field_type = field['type']
            
            # Skip special methods and private fields
            if field_name.startswith('__') or field_name.startswith('_'):
                continue
                
            # Use special field value if available
            if field_name in MockDataGenerator.SPECIAL_FIELD_VALUES:
                mock_data[field_name] = MockDataGenerator.SPECIAL_FIELD_VALUES[field_name]
                continue
            
            # Generate value based on type
            if field_type:
                # Extract base type from complex types (e.g., Optional[str] -> str)
                base_type = field_type.split('[')[-1].split(']')[0].strip() if '[' in field_type else field_type
                
                # Find the appropriate type generator
                for type_pattern, generator in MockDataGenerator.TYPE_MAPPING.items():
                    if type_pattern in field_type:
                        mock_data[field_name] = generator(field_name)
                        break
                else:
                    # Default to string for unknown types
                    mock_data[field_name] = f"test-{field_name}"
            else:
                # No type information, use field name hints
                mock_data[field_name] = f"test-{field_name}"
        
        return mock_data
    
    @staticmethod
    def generate_mock_data_examples(models: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """
        Generate mock data examples for all models
        """
        mock_examples = {}
        
        for model_name, fields in models.items():
            mock_examples[model_name] = MockDataGenerator.generate_mock_data(model_name, fields)
        
        return mock_examples

class TestCodePostProcessor:
    """
    Post-processes generated test code to fix common issues
    """
    
    @staticmethod
    def fix_async_issues(test_code: str, requirements: Dict[str, Any]) -> str:
        """
        Fix common issues with async code in tests
        """
        if not requirements.get('async', False):
            return test_code
        
        # Fix 1: Add pytest.mark.asyncio decorators to async test functions
        lines = test_code.split('\n')
        processed_lines = []
        
        for i, line in enumerate(lines):
            if line.strip().startswith("async def test_") and i > 0 and not lines[i-1].strip().startswith("@pytest.mark.asyncio"):
                processed_lines.append("@pytest.mark.asyncio")
            processed_lines.append(line)
        
        test_code = '\n'.join(processed_lines)
        
        # Fix 2: Correct await expression issues
        await_list_pattern = re.compile(r'await\s+(\[.*?\])', re.DOTALL)
        for match in await_list_pattern.finditer(test_code):
            list_code = match.group(1)
            fixed_code = f"[await item for item in {list_code}]"
            test_code = test_code.replace(f"await {list_code}", fixed_code)
        
        # Fix 3: Fix AsyncMock return values
        mock_pattern = re.compile(r'(mock\w+)\.return_value\s*=\s*\[', re.DOTALL)
        for match in mock_pattern.finditer(test_code):
            mock_name = match.group(1)
            if "return_value" in test_code and f"await {mock_name}" in test_code:
                # Fix the return value assignment for async iterables
                test_code = test_code.replace(
                    f"{mock_name}.return_value = [", 
                    f"{mock_name}.__aiter__.return_value = ["
                )
                
                # Add mock aiter method if needed
                if "AsyncMock" in test_code and f"{mock_name}.__aiter__" in test_code and "__aiter__ = AsyncMock()" not in test_code:
                    # Find a good insertion point
                    pos = test_code.find(f"{mock_name}.__aiter__")
                    if pos > 0:
                        insertion_pos = test_code.rfind("\n    ", 0, pos)
                        if insertion_pos > 0:
                            test_code = (
                                test_code[:insertion_pos] + 
                                "\n    # Set up proper async iterator behavior\n" +
                                f"    {mock_name}.__aiter__ = AsyncMock()\n" +
                                test_code[insertion_pos:]
                            )
        
        # Fix 4: Ensure AsyncMock is imported
        if "AsyncMock" in test_code and "from unittest.mock import AsyncMock" not in test_code:
            # Add import if mock is already imported
            if "from unittest.mock import " in test_code:
                test_code = test_code.replace(
                    "from unittest.mock import ", 
                    "from unittest.mock import AsyncMock, "
                )
            else:
                # Add new import line after existing imports
                import_end = 0
                for line in test_code.split('\n'):
                    if line.startswith(('import ', 'from ')):
                        import_end = test_code.find(line) + len(line)
                
                if import_end > 0:
                    test_code = (
                        test_code[:import_end] + 
                        "\nfrom unittest.mock import AsyncMock" + 
                        test_code[import_end:]
                    )
        
        return test_code
    
    @staticmethod
    def fix_model_validation_issues(test_code: str, models: Dict[str, Dict[str, Any]]) -> str:
        """
        Fix issues with model validation in tests
        """
        if not models:
            return test_code
        
        # Look for mock data dictionaries
        mock_data_pattern = re.compile(r'(\w+)_data\s*=\s*\{([^}]*)\}', re.DOTALL)
        
        for match in mock_data_pattern.finditer(test_code):
            var_name = match.group(1)
            mock_data_content = match.group(2)
            
            # Determine which model this might be for
            model_name = None
            for name in models.keys():
                if name.lower() in var_name.lower():
                    model_name = name
                    break
            
            if model_name and model_name in models:
                # Get the mock example
                mock_example = models[model_name]
                if not mock_example:
                    continue
                
                # Check for missing required fields
                missing_fields = []
                for field_name, field_value in mock_example.items():
                    field_pattern = fr'[\'"]({field_name})[\'"]'
                    if not re.search(field_pattern, mock_data_content):
                        missing_fields.append((field_name, field_value))
                
                if missing_fields:
                    # Add missing fields to the mock data
                    field_additions = ""
                    for field_name, field_value in missing_fields:
                        if isinstance(field_value, str):
                            field_additions += f'    "{field_name}": "{field_value}",\n'
                        else:
                            field_additions += f'    "{field_name}": {field_value},\n'
                    
                    # Find the closing brace of the dictionary
                    closing_brace_pos = test_code.find("}", match.start())
                    if closing_brace_pos > 0:
                        # Add missing fields before the closing brace
                        test_code = (
                            test_code[:closing_brace_pos] + 
                            "\n" + field_additions + 
                            test_code[closing_brace_pos:]
                        )
        
        # Fix type issues in mock data
        type_fixes = [
            # Fix string version fields that should be integers
            (r'("version"\s*:\s*)(["\'])(\d+)\.0(["\'])', r'\1\3'),
            (r'("supported_version"\s*:\s*)(["\'])(\d+)\.0(["\'])', r'\1\3'),
        ]
        
        for pattern, replacement in type_fixes:
            test_code = re.sub(pattern, replacement, test_code)
        
        return test_code
    
    @staticmethod
    def add_mock_examples(test_code: str, mock_examples: Dict[str, Dict[str, Any]]) -> str:
        """
        Add mock data examples to the test code
        """
        if not mock_examples:
            return test_code
        
        # Create example section
        examples_section = "\n# Example mock data for models\n"
        
        for model_name, mock_data in mock_examples.items():
            examples_section += f"\nMOCK_{model_name.upper()}_DATA = {{\n"
            
            for field_name, field_value in mock_data.items():
                if isinstance(field_value, str):
                    examples_section += f'    "{field_name}": "{field_value}",\n'
                else:
                    examples_section += f'    "{field_name}": {field_value},\n'
            
            examples_section += "}\n"
        
        # Add examples after imports
        import_end = 0
        for line in test_code.split('\n'):
            if line.startswith(('import ', 'from ')):
                import_end = test_code.find(line) + len(line)
        
        if import_end > 0:
            # Find the end of the import section
            next_line_pos = test_code.find('\n', import_end)
            if next_line_pos > 0:
                import_end = next_line_pos
            
            test_code = (
                test_code[:import_end] + 
                "\n" + examples_section + 
                test_code[import_end:]
            )
        else:
            # Add at the beginning if no imports found
            test_code = examples_section + "\n" + test_code
        
        return test_code
    
    @staticmethod
    def ensure_framework_imports(test_code: str, requirements: Dict[str, Any]) -> str:
        """
        Ensure all necessary framework imports are included
        """
        needed_imports = []
        
        # Check for pytest
        if requirements.get('framework') == 'pytest' and 'import pytest' not in test_code:
            needed_imports.append('import pytest')
        
        # Check for pytest-asyncio
        if requirements.get('needs_pytest_asyncio', False) and '@pytest.mark.asyncio' in test_code:
            if 'import pytest' not in test_code:
                needed_imports.append('import pytest')
        
        # Check for mocking imports
        if requirements.get('needs_mocking', False):
            if 'unittest.mock' not in test_code:
                mock_imports = ['patch', 'MagicMock']
                if requirements.get('async', False):
                    mock_imports.append('AsyncMock')
                needed_imports.append(f"from unittest.mock import {', '.join(mock_imports)}")
        
        # Add imports at the beginning if needed
        if needed_imports:
            test_code = '\n'.join(needed_imports) + '\n\n' + test_code
        
        return test_code
    
    @staticmethod
    def post_process_test_code(
        test_code: str,
        requirements: Dict[str, Any],
        models: Dict[str, List[Dict[str, Any]]],
        module_path: str,
        import_path: str
    ) -> str:
        """
        Apply all post-processing fixes to the generated test code
        """
        # Check if code seems invalid (conversational)
        if (not test_code.strip().startswith(("import ", "from ", "def ", "class ", "#")) and
            ("would you like" in test_code.lower() or "i understand" in test_code.lower())):
            logger.warning("Detected conversational text in output, generating fallback test")
            return TestCodePostProcessor.generate_fallback_test(module_path, import_path, requirements, models)
        
        # Generate mock data examples
        mock_examples = MockDataGenerator.generate_mock_data_examples(models)
        
        # Apply fixes in sequence
        test_code = TestCodePostProcessor.ensure_framework_imports(test_code, requirements)
        test_code = TestCodePostProcessor.fix_async_issues(test_code, requirements)
        test_code = TestCodePostProcessor.fix_model_validation_issues(test_code, mock_examples)
        test_code = TestCodePostProcessor.add_mock_examples(test_code, mock_examples)
        
        # Validate that the result is proper Python syntax
        try:
            compile(test_code, "<string>", "exec")
            logger.info("Successfully generated and validated test code")
        except SyntaxError as e:
            logger.warning(f"Generated code has syntax error: {e}")
            # Fallback to basic test scaffold
            test_code = TestCodePostProcessor.generate_fallback_test(module_path, import_path, requirements, models)
        
        return test_code
    
    @staticmethod
    def generate_fallback_test(
        module_path: str, 
        import_path: str, 
        requirements: Dict[str, Any], 
        models: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """
        Generate a simple test scaffold as fallback when AI generation fails
        """
        logger.info("Generating fallback test scaffold")
        
        # Generate mock examples for models
        mock_examples = MockDataGenerator.generate_mock_data_examples(models)
        
        # Create basic test scaffold
        framework = requirements.get('framework', 'pytest')
        has_async = requirements.get('async', False)
        
        test_code = f"""# AUTO-GENERATED TEST SCAFFOLD
import {framework}
"""
        
        # Add necessary imports
        if has_async:
            test_code += "import asyncio\n"
        
        if requirements.get('needs_mocking', False):
            mock_imports = ['patch', 'MagicMock']
            if has_async:
                mock_imports.append('AsyncMock')
            test_code += f"from unittest.mock import {', '.join(mock_imports)}\n"
        
        # Add import for the module under test
        test_code += f"from {import_path} import *\n\n"
        
        # Add mock examples
        if mock_examples:
            test_code += "# Example mock data for models\n"
            for model_name, mock_data in mock_examples.items():
                test_code += f"\nMOCK_{model_name.upper()}_DATA = {{\n"
                
                for field_name, field_value in mock_data.items():
                    if isinstance(field_value, str):
                        test_code += f'    "{field_name}": "{field_value}",\n'
                    else:
                        test_code += f'    "{field_name}": {field_value},\n'
                
                test_code += "}\n"
        
        # If asyncio is used, add the needed marker
        if has_async and framework == 'pytest':
            test_code += """
# Configure pytest for asyncio tests
pytestmark = pytest.mark.asyncio
"""
        
        # Extract classes and functions from the module path
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                module_code = f.read()
            
            # Use AST to extract classes and functions
            tree = ast.parse(module_code)
            classes = []
            functions = []
            
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    methods = []
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            methods.append({
                                'name': child.name,
                                'async': isinstance(child, ast.AsyncFunctionDef),
                                'args': [arg.arg for arg in child.args.args if arg.arg != 'self'],
                            })
                    
                    classes.append({
                        'name': node.name,
                        'methods': methods,
                    })
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append({
                        'name': node.name,
                        'async': isinstance(node, ast.AsyncFunctionDef),
                        'args': [arg.arg for arg in node.args.args],
                    })
        except Exception as e:
            logger.warning(f"Error extracting classes and functions: {e}")
            classes = []
            functions = []
        
        # Add tests for each class
        for cls in classes:
            cls_name = cls['name']
            test_code += f"""
class Test{cls_name}:
    @pytest.fixture
    def {cls_name.lower()}_instance(self):
        # TODO: Customize fixture with appropriate initialization
        return {cls_name}()
    
"""
            # Add test for initialization 
            test_code += f"""    {'@pytest.mark.asyncio' if has_async else ''}
    {'async ' if has_async else ''}def test_{cls_name.lower()}_initialization(self, {cls_name.lower()}_instance):
        # Test basic initialization
        assert {cls_name.lower()}_instance is not None
    
"""

            # Add tests for methods
            for method in cls['methods']:
                if method['name'].startswith('_'):
                    continue  # Skip private methods
                
                test_code += f"""    {'@pytest.mark.asyncio' if has_async or method.get('async', False) else ''}
    {'async ' if has_async or method.get('async', False) else ''}def test_{cls_name.lower()}_{method['name']}(self, {cls_name.lower()}_instance):
        # TODO: Set up appropriate test parameters and mocks
        {'mock_result = AsyncMock()' if method.get('async', False) else 'mock_result = MagicMock()'}
        # TODO: Adjust expected parameters and return values
        with patch('some.module.path', mock_result):
            {'result = await ' + cls_name.lower() + '_instance.' + method['name'] + '()' if method.get('async', False) else 'result = ' + cls_name.lower() + '_instance.' + method['name'] + '()'}
            assert result is not None  # Replace with appropriate assertions
"""
        
        return test_code


class EnhancedTestGenerator:
    """
    Main class for generating enhanced tests with improved workflow
    """
    
    def __init__(
        self, 
        api_key: str,
        model: str = "anthropic/claude-3-opus-20240229",
        rate_limiter: Optional[Any] = None,
        site_url: str = "https://test-generator.com",
        site_name: str = "Enhanced Test Generator"
    ):
        """
        Initialize the enhanced test generator
        """
        self.api_key = api_key
        self.model = model
        self.rate_limiter = rate_limiter
        self.site_url = site_url
        self.site_name = site_name
        
        # Initialize API client based on model provider
        if "anthropic" in model.lower():
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=api_key)
                self.api_type = "anthropic"
            except ImportError:
                logger.error("Anthropic package not installed. Install with: pip install anthropic")
                raise
        elif "openai" in model.lower() or "gpt" in model.lower():
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                self.api_type = "openai"
            except ImportError:
                logger.error("OpenAI package not installed. Install with: pip install openai")
                raise
        else:
            # Default to OpenRouter for other models
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key
                )
                self.api_type = "openrouter"
            except ImportError:
                logger.error("OpenAI package not installed. Install with: pip install openai")
                raise
    
    def generate_test(self, module_path: str, coverage_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate test cases for a module with enhanced workflow
        """
        try:
            # Read module code
            with open(module_path, 'r', encoding='utf-8') as f:
                module_code = f.read()
            
            # Get module name and import path
            module_name = os.path.basename(module_path)
            if module_name.endswith('.py'):
                module_name = module_name[:-3]
            
            # Get module directory for import path construction
            module_dir = os.path.dirname(module_path)
            if module_dir.startswith('src/'):
                module_dir = module_dir[4:]
            
            # Construct import path
            import_path = module_dir.replace('/', '.')
            if import_path:
                full_import_path = f"{import_path}.{module_name}"
            else:
                full_import_path = module_name
            
            # Step 1: Analyze code to detect requirements
            logger.info(f"Analyzing module: {module_path}")
            requirements = TestPatternDetector.detect_requirements(module_code)
            
            # Step 2: Extract model fields and generate mock data examples
            models = TestPatternDetector.extract_model_fields(module_code)
            mock_examples = MockDataGenerator.generate_mock_data_examples(models)
            
            # Step 3: Create AI prompts for test generation
            system_prompt = self._create_system_prompt(requirements)
            user_prompt = self._create_user_prompt(
                module_path, 
                module_code,
                full_import_path,
                coverage_data,
                requirements,
                mock_examples
            )
            
            # Step 4: Generate test code with AI
            test_code = self._generate_with_ai(system_prompt, user_prompt)
            
            # Step 5: Post-process and improve the generated test code
            enhanced_test_code = TestCodePostProcessor.post_process_test_code(
                test_code,
                requirements,
                models,
                module_path,
                full_import_path
            )
            
            return enhanced_test_code
            
        except Exception as e:
            logger.error(f"Error generating test: {e}")
            # Generate fallback test if anything fails
            return TestCodePostProcessor.generate_fallback_test(
                module_path, 
                full_import_path,
                requirements,
                models
            )
    
    def _create_system_prompt(self, requirements: Dict[str, Any]) -> str:
        """
        Create system prompt for AI based on detected requirements
        """
        framework = requirements.get('framework', 'pytest')
        
        system_prompt = (
            "You are a Python testing expert specializing in writing high-quality test code. "
            "You ONLY respond with valid, executable Python test code without explanations, commentary, or markdown. "
            "Never start with explanations or questions - just provide working test code that can be directly saved to a file."
        )
        
        # Add framework-specific instructions
        if framework == 'pytest':
            system_prompt += (
                "\nCreate pytest-compatible tests with appropriate fixtures and assertions. "
                "Use pytest best practices including parametrization for similar test cases."
            )
        else:
            system_prompt += (
                "\nCreate unittest-compatible tests with proper TestCase classes and setup/teardown methods. "
                "Follow unittest best practices for test organization."
            )
        
        # Add async-specific instructions
        if requirements.get('async', False):
            system_prompt += (
                "\nThis code uses async/await patterns. Be sure to:"
                "\n- Mark async test functions with @pytest.mark.asyncio"
                "\n- Use AsyncMock for mocking async functions"
                "\n- Handle async iterables correctly with __aiter__ mocks"
                "\n- Properly await async function calls"
            )
        
        # Add model validation instructions
        if requirements.get('strict_typing', False) or requirements.get('needs_model_validation', False):
            system_prompt += (
                "\nThis code uses strict data models. Important requirements:"
                "\n- All model fields must be included in mock data"
                "\n- Type validation must be respected (strings for strings, ints for ints, etc.)"
                "\n- Required fields must never be omitted"
                "\n- Ensure all enum values are valid choices"
            )
        
        return system_prompt
    
    def _create_user_prompt(
        self,
        module_path: str,
        module_code: str,
        import_path: str,
        coverage_data: Optional[Dict[str, Any]],
        requirements: Dict[str, Any],
        mock_examples: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Create detailed user prompt for AI
        """
        # Get module name for imports
        module_name = os.path.basename(module_path)
        if module_name.endswith('.py'):
            module_name = module_name[:-3]
        
        # Format coverage info if available
        coverage_info = ""
        if coverage_data:
            coverage_pct = coverage_data.get("coverage_pct", 0)
            low_coverage_functions = coverage_data.get("low_coverage_functions", [])
            
            coverage_info = f"Current coverage: {coverage_pct:.1f}%\n\n"
            
            if low_coverage_functions:
                coverage_info += "Low coverage functions:\n"
                for func in low_coverage_functions:
                    class_prefix = f"{func.get('class')}." if func.get('class') else ""
                    coverage_info += f"- {class_prefix}{func['name']} (line {func['line']}): {func['coverage_pct']:.1f}% coverage\n"
            else:
                coverage_info += "No specific low coverage areas identified.\n"
        else:
            coverage_info = "No coverage data available. Focus on complete test coverage for all public functions.\n"
        
        # Create framework guidance
        framework = requirements.get('framework', 'pytest')
        framework_guidance = f"Testing Framework: {framework}\n"
        
        if requirements.get('async', False):
            framework_guidance += "- This code uses async/await. Use pytest.mark.asyncio for async tests.\n"
            framework_guidance += "- For async mocks, use AsyncMock and __aiter__ for async iterables.\n"
        
        if requirements.get('needs_mocking', False):
            framework_guidance += "- Use unittest.mock for all mocking needs (not pytest-mock).\n"
        
        # Create mock data examples if available
        mock_data_examples = ""
        if mock_examples:
            mock_data_examples = "## MOCK DATA EXAMPLES\n"
            mock_data_examples += "Use these examples to ensure proper model validation:\n\n"
            
            for model_name, mock_data in mock_examples.items():
                mock_data_examples += f"### {model_name} Example\n"
                mock_data_examples += "```python\n"
                mock_data_examples += f"{model_name.upper()}_MOCK_DATA = {{\n"
                
                for field_name, field_value in mock_data.items():
                    if isinstance(field_value, str):
                        mock_data_examples += f'    "{field_name}": "{field_value}",\n'
                    else:
                        mock_data_examples += f'    "{field_name}": {field_value},\n'
                
                mock_data_examples += "}\n```\n\n"
                mock_data_examples += "IMPORTANT: Include ALL fields shown above to avoid validation errors.\n\n"
        
        # Create the full prompt
        prompt = f"""
# TEST GENERATION ASSIGNMENT

Write clean, production-quality {framework} test code for this Python module.

## MODULE DETAILS
- Filename: {module_name}.py
- Import path: {import_path}
- {coverage_info}

## MODULE CODE
```python
{module_code}
```

## FRAMEWORK REQUIREMENTS
{framework_guidance}

{mock_data_examples}

## STRICT GUIDELINES
1. Write ONLY valid Python test code with no explanations or markdown
2. Include proper imports for ALL required packages and modules
3. Import the module under test correctly: `from {import_path} import *`
4. Focus on complete test coverage for all public functions
5. For mock responses, include ALL required fields in model dictionaries
6. Use appropriate fixtures and test setup for the testing framework
7. When asserting values, ensure case sensitivity and exact type matching
8. Create descriptive test function names that indicate what is being tested
9. Include proper error handling and edge case testing
10. IMPORTANT: For async code, use AsyncMock properly and await all async calls
11. IMPORTANT: Use unittest.mock directly instead of pytest-mock

Your response MUST be valid Python code that can be directly saved and executed with {framework}.
"""
        return prompt.strip()
    
    def _generate_with_ai(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate test code using AI with appropriate API calls
        """
        logger.info("Generating test code with AI...")
        
        if self.api_type == "anthropic":
            return self._generate_with_anthropic(system_prompt, user_prompt)
        else:  # OpenAI or OpenRouter
            return self._generate_with_openai(system_prompt, user_prompt)
    
    def _generate_with_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate test code using Anthropic Claude API
        """
        try:
            response = self.client.messages.create(
                model=self.model.split('/')[-1],
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,
            )
            
            test_code = response.content[0].text
            
            # Extract only Python code from response if wrapped in code blocks
            if "```python" in test_code and "```" in test_code:
                test_code = test_code.split("```python")[1].split("```")[0].strip()
            elif "```" in test_code:
                test_code = test_code.split("```")[1].split("```")[0].strip()
            
            return test_code
            
        except Exception as e:
            logger.error(f"Error generating with Anthropic: {e}")
            return ""
    
    def _generate_with_openai(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate test code using OpenAI or OpenRouter API
        """
        try:
            # Extra headers for OpenRouter
            extra_headers = {}
            if self.api_type == "openrouter":
                extra_headers = {
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                }
            
            response = self.client.chat.completions.create(
                extra_headers=extra_headers if self.api_type == "openrouter" else {},
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,
            )
            
            test_code = response.choices[0].message.content
            
            # Extract only Python code from response if wrapped in code blocks
            if "```python" in test_code and "```" in test_code:
                test_code = test_code.split("```python")[1].split("```")[0].strip()
            elif "```" in test_code:
                test_code = test_code.split("```")[1].split("```")[0].strip()
            
            return test_code
            
        except Exception as e:
            logger.error(f"Error generating with OpenAI/OpenRouter: {e}")
            return ""
    
    def write_test_file(self, module_path: str, test_code: str) -> str:
        """
        Write test code to an appropriate file based on module path
        """
        try:
            # Determine output file structure
            module_rel_path = module_path
            
            # If module_path starts with 'src/', remove it
            if module_rel_path.startswith('src/'):
                module_rel_path = module_rel_path[4:]
            
            # Get module name and directory
            module_dir = os.path.dirname(module_rel_path)
            module_name = os.path.basename(module_path)
            if module_name.endswith('.py'):
                module_name = module_name[:-3]
            
            # Create test directory
            test_dir = os.path.join('tests', module_dir)
            os.makedirs(test_dir, exist_ok=True)
            
            # Determine test file name
            test_file = os.path.join(test_dir, f'test_{module_name}.py')
            
            # If file already exists, create a new version
            if os.path.exists(test_file):
                version = 1
                while os.path.exists(f"{test_dir}/test_{module_name}_v{version}.py"):
                    version += 1
                test_file = f"{test_dir}/test_{module_name}_v{version}.py"
            
            # Write test code to file
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_code)
            
            logger.info(f"Test file saved to: {test_file}")
            return test_file
            
        except Exception as e:
            logger.error(f"Error writing test file: {e}")
            raise


def generate_and_write_test(
    module_path: str, 
    api_key: str,
    model: str = "anthropic/claude-3-haiku-20240307",
    output_dir: str = None
) -> str:
    """
    High-level function to generate and write a test for a module
    """
    # Validate module path
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Module not found: {module_path}")
    
    # Initialize the enhanced test generator
    generator = EnhancedTestGenerator(api_key, model)
    
    # Generate the test code
    test_code = generator.generate_test(module_path)
    
    # Write the test to a file
    test_file = generator.write_test_file(module_path, test_code)
    
    return test_file


def fix_existing_test(test_file_path: str) -> bool:
    """
    Fix an existing test file with common issues
    """
    if not os.path.exists(test_file_path):
        logger.error(f"Test file not found: {test_file_path}")
        return False
    
    try:
        # Read the test file
        with open(test_file_path, 'r', encoding='utf-8') as f:
            test_code = f.read()
        
        # Try to determine the module path from the test file
        module_path = None
        import_pattern = re.compile(r'from\s+([\w.]+)\s+import\s+\*')
        for line in test_code.split('\n'):
            match = import_pattern.search(line)
            if match:
                import_path = match.group(1)
                # Convert import path to module path
                module_path = import_path.replace('.', '/') + '.py'
                if os.path.exists(module_path):
                    break
                elif os.path.exists(f"src/{module_path}"):
                    module_path = f"src/{module_path}"
                    break
        
        # If module path found, analyze it
        requirements = {}
        models = {}
        
        if module_path and os.path.exists(module_path):
            with open(module_path, 'r', encoding='utf-8') as f:
                module_code = f.read()
            
            # Analyze the module
            requirements = TestPatternDetector.detect_requirements(module_code)
            models = TestPatternDetector.extract_model_fields(module_code)
        else:
            # Try to infer requirements from the test file itself
            requirements = TestPatternDetector.detect_requirements(test_code)
        
        # Apply fixes
        enhanced_test_code = TestCodePostProcessor.fix_async_issues(test_code, requirements)
        mock_examples = MockDataGenerator.generate_mock_data_examples(models)
        enhanced_test_code = TestCodePostProcessor.fix_model_validation_issues(enhanced_test_code, mock_examples)
        enhanced_test_code = TestCodePostProcessor.ensure_framework_imports(enhanced_test_code, requirements)
        
        # Write the fixed test file
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_test_code)
        
        logger.info(f"Successfully fixed test file: {test_file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error fixing test file: {e}")
        return False


def main():
    """
    Main function for command-line usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Test Generator")
    parser.add_argument('--module', help='Path to module for test generation')
    parser.add_argument('--api-key', help='API key for AI service')
    parser.add_argument('--model', default='anthropic/claude-3-haiku-20240307', help='Model to use')
    parser.add_argument('--fix-test', help='Path to test file to fix')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine API key from environment if not provided
    api_key = args.api_key
    if not api_key:
        if "anthropic" in args.model.lower():
            api_key = os.environ.get('ANTHROPIC_API_KEY')
        else:
            api_key = os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENROUTER_API_KEY')
    
    if not api_key:
        logger.error("API key not provided. Use --api-key or set appropriate environment variable.")
        return 1
    
    # Fix existing test
    if args.fix_test:
        success = fix_existing_test(args.fix_test)
        return 0 if success else 1
    
    # Generate new test
    if args.module:
        try:
            test_file = generate_and_write_test(args.module, api_key, args.model)
            logger.info(f"Test generated at: {test_file}")
            return 0
        except Exception as e:
            logger.error(f"Error generating test: {e}")
            return 1
    
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main()) 
    
