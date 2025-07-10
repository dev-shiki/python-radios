#!/usr/bin/env python
"""
Generator test berbasis AI yang bekerja dengan struktur project Python apapun.
Fokus pada sistem asinkron dan coverage analysis.
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
    """Generator test universal sederhana untuk project Python apapun."""
    
    def __init__(self, 
                 api_key: str,
                 coverage_threshold: float = 80.0,
                 model: str = "anthropic/claude-3.7-sonnet"):
        """Inisialisasi dengan konfigurasi minimal."""
        self.api_key = api_key
        self.coverage_threshold = coverage_threshold
        self.model = model
        self.openai_client = self._setup_openai()
        self.module_name = ""
        self.test_file_path = None
        self.api_logs = []
        self.start_time = time.time()
    
    def _setup_openai(self):
        """Konfigurasi client OpenAI untuk berbagai provider."""
        # Validasi API key
        if not self.api_key or self.api_key.strip() == "":
            raise ValueError("API key diperlukan tetapi tidak disediakan")
        
        # Dukungan multiple provider dengan pengecekan pattern API key
        if "openrouter" in self.api_key.lower() or os.getenv("OPENROUTER_API_KEY"):
            try:
                client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
                # Skip test API key untuk menghindari masalah rate limiting
                return client
            except Exception as e:
                raise ValueError(f"API key OpenRouter tidak valid: {e}")
        else:
            try:
                client = openai.OpenAI(api_key=self.api_key)
                # Skip test API key untuk menghindari masalah rate limiting
                return client
            except Exception as e:
                raise ValueError(f"API key OpenAI tidak valid: {e}")
    
    def find_files_needing_tests(self, 
                               coverage_data: Dict = None,
                               target_files: List[str] = None) -> List[Path]:
        """
        Mencari file Python yang memerlukan test.
        
        Args:
            coverage_data: Data coverage dari pytest-cov
            target_files: File spesifik yang ditargetkan
        
        Returns:
            List path file Python yang memerlukan test
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
        prompt = self.create_prompt(file_path, source_code)
        
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
        
        # Remove .py extension
        if parts[-1].endswith('.py'):
            parts[-1] = parts[-1][:-3]
        
        # Convert to module notation
        return '.'.join(parts)
    
    def _get_function_args(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[Dict]:
        """Extract function arguments with types and defaults."""
        args = []
        
        # Regular arguments
        for i, arg in enumerate(node.args.args):
            arg_info = {
                'name': arg.arg,
                'annotation': ast.unparse(arg.annotation) if arg.annotation else None,
                'default': None
            }
            
            # Check for defaults (defaults are for the last N args)
            defaults = node.args.defaults
            if defaults and i >= len(node.args.args) - len(defaults):
                default_idx = i - (len(node.args.args) - len(defaults))
                arg_info['default'] = ast.unparse(defaults[default_idx])
            
            args.append(arg_info)
        
        # Keyword-only arguments
        for i, arg in enumerate(node.args.kwonlyargs):
            arg_info = {
                'name': arg.arg,
                'annotation': ast.unparse(arg.annotation) if arg.annotation else None,
                'default': None
            }
            
            # Check for kw_defaults
            if node.args.kw_defaults and i < len(node.args.kw_defaults):
                if node.args.kw_defaults[i] is not None:
                    arg_info['default'] = ast.unparse(node.args.kw_defaults[i])
            
            args.append(arg_info)
        
        return args
    
    def _extract_return_type(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[str]:
        """Extract return type annotation."""
        if node.returns:
            return ast.unparse(node.returns)
        return None
    
    def _extract_uncovered_functions(self, source_code: str) -> Dict[str, Dict]:
        """Extract detailed information about functions/methods that need testing."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return {}
        
        functions = {}
        
        class FunctionVisitor(ast.NodeVisitor):
            def __init__(self):
                self.class_name = None
            
            def visit_ClassDef(self, node):
                old_class = self.class_name
                self.class_name = node.name
                self.generic_visit(node)
                self.class_name = old_class
            
            def visit_FunctionDef(self, node):
                self._process_function(node)
                self.generic_visit(node)
            
            def visit_AsyncFunctionDef(self, node):
                self._process_function(node)
                self.generic_visit(node)
            
            def _process_function(self, node):
                full_name = f"{self.class_name}.{node.name}" if self.class_name else node.name
                
                # Skip private methods and special methods for now
                if node.name.startswith('_') and not node.name.startswith('__'):
                    return
                
                # Extract docstring
                docstring = ast.get_docstring(node)
                
                functions[full_name] = {
                    'name': node.name,
                    'class_name': self.class_name,
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'args': self._get_function_args(node),
                    'return_type': self._extract_return_type(node),
                    'docstring': docstring,
                    'decorators': [ast.unparse(dec) for dec in node.decorator_list],
                    'line_number': node.lineno
                }
        
        visitor = FunctionVisitor()
        visitor.visit(tree)
        
        return functions
    
    def _identify_used_libraries(self, source_code: str) -> List[str]:
        """Identify libraries used in the source code."""
        libraries = set()
        
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    libraries.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    libraries.add(node.module.split('.')[0])
        
        # Common async and testing related libraries
        common_async_libs = ['asyncio', 'aiohttp', 'aiofiles', 'asyncpg', 'motor']
        testing_libs = ['pytest', 'unittest', 'mock']
        
        # Filter to commonly used libraries that affect testing
        relevant_libs = [lib for lib in libraries 
                        if lib in common_async_libs + testing_libs + 
                        ['requests', 'httpx', 'fastapi', 'flask', 'django', 'sqlalchemy', 'pydantic']]
        
        return relevant_libs
    
    def _extract_detailed_function_info(self, uncovered_functions: Dict[str, Dict]) -> str:
        """Extract detailed information about functions for the prompt."""
        if not uncovered_functions:
            return "No specific functions identified - generate comprehensive tests for all public methods."
        
        details = []
        for func_name, func_info in uncovered_functions.items():
            args_str = ", ".join([
                f"{arg['name']}: {arg['annotation'] or 'Any'}" + 
                (f" = {arg['default']}" if arg['default'] else "")
                for arg in func_info['args']
            ])
            
            return_type = func_info['return_type'] or 'None'
            async_marker = "async " if func_info['is_async'] else ""
            class_prefix = f"Class: {func_info['class_name']}, " if func_info['class_name'] else ""
            
            details.append(
                f"- {class_prefix}{async_marker}def {func_info['name']}({args_str}) -> {return_type}\n"
                f"  Line: {func_info['line_number']}\n" +
                (f"  Docstring: {func_info['docstring'][:100]}..." if func_info['docstring'] else "")
            )
        
        return "\n".join(details)
    
    def find_model_references(self, source_file: Path) -> List[Path]:
        """Find model reference files in the project."""
        model_files = set()
        source_dir = self._find_project_root(source_file)
        
        # Read source code to find imports
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except Exception:
            return []
        
        # Extract imports
        imports = self._extract_imports(source_code)
        
        # For each import, try to find the corresponding file
        for import_name in imports:
            # Skip standard library and third-party imports
            if import_name in ['os', 'sys', 'json', 'time', 'datetime', 'typing', 'pathlib',
                             'asyncio', 'aiohttp', 'requests', 'pydantic', 'dataclasses']:
                continue
            
            # Convert import to file path
            import_parts = import_name.split('.')
            
            # Try different combinations
            for i in range(len(import_parts), 0, -1):
                parts = import_parts[:i]
                
                # Try as package/__init__.py
                package_path = source_dir.joinpath(*parts) / '__init__.py'
                if package_path.exists() and self._contains_model_definitions(package_path):
                    model_files.add(package_path)
                
                # Try as module.py
                file_path = source_dir.joinpath(*parts).with_suffix('.py')
                if file_path.exists() and self._contains_model_definitions(file_path):
                    model_files.add(file_path)
        
        # Also check common model file names in the same directory
        source_parent = source_file.parent
        common_model_files = ['models.py', 'model.py', 'schemas.py', 'schema.py', 'types.py']
        for model_file in common_model_files:
            file_path = source_parent / model_file
            if file_path.exists() and file_path != source_file:
                model_files.add(file_path)
        
        # Check parent directories for models
        current_dir = source_parent
        for _ in range(3):  # Look up to 3 levels up
            for model_file in common_model_files:
                file_path = current_dir / model_file
                if file_path.exists():
                    model_files.add(file_path)
            current_dir = current_dir.parent
            if current_dir == source_dir:
                break
        
        return list(model_files)
    
    def _find_project_root(self, file_path: Path) -> Path:
        """Find the project root directory."""
        current = file_path.parent
        
        # Look for common project root indicators
        root_indicators = ['pyproject.toml', 'setup.py', 'requirements.txt', '.git', 'poetry.lock']
        
        while current.parent != current:  # Not at filesystem root
            if any((current / indicator).exists() for indicator in root_indicators):
                return current
            current = current.parent
        
        return file_path.parent
    
    def _extract_imports(self, source_code: str) -> List[str]:
        """Extract import statements from source code."""
        imports = []
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except SyntaxError:
            pass
        return imports
    
    def _contains_model_definitions(self, file_path: Path) -> bool:
        """Check if a file contains model definitions."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple heuristics for model definitions
            model_indicators = [
                'class.*BaseModel',
                'class.*Model',
                '@dataclass',
                'class.*Schema',
                'TypedDict',
                'NamedTuple',
                'Enum',
                'dataclasses.dataclass',
                'pydantic',
                'from typing import',
                'Union',
                'Optional',
                'List',
                'Dict'
            ]
            
            for indicator in model_indicators:
                if re.search(indicator, content, re.IGNORECASE):
                    return True
                    
            return False
        except Exception:
            return False
    
    def extract_model_definitions(self, file_path: Path) -> str:
        """Extract model definitions from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse and extract class definitions
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return ""
            
            model_code = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it's likely a model class
                    class_code = ast.get_source_segment(content, node)
                    if class_code and any(keyword in class_code.lower() for keyword in 
                                        ['basemodel', 'schema', 'dataclass', 'model', 'enum']):
                        model_code.append(class_code)
                
                # Also extract TypedDict, NamedTuple, etc.
                elif isinstance(node, ast.Assign):
                    if hasattr(node, 'value') and isinstance(node.value, ast.Call):
                        if hasattr(node.value.func, 'id'):
                            if node.value.func.id in ['TypedDict', 'NamedTuple']:
                                assign_code = ast.get_source_segment(content, node)
                                if assign_code:
                                    model_code.append(assign_code)
            
            # If no specific classes found, return imports and type hints
            if not model_code:
                lines = content.split('\n')
                relevant_lines = []
                for line in lines:
                    if (line.strip().startswith('from typing') or 
                        line.strip().startswith('import typing') or
                        line.strip().startswith('from pydantic') or
                        line.strip().startswith('import pydantic') or
                        line.strip().startswith('from dataclasses') or
                        'Union[' in line or 'Optional[' in line or 'List[' in line):
                        relevant_lines.append(line)
                
                if relevant_lines:
                    return '\n'.join(relevant_lines[:10])  # First 10 relevant lines
            
            return '\n\n'.join(model_code[:5])  # First 5 model definitions
            
        except Exception as e:
            return f"# Error reading {file_path}: {e}"
    
    def find_enum_references(self, source_file: Path) -> List[Path]:
        """Find enum reference files in the project."""
        enum_files = set()
        source_dir = self._find_project_root(source_file)
        
        # Read source code to find imports
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except Exception:
            return []
        
        # Extract imports
        imports = self._extract_imports(source_code)
        
        # For each import, try to find the corresponding file
        for import_name in imports:
            # Skip standard library imports
            if import_name in ['os', 'sys', 'json', 'time', 'datetime', 'typing', 'pathlib']:
                continue
            
            # Convert import to file path
            import_parts = import_name.split('.')
            
            # Try different combinations
            for i in range(len(import_parts), 0, -1):
                parts = import_parts[:i]
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

    def create_prompt(self, source_file: Path, source_code: str) -> str:
        """
        Membuat prompt hybrid: struktur multi-stage MSPP + requirements dari prompt utama.
        """
        uncovered_functions = self._extract_uncovered_functions(source_code)
        used_libraries = self._identify_used_libraries(source_code)
        model_reference_files = self.find_model_references(source_file)
        enum_reference_files = self.find_enum_references(source_file)
        all_reference_files = list(set(model_reference_files + enum_reference_files))
        model_reference_code = ""
        if all_reference_files:
            model_reference_code = "\nMODELS AND TYPES:\n"
            for file_path in all_reference_files:
                model_definitions = self.extract_model_definitions(file_path)
                if model_definitions:
                    model_reference_code += f"\n# From {file_path}\n```python\n{model_definitions}\n```\n"
        function_details = self._extract_detailed_function_info(uncovered_functions)
        prompt = f"""Buatlah tes pytest lengkap untuk kode di bawah ini menggunakan pendekatan multi-tahap.

TAHAP 1 - KONTEKSTUALISASI:
- File: {source_file}
- Path modul: {self.module_name}
- Path file tes: {self.test_file_path}
- Kode sumber:
```python
{source_code}
```
{model_reference_code}- Fungsi yang memerlukan tes:
{function_details}- Library yang digunakan: {', '.join(used_libraries)}

TAHAP 2 - DESAIN STRUKTUR TES:
- Pilih framework pytest
- Buat kerangka tes dengan fixture yang sesuai
- Inisialisasi objek mock (Mock/AsyncMock dari unittest.mock)
- Organisasi kelas tes dan impor yang diperlukan

TAHAP 3 - IMPLEMENTASI LOGIKA TES:
- Implementasikan assertion untuk semua skenario
- Tangani kasus khusus dan error
- Implementasi pola async (pytest.mark.asyncio, await, AsyncMock)
- Setup perilaku mock (return_value, side_effect, context manager, iterator)
- Validasi parameter dan hasil

TAHAP 4 - OPTIMASI & VALIDASI:
- Optimalkan tes agar mudah dipelihara dan coverage maksimal
- Validasi syntax dan kualitas kode
- Pastikan tes siap diintegrasikan dan mengikuti praktik terbaik pytest

PERSYARATAN:
- Sertakan SEMUA field saat inisialisasi model (wajib & opsional)
- Pastikan nama field sesuai dengan definisi model (case-sensitive)
- Gunakan Mock/AsyncMock dari unittest.mock, atur return_value/side_effect SEBELUM digunakan
- Untuk async: gunakan pytest.mark.asyncio, selalu await panggilan async, konfigurasi AsyncMock dengan benar
- Untuk validasi: gunakan unittest.mock.ANY untuk parameter, perbandingan set untuk koleksi, uji struktur & tipe respons, uji pagination
- Gunakan parametrize untuk variasi, pytest.raises untuk exception, dan kelompokkan tes dalam kelas jika logis

Kembalikan HANYA kode pytest yang dapat dijalankan tanpa penjelasan atau markdown. Kode harus dapat digunakan langsung tanpa modifikasi apa pun.
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate tests for Python modules')
    parser.add_argument('--module', required=False, help='Specific module path to generate tests for')
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
    
    # Initialize generator
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