#!/usr/bin/env python3
"""
Generator Test Otomatis dengan Claude API via OpenRouter

Utilitas untuk menghasilkan test case untuk modul Python dengan coverage rendah
menggunakan Claude API melalui OpenRouter dengan pengelolaan rate limit.

Penggunaan:
  python test_generator.py --module path/to/module.py --coverage-threshold 80 \
    --api-key YOUR_OPENROUTER_API_KEY --rate-limit-rpm 5 --rate-limit-input-tpm 25000 \
    --rate-limit-output-tpm 5000
"""

import os
import sys
import time
import argparse
import logging
import xml.etree.ElementTree as ET
import re
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import openai
from openai import OpenAI

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("claude-test-generator")

@dataclass
class RateLimitConfig:
    """Konfigurasi Rate Limit untuk API"""
    requests_per_minute: int
    input_tokens_per_minute: int 
    output_tokens_per_minute: int

class ApiRateLimiter:
    """
    Rate limiter untuk API yang mempertimbangkan:
    - Jumlah permintaan per menit
    - Batas token input per menit
    - Batas token output per menit
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_timestamps = deque()
        self.input_token_usage = deque()
        self.output_token_usage = deque()
        self.lock = threading.RLock()
        logger.info(f"Rate limiter initiated with config: {config}")
        
    def _cleanup_old_entries(self, queue_data, time_window=60):
        """Membersihkan entri yang lebih lama dari time_window detik"""
        current_time = time.time()
        while queue_data and (current_time - queue_data[0][0]) > time_window:
            queue_data.popleft()
    
    def _get_current_usage(self, queue_data):
        """Mendapatkan penggunaan saat ini dalam periode rate limiting"""
        self._cleanup_old_entries(queue_data)
        return sum(item[1] for item in queue_data)
        
    def wait_for_capacity(self, input_tokens: int, estimated_output_tokens: int) -> None:
        """
        Menunggu sampai ada kapasitas yang cukup untuk mengirim permintaan baru.
        Memblokir sampai semua kondisi rate limit terpenuhi.
        """
        while True:
            with self.lock:
                # Membersihkan entri lama
                self._cleanup_old_entries(self.request_timestamps)
                self._cleanup_old_entries(self.input_token_usage)
                self._cleanup_old_entries(self.output_token_usage)
                
                # Mengecek penggunaan saat ini
                current_requests = len(self.request_timestamps)
                current_input_tokens = self._get_current_usage(self.input_token_usage)
                current_output_tokens = self._get_current_usage(self.output_token_usage)
                
                # Cek apakah semua kondisi terpenuhi
                requests_ok = current_requests < self.config.requests_per_minute
                input_tokens_ok = (current_input_tokens + input_tokens) <= self.config.input_tokens_per_minute
                output_tokens_ok = (current_output_tokens + estimated_output_tokens) <= self.config.output_tokens_per_minute
                
                if requests_ok and input_tokens_ok and output_tokens_ok:
                    # Cukup kapasitas, catat penggunaan baru
                    current_time = time.time()
                    self.request_timestamps.append((current_time, 1))
                    self.input_token_usage.append((current_time, input_tokens))
                    self.output_token_usage.append((current_time, estimated_output_tokens))
                    
                    logger.info(f"Request allowed - Usage stats: Requests={current_requests+1}/{self.config.requests_per_minute}, " +
                                f"Input tokens={(current_input_tokens + input_tokens)}/{self.config.input_tokens_per_minute}, " +
                                f"Output tokens={(current_output_tokens + estimated_output_tokens)}/{self.config.output_tokens_per_minute}")
                    return
            
            # Jika tidak ada kapasitas yang cukup, tunggu sebentar
            time_to_wait = 2.0  # Tunggu 2 detik sebelum mencoba lagi
            logger.info(f"Rate limit tercapai, menunggu {time_to_wait} detik sebelum mencoba lagi...")
            time.sleep(time_to_wait)
    
    def record_actual_usage(self, actual_output_tokens: int):
        """
        Memperbarui penggunaan token output aktual setelah permintaan selesai.
        Ini untuk memastikan akurasi pelacakan token output.
        """
        with self.lock:
            if self.output_token_usage:
                # Perbarui entri terakhir dengan nilai sebenarnya
                timestamp, _ = self.output_token_usage.pop()
                self.output_token_usage.append((timestamp, actual_output_tokens))
                logger.info(f"Updated output token usage to {actual_output_tokens}")

class TokenCounter:
    """Utilitas untuk menghitung token dalam teks"""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Perkiraan kasar berapa token dalam teks.
        Pendekatan sederhana: ~4 karakter per token untuk bahasa Inggris.
        """
        return max(1, len(text) // 4)
    
    @staticmethod
    def count_tokens_with_openai(client: OpenAI, text: str, model: str) -> int:
        """
        Menghitung token menggunakan tiktoken atau perkiraan.
        """
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(model.split('/')[-1])
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Gagal menghitung token dengan tiktoken: {e}")
            # Fallback ke estimasi kasar
            return TokenCounter.estimate_tokens(text)

class CoverageAnalyzer:
    """Menganalisis laporan coverage untuk mengidentifikasi area dengan coverage rendah"""
    
    @staticmethod
    def parse_coverage_data(module_path: str, coverage_file: str = "coverage.xml") -> dict:
        """
        Mengekstrak data coverage untuk modul tertentu dari laporan coverage XML
        """
        if not os.path.exists(coverage_file):
            logger.error(f"File coverage tidak ditemukan: {coverage_file}")
            return {}
        
        try:
            tree = ET.parse(coverage_file)
            root = tree.getroot()
            
            coverage_data = {
                "line_rate": 0.0,
                "coverage_pct": 0.0,
                "uncovered_lines": [],
                "covered_lines": [],
                "low_coverage_functions": []
            }
            
            # Temukan modul dalam laporan
            for class_elem in root.findall('.//class'):
                filename = class_elem.attrib.get('filename')
                
                if filename == module_path:
                    # Dapatkan coverage rate
                    line_rate = float(class_elem.attrib.get('line-rate', 0))
                    coverage_data["line_rate"] = line_rate
                    coverage_data["coverage_pct"] = line_rate * 100
                    
                    # Dapatkan line coverage detail
                    for line in class_elem.findall('.//line'):
                        line_num = int(line.attrib.get('number', 0))
                        hits = int(line.attrib.get('hits', 0))
                        
                        if hits > 0:
                            coverage_data["covered_lines"].append(line_num)
                        else:
                            coverage_data["uncovered_lines"].append(line_num)
                    
                    break
            
            # Identifikasi fungsi dengan coverage rendah
            if os.path.exists(module_path):
                coverage_data["low_coverage_functions"] = CoverageAnalyzer.identify_low_coverage_functions(
                    module_path, coverage_data["covered_lines"], coverage_data["uncovered_lines"]
                )
            
            return coverage_data
            
        except Exception as e:
            logger.error(f"Error saat parsing laporan coverage: {e}")
            return {}
    
    @staticmethod
    def identify_low_coverage_functions(module_path: str, covered_lines: list, uncovered_lines: list) -> list:
        """
        Mengidentifikasi fungsi-fungsi dengan coverage rendah dalam modul
        """
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                code = f.read()
                
            # Regex untuk menemukan definisi fungsi
            function_pattern = re.compile(r'^\s*def\s+(\w+)\s*\(', re.MULTILINE)
            
            functions = []
            lines = code.split('\n')
            
            # Temukan semua definisi fungsi
            for match in function_pattern.finditer(code):
                func_name = match.group(1)
                line_num = code[:match.start()].count('\n') + 1
                
                # Temukan akhir fungsi (estimasi dengan indentasi)
                end_line = line_num
                indent_level = len(lines[line_num - 1]) - len(lines[line_num - 1].lstrip())
                
                for i in range(line_num, len(lines)):
                    line = lines[i]
                    if line.strip() and not line.isspace():
                        # Jika ini bukan baris kosong dan indentasi kurang dari atau sama dengan fungsi induk
                        if len(line) - len(line.lstrip()) <= indent_level and line.lstrip().startswith(('def ', 'class ')):
                            end_line = i - 1
                            break
                    if i == len(lines) - 1:
                        end_line = i
                
                # Hitung baris kode (skip docstring dan komentar)
                code_lines = []
                inside_docstring = False
                docstring_delim = None
                
                for i in range(line_num - 1, end_line):
                    line = lines[i].strip()
                    
                    # Skip docstring
                    if inside_docstring:
                        if docstring_delim in line:
                            inside_docstring = False
                        continue
                    
                    if line.startswith('"""') or line.startswith("'''"):
                        inside_docstring = True
                        docstring_delim = line[:3]
                        continue
                    
                    # Skip komentar
                    if not line.startswith('#') and line:
                        code_lines.append(i + 1)
                
                # Hitung coverage
                code_line_count = len(code_lines)
                covered_count = sum(1 for line in code_lines if line in covered_lines)
                
                if code_line_count > 0:
                    coverage_pct = covered_count / code_line_count * 100
                else:
                    coverage_pct = 100  # Jika tidak ada baris kode (misalnya hanya docstring)
                
                functions.append({
                    "name": func_name,
                    "start_line": line_num,
                    "end_line": end_line,
                    "code_lines": code_line_count,
                    "covered_lines": covered_count,
                    "coverage_pct": coverage_pct
                })
            
            # Filter fungsi dengan coverage rendah (<80%)
            low_coverage_funcs = [f for f in functions if f["coverage_pct"] < 80]
            
            return sorted(low_coverage_funcs, key=lambda x: x["coverage_pct"])
            
        except Exception as e:
            logger.error(f"Error saat mengidentifikasi fungsi dengan coverage rendah: {e}")
            return []

class TestGenerator:
    """
    Generator test yang menggunakan Claude API melalui OpenRouter untuk membuat test case
    """
    
    def __init__(self, api_key: str, model: str = "anthropic/claude-3.5-haiku-20241022", 
                 site_url: str = "https://test-generator-app.com", site_name: str = "Test Generator",
                 rate_limiter: ApiRateLimiter = None):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model
        self.site_url = site_url
        self.site_name = site_name
        self.rate_limiter = rate_limiter
    
    def generate_test(self, module_path: str, coverage_data: dict, test_framework: str = "pytest") -> str:
        """
        Generate test cases using Claude API via OpenRouter with improved validation
        """
        try:
            # Read module code
            with open(module_path, 'r', encoding='utf-8') as f:
                module_code = f.read()
            
            # Get module name without extension for imports
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
                
            # Prepare coverage information for prompt
            coverage_pct = coverage_data.get("coverage_pct", 0)
            low_coverage_functions = coverage_data.get("low_coverage_functions", [])
            
            # Build string of low coverage functions information
            low_coverage_info = ""
            if low_coverage_functions:
                low_coverage_info = "Low coverage functions:\n"
                for func in low_coverage_functions:
                    low_coverage_info += f"- {func['name']} (lines {func['start_line']}-{func['end_line']}): {func['coverage_pct']:.1f}% coverage\n"
            else:
                low_coverage_info = "All functions have partial coverage that needs improvement."
            
            # Determine module type based on code analysis
            is_class_based = "class " in module_code
            has_async = "async " in module_code
            
            # Look for specific package imports that might be needed
            needs_pytest_mock = "mocker" in module_code
            needs_async_mock = "AsyncMock" in module_code or "async def" in module_code
            
            # Craft system prompt to set expectations
            system_prompt = (
                "You are a Python testing expert specializing in writing high-quality test code. "
                "You ONLY respond with valid, executable Python code without any explanations, commentary, or markdown. "
                "Never start with explanations or questions - just provide working test code that can be directly saved to a file."
            )
            
            # Identify specific test dependencies based on module analysis
            test_dependencies = []
            if "pytest" in module_code or test_framework == "pytest":
                test_dependencies.append("pytest")
            if "async" in module_code:
                test_dependencies.append("pytest-asyncio")
            if "mock" in module_code or "patch" in module_code:
                test_dependencies.append("pytest-mock")
            
            # Craft the optimized user prompt for Claude
            prompt_text = f"""
    # TEST GENERATION ASSIGNMENT

    Write complete, production-ready {test_framework} test code for this Python module that currently has {coverage_pct:.1f}% test coverage.

    ## MODULE DETAILS
    - Filename: {module_name}.py
    - Import path: {full_import_path}
    - Current coverage: {coverage_pct:.1f}%
    - {'Contains async code - use pytest.mark.asyncio and AsyncMock' if has_async else ''}

    ## MODULE CODE
    ```python
    {module_code}
    ```

    ## COVERAGE ANALYSIS
    {low_coverage_info}

    ## REQUIRED DEPENDENCIES
    {'- This project uses pytest-mock: import pytest_mock and use the mocker fixture' if needs_pytest_mock else ''}
    {'- AsyncMock should be imported from unittest.mock' if needs_async_mock else ''}
    {'- For pytest.mark.asyncio, add proper fixture configuration' if has_async else ''}

    ## STRICT REQUIREMENTS
    1. Write ONLY runnable Python test code, with no explanations or questions
    2. Include proper imports at the top:
    - Import pytest, unittest.mock.patch, unittest.mock.AsyncMock if needed
    - Add 'import pytest_mock' if using the mocker fixture
    - Import the module under test using: `from {full_import_path} import *`
    3. If using 'mocker' fixture, add appropriate fixture setup for pytest-mock
    4. Focus primarily on functions with low/no coverage
    5. Use appropriate fixtures, mocks, and patches where needed
    6. Cover edge cases, boundary conditions, and error handling
    7. Follow standard {test_framework} naming and organization conventions
    8. Ensure all tests are independent and do not rely on external resources
    9. If using AsyncMock, import it properly: `from unittest.mock import AsyncMock`
    10. Include descriptive docstrings for test functions

    IMPORTANT: Your response MUST be valid Python code that can be executed with pytest. 
    DO NOT include any explanatory text, markdown formatting, or conversational elements.
    """
            
            # Count tokens and estimate output
            input_tokens = TokenCounter.count_tokens_with_openai(self.client, prompt_text, self.model)
            estimated_output_tokens = 3000  # Conservative estimate
            
            logger.info(f"Prompt contains {input_tokens} tokens, estimated output {estimated_output_tokens} tokens")
            
            # Rate limiting if enabled
            if self.rate_limiter:
                self.rate_limiter.wait_for_capacity(input_tokens, estimated_output_tokens)
            
            # Send request to OpenRouter for Claude API
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ],
                max_tokens=4000,
            )
            
            # Estimate actual output token usage
            actual_output_tokens = len(response.choices[0].message.content) // 4  # Rough estimate
            
            if self.rate_limiter:
                self.rate_limiter.record_actual_usage(actual_output_tokens)
            
            logger.info(f"Claude used approximately {actual_output_tokens} output tokens")
            
            # Extract code from response
            test_code = response.choices[0].message.content
            
            # Extract only Python code from response if wrapped in code blocks
            if "```python" in test_code and "```" in test_code:
                test_code = test_code.split("```python")[1].split("```")[0].strip()
            elif "```" in test_code:
                test_code = test_code.split("```")[1].split("```")[0].strip()
            
            # POST-PROCESSING: Ensure we have valid Python code
            # Remove any conversational elements or non-Python content
            if not test_code.startswith("import ") and not test_code.startswith("from ") and not test_code.startswith("#!/"):
                lines = test_code.split('\n')
                # Find first line that looks like valid Python code (import or class/function definition)
                for i, line in enumerate(lines):
                    if (line.startswith("import ") or line.startswith("from ") or 
                        line.startswith("def ") or line.startswith("class ")):
                        test_code = '\n'.join(lines[i:])
                        break
            
            # Ensure we have proper imports
            imports_to_include = []
            
            # Add core imports if not present
            if test_framework == "pytest" and "import pytest" not in test_code:
                imports_to_include.append("import pytest")
            
            # Check for pytest-mock 
            if "mocker" in test_code and "pytest_mock" not in test_code:
                imports_to_include.append("import pytest_mock")
            
            # Check for AsyncMock
            if "AsyncMock" in test_code and "from unittest.mock import AsyncMock" not in test_code:
                imports_to_include.append("from unittest.mock import AsyncMock, patch")
            elif "patch" in test_code and "unittest.mock" not in test_code:
                imports_to_include.append("from unittest.mock import patch")
            
            # Add import for the module under test if not present
            module_import = f"from {full_import_path} import *"
            if module_import not in test_code:
                imports_to_include.append(module_import)
            
            # Add imports at the beginning if needed
            if imports_to_include:
                test_code = '\n'.join(imports_to_include) + '\n\n' + test_code
            
            # Ensure mocker fixture is available if used
            if "def test_" in test_code and "mocker" in test_code and "@pytest.fixture" not in test_code and "pytest_mock" not in test_code:
                # Add pytest-mock fixture setup
                fixture_setup = """
    # Add pytest-mock fixture if not automatically available
    pytest_plugins = ['pytest_mock']
    """
                test_code = fixture_setup + "\n" + test_code
            
            # Ensure async properly set up if needed
            if "async def test_" in test_code and "pytest.mark.asyncio" not in test_code:
                test_code = test_code.replace("async def test_", "@pytest.mark.asyncio\nasync def test_")
            
            # Check if session initialization is needed for RadioBrowser class
            if "RadioBrowser" in test_code and "session=None" in test_code and "radio_browser.session" in test_code:
                # Fix the session initialization issue
                if "session=None" in test_code:
                    test_code = test_code.replace("session=None", "session=MagicMock()")
                    if "from unittest.mock import MagicMock" not in test_code:
                        test_code = "from unittest.mock import MagicMock\n" + test_code
            
            # Validate that the result is proper Python syntax
            try:
                compile(test_code, "<string>", "exec")
                logger.info("Successfully generated and validated test code")
            except SyntaxError as e:
                logger.warning(f"Generated code has syntax error: {e}")
                # Attempt to fix common issues
                if "would you like" in test_code.lower() or "i understand" in test_code.lower():
                    # This looks like conversational text, not code
                    logger.warning("Detected conversational text in output, attempting fallback generation")
                    # Create a basic test scaffold
                    test_code = self._generate_fallback_test(module_path, full_import_path)
            
            return test_code
            
        except Exception as e:
            logger.error(f"Error generating test: {e}")
            # If rate limiter exists, record minimal token usage
            if self.rate_limiter:
                self.rate_limiter.record_actual_usage(1)
            raise

    def _generate_fallback_test(self, module_path: str, import_path: str) -> str:
        """
        Generate a basic test scaffold as fallback when API generation fails
        """
        module_name = os.path.basename(module_path)
        if module_name.endswith('.py'):
            module_name = module_name[:-3]
        
        # Read module to identify classes and functions
        with open(module_path, 'r') as f:
            content = f.read()
        
        # Check for async code
        has_async = "async " in content
        has_radio_browser = "RadioBrowser" in content
        
        # Simple regex to find classes and functions
        classes = re.findall(r'class\s+(\w+)', content)
        functions = re.findall(r'def\s+(\w+)', content)
        
        # Remove private methods
        functions = [f for f in functions if not f.startswith('_')]
        
        # Create basic test scaffold
        test_code = f"""
    import pytest
    from unittest.mock import patch, AsyncMock, MagicMock
    import pytest_mock  # Add pytest-mock for mocker fixture
    from {import_path} import *

    # Add pytest-mock plugin
    pytest_plugins = ['pytest_mock']

    """
        
        # If asyncio is used, add the needed marker
        if has_async:
            test_code += """
    # Configure pytest for asyncio tests
    pytestmark = pytest.mark.asyncio
    """
        
        # Add fix for RadioBrowser tests if needed
        if has_radio_browser:
            test_code += """
    @pytest.fixture
    def radio_browser():
        # Create RadioBrowser with a mock session
        browser = RadioBrowser(user_agent="test_agent")
        browser.session = MagicMock()  # Ensure session is not None
        return browser
    """
        
        # Add test class for each class found
        for cls in classes:
            test_code += f"""
    class Test{cls}:
        @pytest.fixture
        def {cls.lower()}_instance(self, mocker):
            return {cls}()
            
        {'@pytest.mark.asyncio' if has_async else ''}
        {'async ' if has_async else ''}def test_{cls.lower()}_initialization(self, {cls.lower()}_instance):
            assert {cls.lower()}_instance is not None
    """
        
        # Add test for standalone functions
        standalone_funcs = [f for f in functions if not any(f"{c}.{f}" in content for c in classes)]
        for func in standalone_funcs:
            test_code += f"""
    {'@pytest.mark.asyncio' if has_async else ''}
    {'async ' if has_async else ''}def test_{func}(mocker):
        # TODO: Replace with appropriate test parameters and assertions
        {'mock_result = MagicMock()' if not has_async else 'mock_result = AsyncMock()'}
        {'result = ' + func + '()' if not has_async else 'result = await ' + func + '()'}
        assert result is not None
    """
        
        return test_code
    
    def write_test_file(self, module_path: str, test_code: str) -> str:
        """
        Menulis test code ke file yang sesuai
        """
        try:
            # Tentukan struktur direktori untuk test file
            module_rel_path = module_path
            
            # Jika module_path dimulai dengan 'src/', hilangkan
            if module_rel_path.startswith('src/'):
                module_rel_path = module_rel_path[4:]
            
            # Dapatkan nama modul dan direktori
            module_dir = os.path.dirname(module_rel_path)
            module_name = os.path.basename(module_path)
            if module_name.endswith('.py'):
                module_name = module_name[:-3]
            
            # Buat direktori test
            test_dir = os.path.join('tests', module_dir)
            os.makedirs(test_dir, exist_ok=True)
            
            # Tentukan nama file test
            test_file = os.path.join(test_dir, f'test_{module_name}.py')
            
            # Jika file sudah ada, buat versi baru
            if os.path.exists(test_file):
                version = 1
                while os.path.exists(f"{test_dir}/test_{module_name}_v{version}.py"):
                    version += 1
                test_file = f"{test_dir}/test_{module_name}_v{version}.py"
            
            # Tulis test code ke file
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_code)
            
            logger.info(f"Test file disimpan ke: {test_file}")
            return test_file
            
        except Exception as e:
            logger.error(f"Error saat menulis test file: {e}")
            raise

def main():
    """Fungsi utama untuk menjalankan test generator"""
    parser = argparse.ArgumentParser(description='Generate test cases untuk modul Python dengan Claude API via OpenRouter')
    
    parser.add_argument('--module', required=True, help='Path ke modul yang akan dibuatkan test')
    parser.add_argument('--coverage-threshold', type=float, default=80.0, 
                        help='Threshold persentase minimum coverage yang ditargetkan')
    parser.add_argument('--coverage-file', default='coverage.xml', help='Path ke file laporan coverage XML')
    parser.add_argument('--test-framework', default='pytest', choices=['pytest', 'unittest'], 
                        help='Framework test yang digunakan')
    
    parser.add_argument('--api-key', help='OpenRouter API key (default: dari env OPENROUTER_API_KEY)')
    parser.add_argument('--model', default='anthropic/claude-3.5-haiku-20241022', help='Model yang digunakan')
    parser.add_argument('--site-url', default='https://test-generator-app.com', 
                        help='URL situs untuk header HTTP-Referer (OpenRouter)')
    parser.add_argument('--site-name', default='Test Generator',
                        help='Nama situs untuk header X-Title (OpenRouter)')
    
    parser.add_argument('--rate-limit-rpm', type=int, default=5, help='Request per menit')
    parser.add_argument('--rate-limit-input-tpm', type=int, default=25000, help='Token input per menit')
    parser.add_argument('--rate-limit-output-tpm', type=int, default=5000, help='Token output per menit')
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Mode verbose untuk logging')
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validasi modul path
    if not os.path.exists(args.module):
        logger.error(f"Modul tidak ditemukan: {args.module}")
        sys.exit(1)
    
    # Dapatkan API key
    api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        logger.error("OpenRouter API key tidak ditemukan. Gunakan --api-key atau set env OPENROUTER_API_KEY")
        sys.exit(1)
    
    # Buat rate limiter
    rate_limit_config = RateLimitConfig(
        requests_per_minute=args.rate_limit_rpm,
        input_tokens_per_minute=args.rate_limit_input_tpm,
        output_tokens_per_minute=args.rate_limit_output_tpm
    )
    rate_limiter = ApiRateLimiter(rate_limit_config)
    
    # Analisis coverage
    logger.info(f"Menganalisis coverage untuk modul: {args.module}")
    coverage_data = CoverageAnalyzer.parse_coverage_data(args.module, args.coverage_file)
    
    coverage_pct = coverage_data.get("coverage_pct", 0)
    logger.info(f"Coverage saat ini: {coverage_pct:.1f}%")
    
    if coverage_pct >= args.coverage_threshold:
        logger.info(f"Coverage sudah mencapai threshold ({args.coverage_threshold}%). Tidak perlu generate test.")
        sys.exit(0)
    
    # Generate test
    logger.info(f"Membuat test dengan {args.model} via OpenRouter")
    generator = TestGenerator(
        api_key, 
        args.model, 
        args.site_url,
        args.site_name,
        rate_limiter
    )
    
    try:
        test_code = generator.generate_test(args.module, coverage_data, args.test_framework)
        test_file = generator.write_test_file(args.module, test_code)
        logger.info(f"Test berhasil dibuat: {test_file}")
    except Exception as e:
        logger.error(f"Gagal membuat test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()