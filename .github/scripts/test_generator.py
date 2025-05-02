"""
Claude Test Generator

Sistem untuk menggunakan Claude 3.5 Haiku API untuk generate test case otomatis
dengan mekanisme rate limiting berdasarkan batas:
- 5 request per menit
- 25,000 token input per menit
- 5,000 token output per menit

Kode ini termasuk:
1. Rate limiter untuk Claude API
2. Test case generator
3. Coverage analyzer
4. GitHub Action workflow
"""

import os
import time
import json
import logging
import anthropic
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import queue
import re
import glob
from collections import deque
import math

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("claude-test-generator")

# Konstanta untuk model
CLAUDE_MODEL = "claude-3-5-haiku-20240307"

# Konfigurasi rate limiting
@dataclass
class RateLimitConfig:
    requests_per_minute: int = 5
    input_tokens_per_minute: int = 25000
    output_tokens_per_minute: int = 5000

class ClaudeRateLimiter:
    """
    Rate limiter untuk API Claude yang mempertimbangkan:
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
    def count_tokens_with_anthropic(client: anthropic.Anthropic, text: str) -> int:
        """
        Menghitung token menggunakan API Anthropic jika tersedia.
        """
        try:
            token_count = client.count_tokens(text)
            return token_count.tokens
        except Exception as e:
            logger.warning(f"Gagal menghitung token dengan API: {e}")
            # Fallback ke estimasi kasar
            return TokenCounter.estimate_tokens(text)

class ClaudeClient:
    """Wrapper untuk berinteraksi dengan Claude API"""
    
    def __init__(self, api_key: str, rate_limiter: ClaudeRateLimiter):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.rate_limiter = rate_limiter
    
    def generate_response(self, prompt: str, max_tokens: int = 4000) -> str:
        """
        Menghasilkan respons dari Claude API dengan mempertimbangkan rate limiting.
        """
        # Perkiraan token input dan output
        input_tokens = TokenCounter.count_tokens_with_anthropic(self.client, prompt)
        estimated_output_tokens = min(max_tokens, 2000)  # Perkiraan default 2000 atau max_tokens jika lebih kecil
        
        # Tunggu sampai kapasitas rate limit tersedia
        self.rate_limiter.wait_for_capacity(input_tokens, estimated_output_tokens)
        
        try:
            # Kirim permintaan ke Claude API
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Dapatkan respons teks
            response_text = response.content[0].text
            
            # Catat penggunaan token output aktual
            actual_output_tokens = response.usage.output_tokens
            self.rate_limiter.record_actual_usage(actual_output_tokens)
            
            return response_text
        
        except Exception as e:
            logger.error(f"Error saat memanggil Claude API: {e}")
            # Jika terjadi error, catat penggunaan token minimum untuk menghindari overestimasi
            self.rate_limiter.record_actual_usage(1)
            raise

class CoverageAnalyzer:
    """
    Menganalisis laporan coverage untuk mengidentifikasi bagian kode yang perlu ditingkatkan
    """
    
    @staticmethod
    def parse_coverage_report(coverage_path: str = "coverage.xml") -> List[Dict[str, Any]]:
        """
        Mem-parsing laporan coverage XML dan mengidentifikasi modul dengan coverage rendah
        """
        if not os.path.exists(coverage_path):
            logger.error(f"File coverage tidak ditemukan: {coverage_path}")
            return []
        
        try:
            tree = ET.parse(coverage_path)
            root = tree.getroot()
            
            low_coverage_modules = []
            
            # Temukan kelas dengan coverage rendah
            for class_element in root.findall('.//class'):
                filename = class_element.get('filename')
                line_rate = float(class_element.get('line-rate', 0))
                
                # Jika file ada dan coverage kurang dari 70%
                if os.path.exists(filename) and line_rate < 0.7:
                    try:
                        with open(filename, 'r', encoding='utf-8') as f:
                            code = f.read()
                        
                        # Dapatkan informasi per-line jika tersedia
                        lines = []
                        for line in class_element.findall('.//line'):
                            line_number = int(line.get('number'))
                            hits = int(line.get('hits', 0))
                            lines.append({
                                'number': line_number,
                                'hits': hits
                            })
                        
                        # Sort lines by number for easier processing
                        lines.sort(key=lambda x: x['number'])
                        
                        low_coverage_modules.append({
                            'filename': filename,
                            'coverage': line_rate * 100,
                            'code': code,
                            'lines': lines
                        })
                    except Exception as e:
                        logger.warning(f"Tidak dapat membaca file {filename}: {e}")
            
            return low_coverage_modules
            
        except Exception as e:
            logger.error(f"Error saat parsing laporan coverage: {e}")
            return []
    
    @staticmethod
    def identify_uncovered_functions(module: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Mengidentifikasi fungsi-fungsi yang tidak tercakup test dalam suatu modul
        """
        code = module['code']
        lines = module['lines']
        
        # Set line numbers yang telah ditest (hits > 0)
        covered_lines = {line['number'] for line in lines if line['hits'] > 0}
        
        # Pola regex sederhana untuk mendeteksi fungsi
        function_pattern = re.compile(r'^\s*def\s+(\w+)\s*\(', re.MULTILINE)
        
        # Temukan semua fungsi
        functions = []
        for match in function_pattern.finditer(code):
            func_name = match.group(1)
            line_number = code[:match.start()].count('\n') + 1
            
            # Perkirakan akhir fungsi (Ini pendekatan sederhana, tidak menangani semua kasus)
            next_match = function_pattern.search(code, match.end())
            if next_match:
                end_line = code[:next_match.start()].count('\n')
            else:
                end_line = code.count('\n') + 1
            
            # Cek apakah fungsi setidaknya memiliki satu baris yang tercakup
            function_lines = set(range(line_number, end_line + 1))
            has_coverage = bool(function_lines.intersection(covered_lines))
            
            functions.append({
                'name': func_name,
                'start_line': line_number,
                'end_line': end_line,
                'has_coverage': has_coverage
            })
        
        # Filter fungsi yang tidak tercakup
        uncovered_functions = [f for f in functions if not f['has_coverage']]
        return uncovered_functions

class TestGenerator:
    """
    Membuat test case menggunakan Claude API
    """
    
    def __init__(self, claude_client: ClaudeClient):
        self.claude_client = claude_client
    
    def generate_tests_for_module(self, module: Dict[str, Any], test_framework: str = "pytest") -> str:
        """
        Menghasilkan test case untuk modul berdasarkan code dan coverage data
        """
        filename = module['filename']
        code = module['code']
        coverage = module['coverage']
        
        # Identifikasi fungsi yang tidak tercakup
        uncovered_functions = CoverageAnalyzer.identify_uncovered_functions(module)
        uncovered_names = [f['name'] for f in uncovered_functions]
        
        # Buat prompt untuk Claude
        prompt = f"""
        Saya memiliki modul Python dengan coverage test yang rendah ({coverage:.1f}%).
        
        Nama file: {filename}
        
        ```python
        {code}
        ```
        
        Fungsi-fungsi yang tidak tercakup test:
        {', '.join(uncovered_names) if uncovered_names else 'Semua fungsi hanya memiliki coverage parsial'}
        
        Tolong buatkan test case menggunakan {test_framework} untuk meningkatkan coverage.
        Fokuskan pada fungsi yang tidak tercakup dan edge case.
        Test harus mengikuti best practice dan menyertakan assertion yang tepat.
        Jangan sertakan penjelasan, hanya kode Python untuk test saja.
        
        Pastikan test yang dihasilkan dapat dijalankan dan valid sesuai dengan {test_framework}.
        """
        
        # Generate test dengan Claude
        test_code = self.claude_client.generate_response(prompt)
        
        # Ekstrak kode Python dari respons jika perlu
        if "```python" in test_code and "```" in test_code:
            test_code = test_code.split("```python")[1].split("```")[0].strip()
        elif "```" in test_code:
            test_code = test_code.split("```")[1].split("```")[0].strip()
        
        return test_code
    
    def write_test_file(self, module: Dict[str, Any], test_code: str) -> str:
        """
        Menulis kode test ke file
        """
        filename = module['filename']
        
        # Tentukan path test file
        module_name = os.path.basename(filename)
        test_dir = "tests"
        
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        
        if module_name.endswith('.py'):
            module_name = module_name[:-3]
        
        test_file = f"{test_dir}/test_{module_name}.py"
        
        # Jangan timpa file yang ada jika formatnya sama
        if os.path.exists(test_file):
            new_version = 1
            while os.path.exists(f"{test_dir}/test_{module_name}_v{new_version}.py"):
                new_version += 1
            test_file = f"{test_dir}/test_{module_name}_v{new_version}.py"
        
        # Tulis ke file
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_code)
        
        logger.info(f"Test untuk {filename} telah dibuat di {test_file}")
        return test_file

class TestImprovementPipeline:
    """
    Pipeline lengkap untuk meningkatkan coverage test dengan Claude
    """
    
    def __init__(self, api_key: str, rate_limit_config: RateLimitConfig = None):
        if rate_limit_config is None:
            rate_limit_config = RateLimitConfig()
            
        self.rate_limiter = ClaudeRateLimiter(rate_limit_config)
        self.claude_client = ClaudeClient(api_key, self.rate_limiter)
        self.test_generator = TestGenerator(self.claude_client)
    
    def run(self, coverage_path: str = "coverage.xml", min_coverage_threshold: float = 70.0):
        """
        Menjalankan pipeline lengkap
        """
        logger.info("Memulai pipeline peningkatan test...")
        
        # Analisis coverage report
        low_coverage_modules = CoverageAnalyzer.parse_coverage_report(coverage_path)
        logger.info(f"Menemukan {len(low_coverage_modules)} modul dengan coverage rendah")
        
        if not low_coverage_modules:
            logger.info("Tidak ada modul dengan coverage rendah ditemukan")
            return
        
        # Urutkan modul berdasarkan coverage (paling rendah lebih dulu)
        low_coverage_modules.sort(key=lambda m: m['coverage'])
        
        # Proses setiap modul
        for module in low_coverage_modules:
            filename = module['filename']
            coverage = module['coverage']
            
            logger.info(f"Memproses {filename} dengan coverage {coverage:.1f}%")
            
            # Generate test
            test_code = self.test_generator.generate_tests_for_module(module)
            
            # Tulis test ke file
            test_file = self.test_generator.write_test_file(module, test_code)
            
            logger.info(f"Test baru ditulis ke {test_file}")
        
        logger.info("Pipeline selesai!")


# Fungsi utama untuk menjalankan pipeline dari CLI
def main():
    """
    Fungsi utama untuk menjalankan test generator dari command line
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Claude Test Generator')
    parser.add_argument('--coverage-path', type=str, default='coverage.xml',
                        help='Path ke file laporan coverage XML')
    parser.add_argument('--min-coverage', type=float, default=70.0,
                        help='Threshold minimum coverage (persentase)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Anthropic API key (default: dari env ANTHROPIC_API_KEY)')
    parser.add_argument('--export-workflow', action='store_true',
                        help='Export GitHub Action workflow YAML ke file')
    args = parser.parse_args()
    
    # Export GitHub workflow jika diminta
    if args.export_workflow:
        with open('.github/workflows/ai-test-coverage.yml', 'w') as f:
            f.write(GITHUB_WORKFLOW_YAML)
        logger.info("GitHub Action workflow telah diekspor ke .github/workflows/ai-test-coverage.yml")
        return
    
    # Dapatkan API key
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("Anthropic API key tidak ditemukan. Gunakan --api-key atau set env ANTHROPIC_API_KEY")
        return
    
    # Rate limit config untuk Claude 3.5 Haiku
    rate_limit_config = RateLimitConfig(
        requests_per_minute=5,
        input_tokens_per_minute=25000,
        output_tokens_per_minute=5000
    )
    
    # Buat dan jalankan pipeline
    pipeline = TestImprovementPipeline(api_key, rate_limit_config)
    pipeline.run(args.coverage_path, args.min_coverage)

if __name__ == "__main__":
    main()