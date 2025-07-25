#!/usr/bin/env python3
"""
Test Runner for Audio Engine

Comprehensive test runner that executes all test suites with detailed reporting
and performance analysis.
"""

import unittest
import sys
import time
from pathlib import Path
import json
from io import StringIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test modules
from tests.test_noise_generation import TestNoiseAlgorithms, TestNoiseGenerator, TestAudioQuality
from tests.test_therapeutic_processing import (
    TestTherapeuticProcessor, TestTherapeuticIntegration, TestTherapeuticValidation
)
from tests.test_youtube_compliance import (
    TestLoudnessProcessor, TestYouTubeCompliance, TestProfessionalStandards
)


class TestRunner:
    """Custom test runner with detailed reporting."""
    
    def __init__(self):
        """Initialize test runner."""
        self.results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0,
            'execution_time': 0,
            'test_results': {},
            'performance_data': {}
        }
        
    def run_test_suite(self, test_suite_class, suite_name):
        """Run a specific test suite and collect results."""
        print(f"\n{'='*60}")
        print(f"Running {suite_name}")
        print(f"{'='*60}")
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_suite_class)
        
        # Capture output
        stream = StringIO()
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=2,
            buffer=True
        )
        
        # Run tests with timing
        start_time = time.time()
        result = runner.run(suite)
        execution_time = time.time() - start_time
        
        # Collect results
        suite_results = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
            'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0,
            'execution_time': execution_time,
            'output': stream.getvalue()
        }
        
        # Update totals
        self.results['total_tests'] += result.testsRun
        self.results['passed'] += result.testsRun - len(result.failures) - len(result.errors)
        self.results['failed'] += len(result.failures)
        self.results['errors'] += len(result.errors)
        self.results['skipped'] += len(result.skipped) if hasattr(result, 'skipped') else 0
        self.results['execution_time'] += execution_time
        
        # Store suite results
        self.results['test_results'][suite_name] = suite_results
        
        # Print summary for this suite
        print(f"\n{suite_name} Results:")
        print(f"  Tests Run: {result.testsRun}")
        print(f"  Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
        print(f"  Failed: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Success Rate: {suite_results['success_rate']:.1f}%")
        print(f"  Execution Time: {execution_time:.2f}s")
        
        # Print failures and errors if any
        if result.failures:
            print(f"\n  FAILURES:")
            for test, traceback in result.failures:
                print(f"    - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
                
        if result.errors:
            print(f"\n  ERRORS:")
            for test, traceback in result.errors:
                print(f"    - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
        
        return result.wasSuccessful()
    
    def run_all_tests(self):
        """Run all test suites."""
        print("🎵 Audio Engine - Comprehensive Test Suite")
        print("=" * 60)
        print("Testing professional therapeutic noise generation capabilities")
        print(f"Python version: {sys.version}")
        
        # Import torch to check CUDA availability
        try:
            import torch
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        except ImportError:
            print("PyTorch not available")
        
        print()
        
        # Define test suites
        test_suites = [
            # Core functionality tests
            (TestNoiseAlgorithms, "Noise Generation Algorithms"),
            (TestNoiseGenerator, "Core Noise Generator"),
            (TestAudioQuality, "Audio Quality Validation"),
            
            # Therapeutic processing tests
            (TestTherapeuticProcessor, "Therapeutic Processing"),
            (TestTherapeuticIntegration, "Therapeutic Integration"),
            (TestTherapeuticValidation, "Therapeutic Validation"),
            
            # YouTube compliance tests
            (TestLoudnessProcessor, "LUFS/Loudness Processing"),
            (TestYouTubeCompliance, "YouTube Compliance"),
            (TestProfessionalStandards, "Professional Audio Standards")
        ]
        
        # Run all test suites
        overall_success = True
        
        for test_class, suite_name in test_suites:
            try:
                success = self.run_test_suite(test_class, suite_name)
                if not success:
                    overall_success = False
            except Exception as e:
                print(f"\n❌ Error running {suite_name}: {e}")
                overall_success = False
                self.results['errors'] += 1
        
        # Generate final report
        self.generate_final_report(overall_success)
        
        return overall_success
    
    def generate_final_report(self, overall_success):
        """Generate comprehensive final test report."""
        print(f"\n{'='*60}")
        print("FINAL TEST REPORT")
        print(f"{'='*60}")
        
        # Overall statistics
        print(f"📊 Overall Results:")
        print(f"  Total Tests: {self.results['total_tests']}")
        print(f"  Passed: {self.results['passed']} ({self.results['passed']/self.results['total_tests']*100:.1f}%)")
        print(f"  Failed: {self.results['failed']}")
        print(f"  Errors: {self.results['errors']}")
        print(f"  Skipped: {self.results['skipped']}")
        print(f"  Total Execution Time: {self.results['execution_time']:.2f}s")
        
        # Success rate
        if self.results['total_tests'] > 0:
            success_rate = (self.results['passed'] / self.results['total_tests']) * 100
            print(f"  Success Rate: {success_rate:.1f}%")
        
        # Per-suite breakdown
        print(f"\n📋 Per-Suite Results:")
        for suite_name, suite_results in self.results['test_results'].items():
            status = "✅ PASS" if suite_results['failures'] == 0 and suite_results['errors'] == 0 else "❌ FAIL"
            print(f"  {status} {suite_name}: {suite_results['tests_run']} tests, {suite_results['success_rate']:.1f}% success, {suite_results['execution_time']:.2f}s")
        
        # Performance analysis
        print(f"\n⚡ Performance Analysis:")
        fastest_suite = min(self.results['test_results'].items(), key=lambda x: x[1]['execution_time'])
        slowest_suite = max(self.results['test_results'].items(), key=lambda x: x[1]['execution_time'])
        
        print(f"  Fastest Suite: {fastest_suite[0]} ({fastest_suite[1]['execution_time']:.2f}s)")
        print(f"  Slowest Suite: {slowest_suite[0]} ({slowest_suite[1]['execution_time']:.2f}s)")
        
        avg_time_per_test = self.results['execution_time'] / self.results['total_tests'] if self.results['total_tests'] > 0 else 0
        print(f"  Average Time per Test: {avg_time_per_test:.3f}s")
        
        # Overall status
        print(f"\n🎯 Overall Status:")
        if overall_success:
            print("  ✅ ALL TESTS PASSED - Audio Engine is ready for production!")
            print("  🎵 Therapeutic noise generation meets all quality standards")
            print("  📺 YouTube compliance verified")
            print("  🏥 Therapeutic effectiveness validated")
        else:
            print("  ❌ SOME TESTS FAILED - Please review failures before production")
            
            # Provide guidance on failures
            critical_failures = []
            if 'YouTube Compliance' in self.results['test_results']:
                if self.results['test_results']['YouTube Compliance']['failures'] > 0:
                    critical_failures.append("YouTube compliance issues detected")
            
            if 'Therapeutic Validation' in self.results['test_results']:
                if self.results['test_results']['Therapeutic Validation']['failures'] > 0:
                    critical_failures.append("Therapeutic effectiveness concerns")
            
            if critical_failures:
                print("  🚨 Critical Issues:")
                for issue in critical_failures:
                    print(f"    - {issue}")
        
        # Export results to JSON
        self.export_test_results()
        
        print(f"\n📄 Detailed results exported to: test_results.json")
        
    def export_test_results(self):
        """Export test results to JSON file."""
        export_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform
            },
            'results': self.results
        }
        
        # Add torch info if available
        try:
            import torch
            export_data['system_info']['torch_version'] = torch.__version__
            export_data['system_info']['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                export_data['system_info']['cuda_device'] = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        
        with open('test_results.json', 'w') as f:
            json.dump(export_data, f, indent=2)


def main():
    """Main test runner function."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Audio Engine Test Suite')
    parser.add_argument('--suite', choices=[
        'algorithms', 'generator', 'quality', 'therapeutic', 
        'integration', 'validation', 'loudness', 'youtube', 'standards'
    ], help='Run specific test suite only')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner()
    
    if args.suite:
        # Run specific suite
        suite_map = {
            'algorithms': (TestNoiseAlgorithms, "Noise Generation Algorithms"),
            'generator': (TestNoiseGenerator, "Core Noise Generator"),
            'quality': (TestAudioQuality, "Audio Quality Validation"),
            'therapeutic': (TestTherapeuticProcessor, "Therapeutic Processing"),
            'integration': (TestTherapeuticIntegration, "Therapeutic Integration"),
            'validation': (TestTherapeuticValidation, "Therapeutic Validation"),
            'loudness': (TestLoudnessProcessor, "LUFS/Loudness Processing"),
            'youtube': (TestYouTubeCompliance, "YouTube Compliance"),
            'standards': (TestProfessionalStandards, "Professional Audio Standards")
        }
        
        test_class, suite_name = suite_map[args.suite]
        success = runner.run_test_suite(test_class, suite_name)
        sys.exit(0 if success else 1)
    else:
        # Run all tests
        success = runner.run_all_tests()
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()