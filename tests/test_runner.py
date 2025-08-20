#!/usr/bin/env python3
"""
Test runner for Pilot Quantum framework.

This script provides a convenient way to run different categories of tests
and generate reports.
"""

import unittest
import sys
import os
import argparse
import time
from datetime import datetime

# Add the parent directory to the path to import pilot modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_all_tests():
    """Run all tests in the test suite."""
    print("🧪 Running all tests...")
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def run_executor_tests():
    """Run executor-specific tests."""
    print("🔧 Running executor tests...")
    
    # Import specific test modules
    from tests.test_all_executors import TestAllExecutors, TestExecutorPerformance
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add executor tests
    suite.addTest(unittest.makeSuite(TestAllExecutors))
    suite.addTest(unittest.makeSuite(TestExecutorPerformance))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def run_qdreamer_tests():
    """Run QDREAMER integration tests."""
    print("🧠 Running QDREAMER integration tests...")
    
    # Import QDREAMER test modules
    from tests.test_qdreamer_integration import TestQDREAMERIntegration, TestQDREAMERPerformance
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add QDREAMER tests
    suite.addTest(unittest.makeSuite(TestQDREAMERIntegration))
    suite.addTest(unittest.makeSuite(TestQDREAMERPerformance))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def run_performance_tests():
    """Run performance tests."""
    print("⚡ Running performance tests...")
    
    from tests.test_all_executors import TestExecutorPerformance
    
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestExecutorPerformance))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def run_basic_tests():
    """Run basic functionality tests."""
    print("🔍 Running basic tests...")
    
    from tests.test_basic import TestBasicImports, TestConfigurationValidation
    
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestBasicImports))
    suite.addTest(unittest.makeSuite(TestConfigurationValidation))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def run_specific_test(test_name):
    """Run a specific test by name."""
    print(f"🎯 Running specific test: {test_name}")
    
    # Import all test modules
    from tests.test_basic import TestBasicImports, TestConfigurationValidation
    from tests.test_all_executors import TestAllExecutors, TestExecutorPerformance
    from tests.test_qdreamer_integration import TestQDREAMERIntegration, TestQDREAMERPerformance
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add the specific test
    loader = unittest.TestLoader()
    suite.addTest(loader.loadTestsFromName(test_name))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def generate_report(result, test_type, duration):
    """Generate a test report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
=== Pilot Quantum Test Report ===
Timestamp: {timestamp}
Test Type: {test_type}
Duration: {duration:.2f} seconds

Results:
- Tests Run: {result.testsRun}
- Failures: {len(result.failures)}
- Errors: {len(result.errors)}
- Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}

Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%

"""
    
    if result.failures:
        report += "\nFailures:\n"
        for test, traceback in result.failures:
            report += f"- {test}: {traceback}\n"
    
    if result.errors:
        report += "\nErrors:\n"
        for test, traceback in result.errors:
            report += f"- {test}: {traceback}\n"
    
    return report

def save_report(report, filename=None):
    """Save the test report to a file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_report_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(report)
    
    print(f"📄 Test report saved to: {filename}")

class TestRunner:
    """Main test runner class."""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self):
        """Create command line argument parser."""
        parser = argparse.ArgumentParser(
            description='Pilot Quantum Test Runner',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python test_runner.py --all                    # Run all tests
  python test_runner.py --executors              # Run executor tests only
  python test_runner.py --performance            # Run performance tests only
  python test_runner.py --basic                  # Run basic tests only
  python test_runner.py --test TestBasicImports.test_pilot_imports  # Run specific test
  python test_runner.py --all --report           # Run all tests and save report
            """
        )
        
        parser.add_argument(
            '--all',
            action='store_true',
            help='Run all tests'
        )
        
        parser.add_argument(
            '--executors',
            action='store_true',
            help='Run executor tests only'
        )
        
        parser.add_argument(
            '--qdreamer',
            action='store_true',
            help='Run QDREAMER integration tests only'
        )
        
        parser.add_argument(
            '--performance',
            action='store_true',
            help='Run performance tests only'
        )
        
        parser.add_argument(
            '--basic',
            action='store_true',
            help='Run basic tests only'
        )
        
        parser.add_argument(
            '--test',
            type=str,
            help='Run a specific test by name'
        )
        
        parser.add_argument(
            '--report',
            action='store_true',
            help='Generate and save a test report'
        )
        
        parser.add_argument(
            '--output',
            type=str,
            help='Output file for test report'
        )
        
        return parser
    
    def run(self, args=None):
        """Run tests based on command line arguments."""
        if args is None:
            args = self.parser.parse_args()
        
        # Determine which tests to run
        if args.all:
            test_type = "All Tests"
            start_time = time.time()
            result = run_all_tests()
            duration = time.time() - start_time
        elif args.executors:
            test_type = "Executor Tests"
            start_time = time.time()
            result = run_executor_tests()
            duration = time.time() - start_time
        elif args.qdreamer:
            test_type = "QDREAMER Integration Tests"
            start_time = time.time()
            result = run_qdreamer_tests()
            duration = time.time() - start_time
        elif args.performance:
            test_type = "Performance Tests"
            start_time = time.time()
            result = run_performance_tests()
            duration = time.time() - start_time
        elif args.basic:
            test_type = "Basic Tests"
            start_time = time.time()
            result = run_basic_tests()
            duration = time.time() - start_time
        elif args.test:
            test_type = f"Specific Test: {args.test}"
            start_time = time.time()
            result = run_specific_test(args.test)
            duration = time.time() - start_time
        else:
            # Default to running all tests
            test_type = "All Tests"
            start_time = time.time()
            result = run_all_tests()
            duration = time.time() - start_time
        
        # Generate and display report
        report = generate_report(result, test_type, duration)
        print(report)
        
        # Save report if requested
        if args.report:
            save_report(report, args.output)
        
        # Return exit code
        return 0 if result.wasSuccessful() else 1

def main():
    """Main entry point."""
    runner = TestRunner()
    exit_code = runner.run()
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
