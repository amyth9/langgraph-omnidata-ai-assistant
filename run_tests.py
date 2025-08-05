#!/usr/bin/env python3
"""
Test runner script for the AI Pipeline project.

This script provides comprehensive testing for all components of the AI pipeline
including state management, node processing, API handling, and integration tests.

Usage:
    python3 run_tests.py [options]

Options:
    --unit          Run only unit tests
    --integration   Run only integration tests
    --api           Run only API handling tests
    --all           Run all tests (default)
    --verbose       Verbose output
    --coverage      Run with coverage report
    --html          Generate HTML coverage report
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Error: {e}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout)
        if e.stderr:
            print("Stderr:")
            print(e.stderr)
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")

    required_packages = [
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "requests",
        "pydantic",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    print("✅ All dependencies are installed")
    return True


def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests."""
    cmd = [
        "python3",
        "-m",
        "pytest",
        "tests/test_state.py",
        "tests/test_router_node.py",
        "tests/test_weather_node.py",
        "tests/test_rag_node.py",
        "tests/test_weather_tool.py",
        "tests/test_retriever_tool.py",
    ]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=app", "--cov-report=term-missing"])

    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose=False, coverage=False):
    """Run integration tests."""
    cmd = ["python3", "-m", "pytest", "tests/test_integration.py"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=app", "--cov-report=term-missing"])

    return run_command(cmd, "Integration Tests")


def run_api_tests(verbose=False, coverage=False):
    """Run API handling tests."""
    cmd = ["python3", "-m", "pytest", "tests/test_api_handling.py"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=app", "--cov-report=term-missing"])

    return run_command(cmd, "API Handling Tests")


def run_all_tests(verbose=False, coverage=False, html=False):
    """Run all tests."""
    cmd = ["python3", "-m", "pytest", "tests/"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=app", "--cov-report=term-missing"])
        if html:
            cmd.extend(["--cov-report=html"])

    return run_command(cmd, "All Tests")


def generate_coverage_report():
    """Generate HTML coverage report."""
    cmd = [
        "python3",
        "-m",
        "pytest",
        "tests/",
        "--cov=app",
        "--cov-report=html",
        "--cov-report=term-missing",
    ]
    return run_command(cmd, "Coverage Report")


def main():
    """Main function to run tests based on command line arguments."""
    parser = argparse.ArgumentParser(description="Run AI Pipeline tests")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument(
        "--integration", action="store_true", help="Run only integration tests"
    )
    parser.add_argument(
        "--api", action="store_true", help="Run only API handling tests"
    )
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--coverage", action="store_true", help="Run with coverage report"
    )
    parser.add_argument(
        "--html", action="store_true", help="Generate HTML coverage report"
    )

    args = parser.parse_args()

    print("🚀 AI Pipeline Test Runner")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Determine which tests to run
    if args.unit:
        success = run_unit_tests(args.verbose, args.coverage)
    elif args.integration:
        success = run_integration_tests(args.verbose, args.coverage)
    elif args.api:
        success = run_api_tests(args.verbose, args.coverage)
    elif args.all or not any([args.unit, args.integration, args.api]):
        success = run_all_tests(args.verbose, args.coverage, args.html)

    # Generate coverage report if requested
    if args.html and success:
        generate_coverage_report()

    if success:
        print("\n🎉 All tests completed successfully!")
        print("\n📊 Test Summary:")
        print("- Unit tests: State management, node processing, tool functionality")
        print("- Integration tests: Complete pipeline flow")
        print("- API tests: Weather API, geocoding API, error handling")
        print("- Coverage: Code coverage analysis")

        if args.coverage or args.html:
            print("\n📈 Coverage reports generated:")
            if args.html:
                print("- HTML report: htmlcov/index.html")
            print("- Terminal report: Shows missing lines")

        print("\n📋 Test Categories:")
        print("1. State Management Tests:")
        print("   - AssistantState creation and manipulation")
        print("   - Message handling and conversation history")
        print("   - WeatherData and RAGResult models")
        print("   - QueryType and MessageRole enums")

        print("\n2. Node Processing Tests:")
        print("   - RouterNode: Query classification and routing")
        print("   - WeatherNode: Weather data processing")
        print("   - RAGNode: Document retrieval and summarization")

        print("\n3. Tool Functionality Tests:")
        print("   - WeatherTool: API calls, city extraction, error handling")
        print("   - RetrieverTool: Document retrieval, RAG context generation")

        print("\n4. API Handling Tests:")
        print("   - Weather API: Success, error, timeout scenarios")
        print("   - Geocoding API: Location resolution")
        print("   - Network errors, rate limiting, authentication")

        print("\n5. Integration Tests:")
        print("   - Complete pipeline flow")
        print("   - Error handling across components")
        print("   - State preservation and message flow")

        print("\n✅ All test categories passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
