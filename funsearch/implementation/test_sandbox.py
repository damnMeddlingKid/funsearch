#!/usr/bin/env python3
"""Test script for the Sandbox run method."""

from evaluator import Sandbox

def test_basic_function():
    """Test basic function execution."""
    sandbox = Sandbox()
    
    program = """
def add_numbers(x):
    return x + 5
"""
    
    result, success = sandbox.run(program, "add_numbers", 10, 5)
    print(f"Basic test: result={result}, success={success}")
    assert success and result == 15, f"Expected 15, got {result}"

def test_math_operations():
    """Test mathematical operations."""
    sandbox = Sandbox()
    
    program = """
def calculate_score(nums):
    return sum(nums) * len(nums)
"""
    
    result, success = sandbox.run(program, "calculate_score", [1, 2, 3], 5)
    print(f"Math test: result={result}, success={success}")
    assert success and result == 18, f"Expected 18, got {result}"

def test_timeout():
    """Test timeout protection."""
    sandbox = Sandbox()
    
    program = """
def infinite_loop(x):
    while True:
        x += 1
    return x
"""
    
    result, success = sandbox.run(program, "infinite_loop", 1, 2)
    print(f"Timeout test: result={result}, success={success}")
    assert not success, "Should have failed due to timeout"

def test_security_restriction():
    """Test that dangerous operations are blocked."""
    sandbox = Sandbox()
    
    program = """
import os
def dangerous_function(x):
    os.system("echo 'hacked'")
    return x
"""
    
    result, success = sandbox.run(program, "dangerous_function", 1, 5)
    print(f"Security test: result={result}, success={success}")
    assert not success, "Should have failed due to security restriction"

if __name__ == "__main__":
    print("Testing sandbox run method...")
    
    try:
        test_basic_function()
        print("✓ Basic function test passed")
        
        test_math_operations()
        print("✓ Math operations test passed")
        
        test_timeout()
        print("✓ Timeout test passed")
        
        test_security_restriction()
        print("✓ Security restriction test passed")
        
        print("\nAll tests passed! 🎉")
        
    except Exception as e:
        print(f"Test failed: {e}")