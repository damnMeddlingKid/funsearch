#!/usr/bin/env python3
"""Basic test for LLM sampler."""

import os
from sampler import LLM

def test_basic_llm():    
    llm = LLM(samples_per_prompt=1)
    prompt = """
def add_numbers_v0(a, b):
    return 0  # Poor implementation - always returns 0

def add_numbers_v1(a, b):
    return a  # Poor implementation - ignores b

def add_numbers_v2(a, b):
    # Complete this function to properly add two numbers
"""
    
    try:
        samples = llm.draw_samples(prompt)
        print(f"Generated {len(samples)} sample(s)")
        print(f"Sample: {samples[0]}")
        print("✓ LLM test passed")
    except Exception as e:
        print(f"✗ LLM test failed: {e}")

if __name__ == "__main__":
    test_basic_llm()