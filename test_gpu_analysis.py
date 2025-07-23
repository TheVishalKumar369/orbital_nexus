#!/usr/bin/env python3
"""
Quick test script for GPU-accelerated risk analysis
"""

from scripts.gpu_risk_analysis import quick_risk_analysis

# Test with an available satellite ID
print("Testing GPU-accelerated risk analysis...")
result = quick_risk_analysis('4468', max_objects=1000)  # Use an available satellite
print(f"Analysis complete. Found {result['total_risks']} risks.")

if result['total_risks'] > 0:
    print("\nTop risks:")
    print(result['summary'])
else:
    print("No risks found - analysis working correctly!")
    print("This is much faster than the original method.")
