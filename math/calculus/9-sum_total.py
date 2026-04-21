#!/usr/bin/env python3

def summation_i_squared(n):
    if not isinstance(n, (int, float)) or n < 0:
        return None    
    # Using the square pyramidal number formula
    return int((n * (n + 1) * (2 * n + 1)) / 6)
