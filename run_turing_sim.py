# File name: run_turing_sim.py

"""
This script is designed to run simulations from the turing_sim.py module.
It imports and executes functions that test neural assembly dynamics in a simulated brain model.
"""

import turing_sim

def main():
    print("Running Larger K Simulation...")
    turing_sim.larger_k(n=10000, k=100, p=0.01, beta=0.05, bigger_factor=10)
    
    print("\nRunning Turing Erase Simulation...")
    turing_sim.turing_erase(n=50000, k=100, p=0.01, beta=0.05, r=1.0, bigger_factor=20)

if __name__ == "__main__":
    main()
