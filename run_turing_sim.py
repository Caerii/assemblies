import time
from "../turing_sim" import larger_k, turing_erase

def run_simulations():
    print("Running Larger K Simulation...")
    start_time = time.time()
    larger_k(n=10000, k=100, p=0.01, beta=0.05, bigger_factor=10)
    print(f"Larger K Simulation took {time.time() - start_time:.2f} seconds.\n")

    print("Running Turing Erase Simulation...")
    start_time = time.time()
    turing_erase(n=50000, k=100, p=0.01, beta=0.05, r=1.0, bigger_factor=20)
    print(f"Turing Erase Simulation took {time.time() - start_time:.2f} seconds.\n")

if __name__ == "__main__":
    run_simulations()
