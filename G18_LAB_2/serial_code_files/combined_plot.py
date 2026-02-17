import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Dictionary of all the algorithms and their corresponding cluster CSV files
csv_files = {
    'Standard IJK': 'times_ijk.csv',
    'Standard IKJ': 'times_ikj.csv',
    'Standard JIK': 'times_jik.csv',
    'Standard JKI': 'times_jki.csv',
    'Standard KIJ': 'times_kij.csv',
    'Standard KJI': 'times_kji.csv',
    'Block Matrix': 'times_block_cluster.csv',
    'Transpose Matrix': 'times_transpose_cluster.csv'
}

# ========================================================
# 1. Plot: Execution Time vs Problem Size (LINEAR SCALE)
# ========================================================
plt.figure(figsize=(10, 6))
for label, filename in csv_files.items():
    if os.path.exists(filename):
        df = pd.read_csv(filename, skipinitialspace=True)
        df.columns = ["problem_size", "avg_e2e_time", "avg_algo_time"]
        
        plt.plot(df['problem_size'], df['avg_algo_time'], marker='o', label=label)
    else:
        print(f"Warning: {filename} not found.")

plt.xlabel('Problem Size (N)')
plt.ylabel('Execution Time (seconds)')
plt.title('Combined Execution Time vs Problem Size (Linear Scale)')
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('combined_time_linear.png')
plt.close()
print("Saved: combined_time_linear.png")

# ========================================================
# 2. Plot: Execution Time vs Problem Size (LOG SCALE)
# ========================================================
plt.figure(figsize=(10, 6))
for label, filename in csv_files.items():
    if os.path.exists(filename):
        df = pd.read_csv(filename, skipinitialspace=True)
        df.columns = ["problem_size", "avg_e2e_time", "avg_algo_time"]
        
        # log2 for x-axis, log10 for y-axis
        plt.plot(np.log2(df['problem_size']), np.log10(df['avg_algo_time']), marker='o', label=label)

plt.xlabel('Problem Size (log2 scale)')
plt.ylabel('Execution Time (log10 seconds)')
plt.title('Combined Execution Time vs Problem Size (Log Scale)')
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('combined_time_log.png')
plt.close()
print("Saved: combined_time_log.png")

# ========================================================
# 3. Plot: Performance (MFLOPS) vs Problem Size
# ========================================================
plt.figure(figsize=(10, 6))
for label, filename in csv_files.items():
    if os.path.exists(filename):
        df = pd.read_csv(filename, skipinitialspace=True)
        df.columns = ["problem_size", "avg_e2e_time", "avg_algo_time"]
        
        # Calculate MFLOPS
        # For N x N matrix multiplication, operations = 2 * N^3
        ops = 2.0 * (df['problem_size'] ** 3)
        mflops = ops / (df['avg_algo_time'] * 1e6)
        
        # We plot problem size on log2 scale for readability, MFLOPS on linear
        plt.plot(np.log2(df['problem_size']), mflops, marker='o', label=label)

plt.xlabel('Problem Size (log2 scale)')
plt.ylabel('Performance (MFLOPS/sec)')
plt.title('Combined Performance vs Problem Size')
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('combined_performance.png')
plt.close()
print("Saved: combined_performance.png")

print("\nAll combined plots generated successfully!")