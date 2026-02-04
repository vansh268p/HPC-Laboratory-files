import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(
    "results/avg_algo_times.csv",
    skipinitialspace=True
)

# Rename columns 
df.columns = ["problem_size", "runs", "total_ops", "avg_e2e_time", "avg_algo_time"]
# df.columns = ["problem_size", "runs", "total_ops", "e2e_time", "algo_time"]

# ensure numeric types
for col in df.columns:
    df[col] = pd.to_numeric(df[col])

# Throughput (MFLOPs)
# 2 FLOPs per element (mul + add)
df["Throughput (MFLOPs)"] = (3 * df["problem_size"] * df["runs"]) / (df["avg_algo_time"] * (2 ** 20))

# Bandwidth calculation (bits / second)
# Vector triad: 3 reads + 1 write = 4 doubles = 32 bytes
bytes_moved = 16 * df["problem_size"] * df["runs"]
df["Bandwidth (GB/s)"] = bytes_moved / df["avg_algo_time"] / 1e9

# Plot 1: Throughput vs problem size
plt.figure()
plt.plot(
    np.log2(df["problem_size"]),
    np.log10(df["Throughput (MFLOPs)"]),
    marker="o"
)
plt.xlabel("log2(Problem Size)")
plt.ylabel("log10(Throughput) [MFLOPs]")
plt.title("Compute Throughput vs Problem Size")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("results/throughput_vs_problem_size.png")
plt.close()

# -----------------------------------
# Plot 2: Bandwidth vs problem size
# -----------------------------------
plt.figure()
plt.plot(
    np.log2(df["problem_size"]),
    df["Bandwidth (GB/s)"],
    marker="o"
)
plt.xlabel("log2(Problem Size)")
plt.ylabel("Bandwidth (GB/s)")
plt.title("Memory Bandwidth vs Problem Size")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("results/bandwidth_vs_problem_size.png")
plt.close()

print("Plots saved successfully:")
print(" - throughput_vs_problem_size.png")
print(" - bandwidth_vs_problem_size.png")
