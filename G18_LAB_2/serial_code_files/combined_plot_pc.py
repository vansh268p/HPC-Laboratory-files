import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
experiments = [
    {"file": "results/times_standard.csv", "label": "Standard (A)", "color": "red", "marker": "o"},
    {"file": "results/times_transpose.csv", "label": "Transpose (B)", "color": "blue", "marker": "s"},
    {"file": "results/times_block.csv", "label": "Block (C)", "color": "green", "marker": "^"}
]

OUTPUT_IMG = "results/plot_linear.png"

plt.figure(figsize=(10, 6))

for exp in experiments:
    if not os.path.exists(exp["file"]):
        continue
        
    df = pd.read_csv(exp["file"], skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns] # Clean column names
    
    # Sort by problem size just in case
    df = df.sort_values(by="ProblemSize")

    # Plotting
    # X-Axis: Log Scale (Base 2) to space out 4, 8, 16... evenly
    # Y-Axis: Linear Scale (Actual Seconds) to show the huge gap
    plt.plot(
        df["ProblemSize"],
        df["AvgAlgoTime"],
        marker=exp["marker"],
        color=exp["color"],
        label=exp["label"],
        linewidth=2
    )

# Set X-axis to Log Scale (Base 2)
plt.xscale('log', base=2)

# Set Y-axis to Linear (Standard 0, 50, 100...)
plt.yscale('linear')

plt.xlabel("Problem Size ($N$) - Log Scale", fontsize=12)
plt.ylabel("Execution Time (Seconds) - Linear Scale", fontsize=12)
plt.title("Matrix Multiplication: Actual Runtime Difference", fontsize=14)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_IMG)
plt.close()

print(f"Linear scale plot saved to: {OUTPUT_IMG}")
