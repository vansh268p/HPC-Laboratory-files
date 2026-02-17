import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- OPTION A: Standard ---
INPUT_CSV = "results/times_jki.csv"
OUTPUT_IMG = "results/plot_jki.png"
TITLE = "Standard Matrix Multiplication with jki loop"

# --- OPTION B: Transpose ---
# INPUT_CSV = "results/times_transpose.csv"
# OUTPUT_IMG = "results/plot_transpose.png"
# TITLE = "Transpose Matrix Multiplication"

# --- OPTION C: Block ---
#INPUT_CSV = "results/times_block.csv"
#OUTPUT_IMG = "results/plot_block.png"
#TITLE = "Block Matrix Multiplication"

# ==========================================

# Check if file exists first
if not os.path.exists(INPUT_CSV):
    print(f"Error: {INPUT_CSV} not found. Did you run the correct experiment?")
    exit()

df = pd.read_csv(INPUT_CSV, skipinitialspace=True)
df.columns = ["problem_size", "avg_e2e_time", "avg_algo_time"]

for col in df.columns:
    df[col] = pd.to_numeric(df[col])

# Plotting
plt.figure()
plt.plot(
    np.log2(df["problem_size"]),
    np.log10(df["avg_algo_time"]),
    marker="o",
    label=TITLE
)

plt.xlabel("Problem Size (log2 scale)")
plt.ylabel("Execution Time (log10 scale)")
plt.title(f"{TITLE} - Time vs Problem Size")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_IMG)
plt.close()

print(f"Plot saved successfully to: {OUTPUT_IMG}")
