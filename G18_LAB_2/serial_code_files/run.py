import os
import csv
import subprocess
from collections import defaultdict

# --- Problem A: Six Permutations ---
# EXECUTABLE = "./out_ijk"
# OUTPUT_CSV = "results/times_ijk.csv"

#EXECUTABLE = "./out_ikj"
#OUTPUT_CSV = "results/times_ikj.csv"

#EXECUTABLE = "./out_jik"
#OUTPUT_CSV = "results/times_jik.csv"

EXECUTABLE = "./out_jki"
OUTPUT_CSV = "results/times_jki.csv"

# EXECUTABLE = "./out_kij"
# OUTPUT_CSV = "results/times_kij.csv"

# EXECUTABLE = "./out_kji"
# OUTPUT_CSV = "results/times_kji.csv"

# --- Problem B: Transpose ---
# EXECUTABLE = "./out_transpose"
# OUTPUT_CSV = "results/times_transpose.csv"

# --- Problem C: Block ---
# EXECUTABLE = "./out_block"
# OUTPUT_CSV = "results/times_block.csv"

# ==========================================

NUM_RUNS = 5 

# Ensure results directory exists
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

results = defaultdict(lambda: {
    "e2e_times": [],
    "algo_times": []
})

print(f"Running {EXECUTABLE} {NUM_RUNS} times...")
print(f"Saving to {OUTPUT_CSV}...\n")

for run_id in range(NUM_RUNS):
    print(f"=== Run {run_id + 1}/{NUM_RUNS} ===")

    proc = subprocess.Popen(
        [EXECUTABLE],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    header_seen = False

    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if not line:
            continue
        
        line = line.strip()
        # print(line)

        if "ProblemSize" in line:
            header_seen = True
            continue

        if not header_seen:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue

        problem_size = int(parts[0])
        e2e_time = float(parts[1])
        algo_time = float(parts[2])

        results[problem_size]["e2e_times"].append(e2e_time)
        results[problem_size]["algo_times"].append(algo_time)

# Write averaged results
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ProblemSize", "AvgE2ETime", "AvgAlgoTime"])

    for problem_size in sorted(results.keys()):
        entry = results[problem_size]
        avg_e2e = sum(entry["e2e_times"]) / len(entry["e2e_times"])
        avg_algo = sum(entry["algo_times"]) / len(entry["algo_times"])

        writer.writerow([problem_size, f"{avg_e2e:.9f}", f"{avg_algo:.9f}"])

print(f"\nDone! Results saved to: {OUTPUT_CSV}")

