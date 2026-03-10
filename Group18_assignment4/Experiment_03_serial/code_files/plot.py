import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Use a clean style for all graphs
plt.style.use('bmh')

def plot_exp1_scaling():
    print("Generating Experiment 1 plots...")
    
    # Read the data for PC and Cluster
    df_pc = pd.read_csv('exp1_results_PC.csv')
    df_cluster = pd.read_csv('exp1_results_cluster.csv')
    
    configs = [1, 2, 3]
    
    # Requirement: Exactly three comparison plots (one for each configuration)
    for config in configs:
        pc_data = df_pc[df_pc['Config'] == config]
        cl_data = df_cluster[df_cluster['Config'] == config]
        
        plt.figure(figsize=(8, 6))
        
        # Plot PC and Cluster lines on the same graph
        plt.plot(pc_data['Points'], pc_data['Total_Interp_Time'], marker='o', color='blue', linewidth=2, label='Lab PC')
        plt.plot(cl_data['Points'], cl_data['Total_Interp_Time'], marker='s', color='red', linewidth=2, label='HPC Cluster')
        
        # Requirement: log scale for both x and y axes
        plt.xscale('log')
        plt.yscale('log')
        
        plt.xlabel('Number of Particles (log scale)', fontweight='bold')
        plt.ylabel('Execution Time in seconds (log scale)', fontweight='bold')
        plt.title(f'Experiment 01: Scaling - Config {config}')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'Exp1_Config_{config}_Comparison.png')
        plt.close()

def plot_exp2_consistency():
    print("Generating Experiment 2 plots...")
    
    df_pc = pd.read_csv('exp2_results_PC.csv')
    df_cluster = pd.read_csv('exp2_results_cluster.csv')
    
    labels = ['Config 1\n(250x100)', 'Config 2\n(500x200)', 'Config 3\n(1000x400)']
    x = np.arange(len(labels))
    width = 0.35  

    # Requirement: Bar plot of execution time vs. problem index for both PC and Cluster
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Create grouped bars side-by-side
    bars_pc = ax.bar(x - width/2, df_pc['Total_Interp_Time'], width, label='Lab PC', color='skyblue', edgecolor='black')
    bars_cl = ax.bar(x + width/2, df_cluster['Total_Interp_Time'], width, label='HPC Cluster', color='salmon', edgecolor='black')
    
    ax.set_xlabel('Problem Index (Grid Configuration)', fontweight='bold')
    ax.set_ylabel('Total Interpolation Time (seconds)', fontweight='bold')
    ax.set_title('Experiment 02: Execution Time vs. Problem Index')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add number labels on top of the bars
    for bar in bars_pc + bars_cl:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom', fontweight='bold', fontsize=9)
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Exp2_Bar_Comparison.png')
    plt.close()

def plot_exp3(machine_name, file_name):
    print(f"Generating Experiment 3 plots for {machine_name}...")
    
    df = pd.read_csv(file_name)
    df_serial = df[df['Mode'] == 'Serial']
    df_parallel = df[df['Mode'] == 'Parallel']

    # Requirement 1: Iteration vs. Interpolation Time, Mover Time, and Total Time
    plt.figure(figsize=(9, 6))
    plt.plot(df_serial['Iteration'], df_serial['InterpTime'], marker='o', label='Interpolation Time', linewidth=2)
    plt.plot(df_serial['Iteration'], df_serial['MoverTime'], marker='s', label='Mover Time', linewidth=2)
    plt.plot(df_serial['Iteration'], df_serial['TotalTime'], marker='^', label='Total Time', linewidth=2)
    
    plt.xlabel('Iteration', fontweight='bold')
    plt.ylabel('Time (seconds)', fontweight='bold')
    plt.title(f'Experiment 03: Iteration vs. Timings ({machine_name} - Serial Mode)')
    plt.xticks(df_serial['Iteration'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'Exp3_Iteration_Timings_{machine_name}.png')
    plt.close()

    # Requirement 2: Comparison plot showing Mover's serial vs. parallel execution times
    plt.figure(figsize=(9, 6))
    plt.plot(df_serial['Iteration'], df_serial['MoverTime'], marker='o', color='red', label='Serial Mover', linewidth=2)
    plt.plot(df_parallel['Iteration'], df_parallel['MoverTime'], marker='s', color='green', label='Parallel Mover (4 Threads)', linewidth=2)
    
    plt.xlabel('Iteration', fontweight='bold')
    plt.ylabel('Mover Execution Time (seconds)', fontweight='bold')
    plt.title(f'Experiment 03: Mover Serial vs. Parallel Time ({machine_name})')
    plt.xticks(df_serial['Iteration'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'Exp3_Serial_Vs_Parallel_{machine_name}.png')
    plt.close()

    # Requirement 3: Plot showing the speedup achieved
    serial_total = df_serial['MoverTime'].sum()
    parallel_total = df_parallel['MoverTime'].sum()
    
    # Speedup math
    speedup = serial_total / parallel_total 

    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(['Serial Total Time', 'Parallel Total Time'], [serial_total, parallel_total], color=['salmon', 'lightgreen'], edgecolor='black')
    
    ax.set_ylabel('Total Time for 10 Iterations (seconds)', fontweight='bold')
    ax.set_title(f'Experiment 03: Mover Speedup ({machine_name})')
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + (yval*0.02), f"{yval:.2f}s", ha='center', va='bottom', fontweight='bold')
        
    # Show the speedup value clearly
    ax.text(0.5, 0.8, f"Calculated Speedup: {speedup:.4f}x",
            ha='center', va='center', transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'), fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'Exp3_Speedup_{machine_name}.png')
    plt.close()

if __name__ == "__main__":
    plot_exp1_scaling()
    plot_exp2_consistency()
    plot_exp3('Lab_PC', 'exp3_results_PC.csv')
    plot_exp3('HPC_Cluster', 'exp3_results_cluster.csv')
    print("Done! All required plots are ready for your lab report.")