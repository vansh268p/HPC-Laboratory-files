import matplotlib.pyplot as plt

configs = ['a', 'b', 'c', 'd', 'e']
times = [0.250000, 1.200000, 0.880000, 4.630000, 3.430000]  

plt.figure(figsize=(10, 6))
bars = plt.bar(configs, times, color='steelblue')
plt.xlabel('Test Configuration')
plt.ylabel('Execution Time (seconds)')
plt.title('Serial Interpolation Performance - Cluster PC')
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}s', ha='center', va='bottom')

plt.savefig('clusterpc_performance.png', dpi=150)
print("Plot saved!")
