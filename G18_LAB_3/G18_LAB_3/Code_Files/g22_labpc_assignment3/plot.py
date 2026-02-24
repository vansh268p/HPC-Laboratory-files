import matplotlib.pyplot as plt

configs = ['a', 'b', 'c', 'd', 'e']
times = [0.042724, 0.227670, 0.170778, 0.919918, 1.040558]  

plt.figure(figsize=(10, 6))
bars = plt.bar(configs, times, color='steelblue')
plt.xlabel('Test Configuration')
plt.ylabel('Execution Time (seconds)')
plt.title('Serial Interpolation Performance - Lab PC')
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}s', ha='center', va='bottom')

plt.savefig('labpc_performance.png', dpi=150)
print("Plot saved!")