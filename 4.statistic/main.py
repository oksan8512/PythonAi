import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10

fig, ax = plt.subplots(figsize=(12,4))
x = np.linspace(-2, 2*np.pi, 200)
print(f"data x {x}")
y = np.sin(x)

ax.plot(x, y, 'b-', linewidth=2, label='sin(x)')
ax.plot(x, np.cos(x), 'r--', linewidth=2, label='cos(x)')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Графіки тригонометричних функцій')

ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('simple.png', dpi=150, bbox_inches='tight')
plt.close()