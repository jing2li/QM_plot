import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn
from scipy.optimize import brentq

# Solve bessel function using Brent method
def find_zeros(l, n_zeros):
    """Finds the first n_zeros for the spherical Bessel function j_l(x)"""
    zeros = []
    x = 0.1  # Start value
    step = 0.5
  
    while len(zeros) < n_zeros:
        # Search for a sign change to bracket the root
        if spherical_jn(l, x) * spherical_jn(l, x + step) < 0:
            root = brentq(f=lambda x: spherical_jn(l, x), a=x, b=x + step, xtol=1e-13, rtol=1e-10)
            zeros.append(root)
        x += step
    return np.array(zeros)

"""Customisable Parameters"""
l_labels = ['s', 'p', 'd', 'f']
n_l = 4
l_values = range(n_l)
n_roots = 3

plt.figure(figsize=(15, 9))
levels = {}

print("Numerical values of knl:")
for l in l_values:
    # solve l-th Bessel function
    roots = find_zeros(l, n_roots)

    # print zeros to Bessel function
    for i in range(n_roots):
        print("k", i, l, " = ", roots[i])

    # Energy E = hbar^2 / (2ma^2) * k^2. 
    # Express E in unit (pi^2 hbar^2)/(2ma^2) makes E_10 = 1.
    # E = (k/pi)^2
    energies = (roots / np.pi)**2
    levels[l] = energies
    
    for n_idx, E in enumerate(energies):
        n = n_idx + 1
        # Draw the level
        plt.hlines(E, l - 0.25, l + 0.25, colors='black', linewidth=2.5)
        # Label the state (1s, 2p, etc.)
        plt.text(l, E + 0.15, f'{n}{l_labels[l]}', ha='center', fontweight='bold')

# Draw connecting dashed lines for groups
for n_idx in range(n_roots):
    for i in range(n_l-1):
        l_curr = l_values[i]
        l_next = l_values[i+1]

        x_start, x_end = l_curr+0.25, l_next-0.25
        y_start, y_end = [levels[l_curr][n_idx]], [levels[l_next][n_idx]]

        plt.plot([x_start, x_end], [y_start, y_end], 'k--', alpha=0.3)
    plt.text(n_l - 0.65, levels[n_l-1][n_idx], f'{n_idx+1}. Nullstelle', va='center', style='italic')

# Aesthetics
plt.xticks(l_values, [f'$l={l}$' for l in l_values])
plt.yticks(range(round(levels[n_l-1][n_roots-1])+1))
plt.ylabel(r'Energy $E_{nl}$ in units of $\frac{\pi^2 \hbar^2}{2ma^2}$', fontsize=12)
plt.title('Energy Levels: Infinite Spherical Potential Well', pad=20)
plt.ylim(0, round(levels[n_l-1][n_roots-1])+1)
plt.xlim(-0.5, n_l)
plt.grid(axis='y', linestyle=':', alpha=0.5)

plt.figtext(0.5, 0.02, r"Note: Each level is $(2l+1)$-fold degenerate, m=(-l, -l+1, ..., l)", 
            ha="center", bbox={"facecolor":"grey", "alpha":0.1, "pad":5})

plt.show()