import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

# Defining necessary constants
L = 1
mu = 0
sigma = 1
num_samples = 100
num_psis = 50
d_sample = L / num_samples
normalization_tolerance = 1e-8
square_well_constant = np.sqrt(2/L)


def plot_1(x, s, s2):
    # Create initial plot for PART 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    axes[0].plot(x, np.zeros(num_samples), color="grey")
    axes[0].plot(x, s, label=r"$\psi(x, 0)$")
    axes[1].plot(x, s2, label=r"$|\psi(x,0)|^{2}$", color="orange")

    fig.suptitle(r"Part 1 - initial plot of $\psi(x, 0)$", fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(loc="upper left", fontsize=13)
    axes[0].set_xlabel("X")
    axes[0].set_ylabel(r"$\psi(x, 0)$")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(loc="upper right", fontsize=13)
    axes[1].set_ylabel(r"$|\psi(x, 0)|^{2}$")
    axes[1].set_xlabel("X")

    fig.tight_layout()
    plt.show()


def plot_2(x, s2, psi_total):
    # Create initial plot for PART 2
    gs = gridspec.GridSpec(4, 4)

    fig = plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(gs[:2, :2])
    ax1.plot(x, s2, label=r"$|\psi(x, 0)|^{2}$")
    ax1.legend(loc="upper right", fontsize=13)
    ax1.set_ylabel(r"$|\psi(x, 0)|^{2}$")
    ax1.set_xlabel("X")

    
    ax2 = plt.subplot(gs[:2, 2:])
    ax2.plot(x, psi_total ** 2, label=fr"$|\psi(x, 0)|^{2}$ reconstructed from {num_psis} waves")
    ax2.legend(loc="upper left", fontsize=13)
    ax2.set_ylabel(r"$|\psi(x, 0)|^{2}$")
    ax2.set_xlabel("X")


    ax3 = plt.subplot(gs[2:4, 1:3])
    ax3.plot(x, s2, label=r"Original $|\psi(x,0)|^{2}$")
    ax3.plot(x, psi_total ** 2, label="Reconstruction")
    ax3.legend(loc="upper left", fontsize=13)


    fig.suptitle("Part 2 - Fourier composition")
    fig.tight_layout()
    plt.show()


def main():
    x = np.linspace(0, L, num_samples)

    s = np.random.normal(mu, sigma, size=num_samples - 2)
    s = np.concatenate(([0.0], s, [0.0]))

    # Getting the modulo squared
    s2 = s ** 2 

    # Normalization using RHand integration rule
    integral = sum([val / d_sample for val in s2])
    normalizer = 1 / np.sqrt(integral)
    s = s * normalizer
    s2 = s ** 2
    
    # Assert that we have actually found the normalization constant
    integral = sum([val / d_sample for val in s ** 2])
    assert(abs(integral - 1) < normalization_tolerance)
    print("Assert that psi has been normalized (==1):", integral)

    plot_1(x, s, s2)

    # Now to compute the Fourier series of psi
    c_ns = np.zeros(num_psis)
    psi_ns = np.zeros((num_psis, num_samples))
    for n in range(num_psis):
        psi = square_well_constant * np.sin((n * np.pi * x) / L)
        cn = np.sum([(p * f) * d_sample for p, f in zip(psi, s)])
        c_ns[n] = cn
        psi_ns[n] = psi * cn

    psi_total = np.sum(psi_ns, axis=0)

    plot_2(x, s2, psi_total)


if __name__ == "__main__":
    main()
