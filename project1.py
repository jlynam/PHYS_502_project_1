import numpy as np

from scipy.constants import hbar, electron_mass

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

# Defining necessary constants
L = 0.01
mu = 0
sigma = 1
num_samples = 100
num_waves = 5
d_sample = L / num_samples
normalization_tolerance = 1e-8
square_well_constant = np.sqrt(2/L)

revival_time = (4 * electron_mass * np.power(L, 2)) / (np.pi * hbar)
dt = 1
# print("Revival time=", revival_time)
time_stop = 10
num_time_steps = int(np.ceil(time_stop / dt))


def plot_1(x, s, s2):
    # Create initial plot for PART 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    axes[0].plot(x, np.zeros(num_samples), color="grey")
    axes[0].plot(x, s, label=r"$\psi(x, 0)$")
    axes[1].plot(x, s2, label=r"$|\psi(x,0)|^{2}$", color="orange")

    fig.suptitle(r"Part 1 - initial plot of $\psi(x, 0)$", fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(loc="upper left", fontsize=13)
    axes[0].set_xlabel("X $(m)$")
    axes[0].set_ylabel(r"$\psi(x, 0)$")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(loc="upper right", fontsize=13)
    axes[1].set_ylabel(r"$|\psi(x, 0)|^{2}$")
    axes[1].set_xlabel("X $(m)$")

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
    ax1.set_xlabel("X $(m)$")

    
    ax2 = plt.subplot(gs[:2, 2:])
    ax2.plot(x, psi_total ** 2, label=fr"$|\psi(x, 0)|^{2}$ reconstructed from {num_waves} waves")
    ax2.legend(loc="upper left", fontsize=13)
    ax2.set_ylabel(r"$|\psi(x, 0)|^{2}$")
    ax2.set_xlabel("X $(m)$")


    ax3 = plt.subplot(gs[2:4, 1:3])
    ax3.plot(x, s2, label=r"Original $|\psi(x,0)|^{2}$")
    ax3.plot(x, psi_total ** 2, label="Reconstruction")
    ax2.set_xlabel("X $(m)$")
    ax3.legend(loc="upper left", fontsize=13)


    fig.suptitle("Part 2 - Fourier composition")
    fig.tight_layout()
    plt.show()


def plot_3(x, s2, psi_td):
    psi_total = psi_td[0]

    # Create initial plot for PART 3
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(r"Part 3 - initial time-dependent $\psi(x, 0)$", fontsize=15)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(loc="upper left", fontsize=13)
    axes[0].set_xlabel("X $(m)$")
    axes[0].set_ylabel(r"Original $|\psi(x, 0)|^{2}$")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(loc="upper right", fontsize=13)
    axes[1].set_ylabel(r"Recreated $|\psi(x, 0)|^{2}$ with time-dependence")
    axes[1].set_xlabel("X $(m)$")

    fig.tight_layout()
    plt.show()


def main():
    x = np.linspace(0, L, num_samples)
    time = np.arange(0, time_stop, dt)

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

    # plot_1(x, s, s2)

    # Now to compute the Fourier series of psi
    c_ns = np.zeros(num_waves)
    e_ns = np.zeros(num_waves)

    psi_ns = np.zeros((num_waves, num_samples))

    phi_ts = np.zeros((num_waves, num_time_steps), dtype="complex_") 
    phi_i_ts = np.zeros((num_waves, num_time_steps), dtype="complex_")

    for n in range(num_waves):
        psi = square_well_constant * np.sin((n * np.pi * x) / L)
        cn = np.sum([(p * f) * d_sample for p, f in zip(psi, s)])
        c_ns[n] = cn
        psi_ns[n] = psi * cn

        e_n = np.power((n * hbar * np.pi / L), 2) / (2 * electron_mass)
        e_ns[n] = e_n

        # Calculating phi and its complex conjugate
        phi_t = np.exp(-1j * e_n * time / hbar)
        phi_ts[n] = phi_t

        phi_i_t = np.exp(1j * e_n * time / hbar)
        phi_i_ts[n] = phi_i_t


    # First showing psi without the time-dependence
    psi_total = np.sum(psi_ns, axis=0)
    # plot_2(x, s2, psi_total)

    # Now calculating the time-dependent solution
    # the full time-dependent solution should be a 3D array
    # of dimensions (n_waves, n_x_samples, n_t_samples)
    print(psi_ns.shape)
    print(phi_ts.shape)
    time_solution = np.meshgrid(psi_ns, phi_ts)
    time_solution_i = np.meshgrid(psi_ns, phi_i_ts)
    print(len(time_solution), time_solution[0].shape)

    print(time_solution)




    # plot_3(x, s2, psi_td)


if __name__ == "__main__":
    main()
