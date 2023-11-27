import numpy as np

from scipy.constants import hbar, electron_mass

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.animation import FuncAnimation


"""
This script will solve Schrodinger's equation for an electron in a wire.
The initial condition is a randomly generated time (actually position in this case) series

Goal: generate and visualize the solution surface as a function of position and time.

Keywords: quantum mechanics, schrodinger's equation, finite square well,
          fourier decomposition
"""

"""
Constants
"""
# Related to solving the square well problem
L = 0.01  # length of wire
square_well_constant = np.sqrt(2/L)  # Normalization constant for the square well
revival_time = (4 * electron_mass * np.power(L, 2)) / (np.pi * hbar)

# Related to initial condition
mu = 0  # Mean of the randomly generated time series
sigma = 0.5  # Standard deviation of the randomly generated time series
num_samples = 200  # Number of samples in the initial condition

# Related to solution implementation
num_waves = 50  # Number of waves to generate the solution 
d_sample = L / num_samples  # Distance between sample points (used for integration)
normalization_tolerance = 1e-8  # Tolerance to assess if the solution is normalized
dt = 0.01  # Define how large each time step is
time_stop = revival_time  # Define how long into the future to compute the solution
num_time_steps = int(np.ceil(time_stop / dt))  # Number of steps actually computed in the solution
# print(f"Revival time={revival_time}")


"""
This first plot visualizes the initial condition of the solution

There are two plots:
    1) The initial condition, denoted s
    2) The modulo squared of the initial condition, denoted s2,
        which represents the probability density function
"""
def plot_1(x, s, s2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    axes[0].plot(x, np.zeros(num_samples), color="grey", alpha=0.5)
    axes[0].plot(x, s, label=r"$\psi(x, 0)$", color="blue")
    axes[1].plot(x, np.zeros(num_samples), color="grey", alpha=0.5)
    axes[1].plot(x, s2, label=r"$|\psi(x,0)|^{2}$", color="blue")

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


"""
This second plot visualizes the recreation of the initial condition with the fourier series solution

There are three plots:
    1) The modulo squared of the actual initial condition, denoted s2
    2) The modulo squared of initial condition as recreated by the fourier series solution, denoted psi_total2
    3) A comparison between the actual and recreated initial conditions 
"""
def plot_2(x, s2, psi_total2):
    # Create initial plot for PART 2
    gs = gridspec.GridSpec(4, 4)

    fig = plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(gs[:2, :2])
    ax1.plot(x, np.zeros(num_samples), color="grey", alpha=0.5)
    ax1.plot(x, s2, label=r"$|\psi(x, 0)|^{2}$", color='blue')
    ax1.legend(loc="upper right", fontsize=13)
    ax1.set_ylabel(r"$|\psi(x, 0)|^{2}$")
    ax1.set_xlabel("X $(m)$")

    
    ax2 = plt.subplot(gs[:2, 2:])
    ax2.plot(x, np.zeros(num_samples), color="grey", alpha=0.5)
    ax2.plot(x, psi_total2, label=fr"$|\psi(x, 0)|^{2}$ reconstructed from {num_waves} waves", color='orange')
    ax2.legend(loc="upper left", fontsize=13)
    ax2.set_ylabel(r"$|\psi(x, 0)|^{2}$")
    ax2.set_xlabel("X $(m)$")

    ax3 = plt.subplot(gs[2:4, 1:3])
    ax3.plot(x, np.zeros(num_samples), color="grey", alpha=0.5)
    ax3.plot(x, s2, label=r"Original $|\psi(x,0)|^{2}$", color='blue')
    ax3.plot(x, psi_total2, label="Reconstruction", color='orange')
    ax2.set_xlabel("X $(m)$")
    ax3.legend(loc="upper left", fontsize=13)

    fig.suptitle("Part 2 - Fourier composition")
    fig.tight_layout()
    plt.show()


"""
This third plot visualizes the modulo squared over time

This creates one plot:
    1) A 3d plot of the total solution, denoted psi_td,
        over the lattice formed by the position, denoted x and time, denoted time
"""
def plot_3(x, time, psi_td):
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection='3d')
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle(r"Part 3 - initial time-dependent $\psi(x, 0)$", fontsize=15)

    X, T = np.meshgrid(x, time)
    ax.plot_surface(X, T, psi_td.T)# , cmap="plasma")
    ax.legend(loc="upper right", fontsize=13)
    ax.set_ylabel(r"Recreated $|\psi(x, 0)|^{2}$ with time-dependence")
    ax.set_xlabel("X $(m)$")

    fig.tight_layout()
    plt.show()



class UpdatePlot:
    def __init__(self, ax, x, solution):
        self.line, = ax.plot([], [], '-', color='orange')
        self.x = x
        self.ax = ax
        self.solution = solution.T.real
        self.ax.set_xlim(0, L)
        self.ax.set_ylim(0, np.max(self.solution))
        self.ax.set_title("Animation of the 1D infinite well solution")

    def start(self):
        # Used for the *init_func* parameter of FuncAnimation; this is called when
        # initializing the animation, and also after resizing the figure.
        return self.solution[0].tolist(),

    def __call__(self, i):
        y = self.solution[i].tolist()
        self.line.set_data(self.x, y)
        return self.line,



"""
This fourth plot animates the animation of the modulo squared over time

This is one animation:
    1) The modulo squared of the actual initial condition, denoted s2
"""
def plot_4(x, time, psi_td):
    fig, ax = plt.subplots()
    ud = UpdatePlot(ax, x, psi_td)
    anim = FuncAnimation(fig, ud, init_func=ud.start, frames=len(time), interval=50)
    plt.show()


"""
Generate the initial condition of the wave function
"""
def get_initial_condition(x):
    # Random generation
    # s = np.random.normal(mu, sigma, size=num_samples - 2)
    # s = np.concatenate(([0.0], s, [0.0]))  # Conditions imply the boundaries must be 0s

    # Tent function
    # s = np.concatenate((np.arange(0, L / 2, d_sample), np.arange(L / 2, 0, -d_sample)))

    # Dirac delta
    # s = np.zeros(num_samples)
    # s[int(len(s) / 2)] = 1

    # Quadratic
    def func(x):
        A = np.sqrt(30 / np.power(L, 5))
        return A * x * (L - x)
    s = func(x)

    # Uniform distribution
    # s = np.ones(num_samples)
    
    return s

"""
The main computational engine for this project

1. Computes the initial condition for the position vector
2. Solves the fourier series solution to the square well problem
3. Checks that the fourier series solution recreates the initial condition
4. Shows the evolution of the solution from time t=0 to the revival time

"""
def main():
    # Compute the position and time lattices
    x = np.linspace(0, L, num_samples)
    time = np.arange(0, time_stop, dt)

    # Generate the initial condition
    s = get_initial_condition(x)

    # Determine the modulo squared (Probability density function of position)
    s2 = s ** 2 

    # Normalize with Reimann summation 
    integral = sum([val / d_sample for val in s2])
    normalizer = 1 / np.sqrt(integral)

    s = s * normalizer

    # Assert this constant normalizes the initial condition as intended
    integral = sum([val / d_sample for val in s ** 2])
    # print("Assert that psi has been normalized (==1):", integral)
    assert(abs(integral - 1) < normalization_tolerance)

    # Recompute the modulo squared
    s2 = s ** 2
    
    # Visualize the initial condition
    plot_1(x, s, s2)

    # Now to compute the Fourier series solution of Psi
    c_ns = np.zeros(num_waves)  # Initialize coefficients of the fourier series
    e_ns = np.zeros(num_waves)  # Intiialize energy states 

    psi_ns = np.zeros((num_waves, num_samples))  # Initialize the time-independent component of the solution

    phi_ts = np.zeros((num_waves, num_time_steps), dtype="complex_")  # Initialize the time-dependent component of the solution
    phi_ts_ = np.zeros((num_waves, num_time_steps), dtype="complex_")  # Initialize the complex conjugate of the time-dependent component

    time_dependent_solution = np.zeros((num_waves, num_samples, num_time_steps), dtype='complex_')  # Initialize full solution
    time_dependent_solution_ = np.zeros((num_waves, num_samples, num_time_steps), dtype='complex_')  # Initialize full solution complex conjugate

    for n in range(num_waves):
        # Calculating PSI (position function) and its c_ns
        psi_n = square_well_constant * np.sin((n * np.pi * x) / L)
        cn = np.sum([(p * f) * d_sample for p, f in zip(psi_n, s)])
        c_ns[n] = cn
        psi_ns[n] = psi_n

        e_n = np.power((n * hbar * np.pi / L), 2) / (2 * electron_mass)
        e_ns[n] = e_n

        # Calculating PHI (time function) and its complex conjugate
        phi_t = np.exp(-1j * e_n * time / hbar)
        phi_ts[n] = phi_t

        phi_t_ = np.exp(1j * e_n * time / hbar)
        phi_ts_[n] = phi_t_

        # Calculating the full time_dependent solution and its complex conjugate
        time_dependent_solution[n] = np.outer(psi_n * cn, phi_t)
        time_dependent_solution_[n] = np.outer(psi_n * cn, phi_t_)

    psi_total_td_solution = np.sum(time_dependent_solution, axis=0)
    psi_total_td_solution_ = np.sum(time_dependent_solution_, axis=0)
    psi_total = (psi_total_td_solution * psi_total_td_solution_)
    
    # Confirm that the initial condition is found at time t=0
    plot_2(x, s2, psi_total.T[0])

    # Visualize the evolution of the fourier series solution over time
    # plot_3(x, time, psi_total)
    
    plot_4(x, time, psi_total)

if __name__ == "__main__":
    main()
