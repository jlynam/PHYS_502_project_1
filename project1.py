import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main():
    # Defining necessary constants
    L = 1
    mu = 0
    sigma = 1
    num_samples = 100
    d_sample = L / num_samples
    normalization_tolerance = 1e-8
    print('space between samples:', d_sample)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.linspace(0, L, num_samples)

    s = np.random.normal(mu, sigma, size=num_samples - 2)
    s = np.concatenate(([0.0], s, [0.0]))

    # Getting the modulo squared
    s2 = s ** 2 

    integral = sum([val / d_sample for val in s2])
    normalizer = 1 / np.sqrt(integral)
    s = s * normalizer
    
    integral = sum([val / d_sample for val in s ** 2])
    assert(abs(integral - 1) < normalization_tolerance)
    print("Assert that psi has been normalized (==1):", integral)

    ax.plot(x, s ** 2)
    plt.show()


if __name__ == "__main__":
    print("The initial function will look like a mess")
    main()
