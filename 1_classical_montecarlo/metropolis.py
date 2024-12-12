import numpy as np
import matplotlib.pyplot as plt
import time

from numba import jit


@jit(nopython=True)
def energy(system, i, j, L):
    """Energy function of spins connected to site (i, j)."""
    return -1. * system[i, j] * (system[np.mod(i - 1, L), j] + system[np.mod(i + 1, L), j] +
                                 system[i, np.mod(j - 1, L)] + system[i, np.mod(j + 1, L)])


@jit
def prepare_system(L):
    """Initialize the system."""
    system = 2 * (0.5 - np.random.randint(0, 2, size=(L, L)))
    return system


@jit(nopython=True)
def measure_energy(system):
    L = system.shape[0]
    E = 0
    for i in range(L):
        for j in range(L):
            E += energy(system, i, j, L) / 2.
    return E

#without magnetization
@jit(nopython=True)
def metropolis_loop(system, T, N_sweeps, N_eq, N_flips):
    """ Main loop doing the Metropolis algorithm."""
    E = measure_energy(system)
    L = system.shape[0]
    E_list = []
    for step in range(N_sweeps + N_eq):
        i = np.random.randint(0, L) #pick random site x coordinate between 0 and L-1 (flip spin on that site)
        j = np.random.randint(0, L) #same for y

        dE = -2. * energy(system, i, j, L) 
        if dE <= 0.: #prob(new configuration) > prob(old configuration)
            system[i, j] *= -1 # we actually flip spin at site (i,j)
            E += dE
        elif np.exp(-1. / T * dE) > np.random.rand():
            system[i, j] *= -1
            E += dE

        if step >= N_eq and np.mod(step, N_flips) == 0:
            # measurement
            E_list.append(E) #instead of creating a list of the accepted "configurations" of the
                             #system used to calculate the energy, we directly create the list of energies 
                             #of the accepted configurations
    return np.array(E_list)

#with magnetization and trasversal field
@jit(nopython=True)
def metropolis_loop_h(system, T, N_sweeps, N_eq, N_flips, h):
    """ Main loop doing the Metropolis algorithm."""
    M = np.sum(system)   # this is new
    E = metropolis.measure_energy(system) - h * M # include the h here
    L = system.shape[0]
    E_list = []
    M_list = []
    for step in range(N_sweeps + N_eq):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        dE = -2. * metropolis.energy(system, i, j, L)  + h * 2* system[i, j]  # and another place where h is needed
        if dE <= 0.:
            system[i, j] *= -1
            E += dE
            M += 2*system[i,j]
        elif np.exp(-1. / T * dE) > np.random.rand():
            system[i, j] *= -1
            E += dE
            M += 2*system[i,j]
        if step >= N_eq and np.mod(step, N_flips) == 0:
            # measurement
            E_list.append(E)
            M_list.append(M)
    assert(M == np.sum(system))
    return np.array(E_list), np.array(M_list)

if __name__ == "__main__":
    """ Scan through some temperatures """
    # Set parameters here
    L = 10  # Linear system size
    N_sweeps = 10000  # Number of steps for the measurements
    N_eq = 1000  # Number of equilibration steps before the measurements start
    N_flips = 10  # Number of steps between measurements
    N_bins = 10  # Number of bins use for the error analysis

    T_range = np.arange(1.5, 3.1, 0.1)

    C_list = [] #specific heat
    system = prepare_system(L)
    for T in T_range:
        C_list_bin = []
        for k in range(N_bins):
            Es = metropolis_loop(system, T, N_sweeps, N_eq, N_flips)

            mean_E = np.mean(Es)
            mean_E2 = np.mean(Es**2)

            #we estimate 10 specific heats for each temperature, each calculated with a new metropolis loop
            C_list_bin.append(1. / T**2. / L**2. * (mean_E2 - mean_E**2))
        C_list.append([np.mean(C_list_bin), np.std(C_list_bin) / np.sqrt(N_bins)]) 
        #we finally calulate the mean and the variance of the 10 specific heats for each temperature

        print(T, mean_E, C_list[-1])

    # Plot the results
    C_list = np.array(C_list)
    plt.errorbar(T_range, C_list[:, 0], C_list[:, 1])
    Tc = 2. / np.log(1. + np.sqrt(2))
    print(Tc)
    plt.axvline(Tc, color='r', linestyle='--')
    plt.xlabel('$T$')
    plt.ylabel('$c$')
    plt.show()
