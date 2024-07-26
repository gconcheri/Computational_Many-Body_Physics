"""

Module for the exact diagonalization of the Hamiltonian 

H = J sum_j [ c_j^{dag} c_{j+1} + h.c. ] + J_2 sum_j [ c_j^{dag} c_{j+2} + h.c. ]
+ J sum_j ( 1 + delta*j )( n_j - 1/2 )( n_{j+1} - 1/2 ) + J_2 sum_j ( 1 + delta*j )( n_j - 1/2 )( n_{j+2} - 1/2 )

"""


import numpy as np
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def calc_basis(L, N):
    basis = []
    ind_in_basis = dict()
    for state in range(2**L):
        if bin(state).count('1') == N:
            ind_in_basis[state] = len(basis)
            basis.append(state)
    return basis, ind_in_basis

def get_bit(number, bit_position): #calculates bit of number in position = bit_position
    # Shift the desired bit to the least significant position and mask all other bits
    return (number >> (bit_position-1)) & 1 # first position = 1

def diagonal(state, delta, J, J2, L):
    sum = 0
    # Iterate through the bits for nearest-neighbor interactions
    for j in range(1,L):
        sum += J*(1+delta*j)*(get_bit(state,j)-0.5)*(get_bit(state,j+1)-0.5)
    # Iterate through the bits for next-nearest-neighbor interactions
    for j in range(1,L-1):
        sum += J2*(1+delta*j)*(get_bit(state,j)-0.5)*(get_bit(state,j+2)-0.5)

    return sum

# kinetic part of the Hamiltonian exchanges occupation numbers iff they're not equal
# (in which case they are mapped to zero)

def flip_nn(s, j, L):
    """Flip the bits of the state s at positions j and (j+1)%L only if they're different."""
    if (get_bit(s, j) & 1) == (get_bit(s, j+1) & 1):
        return 0
    else:
        return s ^ (1 << (j-1) | 1 << j)

def flip_next_to_nn(s, j, L):
    """Flip the bits of the state s at positions j and (j+2)%L only if they're different."""
    if (get_bit(s, j) & 1) == (get_bit(s, j+2) & 1):
        return 0
    else:
        return s ^ (1 << (j-1) | 1 << (j+1))

def calc_H(L, N, delta, J, J2):
    """Determine the Hamiltonian as scipy.sparse.csr_matrix."""
    print("Generating Hamiltonian ... ", end="", flush=True)
    basis, ind_in_basis = calc_basis(L,N)
    M = len(basis)
    H_data = []
    H_inds = []
    a = 0
    for sa in basis:
        H_data.append(diagonal(sa, delta, J, J2, L)) #calculation of diagonal matrix elements 
        H_inds.append((a, a))

        for j in range(1,L-1):
            b = flip_nn(sa, j, L)
            c = flip_next_to_nn(sa,j,L)
            if b > 0: 
                H_data.append(J) #calculation of off-diagonal matrix elements 
                #H_inds.append((basis.index(flip_nn(sa, j, L)), a))
                H_inds.append((ind_in_basis[b], a))
            if c > 0 :
                H_data.append(J2)
                H_inds.append((ind_in_basis[c], a))
                
        if flip_nn(sa,L-1,L) > 0:
            H_data.append(J)
            H_inds.append((ind_in_basis[flip_nn(sa,L-1,L)],a))

        a += 1
    H_inds = np.array(H_inds)
    H_data = np.array(H_data)

    H = csr_matrix((H_data, (H_inds[:, 0], H_inds[:, 1])), shape=(M,M),dtype=complex)

    print("done", flush=True)
    return H

if __name__ == "__main__":

    # Parameters
    J = 1.0
    J2 = 0.64
    delta = 0.05
    L = np.array([12, 16])
    N = np.floor(L * 3 / 4).astype(int)  # Total number of particles (hardcore bosons)

    print("L:", L)
    print("N:", N)

    #transforming Hamiltonian from sparse to array
    Hs = {}
    for i in range(len(L)):
        H = calc_H(L[i], N[i], delta, J, J2)
        Hs[L[i]] = H.toarray()

    #diagonalization of H
    eigvals = {}
    for l in L:
        eig_per_L = np.linalg.eigvalsh(Hs[l])
        eigvals[l] = eig_per_L

    #definition of distribution function we expect
    def GOE(t):
        return np.pi/2*t*np.exp(-np.pi*t**2/4)

    #definition of fitting function
    def pdf(t, a, b):
        return a/2 * t * np.exp(- b*t**2/4) 

    #plotting distribution function of energy level spacings
    s_datas = {}
    for l in L:
        cutoff = int(0.1*len(eigvals[l]))
        print("# of eigvals: ", len(eigvals[l]))
        print("cutoff: ", cutoff)
        eigvals_new = eigvals[l][cutoff:-cutoff]
        print("new # of eigvals: ", len(eigvals_new))
        s_data = []
        for i in range(len(eigvals_new)-1):
            s_data.append((eigvals_new[i+1]-eigvals_new[i]))
            
        s_data = np.array(s_data)/np.mean(s_data)
        s_datas[l] = s_data

        plt.figure()
        entries, bin_edges, patches = plt.hist(s_data, bins=30, density= True, alpha=0.6, color='g', edgecolor='black', label = "Distribution from diagonalization")
        
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        initial_guess = [np.mean(s_data), np.std(s_data)]
        parameters, covmatrix = curve_fit(pdf, bin_centers, entries, p0= initial_guess)
        print(parameters)

        x_fit = np.linspace(min(s_data), max(s_data), 1000)
        y_fit = pdf(x_fit, *parameters)
        plt.plot(x_fit, y_fit, 'r-', lw=2, color = "blue", label = "Fit function")
        plt.plot(bin_centers, GOE(bin_centers), color='orange', label = "GOE distribution function")

        plt.xlabel('normalized level spacings $s_i$')
        plt.ylabel('relative frequency')
        plt.grid(True)
        plt.title("System size L: {}".format(l))
        plt.legend()

plt.show()