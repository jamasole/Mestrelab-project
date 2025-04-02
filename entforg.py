import numpy as np
#from qiskit.opflow import I, X, Y, Z, PauliExpectation, CircuitStateFn, ListOp, CircuitSampler, StateFn
#from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
JW = JordanWignerMapper()
##################################################################################################################################
#
#                                            Classical routines
#    
##################################################################################################################################

# Function that creates a TFD given beta, U V and E
def get_TFD(beta, U, V, E):
    # Create array with exponentials of energies
    exps = np.exp(-beta*E/2.)
    
    # Normalize it so we get the lambdas
    lambdas = exps/np.sqrt(np.sum(exps*exps))
    
    #Create matrix that has as columns |n>|ñ>
    nn = np.array([np.kron(U[:, i], V[:, i]) for i in range(len(U))]).T
    
    #Sum all the columns multiplied by the exponentials
    TFD = np.sum(nn*lambdas, axis = 1)
    return TFD, lambdas

def get_TFD_trunc(beta, U, V, E, mask):
    '''
    Get truncated TFD

    beta: inverse temperature
    U: left eigenvectors
    V: right eigenvectors
    E: energies
    mask: array of indexes to keep
    '''
    E_trunc = E[mask]

    # Create array with exponentials of energies
    exps = np.exp(-beta*E_trunc/2.)
    
    # Normalize it so we get the lambdas
    lambdas = exps/np.sqrt(np.sum(exps*exps))
    
    #Create matrix that has as columns |n>|ñ>
    nn = np.array([np.kron(U[:, i], V[:, i]) for i in mask]).T
    
    #Sum all the columns multiplied by the exponentials
    TFD = np.sum(nn*lambdas, axis = 1)
    return TFD, lambdas

def create_H_int(beta, omega, N_sites):
    a = lambda i: FermionicOp({"-_%i" % i: 1.}, num_spin_orbitals=2*N_sites)
    a_dag = lambda i: a(i).adjoint()
    v = np.exp(-beta*omega/2)/np.sqrt(1+np.exp(-beta*omega))
    u = 1/np.sqrt(1+np.exp(-beta*omega))
    H_int = sum([2*(v[i]*u[i]*omega[i])/(u[i]*u[i]-v[i]*v[i])*(a(i)@a(i+N_sites)+a_dag(i+N_sites)@a_dag(i)) for i in range(N_sites)])
    return H_int

def quita_Z(H):
    '''Function that changes all Zs in H to Is'''
    h_list = H.to_list()
    for i, term in enumerate(h_list):
        h_list[i] = (term[0].replace('Z', 'I'), term[1])
    return SparsePauliOp.from_list(h_list)


##################################################################################################################################
#
#                                            Quantum routines
#    
##################################################################################################################################

"""
def create_diagonal_circuits(n_qubits_site):
    # Function that creates |b_n> circuits
    diagonal_circuits = [0] * 2**n_qubits_site

    for i in range(2**n_qubits_site):
        bin_string = bin(i)[2:]
        qc = QuantumCircuit(n_qubits_site)
        for k, j in enumerate(reversed(bin_string)):
            if j == '1':
                qc.x(k)
        diagonal_circuits[i] = qc
    return diagonal_circuits

def create_phi_circuit(qc, x, y, p):
    # The logic followed here is explained in https://arxiv.org/abs/2104.10220 SM3
    comparator = np.array(list(x))==np.array(list(y))
    #same = [i for i, x in enumerate(comparator) if x]  
    diff = [i for i, x in enumerate(comparator) if not x] # List of positions with differences
    k = diff[0] # First different bit
    if x[k] == '1': # if x[k] == 1 change p to -p, else, don't. This can be done because there is a simmetry p-->-p up to a phase
        x2 = y
        y2 = x
        p2 = -p
    else:
        x2 = x
        y2 = y
        p2 = p
        
    for i in range(len(x2)): # Prepare state x
        if x2[i] == '1' and i != k:
            qc.x(len(x)-1-i)
    # Apply H and P gates
    qc.h(len(x)-1-k)
    qc.p(p2*np.pi/2, len(x)-1-k)
    
    # Apply cnots
    for l in diff:
        if l != k:
            qc.cnot(len(x)-1-k, len(x)-1-l)
    return qc

def create_off_diagonal_circuits(n_qubits_site):
    # Create all phi circuits
    format_str = '{0:0' + str(n_qubits_site) +'b}'

    phi_circuits = [0] * (4*(2**(2*n_qubits_site-1)-2**(n_qubits_site-1)))

    i = 0
    qc = QuantumCircuit(n_qubits_site)
    for n in range(2**n_qubits_site):
        bn = format_str.format(n)
        for m in range(n):
            bm = format_str.format(m)
            phi_circuits[i]   = create_phi_circuit(qc.copy(), bn, bm, 0)
            phi_circuits[i+1] = create_phi_circuit(qc.copy(), bn, bm, 1)
            phi_circuits[i+2] = create_phi_circuit(qc.copy(), bn, bm, 2)
            phi_circuits[i+3] = create_phi_circuit(qc.copy(), bn, bm, 3)
            i+=4
    return phi_circuits


def expectation_h_tot(params, beta, diagonal_circuits_U, diagonal_circuits_V, phi_circuits_U, phi_circuits_V, H_q, H_int_q, sampler):
    #FIXME: In this case U and V have to share parameters
    theta_par = np.array(params)
    #print('beta=', beta, 'theta=', theta_par)

    # Bind parameters to both diagonal and off diagonal circuits
    diagonal_circuits_to_measure_U = [circuit.bind_parameters(theta_par) for circuit in diagonal_circuits_U]
    phi_circuits_to_measure_U      = [circuit.bind_parameters(theta_par) for circuit in phi_circuits_U]
    
    diagonal_circuits_to_measure_V = [circuit.bind_parameters(theta_par) for circuit in diagonal_circuits_V]
    phi_circuits_to_measure_V      = [circuit.bind_parameters(theta_par) for circuit in phi_circuits_V]
    
    # First compute the energy of H_L
    # We only have to compute diagonal terms
    sampled_energies = measure(H_q, diagonal_circuits_to_measure_U, sampler)
    #print('Energies =', sampled_energies)
    
    #print('lambdas= ',lambdas)

    n_energies = len(sampled_energies)
    H_int_array = np.zeros((n_energies, n_energies), dtype='complex')
    
    # Split each term of H_int in halves
    #print(H_int_q)
    for H_int_term in H_int_q.primitive.to_list():
        pauli_string = H_int_term[0]
        n_qubits_total = len(pauli_string)
        n_qubits_site = len(pauli_string)//2
        coef = H_int_term[1]
        #
        if pauli_string == 'I'*n_qubits_total:
            # Nothing to compute, the expectation value will be np.sum(lambdas*lambdas)
            H_int_array += coef*np.eye(n_energies, dtype='complex') #This has to be one 
        elif pauli_string[:n_qubits_total//2] == 'I'*(n_qubits_site):
            # The off diagonal terms will be zero, create O2 and compute its expectation value
            print('NOT IMPLEMENTED')
            pass #TODO
        elif pauli_string[n_qubits_total//2:] == 'I'*(n_qubits_site):
            # The off diagonal terms will be zero, create O1 and compute its expectation value
            print('NOT IMPLEMENTED')
            pass #TODO
        else:
            # This is the general case, create O1 and O2, evaluate expectation values of diagonal and offdiagonal (with the phis) terms
            O1 = eval('^'.join(pauli_string[:n_qubits_site][i:i+1] for i in range(0, len(pauli_string[:n_qubits_site]), 1)))
            O2 = eval('^'.join(pauli_string[n_qubits_site:][i:i+1] for i in range(0, len(pauli_string[n_qubits_site:]), 1)))

            # Compute all expectation values needed
            O1_exps = measure(O1, diagonal_circuits_to_measure_U + phi_circuits_to_measure_U, sampler)
            O2_exps = measure(O2, diagonal_circuits_to_measure_V + phi_circuits_to_measure_V, sampler)
            
            # Diagonal terms
            diagonal_O1_exps = O1_exps[:2**(n_qubits_site)]
            diagonal_O2_exps = O2_exps[:2**(n_qubits_site)]

            # Phi terms
            off_diagonal_O1_exps = O1_exps[2**(n_qubits_site):]
            off_diagonal_O2_exps = O2_exps[2**(n_qubits_site):]
            
            # Manipulate phi expectation values using numpy  to do the p summs
            aux = np.array([1., -1, 1, -1]*(len(off_diagonal_O1_exps)//4))*off_diagonal_O1_exps*off_diagonal_O2_exps
            aux = np.add.reduceat(aux, np.arange(0, len(aux), 4)) 

            arr = np.zeros((2**n_qubits_site, 2**n_qubits_site), dtype='complex')
            indices = np.tril_indices(len(arr), -1)
            arr[indices] = aux
            arr[np.diag_indices(len(arr))] = np.array(diagonal_O1_exps)*np.array(diagonal_O2_exps)
            H_int_array += coef*arr
            
    # Compute lambdas and the complete expectation values
    lambdas = np.exp(-beta*np.array(sampled_energies)/2.)
    lambdas = lambdas/np.sqrt(np.sum(lambdas*lambdas))
    H_int_exp = np.sum(np.transpose(lambdas*np.transpose(lambdas*H_int_array)))
    H_L_exp   = np.sum(sampled_energies*lambdas*lambdas)
    return 2*H_L_exp+H_int_exp

def measure(op, circuits, sampler):
    psi = ListOp([CircuitStateFn(circuit) for circuit in circuits])
    measurable_expression = StateFn(op, is_measurement=True).compose(psi)
    expectation = PauliExpectation(group_paulis=True).convert(measurable_expression)
    return sampler.convert(expectation).eval()

"""