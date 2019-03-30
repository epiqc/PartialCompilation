print("Begin imports...", flush=True)
import sys
sys.path.append('..')
#import config
from fqc import uccsd, qaoa, util

import numpy as np
from datetime import datetime

#data_path = config.DATA_PATH
data_path = '/project/ftchong/qoc/yongshan'  # make a YOUR_NAME folder here
file_name = datetime.today().strftime('%h%d')

from quantum_optimal_control.helper_functions.grape_functions import transmon_gate
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control.core import hamiltonian
print("End imports...", flush=True)

def binary_search_for_shortest_pulse_time(min_time, max_time, convergence, reg_coeffs):
    """Search between [min_time, max_time] up to 1ns tolerance. Assumes 20 steps per ns.""" 
    min_steps, max_steps = min_time * 20, max_time * 20
    while min_steps + 20 < max_steps: # just estimate to +- 1ns
        mid_steps = int((min_steps + max_steps) / 2)
        total_time = mid_steps / 20.0
        print('\n\ntrying total_time: %s for unitary of size %s' % (str(total_time), str(U.shape)))
        SS = Grape(H0, Hops, Hnames, U, total_time, mid_steps, states_concerned_list, convergence,
                    reg_coeffs=reg_coeffs,
                    use_gpu=False, sparse_H=False, method='ADAM', maxA=maxA, show_plots=False, file_name=file_name, data_path=data_path)
        if SS.l < SS.conv.conv_target: # if converged, search lower half 
            max_steps = mid_steps
        else:
            min_steps = mid_steps
    return mid_steps / 20

full_times = []
for N in range(4, 5, 2):
    row = 2
    col = int(N / 2)
    d = 2 # this is the number of energy levels to consider (i.e. d-level qudits) 
    max_iterations = (2**N) * 125
    decay = max_iterations / 2
    convergence = {'rate':0.01, 'max_iterations': max_iterations,
               'conv_target':1e-3, 'learning_rate_decay':decay, 'min_grad': 1e-9, 'update_step': 20}
    reg_coeffs = {}
    for p in range(1,3):
        print("Running 3-regular graph N=%d, p=%d" % (N, p), flush=True)
        circuit = qaoa.get_qaoa_circuit(N, p, '3Reg')
        # TODO: average over 10 graphs
        U = util.circuitutil.get_unitary(circuit)
        connected_qubit_pairs = util.get_nearest_neighbor_coupling_list(row, col, directed=False)
        H0 = hamiltonian.get_H0(N, d)
        Hops, Hnames = hamiltonian.get_Hops_and_Hnames(N, d, connected_qubit_pairs)
        states_concerned_list = hamiltonian.get_full_states_concerned_list(N, d)
        maxA = hamiltonian.get_maxA(N, d, connected_qubit_pairs)
        shortest_time = binary_search_for_shortest_pulse_time(10, int(90*(N/5)*p), convergence, reg_coeffs)
        full_times.append(shortest_time)

print(full_times, flush=True)

## using driver to get fermionic Hamiltonian
## PySCF example
#driver = PySCFDriver(atom='Li .0 .0 .0; H .0 .0 1.6', unit=UnitsType.ANGSTROM,
#                     charge=0, spin=0, basis='sto3g')
#molecule = driver.run()
## please be aware that the idx here with respective to original idx
#freeze_list = [0]
#remove_list = [-3, -2] # negative number denotes the reverse order
#map_type = 'parity'
#
#h1 = molecule.one_body_integrals
#h2 = molecule.two_body_integrals
#nuclear_repulsion_energy = molecule.nuclear_repulsion_energy
#
#num_particles = molecule.num_alpha + molecule.num_beta
#num_spin_orbitals = molecule.num_orbitals * 2
#print("HF energy: {}".format(molecule.hf_energy - molecule.nuclear_repulsion_energy))
#print("# of electrons: {}".format(num_particles))
#print("# of spin orbitals: {}".format(num_spin_orbitals))
#
## prepare full idx of freeze_list and remove_list
## convert all negative idx to positive
#remove_list = [x % molecule.num_orbitals for x in remove_list]
#freeze_list = [x % molecule.num_orbitals for x in freeze_list]
## update the idx in remove_list of the idx after frozen, since the idx of orbitals are changed after freezing
#remove_list = [x - len(freeze_list) for x in remove_list]
#remove_list += [x + molecule.num_orbitals - len(freeze_list)  for x in remove_list]
#freeze_list += [x + molecule.num_orbitals for x in freeze_list]
#
## prepare fermionic hamiltonian with orbital freezing and eliminating, and then map to qubit hamiltonian
## and if PARITY mapping is selected, reduction qubits
#energy_shift = 0.0
#qubit_reduction = True if map_type == 'parity' else False
#
#ferOp = FermionicOperator(h1=h1, h2=h2)
#if len(freeze_list) > 0:
#    ferOp, energy_shift = ferOp.fermion_mode_freezing(freeze_list)
#    num_spin_orbitals -= len(freeze_list)
#    num_particles -= len(freeze_list)
#if len(remove_list) > 0:
#    ferOp = ferOp.fermion_mode_elimination(remove_list)
#    num_spin_orbitals -= len(remove_list)
#
#qubitOp = ferOp.mapping(map_type=map_type, threshold=0.00000001)
#qubitOp = qubitOp.two_qubit_reduced_operator(num_particles) if qubit_reduction else qubitOp
#qubitOp.chop(10**-10)
#
##print(qubitOp.print_operators())
#print(qubitOp)
#
## setup HartreeFock state
#HF_state = HartreeFock(qubitOp.num_qubits, num_spin_orbitals, num_particles, map_type, 
#                       qubit_reduction)
#
## setup UCCSD variational form
#var_form = UCCSD(qubitOp.num_qubits, depth=1, num_orbitals=num_spin_orbitals, num_particles=num_particles, active_occupied=[0], active_unoccupied=[0, 1], initial_state=HF_state, qubit_mapping=map_type, two_qubit_reduction=qubit_reduction, num_time_slices=1)
#
#parameters = [0.9999999] * 8
#circuit = var_form.construct_circuit(parameters, use_basis_gates=False)
#
#circuit.draw(output='text', line_length=350)
#
#backend_sim = Aer.get_backend('unitary_simulator')
#
#result = qiskit.execute(circuit, backend_sim).result()
#
#unitary = result.get_unitary()
#print(unitary.shape)
#print(unitary)


