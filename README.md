# Partial Compilation
Repository with code and data for results presented in "Partial Compilation of Variational Algorithms for Noisy Intermediate-Scale Quantum Machines" by Pranav Gokhale, Yongshan Ding, Thomas Propson, Christopher Winkler, Nelson Leung, Yunong Shi, David I. Schuster, Henry Hoffmann, Frederic T. Chong (University of Chicago).

Files are loosely organized and intended for record-keeping. Contact the authors for details.

Rough organization, with some key files listed:
- [PY3_quantum-optimal-control](https://github.com/epiqc/PartialCompilation/tree/master/PY3_quantum-optimal-control): Python 3 version of https://github.com/SchusterLab/quantum-optimal-control.
  - [/quantum_optimal_control/core/hamiltonian.py](https://github.com/epiqc/PartialCompilation/blob/master/PY3_quantum-optimal-control/quantum_optimal_control/core/hamiltonian.py): gmon system Hamiltonian
  - [/quantum_optimal_control/helper_functions/qutip_verification.py](https://github.com/epiqc/PartialCompilation/blob/master/PY3_quantum-optimal-control/quantum_optimal_control/helper_functions/qutip_verification.py): verification script
- [experiments](https://github.com/epiqc/PartialCompilation/tree/master/experiments): Contains most code for partial compilation techniques
  - [/Binary_Search_Strict_Partial_Compilation.ipynb](https://github.com/epiqc/PartialCompilation/blob/master/experiments/Binary_Search_Strict_Partial_Compilation.ipynb): Binary search for shortest pulse times when performing partial compilation
  - [/QAOA](https://github.com/epiqc/PartialCompilation/tree/master/experiments/QAOA): results for QAOA benchmarks
  - [/Gate_Times.ipynb](https://github.com/epiqc/PartialCompilation/blob/master/experiments/Gate_Times.ipynb): gate times for each gate in gate set, computed using optimal control
  - [/GateBasedTimes.ipynb](https://github.com/epiqc/PartialCompilation/blob/master/experiments/GateBasedTimes.ipynb): total gate-based runtimes for each VQE benchmark
  - [/StrictPartialBasedTimes.ipynb](https://github.com/epiqc/PartialCompilation/blob/master/experiments/StrictPartialBasedTimes.ipynb): compiling pulse time results for VQE benchmarks, using strict partial compilation
  - [/FlexiblePartialBasedTimes.ipynb](https://github.com/epiqc/PartialCompilation/blob/master/experiments/FlexiblePartialBasedTimes.ipynb): compiling pulse time results for VQE benchmarks, using flexible partial compilation
  - [/RealisticPulses.ipynb](https://github.com/epiqc/PartialCompilation/blob/master/experiments/RealisticPulses.ipynb): results using more realistic pulses (include qutrit states, require smooth pulse shapes, lower sampling rate).
- [fqc](https://github.com/epiqc/PartialCompilation/tree/master/fqc/): code for breaking circuits into slices that can be partially compiled, especially in [fqc/util](https://github.com/epiqc/PartialCompilation/tree/master/fqc/util)
  - [/models](https://github.com/epiqc/PartialCompilation/tree/master/fqc/models): class definition for circuit components
  - [/qaoa](https://github.com/epiqc/PartialCompilation/tree/master/fqc/qaoa): QAOA benchmark generation code
  - [/uccsd](https://github.com/epiqc/PartialCompilation/tree/master/fqc/uccsd): VQE (UCCSD) benchmark generation code
- [scripts](https://github.com/epiqc/PartialCompilation/tree/master/scripts): scripts for sending large batches of compile jobs to computing cluster
