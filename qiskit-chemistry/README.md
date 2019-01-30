# Qiskit Chemistry

[![License](https://img.shields.io/github/license/Qiskit/qiskit-chemistry.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)[![Build Status](https://img.shields.io/travis/com/Qiskit/qiskit-chemistry/master.svg?style=popout-square)](https://travis-ci.com/Qiskit/qiskit-chemistry)[![](https://img.shields.io/github/release/Qiskit/qiskit-chemistry.svg?style=popout-square)](https://github.com/Qiskit/qiskit-chemistry/releases)[![](https://img.shields.io/pypi/dm/qiskit-chemistry.svg?style=popout-square)](https://pypi.org/project/qiskit-chemistry/)

**Qiskit** is an open-source framework for working with noisy intermediate-scale quantum computers (NISQ) at the level of pulses, circuits, algorithms, and applications.

Qiskit is made up elements that work together to enable quantum computing. The element **Aqua**
provides a library of cross-domain algorithms upon which domain-specific applications can be
built. The **Qiskit Chemistry** component has
been created to utilize Aqua for quantum chemistry computations. Aqua is also showcased for other
domains, such as Optimization, Artificial Intelligence, and
Finance, with both code and notebook examples available in the
[qiskit/aqua/chemistry](https://github.com/Qiskit/qiskit-tutorials/tree/master/qiskit/aqua/chemistry)
and [community/aqua/chemistry](https://github.com/Qiskit/qiskit-tutorials/tree/master/community/aqua/chemistry)
folders of the [qiskit-tutorials GitHub Repository](https://github.com/Qiskit/qiskit-tutorials).  

Qiskit Aqua and its applications, such as Qiskit Chemistry, were all designed to be extensible,
and use a pluggable framework where algorithms and support objects used
by algorithms—such as optimizers, variational forms, and oracles—are derived from a defined base class
for the type and discovered dynamically at run time.  In particular, Qiskit Chemistry comes with
chemistry-specific Aqua extensions, such as algorithms, variational forms and initial states that
are suited to simulate molecular structures.

## Installation

We encourage installing Qiskit Chemistry via the pip tool (a python package manager):

```bash
pip install qiskit-chemistry
```
pip will handle all dependencies automatically for you, including the other Qiskit elements upon which
Qiskit Chemistry is built, such as [Aqua](https://github.com/Qiskit/qiskit-aqua) and
[Terra](https://github.com/Qiskit/qiskit-terra), and you will always install the latest (and well-tested)
version.

To run chemistry experiments using Qiskit Chemistry, it is recommended that you to install a classical
computation chemistry software program interfaced by Qiskit Chemistry. 
Several such programs are supported, and while logic to
interface these programs is supplied by Qiskit Chemistry via the above pip installation,
the dependent programs themselves need to be installed separately becausea they are not part of the Qiskit
Chemistry installation bundle.
Qiskit Chemistry comes with prebuilt support to interface the following computational chemistry
software programs:

1. [Gaussian 16&trade;](http://gaussian.com/gaussian16/), a commercial chemistry program
2. [PSI4](http://www.psicode.org/), a chemistry program that exposes a Python interface allowing for accessing internal objects
3. [PySCF](https://github.com/sunqm/pyscf), an open-source Python chemistry program
4. [PyQuante](https://github.com/rpmuller/pyquante2), a pure cross-platform open-source Python chemistry program

Please refer to the [Qiskit Chemistry drivers installation instructions](https://qiskit.org/documentation/aqua/aqua_chemistry_drivers.html)
for details on how to integrate these drivers into Qiskit Chemistry.

A useful functionality integrated into Qiskit Chemistry is its ability to serialize a file in Hierarchical Data
Format 5 (HDF5) format representing all the data extracted from one of the drivers listed above when
executing an experiment.  Qiskit Chemistry can then use that data to initiate the conversion of that
data into a fermionic operator and then a qubit operator, which can then be used as an input to a quantum
algorithm.  Therefore, even without installing one of the drivers above, it is still possible to run
chemistry experiments as long as you have a Hierarchical Data Format 5 (HDF5) file that has been previously
created.  Qiskit Chemistry's built-in HDF5 driver accepts such such HDF5 files as input.  
A few sample HDF5 files for different are provided in the 
[chemistry folder](https://github.com/Qiskit/qiskit-tutorials/tree/master/qiskit/aqua/chemistry) of the
[Qiskit Tutorials](https://github.com/Qiskit/qiskit-tutorials) repository.

To install from source, follow the instructions in the [contribution guidelines](.github/CONTRIBUTING.rst).

## Creating Your First Qiskit Chemistry Programming Experiment

Now that Qiskit Chemistry is installed, it's time to begin working with it.  We are ready to try out an experiment using Qiskit Chemistry:

```python
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType

# Use PySCF, a classical computational chemistry software package, to compute the one-body and two-body integrals in
# molecular-orbital basis, necessary to form the Fermionic operator
driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735',
                    unit=UnitsType.ANGSTROM,
                    basis='sto3g')
molecule = driver.run()
num_particles = molecule.num_alpha + molecule.num_beta
num_spin_orbitals = molecule.num_orbitals * 2

# Build the qubit operator, which is the input to the VQE algorithm in Aqua
ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
map_type = 'PARITY'
qubitOp = ferOp.mapping(map_type)
qubitOp = qubitOp.two_qubit_reduced_operator(num_particles)
num_qubits = qubitOp.num_qubits

# set the backend for the quantum computation
from qiskit import Aer
backend = Aer.get_backend('statevector_simulator')

# setup a classical optimizer for VQE
from qiskit.aqua.components.optimizers import L_BFGS_B
optimizer = L_BFGS_B()

# setup the initial state for the variational form
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
init_state = HartreeFock(num_qubits, num_spin_orbitals, num_particles)

# setup the variational form for VQE
from qiskit.aqua.components.variational_forms import RYRZ
var_form = RYRZ(num_qubits, initial_state=init_state)

# setup and run VQE
from qiskit.aqua.algorithms import VQE
algorithm = VQE(qubitOp, var_form, optimizer)
result = algorithm.run(backend)
print(result['energy'])
```

The program above uses a quantum computer to calculate the ground state energy of molecular Hydrogen,
H<sub>2</sub>, where the two atoms are configured to be at a distance of 0.735 angstroms. The molecular
configuration input is generated using PySCF. First, Qiskit Chemisrtry transparently executes PySCF,
and extracts from it the one- and two-body molecular-orbital integrals; an inexpensive operation that scales
well classically and does not require the use of a quantum computer. These integrals are then used to create
a quantum fermionic-operator representation of the molecule. In this specific example, we use a parity mapping
to generate a qubit operator from the fermionic one, with a unique precision-preserving optimization that
allows for two qubits to be tapered off; a reduction in complexity that is particularly advantageous for NISQ
computers. The qubit operator is then passed as an input to the Variational Quantum Eigensolver (VQE) algorithm,
instantiated with a Limited-memory Broyden-Fletcher-Goldfarb-Shanno Bound (L-BFGS-B) classical optimizer and
the RyRz variational form. The Hartree-Fock state is utilized to initialize the variational form.
This example emphasizes the use of Qiskit Aqua and Qiskit Chemistry's programmatic interface by illustrating
the constructor calls that initialize the VQE `QuantumAlgorithm`, along with its supporting
components—consisting of the L-BFGS-B `Optimizer`, RyRz `VariationalForm`, and Hartree-Fock `InitialState`.
The Aer statevector simulator backend is passed as a parameter to the `run` method of the VQE algorithm object,
which means that the backend will be executed with default parameters.
To customize the backend, you can wrap it into a `QuantumInstance` object, and then pass that object to the
`run` method of the QuantumAlgorithm, as explained above. The `QuantumInstance` API allows you to customize
run-time properties of the backend, such as the number of shots, the maximum number of credits to use,
a dictionary with the configuration settings for the simulator, a dictionary with the initial layout of qubits
in the mapping, and the Terra `PassManager` that will handle the compilation of the circuits.
For the full set of options, please refer to the documentation of the Aqua `QuantumInstance` API.

### Qiskit Chemistry Wizard and Command-line Interfaces

Qiskit Chemistry comes with wizard and command-line tools, which may be used when conducting
chemistry simulation experiments on a quantum machine. Both can load and run an input file
specifying both the chemistry and quantum configurations of the ecperiment.
You can find several
input files to experiment with in the
[qiskit/aqua/chemistry/input_files](https://github.com/Qiskit/qiskit-tutorials/tree/master/qiskit/aqua/chemistry)
and [community/aqua/chemistry/input_files](https://github.com/Qiskit/qiskit-tutorials/tree/master/community/aqua/chemistry)
folders of the [qiskit-tutorials GitHub Repository](https://github.com/Qiskit/qiskit-tutorials).

The wizard provides an easy means to load and run an input file specifying your chemistry problem and
the configuration of the quantum experiment.  The wizard verifies that the quantum-chemistry experiment
is not misconfigured and also allows for automatically generating Python code for easily transitioning
into running Qiskit Chemistry experiments programmatically.

The pip installation creates the `qiskit_chemistry_ui` command that allows you to start the wizard.  Similarly,
the command-line tool can be launched by entering the `qiskit_chemistry_cmd` command.

You can also use Qiskit to execute your code on a **real quantum chip**.
In order to do so, you need to configure Qiskit to use the credentials in
your [IBM Q](https://quantumexperience.ng.bluemix.net) account.
Please consult the relevant instructions in the
[Qiskit Terra GitHub repository](https://github.com/Qiskit/qiskit-terra/blob/master/README.md#executing-your-code-on-a-real-quantum-chip)
for more details.  

## Contribution Guidelines

If you'd like to contribute to Qiskit, please take a look at our
[contribution guidelines](.github/CONTRIBUTING.rst). This project adheres to Qiskit's [code of conduct](.github/CODE_OF_CONDUCT.rst).
By participating, you are expected to uphold to this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-aqua/issues) for tracking requests and bugs. Please
[join the Qiskit Slack community](https://join.slack.com/t/qiskit/shared_invite/enQtNDc2NjUzMjE4Mzc0LTMwZmE0YTM4ZThiNGJmODkzN2Y2NTNlMDIwYWNjYzA2ZmM1YTRlZGQ3OGM0NjcwMjZkZGE0MTA4MGQ1ZTVmYzk)
and use the [Aqua Slack channel](https://qiskit.slack.com/messages/aqua)
for discussion and simple questions.
For questions that are more suited for a forum, we use the **Qiskit** tag in
[Stack Overflow](https://stackoverflow.com/questions/tagged/qiskit).

## Next Steps

Now you're set up and ready to check out some of the other examples from the
[qiskit/aqua/chemistry](https://github.com/Qiskit/qiskit-tutorials/tree/master/qiskit/aqua/chemistry)
and [community/aqua/chemistry](https://github.com/Qiskit/qiskit-tutorials/tree/master/community/aqua/chemistry)
folders of the [qiskit-tutorials GitHub Repository](https://github.com/Qiskit/qiskit-tutorials).

## Authors

Qiskit Chemistry was inspired, authored and brought about by the collective work of a team of researchers.
Aqua continues to grow with the help and work of [many people](./CONTRIBUTORS.rst), who contribute
to the project at different levels.

## License

This project uses the [Apache License 2.0](LICENSE.txt).

Some of the code embedded in Qiskit Chemistry to interface some of the computational chemistry
software drivers requires additional licensing:
* The [Gaussian 16 driver](qiskit/chemistry/drivers/gaussiand/README.md) contains work licensed under the
[Gaussian Open-Source Public License](qiskit/chemistry/drivers/gaussiand/gauopen/LICENSE.txt).
* The [Pyquante driver](qiskit/chemistry/drivers/pyquanted/README.md) contains work licensed under the
[modified BSD license](qiskit/chemistry/drivers/pyquanted/LICENSE.txt).```