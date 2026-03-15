r"""

Using PennyLane with PySCF and OpenFermion
==========================================

.. meta::
    :property="og:description": Learn how to integrate external quantum chemistry libraries with PennyLane.
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/thumbnail_tutorial_external_libs.png


.. related::
    tutorial_quantum_chemistry Quantum chemistry with PennyLane
    tutorial_vqe A brief overview of VQE
    tutorial_givens_rotations Givens rotations for quantum chemistry
    tutorial_adaptive_circuits Adaptive circuits for quantum chemistry


The quantum chemistry module in PennyLane, :mod:`qchem  <pennylane.qchem>`, provides built-in
methods to compute molecular integrals, solve Hartree-Fock equations, and construct
:doc:`fully-differentiable <demos/tutorial_differentiable_HF>` molecular
Hamiltonians. PennyLane also lets you take advantage of various
external resources and libraries to build upon existing tools. In this demo we will show you how
to integrate PennyLane with `PySCF <https://github.com/sunqm/pyscf>`_ and
`OpenFermion <https://github.com/quantumlib/OpenFermion>`_ to compute molecular integrals,
construct molecular Hamiltonians, and import initial states.

Building molecular Hamiltonians
-------------------------------
In PennyLane, Hamiltonians for quantum chemistry are built with the
:func:`~.pennylane.qchem.molecular_hamiltonian` function by specifying a backend for solving the
Hartree–Fock equations. The default backend is the differentiable Hartree–Fock solver of the
:mod:`qchem <pennylane.qchem>` module. A molecular Hamiltonian can also be constructed with
non-differentiable backends that use the electronic structure package
`PySCF <https://github.com/sunqm/pyscf>`_ or the
`OpenFermion-PySCF <https://github.com/quantumlib/OpenFermion-PySCF>`_ plugin. These
backends can be selected by setting the keyword argument ``method='pyscf'`` or
``method='openfermion'`` in :func:`~.pennylane.qchem.molecular_hamiltonian`. This requires
``PySCF`` or ``OpenFermion-PySCF`` to be installed by the user depending on the desired backend:

.. code-block:: bash

   pip install pyscf                 # for method='pyscf`
   pip install openfermionpyscf      # for method='openfermion`

For example, the molecular Hamiltonian for a water molecule can be constructed with the ``pyscf``
backend as:
"""

import pennylane as qml
qml.capture.enable()
import numpy as np

symbols = ["H", "O", "H"]
geometry = np.array([[-0.0399, -0.0038, 0.0000],
                     [ 1.5780,  0.8540, 0.0000],
                     [ 2.7909, -0.5159, 0.0000]])
molecule = qml.qchem.Molecule(symbols, geometry)

H, qubits = qml.qchem.molecular_hamiltonian(molecule, method="pyscf")
print(H)

##############################################################################
# This generates a PennyLane :class:`~.pennylane.Hamiltonian` that can be used in a VQE workflow or
# converted to a
# :doc:`sparse matrix <demos/tutorial_adaptive_circuits#sparse-hamiltonians>`
# in the computational basis.
#
# Additionally, if you have built your electronic Hamiltonian independently using
# `OpenFermion <https://github.com/quantumlib/OpenFermion>`_ tools, it can
# be readily converted to a PennyLane observable using the
# :func:`~.pennylane.import_operator` function. Here is an example:

from openfermion.ops import QubitOperator

H = 0.1 * QubitOperator('X0 X1') + 0.2 * QubitOperator('Z0')
H = qml.qchem.import_operator(H)

print(f'Type: \n {type(H)} \n')
print(f'Hamiltonian: \n {H}')

##############################################################################
# Computing molecular integrals
# -----------------------------
# In order to build a
# :doc:`molecular Hamiltonian <demos/tutorial_quantum_chemistry>`, we need
# one- and two-electron integrals in the molecular orbital basis. These integrals are used to
# construct a fermionic Hamiltonian which is then mapped onto the qubit basis. These molecular
# integrals can be computed with the
# :func:`~.pennylane.qchem.electron_integrals` function of PennyLane. Alternatively, the integrals
# can be computed with the `PySCF <https://github.com/sunqm/pyscf>`_ package and used in PennyLane
# workflows such as building a
# :doc:`fermionic Hamiltonian <demos/tutorial_fermionic_operators>` or
# quantum :doc:`resource estimation <demos/tutorial_resource_estimation>`.
# Let's use water as an example.
#
# First, we define the PySCF molecule object and run a restricted Hartree-Fock
# calculation:

from pyscf import gto, ao2mo, scf

mol_pyscf = gto.M(atom = '''H -0.02111417 -0.00201087  0.;
                            O  0.83504162  0.45191733  0.;
                            H  1.47688065 -0.27300252  0.''')
rhf = scf.RHF(mol_pyscf)
energy = rhf.kernel()

##############################################################################
# We obtain the molecular integrals ``one_ao`` and ``two_ao`` in the basis of atomic orbitals
# by following the example `here <https://pyscf.org/quickstart.html#and-2-electron-integrals>`_:

one_ao = mol_pyscf.intor_symmetric('int1e_kin') + mol_pyscf.intor_symmetric('int1e_nuc')
two_ao = mol_pyscf.intor('int2e_sph')

##############################################################################
# These integrals are then mapped to the basis of molecular orbitals:

one_mo = np.einsum('pi,pq,qj->ij', rhf.mo_coeff, one_ao, rhf.mo_coeff)
two_mo = ao2mo.incore.full(two_ao, rhf.mo_coeff)

##############################################################################
# Note that the two-electron integral tensor is represented in
# `chemists' notation <http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf>`_. To use it
# in PennyLane, we need to convert it into the so-called
# *physicists' notation*:

two_mo = np.swapaxes(two_mo, 1, 3)

##############################################################################
# Let's now look at an example where these molecular integrals are used to build the fermionic
# Hamiltonian of water. To do that we also need to compute the nuclear energy contribution:

core_constant = np.array([rhf.energy_nuc()])

##############################################################################
# We now use the integrals to construct a fermionic Hamiltonian with PennyLane's powerful tools
# for creating and manipulating
# :doc:`fermionic operators <demos/tutorial_fermionic_operators>`:

H_fermionic = qml.qchem.fermionic_observable(core_constant, one_mo, two_mo)

##############################################################################
# The Hamiltonian can be mapped to the qubit basis with the :func:`~.pennylane.jordan_wigner`
# function:

H = qml.jordan_wigner(H_fermionic)

##############################################################################
# Importing initial states
# ------------------------
# Simulating molecules with quantum algorithms requires defining an initial state that should have
# non-zero overlap with the molecular ground state. A trivial choice for the initial state is the
# Hartree-Fock state which is obtained by putting the electrons in the lowest-energy molecular
# orbitals. For molecules with a complicated electronic structure, the Hartree-Fock state has
# only a small overlap with the ground state, which makes executing quantum algorithms
# inefficient.
#
# Initial states obtained from affordable post-Hartree-Fock calculations can be used to make the
# quantum workflow more performant. For instance, configuration interaction (CI) and coupled cluster
# (CC) calculations with single and double (SD) excitations can be performed using PySCF and the
# resulting wave function can be used as the initial state in the quantum algorithm. PennyLane
# provides the :func:`~.pennylane.qchem.import_state` function that takes a PySCF solver object,
# extracts the wave function and returns a state vector in the computational basis that can be used
# in a quantum circuit. Let’s look at an example.
#
# First, we run CCSD calculations for the hydrogen molecule to obtain the solver object.

from pyscf import gto, scf, cc

mol = gto.M(atom=[['H', (0, 0, 0)], ['H', (0, 0, 0.7)]])
myhf = scf.RHF(mol).run()
mycc = cc.CCSD(myhf).run()

##############################################################################
# Then, we use the :func:`~.pennylane.qchem.import_state` function to obtain the
# state vector.

state = qml.qchem.import_state(mycc)
print(state)

##############################################################################
# You can verify that this state is a superposition of the Hartree-Fock state and a doubly-excited
# state.
#
# Converting fermionic operators
# ------------------------------
# Fermionic operators are commonly used to construct observables for molecules and spin systems.
# You can easily convert between fermionic operators created with PennyLane and OpenFermion by using
# the :func:`~.pennylane.from_openfermion` and :func:`~.pennylane.to_openfermion` functions. Let's
# look at some examples. First, we create a fermionic operator with OpenFermion and convert it to a
# PennyLane fermionic operator.

from openfermion import FermionOperator
openfermion_op = 0.5 * FermionOperator('0^ 2') + FermionOperator('0 2^')
pennylane_op = qml.from_openfermion(openfermion_op)
print(pennylane_op)

##############################################################################
# The resulting operator can be used in PennyLane like any other fermionic object. We now take this
# PennyLane fermionic operator and convert it back to an OpenFermion operator.

openfermion_op = qml.to_openfermion(pennylane_op)
print(openfermion_op)

##############################################################################
# The :func:`~.pennylane.from_openfermion` and :func:`~.pennylane.to_openfermion` functions support
# converting several operator types. You can look at the function documentations for more details
# and examples.

##############################################################################
# Conclusions
# -----------
# This tutorial demonstrates how to use PennyLane with external quantum chemistry libraries such as
# `PySCF <https://github.com/sunqm/pyscf>`_ and
# `OpenFermion <https://github.com/quantumlib/OpenFermion>`_.
#
# To summarize:
#
# 1. We can construct molecular Hamiltonians in PennyLane by using a user-installed version of PySCF
#    by passing the argument ``method=pyscf`` to the :func:`~.pennylane.qchem.molecular_hamiltonian`
#    function.
# 2. We can directly use one- and two-electron integrals from PySCF, but we need to convert the
#    tensor containing the two-electron integrals from chemists' notation to physicists' notation.
# 3. We can easily convert between OpenFermion operators and PennyLane operators using the
#    :func:`~.pennylane.from_openfermion` and :func:`~.pennylane.to_openfermion` functions.
# 4. Finally, we can convert PySCF wave functions to PennyLane state vectors using the
#    :func:`~.pennylane.qchem.import_state` function.
#
