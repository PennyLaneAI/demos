r"""Quantum Defect Embedding Theory (QDET)
=========================================
Many interesting problems in quantum chemistry and materials science feature a strongly correlated
region embedded within a larger environment. Example of such systems include point defects in
materials [#Galli]_, active site of catalysts [#SJRLee]_ and surface phenomenon such as adsorption
[#Gagliardi]_. Such systems can be accurately simulated with **embedding theories**, which effectively
capture the strong electronic correlations in the active region with high accuracy, while accounting
for the environment in a more approximate manner.

In this demo, we show how to implement quantum defect embedding theory (QDET) [#Galli]_. This method
has been successfully applied to study systems such as defects in calcium oxide [#Galli]_ and to calculate
excitations of the negatively charged nitrogen-vacancy defect in diamond [#Galli2]_. QDET can be used to calculate ground states,
excited states and dynamic properties of materials. These make QDET a powerful method for affordable quantum simulation
of materials. Another important advantage
of QDET is the compatibility of the method with quantum algorithms as we explain in the following
sections.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_how_to_build_qdet_hamiltonian.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

"""

#############################################
# The main component of a QDET simulation is to construct an effective Hamiltonian that describes
# the impurity subsystem and its interaction with the environment. In second quantization, the
# effective Hamiltonian can be represented in terms of electronic creation, :math:`a^{\dagger}`, and
# annihilation , :math:`a`, operators as
#
# .. math::
#
#     H^{eff} = \sum_{ij} t_{ij}^{eff}a_i^{\dagger}a_j + \frac{1}{2}\sum_{ijkl} v_{ijkl}^{eff}a_i^{\dagger}a_{j}^{\dagger}a_ka_l,
#
# where :math:`t_{ij}^{eff}` and :math:`v_{ijkl}^{eff}` represent the effective one-body and
# two-body integrals, respectively, and the indices :math:`ijkl` span over the orbitals inside the impurity.
# This Hamiltonian describes a simplified representation of the quantum system that is more
# computationally tractable, while properly capturing the essential physics of the problem.
#
# Implementation
# --------------
# A QDET simulation typically starts by obtaining a mean field approximation of the whole system
# using efficient quantum chemistry methods such as density functional theory (DFT). These
# calculations provide a set of orbitals that are partitioned into **impurity** and **bath** orbitals.
# The effective Hamiltonian is constructed from the impurity orbitals and is subsequently solved
# by using either a high accuracy classical method or a quantum algorithm. Let's implement these
# steps for an example!
#
# Mean field calculations
# ^^^^^^^^^^^^^^^^^^^^^^^
# We implement QDET to compute the excitation energies of a negatively charged nitrogen-vacancy
# defect in diamond [#Galli2]_. We use DFT to obtain a mean field description of the whole system.
# The DFT calculations are performed with the `QUANTUM ESPRESSO <https://www.quantum-espresso.org/>`_
# package. This requires downloading pseudopotentials [#Modji]_ for each atomic species
# in the system from the QUANTUM ESPRESSO `database <https://www.quantum-espresso.org/pseudopotentials/>`_.
# To prepare our system, the necessary carbon and nitrogen pseudopotentials can be downloaded by executing
# the following commands through the terminal or command prompt:
#
# .. code-block:: bash
#
#    wget -N -q http://www.quantum-simulation.org/potentials/sg15_oncv/upf/C_ONCV_PBE-1.2.upf
#    wget -N -q http://www.quantum-simulation.org/potentials/sg15_oncv/upf/N_ONCV_PBE-1.2.upf
#
# Next, we need to create the input file for running QUANTUM ESPRESSO. This contains
# information about the system and the DFT calculations. More details on how to construct the input
# file can be found in the QUANTUM ESPRESSO `documentation <https://www.quantum-espresso.org/Doc/INPUT_PW.html>`_
# page. For the system taken here, the input file can be downloaded with
#
# .. code-block:: bash
#
#    wget -N -q https://west-code.org/doc/training/nv_diamond_63/pw.in
#
# DFT calculations can now be initiated using the ``pw.x`` executable in ``WEST``, taking ``pw.in`` as the input file
# and directing the output to ``pw.out``. This process is parallelized across 2 cores using ``mpirun``.
#
# .. code-block:: bash
#
#    mpirun -n 2 pw.x -i pw.in > pw.out
#
# Identify the impurity
# ^^^^^^^^^^^^^^^^^^^^^
# Once we have obtained the mean field description, we can identify our impurity by finding
# the states that are localized around the defect region in real space. To do that, we compute the
# localization factor :math:`L_n` for each state ``n``, defined as:
#
# .. math::
#
#     L_n = \int_{V \in \Omega} d^3 r |\Psi_n(r)|^2,
#
# where :math:`V` is the identified volume including the impurity within the supercell volume
# :math:`\Omega` and :math:`\Psi` is the wavefunction [#Galli2]_. We will use the
# `WEST <https://pubs.acs.org/doi/10.1021/ct500958p>`_ program to compute the localization factor.
# This requires the ``westpp.in`` input file, example for which is shown below. Here, we specify the
# box parameters within which the localization factor is being computed; the vectors for this box are provided in
# in atomic units as ``[x_start, x_end, y_start, y_end, z_start, z_end]``.
#
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#    westpp_control:
#      westpp_calculation: L # triggers the calculation of the localization factor
#      westpp_range:         # defines the range of states to compute the localization factor
#      - 1                   # start from the first state
#      - 176                 # use all 176 states
#      westpp_box:           # specifies the parameter of the box in atomic units for integration
#      - 6.19
#      - 10.19
#      - 6.28
#      - 10.28
#      - 6.28
#      - 10.28
#
# The calculation can now be performed by running the ``westpp.x`` executable from WEST using mpirun to
# parallelize it across two cores.
#
# .. code-block:: bash
#
#    mpirun -n 2 westpp.x -i westpp.in > westpp.out
#
# This creates the file ``westpp.json`` which contains the information we need here. Since
# computational resources required to run the calculation are large, for the purpose of this tutorial we just
# download a pre-computed file with:
#
# .. code-block:: bash
#
#    mkdir -p west.westpp.save
#    wget -N -q https://west-code.org/doc/training/nv_diamond_63/box_westpp.json -O west.westpp.save/westpp.json
#
# We can plot the computed localization factor for each of the states:
#
# .. code-block:: bash
#
#    import json
#    import numpy as np
#    import matplotlib.pyplot as plt
#
#    with open('west.westpp.save/westpp.json','r') as f:
#        data = json.load(f)
#
#    y = np.array(data['output']['L']['K000001']['local_factor'],dtype='f8')
#    x = np.array([i+1 for i in range(y.shape[0])])
#
#    plt.plot(x,y,'o')
#    plt.axhline(y=0.08,linestyle='--',color='red')
#
#    plt.xlabel('Kohn-Sham orbital index')
#    plt.ylabel('Localization factor')
#
#    plt.show()
#
#
# .. figure:: ../_static/demonstration_assets/qdet/localization.jpeg
#    :align: center
#    :width: 70%
#    :target: javascript:void(0)
#
# From this plot, it is easy to see that the orbitals can be categorized as orbitals with low and
# high localization factors. For the purpose of defining an impurity, we need highly localized
# orbitals, so we set a cutoff of :math:`0.06`, illustrated by the red dashed line, and choose the orbitals that have a localization
# factor larger than :math:`0.06`. We'll use these orbitals for the calculation of the parameters
# needed to construct the effective Hamiltonian.
#
# Electronic Integrals
# ^^^^^^^^^^^^^^^^^^^^
# The next step in QDET is to define the effective one-body and two-body integrals for the impurity.
# The effective two-body integrals :math:`v^{eff}` are computed first as matrix elements of the
# partially screened static Coulomb potential :math:`W_0^{R}`.
#
# .. math::
#
#     v_{ijkl}^{eff} = [W_0^{R}]_{ijkl},
#
# where :math:`W_0^R` results from screening the bare Coulomb potential :math:`v` with the reduced
# polarizability :math:`P_0^R = P - P_{imp}`, where :math:`P` is the system's polarizability and
# :math:`P_{imp}` is the impurity's polarizability. Since solving the effective Hamiltonian
# accounts for the exchange and correlation interactions between the active electrons, we remove
# these interactions from the Kohn-Sham Hamiltonian :math:`H^{KS}` to avoid double counting them.
#
# The one-body term :math:`t^{eff}` is obtained by subtracting from the Kohn-Sham Hamiltonian the
# double-counting term accounting for electrostatic and exchange-correlation interactions in the
# active space.
#
# .. math::
#
#     t_{ij}^{eff} = H_{ij}^{KS} - t_{ij}^{dc}.
#
# We use the WEST program to compute these parameters. WEST will first compute the
# quasiparticle energies, then the partially screened Coulomb potential, and finally
# the parameters of the effective Hamiltonian. The software offers several execution modes through the
# :code:`wfreq_calculation` keyword. We choose ``XWGQH`` to ensure the full workflow is executed, which computes
# both the quasiparticle corrections and the final parameters for the QDET effective Hamiltonian. The input file
# for such a calculation is shown below:
#
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#    wstat_control:
#      wstat_calculation: S           # starts the calculation from scratch
#      n_pdep_eigen: 512              # number of eigenpotentials; matches number of electrons
#      trev_pdep: 0.00001             # convergence threshold for eigenvalues
#
#    wfreq_control:
#      wfreq_calculation: XWGQH       # compute quasiparticle corrections and Hamiltonian params
#      macropol_calculation: C        # include long-wavelength limit for condensed systems
#      l_enable_off_diagonal: true    # calculate off-diagonal elements of G_0-W_0 self-energy
#      n_pdep_eigen_to_use: 512       # number of PDEP eigenvectors to be used
#      qp_bands: [87,122,123,126,127,128] # impurity orbitals
#      n_refreq: 300                  # number of frequencies on the real axis
#      ecut_refreq: 2.0               # cutoff for the real frequencies
#
# We can now execute the calculation with:
#
# .. code-block:: bash
#
#    mpirun -n 2 wfreq.x -i wfreq.in > wfreq.out
#
# This calculation takes some time and requires computational resources, therefore we download a
# pre-computed output file with
#
# .. code-block:: bash
#
#    mkdir -p west.wfreq.save
#    wget -N -q https://west-code.org/doc/training/nv_diamond_63/wfreq.json -O west.wfreq.save/wfreq.json
#
# This output file contains all the information we need to construct the effective Hamiltonian.
#
# Effective Hamiltonian
# ^^^^^^^^^^^^^^^^^^^^^
# We now construct the effective Hamiltonian by importing the electron integral results and using
# WEST:
#
# .. code-block:: python
#
#    from westpy.qdet import QDETResult
#
#    effective_hamiltonian = QDETResult(filename='west.wfreq.save/wfreq.json')
#
# The effective Hamiltonian can be solved using a high level method such as the full configuration
# interaction (FCI) algorithm from WEST as:
#
# .. code-block:: python
#
#    solution = effective_hamiltonian.solve()
#
# Using :code:`solve()` prints the excitation energies, spin multiplicity and relative occupation of
# the active orbitals.
#
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#    ======================================================================
#    Building effective Hamiltonian...
#    nspin: 1
#    occupations: [[2. 2. 2. 2. 1. 1.]]
#    =====================================================================
#                   diag[1RDM - 1RDM(GS)]
#       E [eV] char                    87    122    123    126   127   128
#    0  0.000   3-                 0.000  0.000  0.000  0.000 0.000 0.000
#    1  0.436   1-                -0.001 -0.009 -0.018 -0.067 0.004 0.091
#    2  0.436   1-                -0.001 -0.009 -0.018 -0.067 0.092 0.002
#    3  1.251   1-                -0.002 -0.019 -0.023 -0.067 0.054 0.057
#    4  1.939   3-                -0.003 -0.010 -0.127 -0.860 1.000 0.000
#    5  1.940   3-                -0.003 -0.010 -0.127 -0.860 0.000 1.000
#    6  2.935   1-                -0.000 -0.032 -0.043 -0.855 0.929 0.002
#    7  2.936   1-                -0.000 -0.032 -0.043 -0.855 0.002 0.929
#    8  4.661   1-                -0.006 -0.054 -0.188 -1.672 0.960 0.960
#    9  5.080   3-                -0.014 -0.698 -0.213 -0.075 1.000 0.000
#    ----------------------------------------------------------------------
#
# The solution object is a dictionary that contains information about the FCI eigenstates of the
# system, which includes various excitation energies, spin multiplicities, eigenvectors etc.
# More importantly, while FCI handles small embedded effective Hamiltonians with ease, it quickly
# hits a wall with larger impurities. This is precisely where quantum computing steps in, offering
# the scalability needed to tackle such complex systems. The first step to solving these effective
# Hamiltonians via quantum algorithms in PennyLane, is to convert them to qubit Hamiltonians.
#
# Quantum Simulation
# ^^^^^^^^^^^^^^^^^^
# We now map the effective Hamiltonian to the qubit basis. Note that the two-electron integrals obtained
# before are represented in Chemists' notation and need to be converted to a notation
# that is compatible with PennyLane. Here's how to construct the :doc:`qubit Hamiltonian <demos/tutorial_fermionic_operators>`:
#
# .. code-block:: python
#
#    from pennylane.qchem import one_particle, two_particle, observable
#    import numpy as np
#
#    effective_hamiltonian = QDETResult(filename="west.wfreq.save/wfreq.json")
#
#    one_e, two_e = effective_hamiltonian.h1e, effective_hamiltonian.eri
#
#    t = one_particle(one_e[0])
#    v = two_particle(np.swapaxes(two_e[0][0], 1, 3))
#    qubit_op = observable([t, v], mapping="jordan_wigner")
#    print("Qubit Hamiltonian: ", qubit_op)
#
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#       Qubit Hamiltonian:  (2.40331309905556+0j) * I(0) + (-0.12208093833046951+0j) * Z(0) + (-0.12208093833046951+0j) * Z(1) + (-0.003330743747901097+0j) * (Y(0) @ Z(1) @ Y(2)) +  ...
#
# We can use this Hamiltonian in a quantum algorithm such as :doc:`quantum phase estimation (QPE) <demos/tutorial_qpe>`.
# You can compare the results and verify that the computed energies from quantum algorithm
# match those that we obtained before.
#
# Conclusion
# ----------
# Quantum defect embedding theory is a novel framework for simulating strongly correlated
# quantum systems and has been successfully used for studying defects in solids. Applicability of
# QDET however is not limited to defects: it can be used for other systems where a strongly
# correlated subsystem is embedded in a weakly correlated environment. Additionally, QDET is able to
# correct the interaction double counting issue within the active space faced by a variety of
# other embedding theories. The Green's function based formulation of QDET ensures
# exact removal of double counting corrections at the `GW <https://en.wikipedia.org/wiki/GW_approximation>`__
# level of theory thus removing the
# approximation present in the initial DFT based formulation. This formulation also helps to capture
# the response properties and provides access to excited state properties. Another major advantage
# of QDET is the ease with which it can be used with quantum computers in a hybrid framework [#Baker]_.
# Given its ability to access excited states and its hybrid quantum compatibility, potential future extensions
# could involve applying QDET to the wider field of spectroscopy, such as investigating
# core-level excited states relevant to :doc:`X-ray Absorption Spectroscopy (XAS) <demos/tutorial_xas>`.
# In conclusion, QDET is a powerful embedding approach for simulating complex quantum systems.
#
# References
# ----------
#
# .. [#Galli]
#    J. Davidsson, M. Onizhuk, C. Vorwerk, G. Galli,
#    "Discovery of atomic clock-like spin defects in simple oxides from first principles",
#    `arXiv:2302.07523 <https://arxiv.org/abs/2302.07523>`__.
#
# .. [#SJRLee]
#    S. J. R. Lee, F. Ding, F. R. Manby, T. F. Miller III,
#    "Analytical Gradients for Projection-Based Wavefunction-in-DFT Embedding"
#    `arXiv:1903.05830 <https://arxiv.org/abs/1903.05830>`__.
#
# .. [#Gagliardi]
#    A. Mitra, Matthew Hermes, M. Cho, V. Agarawal, L. Gagliardi,
#    "Periodic Density Matrix Embedding for CO Adsorption on the MgO(001)Surface"
#    `J. Phys. Chem. Lett. 2022, 13, 7483 <https://pubs.acs.org/doi/10.1021/acs.jpclett.2c01915>`__.
#
# .. [#Galli2]
#    N. Sheng, C. Vorwerk, M. Govoni, G. Galli,
#    "Green's function formulation of quantum defect embedding theory",
#    `arXiv:2203.05493 <https://arxiv.org/abs/2203.05493>`__.
#
# .. [#Modji]
#    M. S. Zini, A. Delgado, *et al.*,
#    "Quantum simulation of battery materials using ionic pseudopotentials"
#    `arXiv:2302.07981 <https://arxiv.org/abs/2302.07981>`__.
#
# .. [#Baker]
#    Jack S. Baker, Pablo A. M. Casares, *et al.*,
#    "Simulating optically-active spin defects with a quantum computer"
#    `arXiv:2405.13115 <https://arxiv.org/abs/2405.13115>`__.
#
