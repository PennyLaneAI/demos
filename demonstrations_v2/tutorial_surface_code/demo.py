r"""

Introducing the Surface Code
============================

abstract

Hero image:

.. figure:: _static/hero_illustrations/pennylane-demo-lattice-surgery-hero.png
    :align: center
    :width: 50%
    :target: javascript:void(0)
    
Text image:

.. figure:: ../_static/demonstration_assets/lattice_surgery/surface_code_qubit1.png
    :align: center
    :width: 25%
    :target: javascript:void(0)

Intro
-----

We are going to take a look at the `planar, two-dimensional, rotated surface code <https://errorcorrectionzoo.org/c/rotated_surface>`__. This is a modern
variant of the `Kitaev surface code <https://errorcorrectionzoo.org/c/surface>`__ [#kitaev1997]_ [#bravyi1998]_ and not to be confused
with the :doc:`toric code <demos/tutorial_toric_code>` or one of the `many other variants <https://errorcorrectionzoo.org/list/quantum_surface>`__
that generalize it.

In particular and most notably, we want to stress the difference to the original planar surface code:

.. figure:: ../_static/demonstration_assets/lattice_surgery/surface_code_qubit2.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

They are very similar, but the rotated surface code effectively halves the required qubits from 
:math:`4d^2 - 4d + 1` to :math:`2d^2-1` for a given code-distance :math:`d`.

For the purpose of this demo, we shall refer to it just as *the* surface code.

Stabilizers
-----------

The surface code is a stabilizer code, which means that quantum information is encoded in the joint :math:`+1` eigenspace
of a set of commuting observables :math:`\mathcal{S}` called *stabilizers*. These stabilizers :math:`S_i \in \mathcal{S}` 
are faces of four :math:`Z` or :math:`X` observables, arranged in a checkerboard formation:

.. figure:: ../_static/demonstration_assets/surface_code/stabilizers.png
    :align: center
    :width: 50%
    :target: javascript:void(0)
    
All these stabilizers commute with each other (that is one the defining properties). That is because generally Pauli words commute when
they overlap on an equal number of sites, and, in particular for the most simple case, :math:`[XX, ZZ] = 0`.

The measurement outcome of such a stabilizer is binary :math:`\pm 1` and we assume that the underlying quantum state
:math:`|\psi\rangle` that is encoded by the qubit of the surface code is in the logical subspace, i.e. the joint 
:math:`+1` eigenspace or the so-called code space. That means that
the action of any stabilizer :math:`S_i \in \mathcal{S}` is equal to the identity, :math:`S_i |\psi\rangle = + 1 |\psi\rangle`.

By continuously measuring all stabilizers we can ensure that the state is not leaving the code space. 
If, however, we do measure :math:`-1` somewhere, we know that an error has occured. In that case, we need to perform
error correction, which we are discussing later.

Logical operators
-----------------

Logical operators :math:`Z_L` or :math:`X_L` need to commute with all stabilizers but must not be stabilizers (or products thereof) 
themselves. On top of that, they of course need to satisfy the fundamental andi-commutation relation :math:`X_L Z_L = -Z_L X_L`.

Error correction
----------------

decoding

FTQC with the Surface Code
--------------------------

lattice surgery



"""


##############################################################################
# 
#
# Conclusion
# ----------
#
# Conclusion
# 
#
# References
# ----------
#
# Gemini references:
#
# .. [#kitaev1997]
#
#     A. Yu. Kitaev,
#     "Fault-tolerant quantum computation by anyons",
#     `arXiv:quant-ph/9707021 <https://arxiv.org/abs/quant-ph/9707021>`__, 1997
# 
# .. [#bravyi1998]
#
#     Sergey B. Bravyi, A. Yu. Kitaev,
#     "Quantum codes on a lattice with boundary",
#     `arXiv:quant-ph/9811052 <https://arxiv.org/abs/quant-ph/9811052>`__, 1998
# 
# .. [#dennis2002]
#
#     Eric Dennis, Alexei Kitaev, Andrew Landahl, John Preskill,
#     "Topological quantum memory",
#     `arXiv:quant-ph/0110143 <https://arxiv.org/abs/quant-ph/0110143>`__, 2002
# 
# .. [#fowler2012]
#
#     Austin G. Fowler, Matteo Mariantoni, John M. Martinis, Andrew N. Cleland,
#     "Surface codes: Towards practical large-scale quantum computation",
#     `arXiv:1208.0928 <https://arxiv.org/abs/1208.0928>`__, 2012
# 
# .. [#acharya2022]
#
#     Rajeev Acharya, Igor Aleiner, Richard Allen, et al. (Google Quantum AI),
#     "Suppressing quantum errors by scaling a surface code logical qubit",
#     `arXiv:2207.06431 <https://arxiv.org/abs/2207.06431>`__, 2022
# 
# .. [#acharya2024]
#
#     Rajeev Acharya, Leyla Aghababaie-Beni, Igor Aleiner, et al. (Google Quantum AI),
#     "Quantum error correction below the surface code threshold",
#     `arXiv:2408.13687 <https://arxiv.org/abs/2408.13687>`__, 2024
#
# My old references:
#
# .. [#surfacecode]
#
#     Austin G. Fowler, Matteo Mariantoni, John M. Martinis, Andrew N. Cleland,
#     "Surface codes: Towards practical large-scale quantum computation",
#     `arXiv:1208.0928 <https://arxiv.org/abs/1208.0928>`__, 2012
#
#
# .. [#braiding]
#
#     Robert Raussendorf, Jim Harrington, Kovid Goyal,
#     "Topological fault-tolerance in cluster state quantum computation",
#     `arXiv:quant-ph/0703143 <https://arxiv.org/abs/quant-ph/0703143>`__, 2007
#
#
# .. [#latticesurgery]
#
#     Dominic Horsman, Austin G. Fowler, Simon Devitt, Rodney Van Meter,
#     "Surface code quantum computing by lattice surgery",
#     `arXiv:1111.4022 <https://arxiv.org/abs/1111.4022>`__, 2011
#
#
# .. [#Fowler]
#
#     Austin G. Fowler, Craig Gidney
#     "Low overhead quantum computation using lattice surgery"
#     `arXiv:1808.06709 <https://arxiv.org/abs/1808.06709>`__, 2018.
#
#
# .. [#Litinski]
#
#     Daniel Litinski
#     "A Game of Surface Codes: Large-Scale Quantum Computing with Lattice Surgery"
#     `arXiv:1808.02892 <https://arxiv.org/abs/1808.02892v3>`__, 2018.
#
#
# .. [#Chamberland]
#
#     Christopher Chamberland, Earl T. Campbell
#     "Universal quantum computing with twist-free and temporally encoded lattice surgery",
#     `arXiv:2109.02746 <https://arxiv.org/abs/2109.02746>`__, 2021
#
#
# .. [#Litinski2]
#
#     Daniel Litinski, Felix von Oppen
#     "Lattice Surgery with a Twist: Simplifying Clifford Gates of Surface Codes",
#     `arXiv:1709.02318 <https://arxiv.org/abs/1709.02318>`__, 2017
#
#
