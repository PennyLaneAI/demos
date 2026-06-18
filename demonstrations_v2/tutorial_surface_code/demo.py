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

The rotated surface code consists of alternating X (blue) and Z (orange) squares that make up a lattice.
The so-called data qubits sit on the vertices (highlighted in pink). These will be the qubits that encode 
the quantum information of the 5x5 patch that is making up a single qubit:

.. figure:: ../_static/demonstration_assets/surface_code/surface_code_syndrome.png
    :align: center
    :width: 50%
    :target: javascript:void(0)
    
We additionally have syndrome qubits (in light blue) that sit in the middle of the squares. 
These syndrome qubits are used to continuously perform measurements on the surrounding data qubits in a non-destructive way:
stabilizers.
We have :math:`X`- and :math:`Z`-stabilizers with either four or two operators on the edges.

.. figure:: ../_static/demonstration_assets/surface_code/surface_code_with_stabilizers.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

These stabilizers are measured by entangling the data qubits via :math:`\text{CNOT}` gates with the corresponding syndrome qubit and then
measuring that (see `Fig. 1` in [#surfacecode]_ for the detailed circuits). The measurement result :math:`\pm 1` of a stabilizer measurement indicates whether or not an error has occurred.
A :math:`d \times d` surface code qubit can detect up to :math:`d-1` errors, and correct up to :math:`\left\lfloor \tfrac{d-1}{2} \right\rfloor`.
When more errors occur, they may go unnoticed or corrected in the wrong way, as we will see later.

We now want to go into more detail and expand on each of these components. 
Before that, we want to stress the difference of the *rotated* surface code to the original planar surface code:

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
are faces of four or two :math:`Z` or :math:`X` observables, arranged in a checkerboard formation:

.. figure:: ../_static/demonstration_assets/surface_code/surface_code_with_stabilizers.png
    :align: center
    :width: 50%
    :target: javascript:void(0)
    
All these stabilizers commute with each other - that is one of the defining properties. 
That all stabilizers in the surface code depicted here commute can be seen from the fact that generally two unequal and non-trivial Pauli words commute iff they anticommute on an even number of sites.
Take for example the most simple case, two sites ``[0, 1]``, such that we have :math:`[X_0 X_1, Z_0 Z_1] = 0`.

The measurement outcome of such a stabilizer is binary :math:`\pm 1` and we assume that the underlying quantum state
:math:`|\psi\rangle` that is encoded by the qubit of the surface code is in the so-called code (sub)space - the joint 
:math:`+1` eigenspace of all stabilizer measurements.
That means that the action of any stabilizer :math:`S_i \in \mathcal{S}` on a state :math:`|\psi\rangle` in the code space 
is equal to the identity, :math:`S_i |\psi\rangle = + 1 |\psi\rangle`.

By continuously measuring all stabilizers we can ensure that the state is not leaving the code space. 
If, however, we do measure :math:`-1` somewhere, we know that an error has occurred. In that case, we need to perform
error correction, which we are discussing later.

Logical operators: Z and X edges
--------------------------------

You may have noticed that we have weight-2 stabilizers (arches) at the edges of our surface code patch. These are crucial for
defining logical operators on the surface code patch.

Logical operators :math:`Z_L` or :math:`X_L` need to commute with all stabilizers so they don't move our state outside the code space.
At the same time, they must not be stabilizers (or products thereof) themselves. 
On top of that, they of course need to satisfy the fundamental anti-commutation relation :math:`X_L Z_L = -Z_L X_L`.

On the rotated surface code, a logical :math:`X_L` operator is a string of measurements of data qubits that connects the two
edges with :math:`X` arches (left and right here). And vice versa for a logical :math:`Z_L` operator, as indicated below.

.. figure:: ../_static/demonstration_assets/surface_code/Z_string.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Multiplying a logical operator by a stabilizer does not change the logical state, 
so the string on the right hand side is an equivalent logical operator (recall that :math:`Z^2=\mathbb{1}`).
We can continue to deform the string to arrive at a logical :math:`Z_L` operator that goes along the right edge.

.. figure:: ../_static/demonstration_assets/surface_code/Z_edge.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

This is why the left and right edge are called :math:`Z` edges, which may be confusing because they contain :math:`X` arches.
The opposite is true for the top and bottom :math:`X` edges with :math:`Z` arches.

These arches define the logical operators, but they are also crucial from a mathematical point of view and restrict our
:math:`d \times d` patch to exactly one logical qubit. That is because the number of encoded qubits :math:`k` is given by
the difference of data qubits :math:`n` and independent stabilizers :math:`s`, so :math:`k = n - s`. 
For the :math:`5 \times 5` patch we are considering, we have :math:`n=25` data qubits and :math:`16` weight-4 stabilizers.
Together with the :math:`8` arches we get :math:`k = 25 - 16 - 8 = 1`.

The arches are also important for coverage of all possible errors. 
E.g., if a :math:`X` error occurred on the top left data qubit, only the top left :math:`Z` arch would catch it.

Quantum computation via lattice surgery
---------------------------------------

There are different variants of how to perform quantum computation with the surface code. Most modern approaches use
:doc:`lattice surgery </demos/tutorial_lattice_surgery>`.
The concept is relatively simple: To measure :math:`Z_L \otimes Z_L` between two surface code qubits, 
simply connect them via their :math:`Z` edge (lattice merging), 
perform a round of error correction, and destructively measure in between the two patches to split them again (lattice splitting).

.. figure:: ../_static/demonstration_assets/surface_code/lattice_surgery.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

This is important because most modern surface code constructions are targeting 
`Pauli based computation <https://pennylane.ai/compilation/pauli-based-computation>`__, where all
logical operations can be reduced to such Pauli product measurements.

In fact, in :doc:`the Game of Surface Codes </demos/tutorial_game_of_surface_codes>`, a popular framework for thinking about
fault tolerant quantum computers, we forget about everything but the :math:`X` and :math:`Z` edges of our qubit patches.
This results in rectangular boxes with solid (:math:`Z`) and dotted (:math:`X`) edges. 
The same :math:`Z_L \otimes Z_L` measurement from above can be portrayed as

.. figure:: ../_static/demonstration_assets/surface_code/gosc.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

This diagram simply says, we measure qubits :math:`|q_1\rangle` and :math:`|q_2\rangle` along their :math:`Z` edges via an intermediate auxiliary qubit region.


Error correction
----------------


Let us first consider what actually happens if a single :math:`X` or :math:`Z` error occurs on one of the data qubits.
Before the error, each stabilizer measurement returns :math:`+1`, confirming the underlying state is in the correct code space.
Now let us assume the central data qubit experiences a :math:`Z` error. The surrounding :math:`Z` stabilizers are unaffected by it, but
the two :math:`X` stabilizers yield a :math:`-1` measurement - a *defect*.

.. figure:: ../_static/demonstration_assets/surface_code/Z_error.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

The tricky part about error correction is that we are only ever given the information of the syndrome measurements and do not
know what *actually* has physically happened. The same error syndromes could have also occurred due to, e.g., the following error pattern.

.. figure:: ../_static/demonstration_assets/surface_code/unlikely_error.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

This scenario is, however, exponentially more unlikely. A common decoding algorithm is minimum-weight perfect matching (MWPM),
that looks for the shortest (and thus most likely) error string and corrects that. In this scenario, the (by far) most likely
error string is simply the central :math:`Z` error.

Consider the next following situation, where two different weight-2 error strings lead to the same error syndrome.

.. figure:: ../_static/demonstration_assets/surface_code/same_weight_two.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Here, both errors have the same minimum distance, so the choice for an MWPM decoder is ambiguous. Luckily, it does not matter
which error we correct, as they are logically equivalent because they are the same error up to a :math:`Z` stabilizer (on the surface between the two defects).

In the following scenario however, we will run into a real problem though.
Both error strings again lead to the same defect syndrome.

.. figure:: ../_static/demonstration_assets/surface_code/fatal_error.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Now, if the (less likely) error with three errors occurs, but we correct for the (more likely) second scenario, we overall
perform a logical :math:`Z_L` operation, and introduce an undetected error in our computation.
This is a manifestation of the fact that a distance :math:`d=5` rotated surface code qubit can only correct

.. math:: t = \left\lfloor \frac{d-1}{2} \right\rfloor = 2

errors deterministically.

Error correction is continuously performed during computation with one clock cycle corresponding to measuring all stabilizers once.
Instead of actually performing the corrective Pauli strings, one typically tracks them in software and multiplies them retrospectively with the final measurement results.


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
