r"""Achieving universality with the Clifford hierarchy
==========================

We all know the 'Gottesman-Knill' rule: Clifford circuits are efficient to simulate but cannot provide quantum advantage on their own. We also know we need non-Clifford gates (like the $T$ gate) to reach universality [#anynonclifford]_. But why the `$T$ gate specifically <https://pennylane.ai/compilation/clifford-t-gate-set>`__? Why not a random rotation?

It turns out there is a rigorous structure hidden beneath these gates. The Clifford hierarchy explains exactly how 'quantum' a gate is, how hard it is to correct, and why specific gates act as the 'magic' fuel for fault-tolerant computation.

In this demo we will dig deeper into the levels of gates that make up the Clifford hierarchy (the Pauli group, the Clifford group, non-Clifford sets, and more), see how they are related (and how to implement them with gate teleportation!), and what this all means for FTQC.


The trouble with universality and quantum error correction
---------------------------------

It would be nice if we were certain that applying a finite sequence of gates could lead to any arbitrary quantum state -- a property called universality [#universality]_. However, Clifford gates such as the `Hadamard <https://docs.pennylane.ai/en/stable/code/api/pennylane.Hadamard.html>`__ $H$, `Phase <https://docs.pennylane.ai/en/stable/code/api/pennylane.S.html>`__ $S$, or `CNOT <https://docs.pennylane.ai/en/stable/code/api/pennylane.CNOT.html>`__ gates and all Pauli gates :math:`\{X,Y,Z\}` are not enough because they can only achieve :math:`90^{\circ}` or :math:`180^{\circ}` rotations. 

It turns out that all you need to achieve universal quantum computing are the Clifford gates and at least one non-Clifford gate [#qecbook]_! In principle, you could select any non-Clifford gate, but a common gate set is `{Clifford}+T <https://pennylane.ai/compilation/clifford-t-gate-set>`__. 

The `T gate <https://docs.pennylane.ai/en/stable/code/api/pennylane.T.html>`__ applies a :math:`45^{\circ}` rotation about the $Z$ axis. On the surface, this doesn’t seem too special. But with the additional of a non-Clifford gate, the `Solovay-Kitaev theorem <https://en.wikipedia.org/wiki/Solovay%E2%80%93Kitaev_theorem>`__ guarantees that any state can be reached by a finite sequence of gates, and the gate sequence can be found with the Solovay-Kitaev algorithm [#SK_alg]_ or gridsynth algorithms [#gridsynth]_. So, you can finally obtain, say, a :math:`1^{\circ}` rotation about the $Z$ axis to a :math:`10^{-2}` error with a sequence of $T$, $H$, and $S$ gates. 

But to achieve fault-tolerant universal quantum computing, quantum states must be encoded with quantum error correction (QEC) codes. Many QEC codes such as the `CSS <https://pennylane.ai/qml/demos/tutorial_stabilizer_codes>`__, colour, surface, and qLDPC [link to Utkarsh's upcoming demo] codes have transversal (therefore fault-tolerant) implementations of Clifford gates. However, the `Eastin-Knill theorem <https://arxiv.org/pdf/0811.4262>`__ dictates that there can be no quantum error correction code that can implement both Clifford and non-Clifford gates transversally. Therefore, it appears that universal quantum computing is impossible to do with error correction. 

If only there was some relationship between non-Clifford gates and Clifford gates that we can exploit...

You've probably seen a similar idea before
---------------------------------

The core idea of the Clifford hierarchy lurks beneath many of the concepts you may know: relationships between different gates can be exploited to simplify computation. For example, Clifford-only quantum circuits are known to be efficiently simulateable classically, as proven by the `Gottesman-Knill theorem <https://en.wikipedia.org/wiki/Gottesman%E2%80%93Knill_theorem>`__. 

`Stabilizer tableau simulation <https://pennylane.ai/qml/demos/tutorial_clifford_circuit_simulations>`__ is one such method. If $Z$ is a stabilizer corresponding to the state :math:`|0 \rangle`, then the application of a Clifford gate such as $H$ transforms the stabilizer to become :math:`HZH^{\dagger} = X` corresponding to the new state :math:`H |0 \rangle = |+ \rangle`. For any Clifford gate, $C$, and for all Pauli gates :math:`\{X,Y,Z\}`, observe that it is always true that the transformation :math:`CPC^{\dagger}` yields a Pauli gate up to a global phase. 

In other words, Clifford gates map Pauli gates to Pauli gates under conjugation. As my colleague wrote in this `demo <https://pennylane.ai/qml/demos/tutorial_clifford_circuit_simulations>`__, one can exploit this fact to efficiently track how stabilizers evolve through a Clifford-only circuit. 

Can we extend this idea to non-Clifford gates? 

The Clifford Hierarchy
---------------------------------

It turns out that there is a structure connecting infinite classes of gates called the Clifford hierarchy [#gottesmanchuang]_. Exploiting this hierarchy can help us implement any non-Clifford gate fault-tolerantly. 


Pauli group ($C_1$)
^^^^^^^^^^^^^^

At the bottom of this hierarchy is the Pauli group, which contains the familiar Pauli gates and their tensor products :math:`C_1 = \{X, Y, Z\}^{\otimes n}`


Clifford group ($C_2$)
^^^^^^^^^^^^^^

Members of the Clifford group map Pauli gates to Pauli gates under conjugation, up to a global phase. Formally, 

.. math::

    C_2 = \{U: UPU^{\dagger} \in C_1,~ \forall P \in C_1\}.
    

Members of this group include the Hadamard gate $H$, phase gate :math:`S = \sqrt{Z}`, and the :math:`\mathrm{CX}`, :math:`\mathrm{CY}`, and :math:`\mathrm{CZ}` gates. As an example, they conjugate Paulis like so: :math:`HZH^{\dagger} = X` and :math:`SXS^{\dagger} = Y`. Notice that the entire Pauli group lives within the Clifford group (e.g., :math:`XZX^{\dagger} = -Z`), but the vernacular is that the Clifford group excludes the Pauli group i.e., :math:`C_2 \backslash C_1`. 


$C_3$ set
^^^^^^^^^^^^^^

Members of the $C_3$ map $C_2$ gates to $C_1$ gates under conjugation, up to a global phase i.e., 

.. math::

    C_3 = \{U: UPU^{\dagger} \in C_2,~ \forall P \in C_1\}.

Examples of members of this group include the :math:`T = \sqrt{S}` gate, Toffoli gate, and :math:`\mathrm{CCX}`, :math:`\mathrm{CCY}`, and :math:`\mathrm{CCZ}` gates. 


$C_k$ set
^^^^^^^^^^^^^^

More generally, the :math:`k^{\mathrm{th}}` level of the Clifford hierarchy for :math:`k\geq 2` is:

.. math::
    
    C_k = \{U: UPU^{\dagger} \in C_{k-1},~ \forall P \in C_1 \}.

The Pauli and Clifford groups constitute the foundation of infinitely nested sets of gates. Note that applying a control to the :math:`C^{(k)}X` or :math:`C^{(k)}Z` gate yields a gate in the :math:`k^{\mathrm{th}}` level, as does taking the square root of the :math:`Z^{(1/2)^{k-1}}` rotation gate [#climbdiagonal]_, [#controlledgates]_, [#climb]_. $C_k$ is non-empty because it contains at least :math:`R_Z(m \pi/2^k)` where $m$ is any integer [#qecbook]_. As the result is an uncountably infinite real number, there are infinitely many $C_k$ sets representing increasingly fine $Z$-rotations as $k$ increases. 


Achieving universal and fault-tolerant quantum computing
---------------------------------

With the Clifford hierarchy, we can fault-tolerantly implement a $C_3$ gate with only $C_2$ gates via gate teleportation [#gottesmanchuang]_. Gate teleportation builds on top of `state teleportation <https://pennylane.ai/qml/demos/tutorial_teleportation>`__ à la Alice and Bob. As shown in Figure 1, suppose we apply a gate :math:`U\in C_3` on Bob’s half of the Bell state pair on the bottom, and proceed with :math:`|\psi\rangle` teleportation as usual. Upon measuring the top two qubits, the bottom qubit becomes :math:`UP|\psi\rangle`, where $P$ is a uniformly random Pauli :math:`\cup ~ I` error. This can be conjugated to become :math:`UPU^{\dagger}U|\psi\rangle = CU|\psi\rangle`. By the Clifford hierarchy, $C$ must be a Clifford gate. As discussed above, many QEC codes can implement Clifford gates fault-tolerantly. Thus, with the knowledge of $P$ from the Bell state measurement, :math:`C^{\dagger}` can be applied to produce :math:`U|\psi\rangle`, the desired non-Clifford gate. This procedure, known as magic state injection, generalises to the n-qubit case. 

.. figure:: ../_static/demonstration_assets/universality_and_clifford_hierarchy/Figure-1-universal-gate-teleportation.png
  :alt: Universal gate teleportation circuit.
  :width: 95%
  :align: center

  Figure 1: *A universal gate teleportation circuit applies a third level gate to the state using only gates in the second level and measurements, given a magic state (left of the dashed line).*

The challenge of implementing the $C_3$ gate, $U$, fault-tolerantly has been shifted to fault-tolerantly preparing the *magic state* :math:`(I \otimes U)(|00\rangle+|11\rangle)/\sqrt{2}`. Magic states for $T$ gates are discussed further `here <https://pennylane.ai/qml/demos/tutorial_magic_states>`__ and their fault-tolerant preparation via magic state distillation is discussed `here <https://pennylane.ai/qml/demos/tutorial_magic_state_distillation>`__. The remainder of the circuit consists of Clifford ($C_2$) gates and Bell basis measurements, which have fault-tolerant implementations in common QEC codes. Therefore, we can avoid the Easton-Knill theorem to fault-tolerantly implement both Clifford and non-Clifford gates! 

What is more, this teleportation circuit provides a systematic method to teleport any $C_k$ gate, so long as you have access to :math:`C_{k-1}` gates. If :math:`C_{k-1}` gates are not fault-tolerantly implemented in your QEC code of choice, then you may use additional teleportation circuits that produce lower level gates until you reach the fault-tolerant gate set. Figure 2 below shows an example of nested teleportation circuits to implement a $C_4$ gate for a QEC code that transversally implements Clifford gates. The first Bell basis measurement applies $C_4$ with some Pauli error $P_4$. Conjugation implies we must apply a :math:`C_3^{\dagger} = (C_4 P_4 C_4^{\dagger})^{\dagger} \in C_3`. For a QEC code that does not implement this type of gate transversally, we use another teleportation circuit. That teleportation circuit induces a Pauli error $P_3$ that must be corrected in the manner described above. It can be confirmed through computation that the final result is :math:`C_4 |\psi\rangle`. Higher level gates may be implemented in a recursive manner. 


.. figure:: ../_static/demonstration_assets/universality_and_clifford_hierarchy/Figure-2-universal-teleportation-c4-gate.png
  :alt: Recursive universal teleportation circuit to apply a C_4 gate.
  :width: 95%
  :align: center

  Figure 2: *A recursive universal gate teleportation circuit that applies a fourth level gate using a nested teleportation gate that implements a third level gate using only gates in the second level and measurements.*

Teleportation is more efficient with semi-Clifford gates
---------------------------------

Teleportation resource cost can be reduced if the gate is semi-Clifford i.e., it can be written as $U = G_b V G_a$, where $V$ is a diagonal matrix in $C_k$ and $G_a$ and $G_b$ are each Clifford gates [#onebit]_. All one- and two-qubit gates in either $C_1$ or $C_2$ are semi-Clifford [#semiclifford]_. The $T$ gate is diagonal, which is a subset of semi-Clifford gates with $G_b = G_a = I$. Figure 3a depicts the general one-bit Z-teleportation circuit for $U$, Figure 3b depicts the general one-bit X-teleportation circuit for $U$, and Figure 3c depicts the one-bit teleportation circuit for the $T$ gate. The section within the dashed box is a `magic state <https://pennylane.ai/qml/demos/tutorial_magic_states>`__. 



.. figure:: ../_static/demonstration_assets/universality_and_clifford_hierarchy/Figure-3-one-bit-teleportation.png
  :alt: One-bit teleportation circuits.
  :width: 95%
  :align: center

  Figure 3: *One-bit teleportation circuits. (a) Z-teleportation, (b) X-teleportation, and (c) $T$ gate teleportation.*


Here, conjugating the $U$ gate across the :math:`D = \{Z,X\}` Pauli error creates the term :math:`UDU^{\dagger}`, which must be Clifford if $U$ is in $C_3$ and $D$ is Pauli as per the Clifford hierarchy. Therefore, if the magic state is available, then a $T$ gate can be implemented fault-tolerantly. 

The one-bit teleportation protocol halves the number of ancilla qubits, measurements, and gates compared to the general two-bit teleportation protocol above. Note that the diagonal $V$ in $U = G_b V G_a$ need not commute with the CNOT gates because you are always free to select $X$-teleportation. All diagonal gates commute with the control part of a CNOT gate. For this reason, this circuit can implement a controlled-Hadamard gate, which does not commute with CNOT [#onebit]_. 

Recursive application of this one-bit teleportation circuit leads to the implementation of semi-Clifford $C_k$ gates. Figure 4 illustrates an example of X-teleportation of a semi-Clifford $C_4$ gate. 


.. figure:: ../_static/demonstration_assets/universality_and_clifford_hierarchy/Figure-4-one-bit-teleportation-c4-gate.png
  :alt: Recursive one-bit X-teleportation circuit for applying a C_4 gate.
  :width: 95%
  :align: center

  Figure 4: *Recursive X-teleportation of a fourth level level gate using a nested X-teleportation circuit that implements a third level gate.*


So, what's so special about the T gate?
---------------------------------
Adding any non-Clifford gate to a set of Clifford gates provides universality. The $T$ gate often appears as the non-Clifford gate of choice, but it’s just a :math:`45^{\circ}` rotation about the $Z$ axis. What’s so special about the $T$ gate? Why not a gate that implements a :math:`1^{\circ}` rotation? 

Gates above $C_3$ in the Clifford hierarchy are eliminated because they require more resources for the nested teleportation gates to implement. A diagonal $C_3$ gate enables more efficient teleportation, which narrows the choices down to the $T$ gate. 

Conclusion
---------------------------------
We have seen how the Clifford hierarchy enables universal and fault-tolerant quantum computing by mapping higher level gates down to lower level gates. A similar idea can be employed in `Pauli frame tracking <https://pennylane.ai/compilation/pauli-frame-tracking>`__ to avoid having to physically execute correction Pauli gates [#pauliframetracking]_. There are other interesting mathematical quirks of the Clifford hierarchy that are related to phase polynomials. The precise nature of the hierarchy and which gates lie in the hierarchy are still being explored. What we do know is that the diagonal $C-U$ gates that perform period finding for Shor’s algorithm & in quantum phase estimation (QPE) can be implemented using the teleportation circuits here. 


#############################################################################
References
---------------------------------
.. [#anynonclifford]

    G. Nebe, E.M. Rains, N.J.A. Sloane
    "The invariants of the Clifford groups"
    `arXiv:math/0001038 <https://arxiv.org/abs/math/0001038>`__, 2000.

.. [#universality]

    D. Deutsch, A. Barenco, and A. Ekert
    "Universality in Quantum Computation"
    `arXiv:quant-ph/9505018 <https://arxiv.org/abs/quant-ph/9505018>`__, 1995.

.. [#qecbook]

    D. Gottesman
    "Surviving as a Quantum Computer in a Classical World"
    `Book <https://www.cs.umd.edu/~dgottesm/QECCbook-2024.pdf>`__, 2024.
    
.. [#SK_alg]

    C.M. Dawnson and M.A. Nielsen
    "The Solovay-Kitaev Algorithm"
    `arXiv:quant-ph/0505030 <https://arxiv.org/abs/quant-ph/0505030>`__, 2005.

.. [#gridsynth]

    N.J. Ross and P. Selinger
    "Optimal ancilla-free Clifford+T approximation of z-rotations"
    `arXiv:1403.2975 <https://arxiv.org/abs/1403.2975>`__, 2016.

.. [#gottesmanchuang]

    D. Gottesman and I.L. Chuang
    "Quantum teleportation is a universal computational primitive"
    `arXiv:quant-ph/9908010 <https://arxiv.org/abs/quant-ph/9908010>`__, 1999.

.. [#onebit]

    X. Zhou, D.W. Leung, and I.L. Chuang
    "Methodology for quantum logic gate construction"
    `quant-ph/0002039 <https://arxiv.org/abs/quant-ph/0002039>`__, 2000.

.. [#semiclifford]

    B. Zeng, X. Chen, and I.L. Chuang
    "Semi-Clifford operations, structure of C_k hierarchy, and gate complexity for fault-tolerant quantum computation"
    `0712.2084v2 <https://arxiv.org/abs/0712.2084v2>`__, 2007.

.. [#diagonal]

    S.X. Cui, D. Gottesman, and A. Krishna
    "Diagonal gates in the Clifford hierarchy"
    `1608.06596v1 <https://arxiv.org/abs/1608.06596v1>`__, 2016.

.. [#pauliframetracking]

    C. Chamberland, P. Iyer, and D. Poulin
    "Fault-tolerant quantum computing in the Pauli or Clifford frame with slow error diagnostics"
    `1704.06662v2 <https://arxiv.org/pdf/1704.06662>`__, 2017.

    
.. [#climbdiagonal]

    J. Hu, Q. Liang, and R. Calderbank
    "Climbing the Diagonal Clifford Hierarchy"
    `2110.11923v2  <https://arxiv.org/pdf/2110.11923>`__, 2021.

.. [#controlledgates]

    J.T. Anderson and M. Weippert
    "Controlled Gates in the Clifford Hierarchy"
    `2410.04711v3   <https://arxiv.org/pdf/2410.04711>`__, 2025.

.. [#climb]

    L. Bastioni, S. Glandon, T. Pllaha, M. Stewart, and P. Waitkevich
    "Climbing the Clifford Hierarchy"
    `2603.12088v1 <https://arxiv.org/pdf/2603.12088>`__, 2026.

.. [#stim]
    C. Gidney
    "Stim: a fast stabilizer circuit simulator"
    `Quantum 5, 497 <https://doi.org/10.22331/q-2021-07-06-497>`__, 2021.


"""