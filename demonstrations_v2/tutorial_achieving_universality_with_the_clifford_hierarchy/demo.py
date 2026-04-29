r"""Achieving universality with the Clifford hierarchy
==========================

Maybe you have heard of the 'Gottesman-Knill' rule: :doc:`Clifford circuits are efficient to simulate <demos/tutorial_clifford_circuit_simulations>` but cannot provide quantum advantage on their own. We also know we need non-Clifford gates (like the $T$ gate) to reach universality [#anynonclifford]_. But why the `$T$ gate specifically <https://pennylane.ai/compilation/clifford-t-gate-set>`__? Why not a random rotation?

It turns out there is a rigorous structure hidden beneath these gates. The Clifford hierarchy explains exactly how 'quantum' a gate is, how hard it is to correct, and why specific gates act as the 'magic' fuel for fault-tolerant computation.

In this demo, we will dig deeper into the levels of gates that make up the Clifford hierarchy (the Pauli group, the Clifford group, non-Clifford sets, and more), see how they are related (and how to implement them with gate teleportation!), and what this all means for fault-tolerant quantum computing (FTQC).


The trouble with universality and quantum error correction
---------------------------------

It would be nice if we were certain that applying a finite sequence of gates could approximate any arbitrary quantum state --- a property called universality [#universality]_. However, Clifford gates such as the :class:`Hadamard <pennylane.Hadamard>` $H$, :class:`Phase<pennylane.S>` $S$, or :class:`CNOT<pennylane.CNOT>` gates and all Pauli gates :math:`\{X,Y,Z\}` are not enough because they can only achieve :math:`90^{\circ}` or :math:`180^{\circ}` rotations on the Bloch sphere. That means, if you had a single qubit initially in the :math:`|0\rangle` state, no matter the sequence of Clifford gates you apply, you can only ever reach 6 points on the Bloch sphere: the :doc:`stabilizer states <demos/tutorial_stabilizer_codes>`. 

It turns out that all you need to achieve universal quantum computing are the Clifford gates and at least one non-Clifford gate [#qecbook]_! In principle, you could select any non-Clifford gate, but a common gate set is `(Clifford+T) <https://pennylane.ai/compilation/clifford-t-gate-set>`__. 

The :class:`T <pennylane.T>` gate applies a :math:`45^{\circ}` rotation about the $Z$ axis. On the surface, this doesn’t seem too special --- but with the addition of a non-Clifford gate, the `Solovay-Kitaev theorem <https://en.wikipedia.org/wiki/Solovay%E2%80%93Kitaev_theorem>`__ guarantees that any state can be approximated by a finite sequence of gates. This gate sequence can be found via methods such as the Solovay-Kitaev algorithm [#SK_alg]_ or gridsynth [#gridsynth]_. Putting all these pieces together, we can now obtain, say, a :math:`1^{\circ}` rotation about the $Z$ axis to a :math:`10^{-2}` error with a sequence of $T$, $H$, and $S$ gates. 

For `noisy intermediate-scale quantum (NISQ) <https://pennylane.ai/blog/2023/06/from-nisq-to-isq>`__ computing, the story ends here. The problem arises when trying to achieve **both** universal **and** fault-tolerant quantum computing. 

Realistic quantum hardware is noisy, and so quantum data must be protected with `quantum error correction (QEC) codes <https://pennylane.ai/codebook/quantum-error-correction>`__ to achieve `fault-tolerance <https://pennylane.ai/topics/fault-tolerant-quantum-computing>`__. Note that a fault-tolerant implementation of a quantum gate can be obtained for free *if* that gate can be implemented transversally, because transversal gates limit the propagation of errors. Many QEC codes such as the :doc:`CSS <demos/tutorial_stabilizer_codes>`, colour, surface, and :doc:`qLDPC <demos/tutorial_qldpc_codes>` codes have transversal implementations of Clifford gates. 

However, the `Eastin-Knill theorem <https://arxiv.org/pdf/0811.4262>`__ dictates that there can be no quantum error correction code that can implement both Clifford and non-Clifford gates transversally. Hence, the same QEC codes as above that implement Clifford gates fault-tolerantly cannot implement non-Clifford gates fault-tolerantly. 

So, it appears that we are stuck: Either we perform fault-tolerant but non-universal quantum computing, or we perform universal but non-fault-tolerant quantum computing. Is there some way we can implement non-Clifford gates in these QEC codes fault-tolerantly and non-transversally?  


If only there was some relationship between non-Clifford gates and Clifford gates that we can exploit...

Clifford and Pauli gate relations
---------------------------------

The core idea of the Clifford hierarchy lurks beneath many of the concepts you may already know: relationships between different gates can be exploited to simplify computation. For example, Clifford-only quantum circuits are known to be efficiently simulatable classically, as proven by the `Gottesman-Knill theorem <https://en.wikipedia.org/wiki/Gottesman%E2%80%93Knill_theorem>`__. 

:doc:`Stabilizer tableau simulation <demos/tutorial_clifford_circuit_simulations>` is one such method. If $Z$ is a stabilizer corresponding to the state :math:`|0 \rangle`, then the application of a Clifford gate such as $H$ transforms the stabilizer to become :math:`HZH^{\dagger} = X` corresponding to the new state :math:`H |0 \rangle = |+ \rangle`. For any Clifford gate, $C$, and for all Pauli gates :math:`P \in \{X,Y,Z\}`, observe that it is always true that the transformation :math:`CPC^{\dagger}` yields a Pauli gate up to a global phase. 

In other words, Clifford gates map Pauli gates to Pauli gates under conjugation. As my colleague wrote in this :doc:`demo <demos/tutorial_clifford_circuit_simulations>`, one can exploit this fact to efficiently track how stabilizers evolve through a Clifford-only circuit. :doc:`Pauli propagation <demos/tutorial_classical_expval_estimation>` exploits a similar idea to accelerate the classical calculation of circuits' expectation values. `Pauli frame tracking <https://pennylane.ai/compilation/pauli-frame-tracking>`__ uses a similar idea to decouple the decoding step of quantum error correction from Clifford gate execution. 

What do non-Clifford gates map Pauli gates to? Does that mapping help us simplify computation too? 

The Clifford Hierarchy
---------------------------------

It turns out that there is a structure connecting infinite classes of gates called the Clifford hierarchy [#gottesmanchuang]_. Exploiting this hierarchy can help us implement any non-Clifford gate fault-tolerantly. 


Pauli group (:math:`\mathcal{C}_1`)
^^^^^^^^^^^^^^

At the bottom of this hierarchy is the Pauli group, which contains the familiar Pauli gates and their tensor products :math:`\mathcal{C}_1 = \{X, Y, Z\}^{\otimes n}` for :math:`n` qubits.


Clifford group (:math:`\mathcal{C}_2`)
^^^^^^^^^^^^^^

Members of the Clifford group map Pauli gates to Pauli gates under conjugation, up to a global phase. Formally, 

.. math::

    \mathcal{C}_2 = \{U: UPU^{\dagger} \in \mathcal{C}_1,~ \forall P \in \mathcal{C}_1\}.
    

Members of this group include the Hadamard gate $H$, phase gate :math:`S = \sqrt{Z}`, and the :math:`\mathrm{CX}`, :math:`\mathrm{CY}`, and :math:`\mathrm{CZ}` gates. As an example, they conjugate Paulis like so: :math:`HZH^{\dagger} = X` and :math:`SYS^{\dagger} = -X`. The global phase of :math:`\pm 1` is neglected when determining if a gate resides in a Clifford hierarchy level. Notice that the entire Pauli group lives within the Clifford group (e.g., :math:`XZX^{\dagger} = -Z`), i.e., :math:`\mathcal{C}_1 \subset \mathcal{C}_2`. 


:math:`\mathcal{C}_3` set
^^^^^^^^^^^^^^

Members of :math:`\mathcal{C}_3` map :math:`\mathcal{C}_1` gates to :math:`\mathcal{C}_2` gates under conjugation, up to a global phase i.e., 

.. math::

    \mathcal{C}_3 = \{U: UPU^{\dagger} \in \mathcal{C}_2,~ \forall P \in \mathcal{C}_1\}.

Examples of members of this group include the :math:`T = \sqrt{S}` gate, the Toffoli gate, and :math:`\mathrm{CCZ}` gate. For example, the $T$ gate conjugates Pauli gates like so: :math:`TXT^{\dagger} = e^{-i \pi/4} SX \sim SX` and :math:`TYT = -e^{-i\pi/4} XS \sim XS`. The :math:`\sim` represents the equivalence up to a global phase. To show that $XS$ is in fact Clifford, we can perform explicit computation. :math:`(XS)X(XS)^{\dagger} = -Y`, :math:`(XS)Y(XS)^{\dagger}=-X`, and :math:`(XS)Z(XS)^{\dagger}=-Z`. If we ignore the global phase of $-1$, we can see that the result of conjugation is a Pauli gate, which proves that $XS$ is in fact a Clifford gate. A similar result follows for $SX$. 


:math:`\mathcal{C}_k` set
^^^^^^^^^^^^^^

More generally, the :math:`k^{\mathrm{th}}` level of the Clifford hierarchy for :math:`k\geq 2` is:

.. math::
    
    \mathcal{C}_k = \{U: UPU^{\dagger} \in \mathcal{C}_{k-1},~ \forall P \in \mathcal{C}_1 \}.

The Pauli and Clifford groups constitute the foundation of infinitely nested sets of gates. Note that applying a control to the :math:`C^{(k-1)}X` or :math:`C^{(k-1)}Z` gate in :math:`\mathcal{C}_k` yields a gate in the :math:`k+1^{\mathrm{th}}` level, as does taking the square root of the :math:`Z^{(1/2)^{k-1}}\in \mathcal{C}_k` rotation gate [#climbdiagonal]_, [#controlledgates]_, [#climb]_. :math:`\mathcal{C}_k` is non-empty because it contains at least :math:`R_Z(m \pi/2^k)` where $m$ is any integer [#qecbook]_. Because there can be infinitesimally fine $Z$-rotations, there are infinitely many non-empty $C_k$ sets, and :math:`\mathcal{C}_1 \subset \mathcal{C}_2 \subset \dots \subset \mathcal{C}_k \subset \mathcal{C}_{k+1} \subset \dots`.


Achieving universal FTQC with the Clifford hierarchy
---------------------------------

With the Clifford hierarchy, we can fault-tolerantly implement a :math:`\mathcal{C}_3` gate with only :math:`\mathcal{C}_2` gates via gate teleportation [#gottesmanchuang]_. Gate teleportation builds on top of :doc:`state teleportation <demos/tutorial_teleportation>` à la Alice and Bob. Recalling that many QEC codes cannot implement a non-Clifford gate transversally, Alice cannot simply apply a transversal non-Clifford gate on her top qubit in :ref:`Figure 1 <fig-1-universal-gate-teleportation>` without possibly introducing irrecoverable noise. 

.. _fig-1-universal-gate-teleportation:

.. figure:: ../_static/demonstration_assets/universality_and_clifford_hierarchy/Figure-1-universal-gate-teleportation.png
  :alt: Universal gate teleportation circuit.
  :width: 95%
  :align: center

  Figure 1: *A universal gate teleportation circuit applies a third level gate to the state using only gates in the second level and measurements, given a magic state (left of the dashed line).*
  
So, as shown in :ref:`Figure 1 <fig-1-universal-gate-teleportation>`, suppose we apply a gate :math:`U\in \mathcal{C}_3` on Bob’s half of the Bell state pair on the bottom, and proceed with :math:`|\psi\rangle` teleportation as usual. We won't worry about how Bob can apply $U$ but Alice can't just yet. Suffice to say, Bob has a :doc:`magic state <demos/tutorial_magic_states>`. 🪄

Upon measuring the top two qubits, like in standard state teleportation, a Pauli error $P$ occurs on the bottom qubit. There is an equal chance of :math:`P \in \{X,Y,Z,I\}` occurring. For state teleportation, applying :math:`P^{\dagger}=P` removes that error. However, the difference here with gate teleportation is that Bob applied $U$ already. So, after measurement, the bottom qubit becomes :math:`UP|\psi\rangle`. Applying :math:`P^{\dagger}` does not remove the Pauli error. 

To see how we can overcome this, let's apply the identity :math:`I=U^{\dagger} U` to obtain :math:`UP I |\psi\rangle = UPU^{\dagger}U|\psi\rangle = CU|\psi\rangle`. Observe that this method has reversed the order of the gates so that $C$ is now exposed while $U$ is applied to the state :math:`|\psi\rangle` first. By the Clifford hierarchy, $C$ must be a Clifford gate. As discussed above, many QEC codes can implement Clifford gates fault-tolerantly. Thus, with the knowledge of $P$ from the Bell state measurement, :math:`C^{\dagger} = UPU^{\dagger} = C` can be applied to produce :math:`U|\psi\rangle`, the desired non-Clifford gate. This procedure, known as magic state injection, generalises to the n-qubit case. 

An example code for a $C_3$ teleportation circuit is given below: 
"""

import pennylane as qp
import numpy as np

dev = qp.device('default.qubit', wires=3)
initial_state = np.array([1, 1]) / np.sqrt(2) # arbitrary initial state

# Define your C_3 gate here. 
# We chose a T gate. Feel free to change it. 
def target_gate(wire):
    qp.T(wires=wire) 

@qp.qnode(dev)
def universal_teleportation(state):
    qp.StatePrep(state, wires=0) # initialise the state

    # Prepare magic state using the bottom 2 qubits
    qp.H(1)
    qp.CNOT(wires=[1, 2])
    target_gate(2) # Apply the C_3 gate
    
    # Teleport
    qp.CNOT(wires=[0, 1])
    qp.H(0)

    # Measure
    m0 = qp.measure(0) # Z error
    m1 = qp.measure(1) # X error

    # We must handle U P U^dagger corrections
    def x_corr():
        qp.adjoint(target_gate)(2)
        qp.X(2)
        target_gate(2)

    def z_corr():
        qp.adjoint(target_gate)(2)
        qp.Z(2)
        target_gate(2)

    def xz_corr():
        qp.adjoint(target_gate)(2)
        qp.X(2) 
        qp.Z(2)
        target_gate(2)

    # 3. Apply conditionally
    qp.cond(m1 & (m0 == 0), x_corr)() # If we measure |01>
    qp.cond(m0 & (m1 == 0), z_corr)() # If we measure |10>
    qp.cond(m0 & m1, xz_corr)() # If we measure  |11>

    return qp.density_matrix(wires=2)

##########################################
# Let's now check the circuit gives the expected result,
# by computing the correct answer directly from the
# density matrix:

U_mat = qp.matrix(target_gate, wire_order=[0])(0)
initial_dm = np.outer(initial_state, np.conj(initial_state))
correct_answer = U_mat @ initial_dm @ np.conj(U_mat).T
print(correct_answer)

##########################################
# Comparing this to our universal teleportation circuit,
# we can see that it matches:

result = universal_teleportation(initial_state)
print(result)
print(np.allclose(result, correct_answer))


##########################################
# Observe that :math:`(UPU^{\dagger})^{\dagger}` in practice is implemented as simply the reverse of the conjugation. 
# 
# 
# 
# The challenge of implementing the :math:`\mathcal{C}_3` gate, $U$, fault-tolerantly has been shifted to fault-tolerantly preparing the *magic state* :math:`(I \otimes U)(|00\rangle+|11\rangle)/\sqrt{2}` offline. The key idea is: We can prepare numerous, potentially noisy, candidate magic states in advance, and only allow the sufficiently clean states to be consumed to enable teleportation. Although magic states are a broader idea, the term is often used to mean :doc:`magic states for $T$ gates <demos/tutorial_magic_states>` specifically. There are a few ways to prepare magic states fault-tolerantly such as :doc:`magic state distillation <demos/tutorial_magic_state_distillation>`. The remainder of the circuit consists of Clifford (:math:`\mathcal{C}_2`) gates and Bell basis measurements, which have fault-tolerant implementations in common QEC codes. Therefore, we can fault-tolerantly implement both Clifford and non-Clifford gates despite the Eastin-Knill theorem! 
# 
# What is more, this teleportation circuit provides a systematic method to teleport any :math:`\mathcal{C}_k` gate, so long as you have access to :math:`\mathcal{C}_{k-1}` gates. If :math:`\mathcal{C}_{k-1}` gates are not fault-tolerantly implemented in your QEC code of choice, then you may use additional teleportation circuits that produce lower level gates until you reach the fault-tolerant gate set. :ref:`Figure 2 <fig-2-universal-teleportation-c4>` below shows an example of nested teleportation circuits to implement a :math:`\mathcal{C}_4` gate for a QEC code that transversally implements Clifford gates. The first Bell basis measurement applies :math:`C_4 \in \mathcal{C}_4` with some Pauli error $P_4$. Conjugation implies we must apply :math:`C_3^{\dagger} = (C_4 P_4 C_4^{\dagger})^{\dagger} = C_3 \in\mathcal{C}_3` correction gate. For a QEC code that does not implement this type of gate transversally, we use another teleportation circuit. That teleportation circuit induces a Pauli error $P_3$ that must be corrected in the manner described above. It can be confirmed through computation that the final result is :math:`C_4 |\psi\rangle`. 
# 
# 
# .. _fig-2-universal-teleportation-c4:
# 
# .. figure:: ../_static/demonstration_assets/universality_and_clifford_hierarchy/Figure-2-universal-teleportation-c4-gate.png
#   :alt: Recursive universal teleportation circuit to apply a C_4 gate.
#   :width: 95%
#   :align: center
# 
#   Figure 2: *A recursive universal gate teleportation circuit that applies a fourth level gate using a nested teleportation gate that implements a third level gate using only gates in the second level and measurements.*
# 
# An example code that implements this doubly-nested $C_4$ teleportation circuit is below: 

import pennylane as qp
import jax.numpy as jnp
from catalyst import qjit

dev = qp.device('lightning.qubit', wires=5)

def conjugate(gate_fn, pauli_fn):
    # Applies U  P  U^dagger
    def conjugated_gate(wire):
        qp.adjoint(gate_fn)(wire)
        pauli_fn(wire)
        gate_fn(wire)
    return conjugated_gate

# Helper redefinitions
def pauli_x(wire): qp.X(wire)
def pauli_z(wire): qp.Z(wire)
def pauli_xz(wire):
    qp.X(wire)
    qp.Z(wire)

# Define the C_4 gate. We chose sqrt(T) here. 
# Feel free to change it
def l4_gate(wire):
    qp.PhaseShift(jnp.pi / 8, wires=wire)

# Automatically generate C_3 corrections (C_4 P_4 C_4^\dagger)
l3_x_corr = conjugate(l4_gate, pauli_x)

# Automatically generate C_2 corrections for the nested C_3 (C_3 P_3 C_3^\dagger)
l2_z_corr = conjugate(l3_x_corr, pauli_z)
l2_x_corr = conjugate(l3_x_corr, pauli_x)
l2_xz_corr = conjugate(l3_x_corr, pauli_xz)

# Nested teleportation
@qjit
@qp.qnode(dev)
def c4_teleportation(state):
    qp.StatePrep(state, wires=0)

    # C_4 Teleportation
    qp.Hadamard(1)
    qp.CNOT(wires=[1, 2])
    l4_gate(2)
    
    qp.CNOT(wires=[0, 1])
    qp.Hadamard(0)

    m0 = qp.measure(0) 
    m1 = qp.measure(1) 

    # 2. Nested C_3 Teleportation
    def apply_nested_l3_correction():
        qp.Hadamard(3)
        qp.CNOT(wires=[3, 4])
        
        # Apply C_3 = C_4 P_4 C_4^\dagger 
        l3_x_corr(4)

        qp.CNOT(wires=[2, 3])
        qp.Hadamard(2)

        n0 = qp.measure(2) 
        n1 = qp.measure(3) 

        # Apply C_2 = C_3 P_3 C_3^\dagger
        def apply_z(): l2_z_corr(4)
        def apply_x(): l2_x_corr(4)
        def apply_xz(): l2_xz_corr(4)

        qp.cond(n0 & (n1 == 0), apply_z)()
        qp.cond(n1 & (n0 == 0), apply_x)()
        qp.cond(n0 & n1, apply_xz)()

        qp.cond(m0 == 1, pauli_z)(4)

    def skip_nested_l3_correction():
        qp.cond(m0 == 1, pauli_z)(2)
        qp.SWAP(wires=[2, 4])

    # 3. Branching Execution
    qp.cond(m1 == 1, apply_nested_l3_correction, skip_nested_l3_correction)()

    return qp.state()

##########################################
# What is the expected correct density matrix? 

initial_state = jnp.array([1, 1]) / jnp.sqrt(2) # arbitrary initial state
correct_state = qp.matrix(l4_gate, wire_order=[0])(0) @ jnp.expand_dims(initial_state, axis=1)
correct_density_matrix = jnp.outer(correct_state , jnp.conj(correct_state))
print(jnp.round(correct_density_matrix , 3))

##########################################
# Let us see if the circuit produces the same density matrix. Indeed, we can see that the results match. 
output_state = qp.math.reduce_statevector(c4_teleportation(initial_state), indices=[4])
print(jnp.round(output_state , 3))
print(np.allclose(output_state, correct_density_matrix))

##########################################
# 
# Most generally, this teleportation circuit can be applied indefinitely to apply arbitrarily high level gates using the same idea outlined in :ref:`Figure 2 <fig-2-universal-teleportation-c4>`. :ref:`Figure 3 <fig-3-recursive-universal-teleportation>` below illustrates this idea artistically. It's turtles all the way down!
# 
# .. _fig-3-recursive-universal-teleportation:
# 
# .. figure:: ../_static/demonstration_assets/universality_and_clifford_hierarchy/Figure-6-recursive-teleportation.png
#   :alt: Recursive universal teleportation circuit to apply an arbitrary high level gate.
#   :width: 95%
#   :align: center
#
#   Figure 3: *A universal teleportation circuit for applying an arbitrary high level gate*
# 
# Teleportation is more efficient with semi-Clifford gates
# ---------------------------------
# 
# While the universal teleportation circuit above can implement any non-Clifford gate in the Clifford hierarchy fault-tolerantly, it still isn't clear why the $T$ gate is commonly used to enable universality. To see that, let's be greedy: How can we teleport gates more efficiently? 
# 
# If a gate is semi-Clifford i.e., it can be written as $U = G_b V G_a$, where $V$ is a diagonal matrix in :math:`\mathcal{C}_k` and $G_a$ and $G_b$ are each Clifford gates, then the resource cost of gate teleportation can be **halved** [#onebit]_. All one- and two-qubit gates in :math:`\mathcal{C}_k` are semi-Clifford, as are three-qubit gates in :math:`\mathcal{C}_3` [#semiclifford]_. Importantly, the $T$ gate is diagonal, which is a subset of semi-Clifford gates with $G_b = G_a = I$.
# 
# To explain why semi-Clifford gates can be teleported more efficiently, we firstly depict these more efficient 'one-bit' teleportation circuits. There are two such flavours: $Z$-teleportation and $X$-teleportation, named after the classically controlled correction these circuits apply. :ref:`Figure 4a <fig-4-one-bit-teleportation>` depicts the general one-bit $Z$-teleportation circuit for $U$, :ref:`Figure 4b <fig-4-one-bit-teleportation>` depicts the general one-bit $X$-teleportation circuit for $U$, and :ref:`Figure 4c <fig-4-one-bit-teleportation>` depicts the one-bit teleportation circuit for the $T$ gate. The section within the dashed box is a :doc:`magic state <demos/tutorial_magic_states>`. 
# 
# .. _fig-4-one-bit-teleportation:
# 
# .. figure:: ../_static/demonstration_assets/universality_and_clifford_hierarchy/Figure-3-one-bit-teleportation.png
#   :alt: One-bit teleportation circuits.
#   :width: 95%
#   :align: center
# 
#   Figure 4: *One-bit teleportation circuits. (a) Z-teleportation, (b) X-teleportation, and (c) $T$ gate teleportation.*
# 
# 
# Here, conjugating the $U$ gate across the :math:`D \in \{Z,X\}` Pauli error creates the term :math:`UDU^{\dagger}`, which must be Clifford if $U$ is in :math:`\mathcal{C}_3` and $D$ is Pauli as per the Clifford hierarchy. Semi-Clifford-ness allows us to move the $U$ gate to before the CNOT gate. Just as before, the part of the circuit up to the $U$ gate is called a magic state, which can be prepared in advance. Therefore, if the magic state is available, then a semi-Clifford gate such as the $T$ gate can be implemented fault-tolerantly. 
# 
# The one-bit teleportation protocol halves the number of ancilla qubits, measurements, and gates compared to the general two-bit teleportation protocol above. Note that the diagonal $V$ in $U = G_b V G_a$ need not commute with the CNOT gates because you are always free to select $X$-teleportation. All diagonal gates commute with the control part of a CNOT gate. For this reason, this circuit can implement a controlled-Hadamard gate, which does not commute with CNOT [#onebit]_. 
# 
# Recursive application of this one-bit teleportation circuit leads to the implementation of semi-Clifford :math:`\mathcal{C}_k` gates. :ref:`Figure 5 <fig-5-one-bit-teleportation-c4>` illustrates an example of X-teleportation of a semi-Clifford :math:`C_4\in\mathcal{C}_4` gate assuming that only :math:`\mathcal{C}_2` gates are implemented transversally. This means we must use two teleportation circuits in total. 
# 
# .. _fig-5-one-bit-teleportation-c4:
# 
# .. figure:: ../_static/demonstration_assets/universality_and_clifford_hierarchy/Figure-4-one-bit-teleportation-c4-gate.png
#   :alt: Recursive one-bit X-teleportation circuit for applying a C_4 gate.
#   :width: 95%
#   :align: center
# 
#   Figure 5: *Recursive X-teleportation of a fourth level level gate using a nested X-teleportation circuit that implements a third level gate.*
# 
# Now, we have an efficient method to teleport certain non-Clifford gates! 
# 
# 
# So, what's so special about the T gate?
# ---------------------------------
# Adding any non-Clifford gate to a set of Clifford gates provides universality. The $T$ gate often appears as the non-Clifford gate of choice, but it’s just a :math:`45^{\circ}` rotation about the $Z$ axis. What’s so special about the $T$ gate? Why not a gate that implements a :math:`1^{\circ}` rotation? Or why not a Toffoli or a controlled-phase gate? 
# 
# Gates above :math:`\mathcal{C}_3` in the Clifford hierarchy are eliminated because they require more resources to implement because of the need for nested teleportation circuits, as shown in the above figures. 
# 
# Within :math:`\mathcal{C}_3`, we should restrict ourselves to semi-Clifford gates to let us use the more efficient teleportation circuits. That means we should only consider one-, two-, or three-qubit gates [#semiclifford]_, such as the $T$ gate, controlled-phase gate, controlled-Hadamard gate, and Toffoli gate. The gate that requires the fewest resources overall is the $T$ gate because it is a single-qubit diagonal gate (i.e., $G_a=G_b=I$). With these arguments, it is clear why the $T$ is often the non-Clifford gate of choice. 
# 
# One can inject a $T$ gate via the circuit presented in :ref:`Figure 4c <fig-4-one-bit-teleportation>`, or using the circuit below. Additional explanation of the circuit below can be found `in this magic states glossary entry <https://pennylane.ai/qml/glossary/what-are-magic-states>`__. 
# 
# .. figure:: ../_static/demonstration_assets/universality_and_clifford_hierarchy/Figure-5-magic-state-circuit.png
#   :alt: Standard magic state injection circuit
#   :width: 95%
#   :align: center
# 
#   Figure 6: *Magic state injection circuit for a T gate.*
# 
# 
# 
# Conclusion
# ---------------------------------
# 
# We have seen how the Clifford hierarchy enables universal and fault-tolerant quantum computing by mapping higher level gates down to lower level gates. The same hierarchy also ranks gates by the number of resources needed to implement them fault-tolerantly, thus how 'quantum' they are and how gates can be considered as magic fuel for fault-tolerance. 
# 
# Although the Clifford hierarchy was first proposed in the context of universality [#gottesmanchuang]_, its ideas lurk underneath other topics. For example, `Pauli frame tracking <https://pennylane.ai/compilation/pauli-frame-tracking>`__ conjugates Clifford gates to avoid having to physically execute correction Pauli gates [#pauliframetracking]_. 
# 
# Not only $T$ gates can be implemented fault-tolerantly; the Clifford hierarchy shows how an enormous class of gates can be implemented fault-tolerantly. For example, the diagonal $C-U$ gates that perform period finding for `Shor’s algorithm <https://pennylane.ai/codebook/shors-algorithm>`__ , the :doc:`quantum Fourier transform <demos/tutorial_qft>`, and in :doc:`quantum phase estimation (QPE) <demos/tutorial_qpe>` can be implemented fault-tolerantly using the teleportation circuits shown in the above sections. 
# 
# 
# 
# References
# ---------------------------------
# .. [#anynonclifford]
# 
#     G. Nebe, E.M. Rains, N.J.A. Sloane
#     "The invariants of the Clifford groups"
#     `arXiv:math/0001038 <https://arxiv.org/abs/math/0001038>`__, 2000.
# 
# .. [#universality]
# 
#     D. Deutsch, A. Barenco, and A. Ekert
#     "Universality in Quantum Computation"
#     `arXiv:quant-ph/9505018 <https://arxiv.org/abs/quant-ph/9505018>`__, 1995.
# 
# .. [#qecbook]
# 
#     D. Gottesman
#     "Surviving as a Quantum Computer in a Classical World"
#     `Book <https://www.cs.umd.edu/~dgottesm/QECCbook-2024.pdf>`__, 2024.
#     
# .. [#SK_alg]
# 
#     C.M. Dawnson and M.A. Nielsen
#     "The Solovay-Kitaev Algorithm"
#     `arXiv:quant-ph/0505030 <https://arxiv.org/abs/quant-ph/0505030>`__, 2005.
# 
# .. [#gridsynth]
# 
#     N.J. Ross and P. Selinger
#     "Optimal ancilla-free Clifford+T approximation of z-rotations"
#     `arXiv:1403.2975 <https://arxiv.org/abs/1403.2975>`__, 2016.
# 
# .. [#gottesmanchuang]
# 
#     D. Gottesman and I.L. Chuang
#     "Quantum teleportation is a universal computational primitive"
#     `arXiv:quant-ph/9908010 <https://arxiv.org/abs/quant-ph/9908010>`__, 1999.
# 
# .. [#onebit]
# 
#     X. Zhou, D.W. Leung, and I.L. Chuang
#     "Methodology for quantum logic gate construction"
#     `arXiv:quant-ph/0002039 <https://arxiv.org/abs/quant-ph/0002039>`__, 2000.
# 
# .. [#semiclifford]
# 
#     B. Zeng, X. Chen, and I.L. Chuang
#     "Semi-Clifford operations, structure of :math:`C_k` hierarchy, and gate complexity for fault-tolerant quantum computation"
#     `arXiv:0712.2084 <https://arxiv.org/abs/0712.2084>`__, 2008.
# 
# .. [#diagonal]
# 
#     S.X. Cui, D. Gottesman, and A. Krishna
#     "Diagonal gates in the Clifford hierarchy"
#     `arXiv:1608.06596 <https://arxiv.org/abs/1608.06596>`__, 2016.
# 
# .. [#pauliframetracking]
# 
#     C. Chamberland, P. Iyer, and D. Poulin
#     "Fault-tolerant quantum computing in the Pauli or Clifford frame with slow error diagnostics"
#     `arXiv:1704.06662 <https://arxiv.org/abs/1704.06662>`__, 2017.
# 
#     
# .. [#climbdiagonal]
# 
#     J. Hu, Q. Liang, and R. Calderbank
#     "Climbing the Diagonal Clifford Hierarchy"
#     `arXiv:2110.11923  <https://arxiv.org/abs/2110.11923>`__, 2021.
# 
# .. [#controlledgates]
# 
#     J.T. Anderson and M. Weippert
#     "Controlled Gates in the Clifford Hierarchy"
#     `arXiv:2410.04711 <https://arxiv.org/abs/2410.04711>`__, 2025.
# 
# .. [#climb]
# 
#     L. Bastioni, S. Glandon, T. Pllaha, M. Stewart, and P. Waitkevich
#     "Climbing the Clifford Hierarchy"
#     `arXiv:2603.12088 <https://arxiv.org/abs/2603.12088>`__, 2026. 
# 
# 
# 