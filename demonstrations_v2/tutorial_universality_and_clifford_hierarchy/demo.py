r"""Achieving universality with the Clifford hierarchy
==========================




We all know the 'Gottesman-Knill' rule: Clifford circuits are efficient to simulate but cannot provide quantum advantage on their own. We also know we need non-Clifford gates (like the $T$ gate) to reach universality. But why the $T$ gate specifically? Why not a random rotation?

It turns out there is a rigorous structure hidden beneath these gates. The Clifford Hierarchy explains exactly how 'quantum' a gate is, how hard it is to correct, and why specific gates act as the 'magic' fuel for fault-tolerant computation.

In this demo we will dig deeper into the levels of gates that make up the clifford hierarchy (the Pauli group, the Clifford group, T gates, and more), see how they are related (and how to implement them with gate teleportation!), and what this all means for FTQC.



Achieving universal fault-tolerant quantum computing requires the ability to implement both Clifford and non-Clifford gates. Unfortunately, the Easton-Knill theorem proves that no quantum error correction scheme, necessary for protecting quantum data from noise, allows both Clifford and non-Clifford gates to be implemented transversally.


The Easton-Knill theorem appears to make universality impossible in fault-tolerant quantum computing. However, the 



What is the problem? 



It appears like achieving the simultaneous goals of (a) universal (b) fault-tolerant quantum computing with the (c) potential for quantum advantage 


On the surface, quantum computing that is simultaneously (a) universal, (b) fault-tolerant, and (c) potentially advantageous over classical methods appears extremely challenging because achieving one goals makes achieving the other harder. We want to protect 

You may have heard statements like:
* You don't ever need to physically execute Pauli correction gates
* It's not possible to implement both Clifford and non-Clifford gates
* You must have magic states for FTQC
* It is possible 


What is the answer?

Executing a quantum program that can represent any unitary operation to acceptable precision, a property called universality, 

## Definition of the Clifford hierarchy

The Clifford hierarchy consists of infinitely nested sets of gates indexed as $k = 1, 2, \dots$ [Gottesman & Chuang]. To set things up, let's define the two lowest levels of the Clifford hierarchy. 

### The Pauli Group ($C_1$)

Members of the Pauli group, $C_1$, include the familiar Pauli gates: $\{X, Y, Z\}$. 

### The Clifford Group ($C_2$)

Members of the Clifford group, $C_2$, conjugate Pauli gates to Pauli gates, up to a global phase. 

Formally, $C_2 = \{ U: U P U^\dagger \in C_1 \forall P \in C_1 \}$

Members of this group include the Hadamard gate $H$, phase gate $S$, and the CNOT gate. As an example, they conjugate Paulis like so: $HZH^\dagger = X$ and $SXS^\dagger = Y$. Notice that the entire Pauli group lives within the Clifford group (e.g., $XZX^\dagger = -Z$), but the vernacular is that the Clifford group excludes the Pauli group i.e., $C_2 \\ C_1$. The Gottesman-Knill theorem states that a circuit composed to entirely Pauli and Clifford gates are efficiently simulateable classically, meaning such circuits do not warrant making a quantum computer to execute. In this manner, these gates aren't too 'quantum'. 

### $C_3$ set 

At this stage, it is important to mention that many quantum error correction codes, such as the CSS, colour, surface, and qLDPC codes, have fault-tolerant and transversal implementations of gates belonging in $C_1$ and $C_2$ groups, making these gates straightforward to implement in a FTQC context. 

However,

1). The Gottesman-Knill theorem states that you must have a gate outside of $C_1$ and $C_2$ to have the chance of quantum advantage.
2). The Eastin-Knill theorem states no quantum error correcting code can implement both Clifford and non-Clifford gates transversally.
3). The proofs by Nebe, Rains, and Sloane (http://arxiv.org/abs/math/0001038v2) / The Solovay-Kitaev theorem show that you must have both Clifford and non-Clifford gates in your gate set to universally perform quantum computing. 

In principle, you could select any gate that is outside of the Pauli and Clifford groups. Arbitrary rotation gates, for example, such as $R_Z(\theta)$ when $\theta\neq \{0, \pi/2, \pi\}$. 

However, we shall see how members of the next level in the hierarchy, $C_3$, can efficiently address all three concerns. Members of $C_3$ are defined to satisfy: 

$C_3 = \{U: U P U^\dagger \in C_{2} \forall P \in C_1\}$

In other words, members of $C_3$ map to members of $C_2$ under conjugation, as in the Heisenberg picture. Examples of members of this group include the $T$ gate, Toffoli gate, and CCZ gate.

### C_k set

More generally, the $k^{\mathrm{th}}$ level of the Clifford hierarchy for $k\geq 2$ is: 

$C_k = \{U: UPU^\dagger in C_{k-1} \forall P \in C_1 \}$. 

The Pauli and Clifford group constitute the foundation of an infinitely nested gates. 


## Sidestepping the Eastin-Knill theorem -- Gate teleportation!

Recall that many QEC codes naturally implement Clifford gates transversally and fault-tolerantly, but not non-Clifford gates such as $T$ gates. That's a consequence of the Eastin-Knill theorem. A non-transversal execution of a $T$ gate risks introducing irrecoverable noise to the encoded data. Transversal means that each qubit of a logical qubit interacts with itself or its counterpart in another logical qubit to prevent the spread of errors. Therefore, we need a method to safely apply a non-Clifford gate to logical qubits. 

Gottesman and Chuang showed that it is always possible to apply a gate in $C_k$ with a gate teleportation circuit using gates and measurements that are at most in $C_{k-1}$. 

Gate teleportation is an extension of state teleportation (recall Alice and Bob). To teleport a $T$ gate, see Figure 1 below. Prepared in advance is a Bell state aka EPR pair where the $T$ gate is applied to half of the pair. The other half undergoes Bell basis measurement with the input state $|\psi\rangle$. Such measurements have a uniform probability of introducing a Pauli u I gate change. 

Knowing the definition of a $C_3$ gate leads to $TPT^\dagger = C$, where $C$ is some Clifford gate that can be classically determined once the Bell basis measurement reveals $P$. Hence, the state may be written as $TP|\psi\rangle = CT|\psi\rangle$. If we apply $C^\dagger$, conditioned on the result of $P$, then the output state becomes $T|\psi\rangle$. 

Fault tolerance becomes a question of careful state preparation rather than 



## Teleportation is more efficient with semi-Clifford gates, like $T$ gates

## Interesting mathematical properties & consequences of the Clifford hierarchy

Note that higher level diagonal gates are interesting to be able to implement too. The multi-controlled Z gates that appear in Shor's period finding algorithm or in QFT, are one example. Here, the $C^kZ$ gate resides in the $k+1$ level of the Clifford hierarchy. 



"""
