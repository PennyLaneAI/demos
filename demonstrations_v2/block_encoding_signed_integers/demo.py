r"""Block encoding signed integers
==========================

Quantum algorithms based on :doc:`block encodings <demos/tutorial_lcu_blockencoding>` serve as a powerhouse for `fault-tolerant quantum computation <https://pennylane.ai/topics/fault-tolerant-quantum-computing>`__. 
Unified under the framework of the :doc:`QSVT <demos/tutorial_intro_qsvt>`, block encodings enable efficient polynomial transformations of matrices, 
allowing for fast simulation algorithms, :doc:`linear system solvers <demos/tutorial_apply_qsvt>`, and more. Herein we present a technique to block encode 
`signed integers <https://en.wikipedia.org/wiki/Signed_number_representations>`__ loaded in a quantum register. This is useful, in the simplest example, to block encode a position operator. 
But it can also be used to block encode any data loaded into a quantum register, such as a discrete approximation to a 
function of a quantum register that can be loaded with :doc:`QROM <demos/tutorial_intro_qrom>`. 

With this construction, we can block encode position or momentum operators, or block encode functions of them. 
These functions can be first loaded with QROM, or computed using :doc:`arithmetic operations <demos/tutorial_how_to_use_quantum_arithmetic_operators>`. 
This is especially useful for :doc:`chemistry simulations in first quantization <demos/tutorial_resource_estimation>`, where our system qubits encode a discrete grid, and store the positions or momenta of the nuclei and electrons. 

In said simulations, one needs an efficient implementation of the kinetic energy operator. 
Once we have a block encoding of the operation $\ket p \to p \ket p$, we can simply square these block encodings by applying them sequentially to get a block encoding of $p^2$. 

However, we can do something more clever by building a walk operator $U_p Z_\Pi$, where $Z_\Pi$ is a reflection about the block encoding subspace (see Ch. 7.1 of `Lin Lin's lecture notes <https://arxiv.org/abs/2201.08309>`__ for more details). 
If we apply $U_p Z_\Pi U_p Z_\Pi$, then we have encoded the second-order Chebyshev polynomial $T(p) = 2p^2 -\mathbb I$. 
Since the identity commutes with the Hamiltonian, it does not influence the dynamics, and given the leading factor of $2$, we can block encode the mass coefficients with a factor of ½ in our prep circuit, thus reducing the overall 1-norm by half. 
Given the 1-norm is usually very large (a bottle neck for :doc:`qubitization <demos/tutorial_qubitization` simulation), this block encoding method trick is a significant improvement with almost negligible additional circuit depth. 

This demo will show how to block encode a register of signed integers by the technique elucidated by Pocrnic et al. [#pocrnic]_. 
While the proof may be found in the `paper <https://arxiv.org/abs/2602.11272>`__, this demo details the action of each part of 
the PREP and SEL operators, provides a working circuit, and details how :doc:`PennyLane can estimate the resources <demos/re_how_to_use_pennylane_for_resource_estimation>` of this block encoding. 

Specifically, we shall show how to go from a superposition of integers $a$ in [two’s complement](https://en.wikipedia.org/wiki/Two%27s_complement) form 
e.g., $\sum_a c_a |a\rangle$, where $c_a$ is a normalisation coefficient, to a block encoding of $a$ such that $\sum_a c_a |a\rangle \rightarrow \frac{1}{2^{n-1}} \sum_a c_a a |a\rangle$, where $n$ is the number of qubits. 

Signed integers
---------------------------------

[Two’s complement](https://en.wikipedia.org/wiki/Two%27s_complement) is the most popular method for encoding signed integers on modern computers, 
with $n$ bits encoding a signed integer $a \in \{-2^{n-1}, \dots, +2^{n-1}-1 \}.$ The bitstring representation 
is $a = \bar{a}_{n-1} \dots \bar{a}_0$ where $\bar{a}_j \in \{ 0,1\}.$ Either the leading or the trailing bit 
encodes the sign (depending on the endianness): 0 means a positive number while 1 means a negative number. 
For example, $a = -3 = 101$ for $n=3$. Quantum binary encoding of $a$, then, is as simple as applying the Pauli $X$ gate 
on the appropriate qubit iff $\bar{a}_j=1$. However, our goal here is to construct a block encoding of 
the operation $|a\rangle \rightarrow a | a \rangle$ where $a$ is said integer. Note that, in practice, encoding a set of signed integers is ordinarily the goal, not just a single integer. 


Circuit structure
---------------------------------

This block encoding circuit consists of two main components: PREP and SEL. 
In turn, PREP consists of $\mathtt{amp}_n$ to prepare a resource state $|\sqrt{\mathtt{amp}_n}\rangle. 
This PREP technique had been previously described in a `paper by Su et al. <https://arxiv.org/abs/2105.12767>`__ [#Su]_, 
but there was no circuit provided for PREP to the best of our knowledge. 

PREP
^^^^^^^^^^^^^^

For this implementation, the following resource state is important for PREP. 

$|\sqrt{\mathtt{amp}_n}\rangle = \frac{1}{2^{(n-1)/2}} \Big[|0\rangle^{n-1}|1\rangle_s + \sum_{b=0}^{n-2} 2^{b/2} |b\rangle |0\rangle_s \Big]$

where $b$ denotes a one-hot encoding of integers. That is, $|0\rangle = |0\dots 1\rangle$, $|1\rangle = |0\dots 10\rangle$, and $|n-2\rangle = |10\dots 0\rangle$. The all-zero state is the only state marked by $|1\rangle_s$ which serves as a flag qubit to help us encode the negative sign later. 

The PREP operator prepares this $|\sqrt{\mathtt{amp}_n}\rangle$ state along with a non-entangled $|h\rangle = |+\rangle$ state to enable destructive interference for amplitudes equalling zero later. 


.. _fig-1-PREP-oracle:

.. figure:: ../_static/demonstration_assets/block_encoding_signed_integers/PREP-oracle.png
  :alt: PREP oracle.
  :width: 95%
  :align: center

  Figure 1: *PREP oracle is composed of $|\sqrt{\mathtt{amp}_n}\rangle$ above in the above circuit. “$\mathtt{HAD}$” refers to the Hadamard gate.*
  

For example, for $n = 3$, $|\sqrt{\mathtt{amp}_n}\rangle = \frac{1}{2}|00\rangle|1\rangle_s + \frac{1}{2} |01\rangle|0\rangle_s + \frac{1}{\sqrt{2}} |10\rangle |0\rangle_s$.


Such a resource state can be prepared by the circuit below: 

.. _fig-2-amp:

.. figure:: ../_static/demonstration_assets/block_encoding_signed_integers/block_encoding_signed_integers_amp.png
  :alt: amp resource state preparation.
  :width: 95%
  :align: center

  Figure 2: *Circuit to prepare |$\sqrt{\mathtt{amp}_n}\rangle$*
  
The initial :class:`.~pennylane.Hadamard` gate and subsequent cascade of controlled-Hadamard gates creates a superposition of some computational basis states 
$|0\dots 0\rangle, |10\dots0\rangle, |110\dots 0\rangle, \dots, |1\dots 1\rangle,$ where the $k^\text{th}$ state has $1/\sqrt{2}$ the amplitude of 
the $k-1^{\text{th}}$ state for $0\leq k \leq n-2.$

For $n=3$, this performs $|000\rangle\rightarrow \Big(\frac{1}{\sqrt{2}}|00\rangle + \frac{1}{2}(|10\rangle + |11\rangle)\Big)|0\rangle.$

This effectively represents a unary encoding. To obtain the complementary encoding, we must flip the bits with $X$ gates, as seen in :ref:`Figure 2 <fig-2-amp>`. 
For our example, we have $\Big(\frac{1}{\sqrt{2}}|11\rangle + \frac{1}{2}(|01\rangle + |00\rangle)\Big)|0\rangle.$

The $n^{\text{th}}$ qubit is used to encode the sign. To do so, we entangle the $n-1$ qubit all-zero state with the $n^{\text{th}}$ qubit 
via an open-controlled CNOT gate. For our $n=3$ example, that is $\frac{1}{\sqrt{2}}|11\rangle|0\rangle_s + \frac{1}{2}|01\rangle|0\rangle_s + \frac{1}{2}|00\rangle|1\rangle_s.$

Finally, we want to convert this unary encoding to a one-hot encoding with a rising cascade of CNOTs. 
This yields the desired state $|\sqrt{\mathtt{amp}_n}\rangle$. For $n=3$, $\frac{1}{\sqrt{2}}|10\rangle|0\rangle_s + \frac{1}{2}|01\rangle|0\rangle_s + \frac{1}{2}|00\rangle|1\rangle_s.$ 

Note that the endianness of this register has now been flipped, ie. $|001\rangle = |0\rangle$, $|1\rangle = |010\rangle$, $|2\rangle = |100\rangle$, etc. 
The all-zero state does not have a meaning in this one-hot encoding. Rather than applying swaps to restore the ordering, 
it is more gate efficient to just reinterpret the endianness in the SEL circuit, inverting the ordering of operations on $|b\rangle$. 

The code snippet below creates the $|\sqrt{\mathtt{amp}_n}\rangle$ state given $n$ qubits, the $b$ register, and the $s$ qubit. 

""" 
import pennylane as qp
import numpy as np

def ampn(n, b, s):
    # Cascade of controlled Hadamards
    qp.H(b[0])
    for index in range(1, n-1):
        qp.ctrl(qp.H, control=b[index-1], control_values=1)(wires=b[index])

    # Apply X to the top n-1 qubits
    for wire in range(n-1):
        qp.X(b[wire])

    # CNOT controlled on 0 instead of 1
    qp.ctrl(qp.X(s), control=b[n-2], control_values=0)

    # Cascade of CNOTs up
    for wire in range(n-3, -1, -1):
        qp.CNOT([b[wire], b[wire+1]])

##########################################
# The following creates the PREP operator from ```ampn```. 
# 
def prepn():
    ampn(n,b,s)
    qp.H(h)

##########################################
# For `fault-tolerant quantum computing <https://pennylane.ai/topics/fault-tolerant-quantum-computing>`__, the non-Clifford gate cost 
# is typically the most burdensome. The sole non-Clifford gates are the controlled-Hadamard gates, which may be constructed by 
# one :class:`.~pennylane.Toffoli` gate each [#pocrnic]_. Therefore, a $n$-qubit PREP circuit uses $n-2$ controlled-Hadamard gates, 
# and thus costs $n-2$ Toffolis. 
# 
# The overall block encoding circuit calls PREP and the adjoint of PREP, so $2n-4$ Toffolis are needed as a result.
# 
# The detailed proof of this operator is listed in the paper by Pocrnic et al. [#pocrnic]_.  
# 
# 
# SEL
# ^^^^^^^^^^^^^^
# 
# .. _fig-3-sel:
# 
# .. figure:: ../_static/demonstration_assets/block_encoding_signed_integers/block_encoding_signed_integers_SEL.png
#   :alt: SEL circuit.
#   :width: 95%
#   :align: center
# 
#   Figure 2: *SEL sets up the relevant interference to encode the signed integers. The Toffoli gates are applied transversally over the $n-1$ qubits in the $a$ and $b$ registers, but act on the same flag qubit (see Figure 12 in [#pocrnic]_ for an example.)*
# 
# With all bitwise amplitudes loaded in the $b$ register, SEL must allow a branch to survive if $\bar{a}_j = \bar{b}_{n-2-j} = 1$, 
# and set up destructive interference otherwise. The adjoint of PREP will square the surviving amplitudes, the sum of which block 
# encodes $a/2^{n-1}$ up to the sign kicked back by CZ.
# 
# Unsigned case 
# =====================
# 
# For ease of explanation, let's first consider the unsigned case when a is non-negative. The sign bit $\bar a_{n-1}=0$, so the initial CZ and 
# CNOTs do nothing. SEL must allow a branch to survive if $\bar{a}_j = \bar{b}_{n-2-j} = 1$, and set up destructive interference otherwise. 
# The action of SEL is as follows: 
# 
# - A Toffoli checks if $\bar{a}_j = \bar{b}_{n-2-j} = 1$, and sets the flag qubit to be $1$ if so. (See the Note below) 
# - A CCZ gate targeting the $|h\rangle=|+\rangle$ qubit is controlled on the $ctl$ qubit and open-controlled on this flag qubit. Only a branch that sets the flag qubit to be $0$ leads to the CCZ gate firing. 
# - The flag is uncomputed by another Toffoli
# 
# Note: While it may seem like we would want to control on $\bar{a}_j$ and $\bar{b}_j$, observe that the nature of PREP encodes $|b\rangle$ with the 
# opposite endianness. Rather than applying SWAP gates to correct this, it is more resource efficient to just reinterpret the endianness of 
# $|b\rangle$ such that we invert the order of the Toffoli gates on that register as written above. 
# 
# Let us consider the scenario when $\bar{a}_j = \bar{b}_{n-2-j} = 1$. We’d like to add its weight $2^j$ to the block encoding. 
# The CCZ gate does not fire, leaving $|h\rangle=|+\rangle$ untouched. When this encounters the outgoing $\langle+|$ from the 
# adjoint of PREP later on, the result is $\langle +|+ \rangle= 1$: the branch survives, contributing its weight to be added up with 
# the other surviving branches’ weights. 
# 
# The other scenario is when $\bar{a}_j = 0$. This means that the Toffoli does not fire, allowing the CCZ to convert $|h\rangle=|+\rangle$ into 
# $|-\rangle$. When this encounters the outgoing $\langle+|$ from the adjoint of PREP later on, the result is $\langle+|-\rangle = 0$: 
# the amplitude is destroyed, so correctly contributes nothing to the final amplitude. In this way, SEL may be thought of as a filter 
# that removes undesirable amplitudes instead of a selector that applies desirable amplitudes. 
# 
# Summing the surviving weights gives $\sum_j \bar a_j\, 2^j = a$, and dividing by
# the $2^{n-1}$ subnormalization yields the block element $a / 2^{n-1}$.
# 
# Signed case
# =====================
# 
# Next, consider the signed case. When the input is negative, the sign bit $\bar{a}_{n-1} = 1$, which triggers two effects. 
# 
# Firstly, the CZ between $|ctl\rangle$ and the sign qubit kicks back a $-1$, giving the block-encoded amplitude the negative sign. 
# 
# Secondly, the CNOTs controlled on the sign bit flip the lower $n-1$ qubits in $|a\rangle$. Now, $\bar{a}_j$ in this section denotes 
# the bit-flipped values. This is the first step of negation in two’s complement: performing the additive inverse. The more familiar 
# reader may notice this is equivalent to taking the one’s complement. 
# 
# Just as in the unsigned case, the action of the following Toffolis and the CCZ gate is to retain the amplitude of the branch 
# if $\bar{a}_j = \bar{b}_{n-2-j} = 1$ and delete the amplitude otherwise (see above). 
# 
# Contrary to the unsigned case, now, we must add $+1$ to complete negation in two's complement. That $+1$ comes from the all-zeros 
# branch $|0\dots0\rangle|1\rangle_s$. Ordinarily, some extra arithmetic must be done, but a clever way comes from the realisation 
# that the amplitude of the all-zeros branch is $2^0 = +1$. No Toffolis fire for this branch, irrespective of $a$, meaning that the 
# flag qubit is $|0\rangle$. That allows CCZ to apply a Z gate. We established above that this Z gate can lead to the elimination of 
# this branch’s amplitude. However, the CCCZ gate controlled on $|ctl\rangle$, $|s\rangle$ (the marker qubit in $\mathtt{amp_n}$), and 
# the sign qubit finally fires when $a$ is negative to apply another Z gate, cancelling the first Z gate from CCZ. Therefore, the 
# amplitude is correctly retained, adding $+1$ during the adjoint of PREP. 
# 
# For example, if $a=-6$, the two's complement binary encoding is $|a\rangle = |1010\rangle$. Flipping all but the sign bit gives 
# $|1101\rangle$. Ignoring the sign qubit, the state is |$101\rangle = |5\rangle$ (note that $5$ is the one’s complement). 
# The all-zeros branch adds $+1$ ($5+1=6=|-6|$) while the CZ provides the minus sign. Thus, $|a=-6\rangle = -6 |-6\rangle$, up to normalisation. 
# 
# In total, the SEL operator requires $2n+1$ Toffoli gates [#pocrnic]_
# 
# The following constructs the SEL operator as well as performs state preparation of the list of integers to be encoded: 

def sel():
    # SEL
    qp.CZ(wires=[ctl,anm1])

    ## CNOTs from the a sign qubit
    for i in range(0, n-1):
        qp.CNOT([anm1, a[i]])

    ## Toffoli from a and b to flag
    for i in range(n-1):
        qp.Toffoli([a[i], b[n-2-i], flag])

    ## CCZ from control and flag to h
    qp.ctrl(qp.Z(h),
        control=[ctl, flag],
        control_values=[1, 0])   # ctrl must be 1, flag must be 0
    ## CCCZ from ctrl, s, and anm1 to h
    qp.ctrl(qp.Z(h),
            control=[ctl, s, anm1],
            control_values=[1,1,1])
    ## Uncompute phase
    ## Toffoli from a and b to flag
    for i in range(n-1):
        qp.Toffoli([a[i], b[n-2-i], flag])
    ## CNOTs from the a sign qubit
    for i in range(0, n-1):
        qp.CNOT([anm1, a[i]])

# a is an integer in two's complement form (aka binary)
# E.g., for n = 3 qubits, a = -3 = 101
# It is encoded in little endian form
def prep_amp(a_num):
    qp.StatePrep(a_num, wires=[wire for sublist in [a, [anm1]] for wire in sublist], normalize=True)

##########################################
# With the code to create PREP and SEL, we consider an implementation with $n=3$ qubits. 
# 

n = 3
b = [f"b_{i}" for i in range(n - 1)]   # ['b_0','b_1']
anm1 = "anm1" # '$a_{n-1}$’
a = [f"a_{i}" for i in range(n - 1)]   # ['a_0','a_1’]
s = "s"
h = "h"
flag = "f"
ctl = "ctl"
dev = qp.device('default.qubit', wires= [ctl] + [s] + b + [h] + [flag] + [anm1] + a)   # define the register order on the device


@qp.qnode(dev)
def block_encoding(a_num):
    qp.X(wires=ctl) # Turn the block encoding "on"
    prep_amp(a_num)
    prepn()
    sel()
    qp.adjoint(prepn)()
    qp.X(wires=ctl) # Reset the ctl qubit
    return qp.state()

# Draw the block encoding circuit with a = -3 =[1,0,1] and +2 = [0,1,0] in equal superposition, in this case. Equal superposition is not generally necessary. 
qp.drawer.use_style('pennylane')
a_value = np.zeros(2**n)
a_value[2] = 1
a_value[5] = 1
qp.draw_mpl(block_encoding)(a_value)


##########################################
# To confirm the circuit works as expected, we calculate the correct amplitudes. We also  identify the relevant amplitude in the statevector, 
# assuming the particular wire ordering shown in the above figure and that the 
# auxiliary qubits must end as $|0\rangle$. 

correct_amplitude_101 = (1/(2**(n-1)))*(1/np.sqrt(2))*-3 
correct_amplitude_010 = (1/(2**(n-1)))*(1/np.sqrt(2))*+2 

index_101 = 6
index_010 = 1

##########################################
# Thus, we ask if the amplitudes are as expected, to which the answer is: 

## Check the correct amplitudes. 
output = block_encoding(a_value)
# Check the -3 case
print("Is the -3 amplitude correct? ", np.allclose(output[index_101], correct_amplitude_101))
# Check the +2 case
print("Is the +2 amplitude correct? ", np.allclose(output[index_010], correct_amplitude_010))

##########################################
# Resource estimation
# ---------------------------------
# 
# Below we build the resource operator for ```prep_amp(n)```.

import pennylane.estimator as qre

class PrepAmp(qre.ResourceOperator):
    """
    For a given number of qubits $n$, calculates the resources required to prepare an |amp> state.
    """

    resource_keys = {"n"}  # the parameters that determine the resources of this operator

    def __init__(self, n, wires=None):
        self.num_wires = n
        # We also usually validate the wires here to make sure they match num_wires
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information
        needed to compute the resources."""
        # the keys should match the resource keys
        return {
            "n": self.num_wires,
        }

    @classmethod
    def resource_rep(cls, n) -> qre.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`:
            the operator in a compressed representation
        """
        params = {"n": n}
        return qre.CompressedResourceOp(cls, n, params)

    @classmethod
    def resource_decomp(cls, n):
        x = qre.X.resource_rep()
        cnot = qre.CNOT.resource_rep()
        h = qre.Hadamard.resource_rep()
        ch = qre.CH.resource_rep()

        gate_cost = [
            qre.GateCount(h),
            qre.GateCount(ch, n - 2),
            qre.GateCount(x, n - 1),
            qre.GateCount(cnot, n - 1),
        ]
        return gate_cost

##########################################
# Next we build the SelAmp resource operator: 

class SelAmp(qre.ResourceOperator):
    """
    Given an amp state and an input state of size $n$, calculates the resources required to apply the select operator that
    block encodes a signed integer.
    """

    resource_keys = {"n"}  # the parameters that determine the resources of this operator

    def __init__(self, n, wires=None):
        self.n = n
        # n from amp state, n from target state, 1 ctrl, 1 plus, ignore allocated qubit
        self.num_wires = n + n + 2
        # we also usually validate the wires here to make sure they match num_wires
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information
        needed to compute the resources."""
        # the keys should match the resource keys
        return {
            "n": self.n,
        }

    @classmethod
    def resource_rep(cls, n) -> qre.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`:
            the operator in a compressed representation
        """
        params = {"n": n}
        return qre.CompressedResourceOp(cls, 2 * n + 2, params)

    @classmethod
    def resource_decomp(cls, n):
        gate_cost = []
        x = qre.X.resource_rep()
        cnot = qre.CNOT.resource_rep()
        cz = qre.CZ.resource_rep()
        ccz = qre.CCZ.resource_rep()

        tof = qre.Toffoli.resource_rep()

        l_elbow = qre.TemporaryAND.resource_rep()
        r_elbow = qre.Adjoint.resource_rep(l_elbow)

        alloc = qre.Allocate(2)
        gate_cost.append(alloc)

        # cost:
        gate_cost.append(qre.GateCount(cz, 1))
        gate_cost.append(qre.GateCount(cnot, n - 1))
        gate_cost.append(qre.GateCount(tof, 2 * (n - 1)))
        gate_cost.append(qre.GateCount(x, 2))  # conjugate zero control
        gate_cost.append(qre.GateCount(l_elbow))  # use a temp and for the triply controlled Z
        gate_cost.append(qre.GateCount(r_elbow))
        gate_cost.append(qre.GateCount(ccz, 2))
        gate_cost.append(qre.GateCount(cnot, n - 1))

        gate_cost.append(qre.Deallocate(2))
        return gate_cost

##########################################
# With these resource estimation operators, we can estimate the resource cost of PREP-SEL-PREP for $n=10$ to be: 

PREP_estimate = PrepAmp(10)
print(PREP_estimate.resource_decomp(10))

##########################################
# and for SEL, 

SEL_estimate = SelAmp(10)
print(SEL_estimate.resource_decomp(10))

##########################################
# Therefore, the total cost of PREP-SEL-PREP is $2\times$ PREP cost + SEL cost from above. 
# 
# In general, the total cost of this block encoding is $4n-3$ Toffoli gates. 
# 
# Using this method allows block encoding of kinetic energy operators via a walk operator with a shifted spectrum, 
# reducing the 1-norm by a factor of 2. See the paper by Pocrnic et al. [#pocrnic]_ for more details. 


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
# .. [#pocrnic]
# 
#     M. Pocrnic, I. Loaiza, J. M. Arrazola, N. Wiebe, and D. Motlagh
#     "Efficient Simulation of Pre-Born-Oppenheimer Dynamics on a Quantum Computer"
#     `arXiv:2602.11272 <https://arxiv.org/abs/2602.11272>`__, 2026.
# 
# .. [#linlin]
# 
#     L. Lin
#     "Lecture Notes on Quantum Algorithms for Scientific Computation"
#     `arXiv:2201.08309 <https://arxiv.org/abs/2201.08309>`__, 2021.
# 
# .. [#Su]
# 
#     Y. Su, D. W. Berry, N. Wiebe, N. Rubin, and R. Babbush
#     "Fault-Tolerant Quantum Simulations of Chemistry in First Quantization"
#     `arXiv:2105.12767 <https://arxiv.org/abs/2105.12767>`__, 2021.
# 
# 
# 