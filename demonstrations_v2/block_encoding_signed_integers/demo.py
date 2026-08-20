r"""Block encoding signed integers
=========================================

Quantum algorithms based on :doc:`block encodings <demos/tutorial_lcu_blockencoding>` serve as a powerhouse for `fault-tolerant quantum computation <https://pennylane.ai/topics/fault-tolerant-quantum-computing>`__. 
Under the unifying framework of the :doc:`QSVT <demos/tutorial_intro_qsvt>`, block encodings enable efficient polynomial transformations of matrices, 
allowing for fast `Hamiltonian simulation algorithms <https://pennylane.ai/topics/hamiltonian-simulation>`__, :doc:`linear system solvers <demos/tutorial_apply_qsvt>`, and more. However, finding resource efficient circuits to block encode linear operators can be a challenging task. 

Herein, we present a technique to block encode 
`signed integers <https://en.wikipedia.org/wiki/Signed_number_representations>`__ loaded in a quantum register i.e., :math:`|a\rangle \rightarrow a|a\rangle`, where :math:`a` is a signed integer stored in the register. In the simplest example, this block encodes a position or momentum operator. 
But it can also be used to block encode any data loaded into a quantum register, such as a discrete approximation to a 
function of that quantum register, which can be loaded with :doc:`QROM <demos/tutorial_intro_qrom>` or computed on the fly with :doc:`arithmetic operators <demos/tutorial_how_to_use_quantum_arithmetic_operators>`. 

Such block encodings are key for :doc:`chemistry simulations in first quantization <demos/tutorial_resource_estimation>`, where the system qubits encode a discrete grid storing the positions or momenta of the nuclei and electrons. 

In said simulations, a block encoding of the momentum operator :math:`\ket p \to p \ket p` is the building block of the kinetic energy operator; applying the block encoding twice yields a block encoding of :math:`p^2`. However, when wrapped in a walk operator the same construction also has a less obvious payoff: it provides a halved 1-norm construction yielding a doubly efficient :doc:`qubitization <demos/tutorial_qubitization>` simulation. 

Applying this walk operator twice encodes the 2nd-order Chebyshev polynomial :math:`2p^2 -\mathbb I` rather than :math:`p^2` alone, and the leading factor of 2 in that polynomial lets us encode the mass coefficients as :math:`1/4m` instead of :math:`1/2m`, thereby halving the 1-norm of the kinetic energy operator :math:`p^2/2m` at almost no additional circuit depth.
Since the 1-norm sets the cost of :doc:`qubitization <demos/tutorial_qubitization>` and is usually the bottleneck, this is a substantial saving. We return to it in the conclusion, once the circuit is in hand.

This demo will show how to block encode a register of signed integers by the technique elucidated by Pocrnic et al. [#pocrnic]_. 
While the proof may be found in the `paper <https://arxiv.org/abs/2602.11272>`__, this demo details the action of each part of 
the PREP and SEL operators, provides a working circuit, and details how :doc:`PennyLane can estimate the resources <demos/re_how_to_use_pennylane_for_resource_estimation>` of this block encoding. 

Specifically, we shall show how to go from a superposition of integers :math:`a` in `two's complement <https://en.wikipedia.org/wiki/Two%27s_complement>`__ form 
e.g., :math:`\sum_a c_a |a\rangle`, where :math:`c_a` is a normalization coefficient, to a block encoding of :math:`a` such that :math:`\sum_a c_a |a\rangle \rightarrow \frac{1}{2^{n-1}} \sum_a c_a a |a\rangle`, where :math:`n` is the number of qubits. 

Signed integers
---------------------------------

`Two's complement <https://en.wikipedia.org/wiki/Two%27s_complement>`__ is the most popular method for encoding signed integers on modern computers, 
with :math:`n` bits encoding a signed integer :math:`a \in \{-2^{n-1}, \dots, +2^{n-1}-1 \}.` The bitstring representation 
is :math:`a = \bar{a}_{n-1} \dots \bar{a}_0` where :math:`\bar{a}_j \in \{ 0,1\}.` Either the leading or the trailing bit 
encodes the sign (depending on the endianness): 0 means a positive number while 1 means a negative number. 
For example, :math:`a = -3 = 101` for :math:`n=3`. Quantum binary encoding of :math:`a`, then, is as simple as applying the Pauli X gate 
on the appropriate qubit iff :math:`\bar{a}_j=1`. However, our goal here is to construct a block encoding of 
the operation :math:`|a\rangle \rightarrow a | a \rangle` where :math:`a` is said integer. Note that, in practice, encoding a superposition of signed integers is ordinarily the goal, not just a single integer. This naturally arises in first quantization due to the anti-symmetrization condition on the wavefunction [#Su]_. 

A helpful fact about the two's complement is that negation (aka finding the additive inverse) consists of two steps: 1) Bit flip all bits and 2) add :math:`+1`. For example, if :math:`a = -3 = 101`, then bit flipping yields :math:`010` and adding :math:`+1=001` yields :math:`011=+3`. 

Circuit structure
---------------------------------

This block encoding circuit consists of the standard two oracles from :doc:`Linear Combinations of Unitaries (LCU) <demos/tutorial_lcu_blockencoding>`: PREP and SEL. 
In turn, PREP consists of :math:`\mathtt{amp}_n` to prepare a resource state :math:`|\sqrt{\mathtt{amp}_n}\rangle`. 

This PREP technique had been previously described in a `paper <https://arxiv.org/abs/2105.12767>`__ by Su et al. [#Su]_, 
but the explicit circuit constructions were not provided. In fact, the techniques introduced in the aforementioned paper largely inspired the constructions presented herein. The expert reader may recall that the discrete momentum encoding in that work is done in the sign-magnitude representation, which leads to a slight excess in the 1-norm of the kinetic energy operator. Not only is two's complement a more natural encoding for general arithmetic, it also avoids this artifact by exactly block encoding the momenta over the entire grid with no probability of failure.

PREP
^^^^^^^^^^^^^^

For this implementation, the following resource state is prepared by the PREP routine 

:math:`|\sqrt{\mathtt{amp}_n}\rangle = \frac{1}{2^{(n-1)/2}} \Big[|0\rangle^{n-1}|1\rangle_s + \sum_{b=0}^{n-2} 2^{b/2} |b\rangle |0\rangle_s \Big]`

where :math:`b` denotes a one-hot encoding of integers. That is, :math:`|0\rangle = |0\dots 1\rangle`, :math:`|1\rangle = |0\dots 10\rangle`, and :math:`|n-2\rangle = |10\dots 0\rangle`. Note that the amplitude :math:`2^{b/2}` encodes the amplitude of the corresponding integer. 

The all-zero state is the only state marked by :math:`|1\rangle_s` which serves as a flag qubit to help us encode the negative sign later because it has an amplitude of :math:`1`, up to normalization. As elucidated in the "Signed integers" section of this demo, this helps us to eventually negate a positive number. 

The PREP operator prepares this :math:`|\sqrt{\mathtt{amp}_n}\rangle` state along with a non-entangled :math:`|h\rangle = |+\rangle` state to enable destructive interference for amplitudes equalling zero later. The circuit is shown below, in :ref:`Figure 1 <fig-1-PREP-oracle>`


.. _fig-1-PREP-oracle:

.. figure:: ../_static/demonstration_assets/block_encoding_signed_integers/PREP-oracle.png
  :alt: PREP oracle.
  :width: 95%
  :align: center

  Figure 1: PREP oracle is composed of :math:`|\sqrt{\mathtt{amp}_n}\rangle` in the circuit above. :math:`\mathtt{Had}` refers to the Hadamard gate.
  

For example, for :math:`n = 3`, :math:`|\sqrt{\mathtt{amp}_n}\rangle = \frac{1}{2}|00\rangle|1\rangle_s + \frac{1}{2} |01\rangle|0\rangle_s + \frac{1}{\sqrt{2}} |10\rangle |0\rangle_s`.


Such a resource state can be prepared by the circuit below: 

.. _fig-2-amp:

.. figure:: ../_static/demonstration_assets/block_encoding_signed_integers/block_encoding_signed_integers_amp.png
  :alt: amp resource state preparation.
  :width: 95%
  :align: center

  Figure 2: Circuit to prepare :math:`|\sqrt{\mathtt{amp}_n}\rangle`
  
The initial :class:`~.pennylane.Hadamard` gate and subsequent cascade of controlled-Hadamard gates create a superposition of some computational basis states 
:math:`|0\dots 0\rangle, |10\dots0\rangle, |110\dots 0\rangle, \dots, |1\dots 1\rangle,` where the :math:`k^\text{th}` state has :math:`1/\sqrt{2}` the amplitude of 
the :math:`k-1^{\text{th}}` state for :math:`0\leq k \leq n-2.`

For :math:`n=3`, this performs :math:`|000\rangle\rightarrow \Big(\frac{1}{\sqrt{2}}|00\rangle + \frac{1}{2}(|10\rangle + |11\rangle)\Big)|0\rangle.`

This effectively represents a unary encoding. To obtain the complementary encoding, we must flip the bits with X gates, as seen in :ref:`Figure 2 <fig-2-amp>`. 
For our example, we have :math:`\Big(\frac{1}{\sqrt{2}}|11\rangle + \frac{1}{2}(|01\rangle + |00\rangle)\Big)|0\rangle.`

The :math:`n^{\text{th}}` (bottom) qubit is used as a flag to encode the sign. To do so, we entangle the :math:`n-1` qubit all-zero state with the :math:`n^{\text{th}}` qubit 
via an open-controlled CNOT gate. For our :math:`n=3` example, that is :math:`\frac{1}{\sqrt{2}}|11\rangle|0\rangle_s + \frac{1}{2}|01\rangle|0\rangle_s + \frac{1}{2}|00\rangle|1\rangle_s.`

Finally, we want to convert this unary encoding to a one-hot encoding with a rising cascade of CNOTs. 
This yields the desired state :math:`|\sqrt{\mathtt{amp}_n}\rangle`. For :math:`n=3`, :math:`\frac{1}{\sqrt{2}}|10\rangle|0\rangle_s + \frac{1}{2}|01\rangle|0\rangle_s + \frac{1}{2}|00\rangle|1\rangle_s.`

Note that the one-hot label runs in the opposite direction to the wire order: Label :math:`|k\rangle` places its single :math:`1` on wire :math:`b_{n-2-k}`. Continuing the :math:`n=3` example, :math:`|0\rangle=|01\rangle` and :math:`|1\rangle=|10\rangle`, while for :math:`n=4`, we would have :math:`|001\rangle = |0\rangle`, :math:`|1\rangle = |010\rangle`, and :math:`|2\rangle = |100\rangle`. 
The all-zero state does not have a meaning in this one-hot encoding. Rather than applying swaps to restore the ordering, 
it is more gate efficient to just reinterpret the endianness in the SEL circuit, inverting the ordering of operations on :math:`|b\rangle`. 

The code snippet below creates the :math:`|\sqrt{\mathtt{amp}_n}\rangle` state given :math:`n` qubits, the :math:`b` register, and the :math:`s` qubit. 

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
# The following creates the PREP operator from ``ampn``. 
# 
def prepn():
    ampn(n,b,s)
    qp.H(h)

##########################################
# For `fault-tolerant quantum computing <https://pennylane.ai/topics/fault-tolerant-quantum-computing>`__, the `non-Clifford gate <https://pennylane.ai/compilation/clifford-t-gate-set>`__ cost 
# is typically the most burdensome. The sole non-Clifford gates are the controlled-Hadamard gates, which may be constructed by 
# one :class:`~.pennylane.Toffoli` gate each [#pocrnic]_. Therefore, an :math:`n`-qubit PREP circuit uses :math:`n-2` controlled-Hadamard gates, 
# and thus costs :math:`n-2` Toffolis. 
# 
# The overall block encoding circuit calls PREP and the adjoint of PREP, so :math:`2n-4` Toffolis are needed as a result.
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
#   Figure 3: SEL sets up the relevant interference to encode the signed integers. The Toffoli gates are applied over the :math:`n-1` qubits in the :math:`a` and :math:`b` registers, but act on the same flag qubit (see Figure 12 in [#pocrnic]_ for an example.)
# 
# With all bitwise amplitudes loaded in the :math:`b` register, SEL must allow a branch to survive if :math:`\bar{a}_j = \bar{b}_{n-2-j} = 1`, 
# and set up destructive interference otherwise. The adjoint of PREP will square the surviving amplitudes, the sum of which block 
# encodes :math:`a/2^{n-1}` up to the sign kicked back by CZ. We first show how SEL, whose circuit may be seen in :ref:`Figure 3 <fig-3-sel>`, meets this criterion for non-negative :math:`a`, then what changes when :math:`a` is negative. 
# 
# Unsigned case 
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 
# For ease of explanation, let's first consider the unsigned case when :math:`a` is non-negative. The sign bit :math:`\bar a_{n-1}=0`, so the initial CZ and 
# CNOTs do nothing. 
# The action of SEL is as follows: 
# 
# - A Toffoli checks if :math:`\bar{a}_j = \bar{b}_{n-2-j} = 1`, and sets the flag qubit to be :math:`1` if so. (See the Note below) 
# - A CCZ gate targeting the :math:`|h\rangle=|+\rangle` qubit is controlled on the :math:`\mathtt{ctl}` qubit and open-controlled on this flag qubit. Only a branch that sets the flag qubit to be :math:`0` leads to the CCZ gate firing. 
# - The flag is uncomputed by another Toffoli
# 
# Note: While it may seem like we would want to control on :math:`\bar{a}_j` and :math:`\bar{b}_j`, observe that the nature of PREP encodes :math:`|b\rangle` with the 
# opposite endianness. Rather than applying SWAP gates to correct this, it is more resource efficient to just reinterpret the endianness of 
# :math:`|b\rangle` such that we invert the order of the Toffoli gates on that register as written above. 
# 
# Let us consider the scenario when :math:`\bar{a}_j = \bar{b}_{n-2-j} = 1`. We'd like to add its weight :math:`2^j` to the block encoding. 
# The CCZ gate does not fire, leaving :math:`|h\rangle=|+\rangle` untouched. When this encounters the outgoing :math:`\langle+|` from the 
# adjoint of PREP later on, the result is :math:`\langle +|+ \rangle= 1`: the branch survives, contributing its weight to be added up with 
# the other surviving branches' weights. 
# 
# The other scenario is when :math:`\bar{a}_j = 0`. This means that the Toffoli does not fire, allowing the CCZ to convert :math:`|h\rangle=|+\rangle` into 
# :math:`|-\rangle`. When this encounters the outgoing :math:`\langle+|` from the adjoint of PREP later on, the result is :math:`\langle+|-\rangle = 0`: 
# the amplitude is destroyed, so correctly contributes nothing to the final amplitude. In this way, SEL may be thought of as a filter 
# that removes undesirable amplitudes instead of a selector that applies desirable amplitudes. 
# 
# Summing the surviving weights gives :math:`\sum_j \bar a_j\, 2^j = a`, and dividing by
# the :math:`2^{n-1}` subnormalization yields the block element :math:`a / 2^{n-1}`.
# 
# Signed case
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Next, consider the signed case. When the input is negative, the sign bit :math:`\bar{a}_{n-1} = 1`, which triggers two effects. 
# 
# Firstly, the CZ between :math:`|\mathtt{ctl}\rangle` and the sign qubit kicks back a :math:`-1`, giving the block-encoded amplitude the negative sign. 
# 
# Secondly, the CNOTs controlled on the sign bit flip the lower :math:`n-1` qubits in :math:`|a\rangle`. Now, :math:`\bar{a}_j` in this section denotes 
# the bit-flipped values. This is the first of two steps to negate a two's complement integer: taking the one's complement of the lower :math`n-1` bits. 
# The additive inverse is only completed by the :math:`+1` that follows, which we obtain for free from the all-zeros branch below. 
# The keen reader may observe that we exploited the helpful fact about negating two's complement mentioned in the "Signed integers" section of this demo. 
# 
# Just as in the unsigned case, the action of the following Toffolis and the CCZ gate is to retain the amplitude of the branch 
# if :math:`\bar{a}_j = \bar{b}_{n-2-j} = 1` and delete the amplitude otherwise (see above). 
# 
# Contrary to the unsigned case, now, we must add :math:`+1` to complete negation in two's complement. That :math:`+1` comes from the all-zeros 
# branch :math:`|0\dots0\rangle|1\rangle_s`. Ordinarily, some extra arithmetic must be done, but a clever way comes from the realization 
# that the amplitude of the all-zeros branch is :math:`2^0 = +1`. No Toffolis fire for this branch, irrespective of :math:`a`, meaning that the 
# target ancilla qubit (initialized as :math:`\ket 0`) is :math:`|0\rangle`. That allows CCZ to apply a Z gate. We established above that this Z gate can lead to the elimination of 
# this branch's amplitude. However, the CCCZ gate controlled on :math:`|\mathtt{ctl}\rangle`, :math:`|s\rangle` (the marker qubit in :math:`\mathtt{amp_n}`), and 
# the sign qubit finally fires when :math:`a` is negative to apply another Z gate, cancelling the first Z gate from CCZ. Therefore, the 
# amplitude is correctly retained, adding :math:`+1` during the adjoint of PREP. 
# 
# For example, if :math:`a=-6`, the two's complement binary encoding is :math:`|a\rangle = |1010\rangle`. Flipping all but the sign bit gives 
# :math:`|1101\rangle`. Ignoring the sign qubit, the state is :math:`|101\rangle = |5\rangle` (note that :math:`5` is the one's complement). 
# The all-zeros branch adds :math:`+1` (:math:`5+1=6=|-6|`) while the CZ provides the minus sign. Thus, :math:`|a=-6\rangle = -6 |-6\rangle`, up to normalization. 
# 
# In total, the SEL operator requires :math:`2n+1` Toffoli gates [#pocrnic]_.
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
# With the code to create PREP and SEL, we consider an implementation with :math:`n=3` qubits. 
# 

n = 3
b = [f"b_{i}" for i in range(n - 1)]   # ['b_0','b_1']
anm1 = "anm1" # '$a_{n-1}$'
a = [f"a_{i}" for i in range(n - 1)]   # ['a_0','a_1']
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
# auxiliary qubits must end as :math:`|0\rangle`. 

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
# Below we build the resource operator for ``ampn``, the circuit :math:`\mathtt{amp_n}` that prepares the resource state :math:`|\sqrt{\mathtt{amp_n}}\rangle`: 

import pennylane.estimator as qre

class AmpN_estimator(qre.ResourceOperator):
    """
    For a given number of qubits n, calculates the resources required to prepare an |\sqrt{amp_n}> state with the amp_n circuit. 
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
    Given an amp state and an input state of size :math:`n`, calculates the resources required to apply the select operator that
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
# With these resource estimation operators, we can estimate the resource cost of PREP-SEL-PREP for :math:`n=10` to be: 

PREP_estimate = AmpN_estimator(10)
print(PREP_estimate.resource_decomp(10))

##########################################
# and for SEL, 

SEL_estimate = SelAmp(10)
print(SEL_estimate.resource_decomp(10))

##########################################
# Therefore, the total cost of PREP-SEL-PREP is :math:`2\times` PREP cost + SEL cost from above. 
# 
# In general, the total cost of this block encoding is :math:`4n-3` Toffoli gates. 
# 
# Using this method allows block encoding of kinetic energy operators via a walk operator with a shifted spectrum, 
# reducing the 1-norm by a factor of 2. See the paper by Pocrnic et al. [#pocrnic]_ for more details. 
#
# 
# Conclusion
# ---------------------------------
# 
# We have shown how to block encode a register of signed integers stored in two's complement form, taking
# :math:`\sum_a c_a |a\rangle \rightarrow \frac{1}{2^{n-1}} \sum_a c_a a |a\rangle` using a PREP-SEL-PREP structure that
# costs only :math:`4n-3` Toffoli gates. Along the way, we saw how the PREP operator prepares the :math:`|\sqrt{\mathtt{amp}_n}\rangle`
# resource state, how SEL acts as a filter that sets up the right interference for both the unsigned and signed cases,
# and how PennyLane's resource estimator lets us count the non-Clifford cost directly.
#
# We can now return to the benefit of this formulation promised in the introduction. 
# Applying the block encoding of the momentum operator 
# :math:`|p\rangle \rightarrow p |p\rangle` twice gives a block encoding of :math:`p^2`, and hence a direct route to the
# kinetic energy operator for :doc:`chemistry simulations in first quantization <demos/tutorial_resource_estimation>`. 
# But this construction leads to a better option than squaring. We can
# build a walk operator :math:`U_p Z_\Pi`, where :math:`U_p` is the block encoding of the momentum operator that we've just detailed and :math:`Z_\Pi` is a reflection about the block encoding subspace (see Ch. 7.1 of
# `Lin Lin's lecture notes <https://arxiv.org/abs/2201.08309>`__ [#linlin]_ for more details). Applying it twice :math:`U_p Z_\Pi U_p Z_\Pi`
# encodes the second-order Chebyshev polynomial :math:`T_2(p) = 2p^2 - \mathbb I`. 
# 
# The identity commutes with the Hamiltonian, so it shifts the spectrum without affecting the dynamics and can thus be ignored. Note that since this construction shifts the kinetic energy to be centered upon zero, were this block encoding called in a phase estimation the shift would need to be accounted for by undoing it classically (similar to how performing phase estimation on the walk operator outputs :math:`\text{arccos}(\lambda)` and this is accounted for classically in most :doc:`qubitized QPE <demos/tutorial_re_for_qubitizedQPE>` workflows).
# The leading factor of 2 lets us block encode the mass coefficients as :math:`1/4m` rather than :math:`1/2m` in the PREP circuit, halving the overall 1-norm of the kinetic energy operator. 
# Since the number of calls to the block encoding in a :doc:`qubitization <demos/tutorial_qubitization>` simulation is directly proportional to the 1-norm, this trick halves the simulation cost at almost negligible additional
# circuit depth relative to the cost of block encoding the first-quantized Hamiltonian itself. 
# 
# This demo discussed a method to block encode signed integers, and alluded to how it can enable a more efficient block encoding of the kinetic energy operator for quantum chemistry applications. But this is just one part of the end-to-end quantum algorithm to simulate the quantum dynamics of various molecules detailed in the paper by Pocrnic et al. [#pocrnic]_. Make sure to check out the paper for the details, like reducing the number of subroutine calls from :math:`O(\eta)` to :math:`O(1)` using swap networks (over :math:`\eta` particles), and how to encode the Coulomb interaction using destructive interference!
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
#     `arXiv:2201.08309 <https://arxiv.org/abs/2201.08309>`__, 2022. 
# 
# .. [#Su]
# 
#     Y. Su, D. W. Berry, N. Wiebe, N. Rubin, and R. Babbush
#     "Fault-Tolerant Quantum Simulations of Chemistry in First Quantization"
#     `arXiv:2105.12767 <https://arxiv.org/abs/2105.12767>`__, 2021.
# 
# 
# 