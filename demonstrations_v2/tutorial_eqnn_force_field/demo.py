r"""Symmetry-invariant quantum machine learning force fields
========================================================


Symmetries are ubiquitous in physics. From condensed matter to particle physics, they have helped us
make connections and formulate new theories. In the context of machine learning, inductive bias has
proven to be successful in the presence of symmetries. This framework, known as geometric deep
learning, often enjoys better generalization and trainability. In this demo, we will learn how to
use geometric quantum machine learning to drive molecular dynamics as introduced in recent research [#Le23]_. We
will take as an example a triatomic molecule of :math:`H_2O.`


Introduction
-----------------------------------

First, let’s introduce the overall playground of this work: **molecular dynamics (MD)**. MD is an
essential computational simulation method to analyze the dynamics of atoms or molecules in a
chemical system. The simulations can be used to obtain macroscopic thermodynamic properties of
ergodic systems. Within the simulation, Newton's equations of motion are numerically integrated. Therefore,
it is crucial to have access to the forces acting on the constituents of the system or, equivalently,
the potential energy surface (PES), from which we can obtain the atomic forces. Previous research by [#Kiss22]_ presented variational
quantum learning models (VQLMs) that were able to learn the potential energy and atomic forces of
a selection of molecules from *ab initio* reference data.


The description of molecules can be greatly simplified by considering inherent **symmetries**. For
example, actions such as translation, rotation, or the interchange of identical atoms or molecules
leave the system unchanged. To achieve better performance, it is thus desirable to include this
information in our model. To do so, the data input can simply be made invariant itself, e.g., by
making use of so-called symmetry functions–hence yielding invariant energy predictions.

In this demo, we instead take the high road and design an intrinsically symmetry-aware model based on
equivariant quantum neural networks. Equivariant machine learning models have demonstrated many advantages
such as being more robust to noisy data and enjoying better generalization capabilities. Moreover, this has the additional
advantage of relaxing the need for data preprocessing, as the raw Cartesian coordinates can be given directly as inputs to the learning
model.

An overview of the workflow is shown in the figure below. First, the relevant symmetries are identified
and used to build the quantum machine model. We then train it on the PES of some molecule, e.g. :math:`H_2O,`
and finally obtain the forces by computing the gradient of the learned PES.

.. figure:: ../_static/demonstration_assets/eqnn_force_field/overview.png
    :align: center
    :width: 60%


Equivariant Quantum Machine learning
-----------------------------------

In order to incorporate symmetries into machine learning models, we need a few concepts from group theory. A formal course on the
subject is out of the scope of the present document, which is why we refer to two other demos, `equivariant graph embedding <https://pennylane.ai/qml/demos/tutorial_equivariant_graph_embedding/>`_
and `geometric quantum machine learning <https://pennylane.ai/qml/demos/tutorial_geometric_qml/#introduction>`_, as well as Ref. [#Meyer23]_ for a more thorough introduction.

In the following, we will denote elements of a symmetry group :math:`G` with :math:`g \in G.` :math:`G` could be for instance the rotation group :math:`SO(3),`
or the permutation group :math:`S_n.` Groups are often easier understood in terms of their representation :math:`V_g : \mathcal{V} \rightarrow \mathcal{V}` which maps group elements
to invertible linear operations, i.e. to :math:`GL(n),` on some vector space :math:`\mathcal{V}.` We call a function :math:`f: \mathcal{V} \rightarrow \mathcal{W}` *invariant* with respect to the action of
the group, if

.. math::  f(V_g(v)) = f(v),  \text{  for all } g \in G.

The concept of *equivariance* is a bit weaker, as it only requires the function to *commute* with the group action, instead of remaining constant.
In mathematical terms, we require that

.. math::  f(V_g(v)) = \mathcal{R}_g(f(v)),  \text{  for all } g \in G,

with :math:`\mathcal{R}` being a representation of :math:`G` on the vector space :math:`\mathcal{W}.` These concepts are important in
machine learning, as they tell us how the internal structure of the data, described by the group, is conserved when passing through the model.
In the remaining, we will refer to :math:`\mathcal{V}` and :math:`V_g` as the data space and the representation on it, respectively,
and :math:`\mathcal{W}` and :math:`\mathcal{R}_g` as the qubit space and the symmetry action on it, respectively.

Now that we have the basics, we will focus on the task at hand: building an equivariant quantum neural network for chemistry!

We use a `quantum reuploading model <https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series/>`__, which consists of a
variational ansatz :math:`M_\Theta(\mathcal{X})` applied to some initial state
:math:`|\psi_0\rangle.` Here, :math:`\mathcal{X}` denotes the description of a molecular configuration, i.e.,
the set of Cartesian coordinates of the atoms. The quantum circuit is given by

.. math:: M_\Theta(\mathcal{X}) = \left[ \prod_{d=D}^1 \Phi(\mathcal{X}) \mathcal{U}_d(\vec{\theta}_d) \right] \Phi(\mathcal{X}),

and depends on both data :math:`\mathcal{X}` and trainable parameters :math:`\Theta = \{\vec{\theta}_d\}_{d=1}^D.`
It is built by interleaving parametrized trainable layers :math:`U_d(\vec{\theta}_d)` with data
encoding layers :math:`\Phi(\mathcal{X}).` The corresponding quantum function
:math:`f_{\Theta}(\mathcal{X})` is then given by the expectation value of a chosen observable
:math:`O`

.. math:: f_\Theta(\mathcal{X}) = \langle \psi_0 | M_\Theta(\mathcal{X})^\dagger O M_\Theta(\mathcal{X}) |\psi_0 \rangle.

For the cases of a diatomic molecule (e.g. :math:`LiH`) and a triatomic molecule of two atom types (e.g. :math:`H_2O`), panel (a)
of the following figure displays the descriptions of the chemical systems by the Cartesian coordinates of their
atoms, while the general circuit formulation of the corresponding symmetry-invariant VQLM for these cases is shown in panel (b). Note that
we will only consider the triatomic molecule :math:`H_2O` in the rest of this demo.

.. figure:: ../_static/demonstration_assets/eqnn_force_field/siVQLM_monomer.jpg
    :align: center
    :width: 90%

An overall invariant model is composed of four ingredients: an invariant initial state, an
equivariant encoding layer, equivariant trainable layers, and finally an invariant observable. Here,
equivariant encoding means that applying the symmetry transformation first on the atomic
configuration :math:`\mathcal{X}` and then encoding it into the qubits produces the same results as
first encoding :math:`\mathcal{X}` and then letting the symmetry act on the qubits, i.e.,

.. math:: \Phi(V_g[\mathcal{X}]) = \mathcal{R}_g \Phi(\mathcal{X}) \mathcal{R}_g^\dagger,

where :math:`V_g` and :math:`\mathcal{R}_g` denote the symmetry representation on the data and qubit level, respectively.

For the trainable layer, equivariance means that the order of applying the symmetry and the
parametrized operations does not matter:

.. math:: \left[\mathcal{U}_d(\vec{\theta}_d), \mathcal{R}_g\right]=0.

Furthermore, we need to find an invariant observable :math:`O = \mathcal{R}_g O \mathcal{R}_g^\dagger` and an initial state
:math:`|\psi_0\rangle = \mathcal{R}_g |\psi_0\rangle,` i.e., which can absorb the symmetry action. Putting all this together
results in a symmetry-invariant VQLM as required.

In this demo, we will consider the example of a triatomic molecule of two atom types, such as a
water molecule. In this case, the system is invariant under translations, rotations, and the
exchange of the two hydrogen atoms. Translational symmetry is included by taking the
central atom as the origin. Therefore, we only need to encode the coordinates of the two identical *active* atoms, which
we will call :math:`\vec{x}_1` and :math:`\vec{x}_2.`

Let’s implement the model depicted above!


"""

######################################################################
# Implementation of the VQLM
# --------------------------
# We start by importing the libraries that we will need.

import pennylane as qml
import numpy as np

import jax

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp

import scipy
import matplotlib.pyplot as plt
import sklearn

######################################################################
# Let us construct Pauli matrices, which are used to build the Hamiltonian.
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1.0j], [1.0j, 0]])
Z = np.array([[1, 0], [0, -1]])

sigmas = jnp.array(np.array([X, Y, Z]))  # Vector of Pauli matrices
sigmas_sigmas = jnp.array(
    np.array(
        [
            np.kron(X, X),
            np.kron(Y, Y),
            np.kron(Z, Z)
        ]  # Vector of tensor products of Pauli matrices
    )
)


######################################################################
# We start by considering **rotational invariance** and building an initial state invariant under
# rotation, such as the singlet state :math:`|S\rangle = \frac{|01⟩−|10⟩}{\sqrt{2}}.` A general
# :math:`2n`-invariant state can be obtained by taking :math:`n`-fold tensor product.
#


def singlet(wires):
    # Encode a 2-qubit rotation-invariant initial state, i.e., the singlet state.

    qml.Hadamard(wires=wires[0])
    qml.PauliZ(wires=wires[0])
    qml.PauliX(wires=wires[1])
    qml.CNOT(wires=wires)


######################################################################
# Next, we need a rotationally equivariant data embedding. We choose to encode a three-dimensional
# data point :math:`\vec{x}\in \mathbb{R}^3` via
#
# .. math:: \Phi(\vec{x}) = \exp\left( -i\alpha_\text{enc} [xX + yY + zZ] \right),
#
# where we introduce a trainable encoding angle :math:`\alpha_\text{enc}\in\mathbb{R}.` This encoding
# scheme is indeed equivariant since embedding a rotated data point is the same as embedding the
# original data point and then letting the rotation act on the qubits:
# :math:`\Phi(r(\psi,\theta,\phi)\vec{x}) = U(\psi,\theta,\phi) \Phi(\vec{x}) U(\psi,\theta,\phi)^\dagger.`
# For this, we have noticed that any rotation on the data level can be parametrized by three angles
# :math:`V_g = r(\psi,\theta,\phi),` which can also be used to parametrize the corresponding
# single-qubit rotation :math:`\mathcal{R}_g = U(\psi,\theta,\phi),` implemented by the usual
# `qml.rot <https://docs.pennylane.ai/en/stable/code/api/pennylane.Rot.html>`__
# operation. We choose to encode each atom
# twice in parallel, resulting in higher expressivity. We can do so by simply using this encoding scheme twice for each
# active atom (the two Hydrogens in our case):
#
# .. math:: \Phi(\vec{x}_1, \vec{x}_2) = \Phi^{(1)}(\vec{x}_1) \Phi^{(2)}(\vec{x}_2) \Phi^{(3)}(\vec{x}_1) \Phi^{(4)}(\vec{x}_2).
#


def equivariant_encoding(alpha, data, wires):
    # data (jax array): cartesian coordinates of atom i
    # alpha (jax array): trainable scaling parameter

    hamiltonian = jnp.einsum("i,ijk", data, sigmas)  # Heisenberg Hamiltonian
    U = jax.scipy.linalg.expm(-1.0j * alpha * hamiltonian / 2)
    qml.fourier.mark(qml.QubitUnitary(U, wires=wires), "E")


######################################################################
# Finally, we require an equivariant trainable map and an invariant observable. We take the Heisenberg
# Hamiltonian, which is rotationally invariant, as an inspiration. We define a single summand of it,
# :math:`H^{(i,j)}(J) = -J\left( X^{(i)}X^{(j)} + Y^{(i)}Y^{(j)} + Z^{(i)}Z^{(j)} \right),` as a
# rotationally invariant two-qubit operator and choose
#
# .. math:: O = X^{(0)}X^{(1)} + Y^{(0)}Y^{(1)} + Z^{(0)}Z^{(1)}
#
# as our observable.
#
# Furthermore, we can obtain an equivariant parametrized operator by exponentiating this Heisenberg
# interaction:
#
# .. math:: RH^{(i,j)}(J) = \exp\left( -iH^{(i,j)}(J) \right),
#
# where :math:`J\in\mathbb{R}` is a trainable parameter. By combining this exponentiated operator for
# different pairs of qubits, we can design our equivariant trainable layer
#
# .. math:: \mathcal{U}(\vec{j}) = RH^{(1,2)}(j_1) RH^{(3,4)}(j_2) RH^{(2,3)}(j_3)
#
# In the case of a triatomic molecule of two atom types, we need to modify the previous VQLM to
# additionally take into account the **invariance under permutations of the same atom types**.
#
# Interchanging two atoms is represented on the data level by simply interchanging the corresponding
# coordinates, :math:`V_g = \sigma(\vec{x}_1, \vec{x}_2) = (\vec{x}_2, \vec{x}_1).` On the Hilbert
# space this is represented by swapping the corresponding qubits,
# :math:`\mathcal{R}_g = U(i,j) = SWAP(i,j).`
#
# The singlet state is not only rotationally invariant but also permutationally invariant under swapping
# certain qubit pairs, so we can keep it. The previous embedding scheme for one data point can
# be extended for embedding two atoms and we see that this is indeed not only rotationally
# equivariant but also equivariant with respect to permutations, since encoding two swapped atoms is
# just the same as encoding the atoms in the original order and then swapping the qubits:
# :math:`\Phi\left( \sigma(\vec{x}_1, \vec{x}_2) \right) = SWAP(i,j) \Phi(\vec{x}_1, \vec{x}_2) SWAP(i,j).`
# Again, we choose to encode each atom twice as depicted above.
#
# For the invariant observable :math:`O,` we note that our Heisenberg interaction is invariant under the swapping
# of the two involved qubits, therefore we can make use of the same observable as before.
#
# For the equivariant parametrized layer we need to be careful when it comes to the selection of qubit
# pairs in order to obtain equivariance, i.e., operations that commute with the swappings. This is
# fulfilled by coupling only the qubits with are neighbors with respect to the 1-2-3-4-1 ring topology, leading to the following operation:
#
# .. math:: \mathcal{U}(\vec{j}) = RH^{(1,2)}(j_1) RH^{(3,4)}(j_2) RH^{(2,3)}(j_3) RH^{(1,4)}(j_3)
#
# In code, we have:
#


def trainable_layer(weight, wires):
    hamiltonian = jnp.einsum("ijk->jk", sigmas_sigmas)
    U = jax.scipy.linalg.expm(-1.0j * weight * hamiltonian)
    qml.fourier.mark(qml.QubitUnitary(U, wires=wires), "U")


# Invariant observable
Heisenberg = [
    qml.PauliX(0) @ qml.PauliX(1),
    qml.PauliY(0) @ qml.PauliY(1),
    qml.PauliZ(0) @ qml.PauliZ(1),
]
Observable = qml.Hamiltonian(np.ones((3)), Heisenberg)

######################################################################
# It has been observed that a small amount of **symmetry-breaking** (SB) can improve the convergence
# of the VQLM. We implement it by adding a small rotation around the :math:`z`-axis.
#


def noise_layer(epsilon, wires):
    for _, w in enumerate(wires):
        qml.RZ(epsilon[_], wires=[w])


######################################################################
# When setting up the model, the hyperparameters such as the number of repetitions of encoding and
# trainable layers have to be chosen suitably. In this demo, we choose six layers (:math:`D=6`) and
# one repetition of trainable gates inside each layer (:math:`B=1`) to reduce long runtimes. Note that this choice differs from the original paper, so the results
# therein will not be fully reproduced
# within this demo. We start by defining the relevant hyperparameters and the VQLM.
#

############ Setup ##############
D = 6  # Depth of the model
B = 1  # Number of repetitions inside a trainable layer
rep = 2  # Number of repeated vertical encoding

active_atoms = 2  # Number of active atoms
# Here we only have two active atoms since we fixed the oxygen (which becomes non-active) at the origin
num_qubits = active_atoms * rep
#################################


dev = qml.device("default.qubit", wires=num_qubits)


@qml.qnode(dev, interface="jax")
def vqlm(data, params):

    weights = params["params"]["weights"]
    alphas = params["params"]["alphas"]
    epsilon = params["params"]["epsilon"]

    # Initial state
    for i in range(rep):
        singlet(wires=np.arange(active_atoms * i, active_atoms * (1 + i)))

    # Initial encoding
    for i in range(num_qubits):
        equivariant_encoding(
            alphas[i, 0], jnp.asarray(data, dtype=complex)[i % active_atoms, ...], wires=[i]
        )

    # Reuploading model
    for d in range(D):
        qml.Barrier()

        for b in range(B):
            # Even layer
            for i in range(0, num_qubits - 1, 2):
                trainable_layer(weights[i, d + 1, b], wires=[i, (i + 1) % num_qubits])

            # Odd layer
            for i in range(1, num_qubits, 2):
                trainable_layer(weights[i, d + 1, b], wires=[i, (i + 1) % num_qubits])

        # Symmetry-breaking
        if epsilon is not None:
            noise_layer(epsilon[d, :], range(num_qubits))

        # Encoding
        for i in range(num_qubits):
            equivariant_encoding(
                alphas[i, d + 1],
                jnp.asarray(data, dtype=complex)[i % active_atoms, ...],
                wires=[i],
            )

    return qml.expval(Observable)


######################################################################
# Simulation for the water molecule
# -----------------------------------
#
# We start by downloading the `dataset <https://zenodo.org/records/2634098>`__, which we have prepared for
# convenience as a Python ndarray. In the following, we will load, preprocess and split the data into a
# training and testing set, following standard practices.
#

# Load the data
energy = np.load("eqnn_force_field_data/Energy.npy")
forces = np.load("eqnn_force_field_data/Forces.npy")
positions = np.load(
    "eqnn_force_field_data/Positions.npy"
)  # Cartesian coordinates shape = (nbr_sample, nbr_atoms,3)
shape = np.shape(positions)

### Scaling the energy to fit in [-1,1]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler((-1, 1))

energy = scaler.fit_transform(energy)
forces = forces * scaler.scale_


# Placing the oxygen at the origin
data = np.zeros((shape[0], 2, 3))
data[:, 0, :] = positions[:, 1, :] - positions[:, 0, :]
data[:, 1, :] = positions[:, 2, :] - positions[:, 0, :]
positions = data.copy()

forces = forces[:, 1:, :]  # Select only the forces on the hydrogen atoms since the oxygen is fixed


# Splitting in train-test set
indices_train = np.random.choice(np.arange(shape[0]), size=int(0.8 * shape[0]), replace=False)
indices_test = np.setdiff1d(np.arange(shape[0]), indices_train)

E_train, E_test = (energy[indices_train, 0], energy[indices_test, 0])
F_train, F_test = forces[indices_train, ...], forces[indices_test, ...]
data_train, data_test = (
    jnp.array(positions[indices_train, ...]),
    jnp.array(positions[indices_test, ...]),
)

#################################
# We will know define the cost function and how to train the model using Jax. We will use the mean-square-error loss function.
# To speed up the computation, we use the decorator ``@jax.jit`` to do just-in-time compilation for this execution. This means the first execution will typically take a little longer with the
# benefit that all following executions will be significantly faster, see the `Jax docs on jitting <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_.

#################################
from jax.example_libraries import optimizers

# We vectorize the model over the data points
vec_vqlm = jax.vmap(vqlm, (0, None), 0)


# Mean-squared-error loss function
@jax.jit
def mse_loss(predictions, targets):
    return jnp.mean(0.5 * (predictions - targets) ** 2)


# Make prediction and compute the loss
@jax.jit
def cost(weights, loss_data):
    data, E_target, F_target = loss_data
    E_pred = vec_vqlm(data, weights)
    l = mse_loss(E_pred, E_target)

    return l


# Perform one training step
@jax.jit
def train_step(step_i, opt_state, loss_data):

    net_params = get_params(opt_state)
    loss, grads = jax.value_and_grad(cost, argnums=0)(net_params, loss_data)

    return loss, opt_update(step_i, grads, opt_state)


# Return prediction and loss at inference times, e.g. for testing
@jax.jit
def inference(loss_data, opt_state):

    data, E_target, F_target = loss_data
    net_params = get_params(opt_state)

    E_pred = vec_vqlm(data, net_params)
    l = mse_loss(E_pred, E_target)

    return E_pred, l


#################################
# **Parameter initialization:**
#
# We initiliase the model at the identity by setting the initial parameters to 0, except the first one which is chosen uniformly.
# This ensures that the circuit is shallow at the beginning and has less chance of suffering from the barren plateau phenomenon. Moreover,
# we disable the symmetry-breaking strategy, as it is mainly useful for larger systems.
np.random.seed(42)
weights = np.zeros((num_qubits, D, B))
weights[0] = np.random.uniform(0, np.pi, 1)
weights = jnp.array(weights)

# Encoding weights
alphas = jnp.array(np.ones((num_qubits, D + 1)))

# Symmetry-breaking (SB)
np.random.seed(42)
epsilon = jnp.array(np.random.normal(0, 0.001, size=(D, num_qubits)))
epsilon = None  # We disable SB for this specific example
epsilon = jax.lax.stop_gradient(epsilon)  # comment if we wish to train the SB weights as well.


opt_init, opt_update, get_params = optimizers.adam(1e-2)
net_params = {"params": {"weights": weights, "alphas": alphas, "epsilon": epsilon}}
opt_state = opt_init(net_params)
running_loss = []
#################################
# We train our VQLM using stochastic gradient descent.


num_batches = 5000  # number of optimization steps
batch_size = 256  # number of training data per batch


for ibatch in range(num_batches):
    # select a batch of training points
    batch = np.random.choice(np.arange(np.shape(data_train)[0]), batch_size, replace=False)

    # preparing the data
    loss_data = data_train[batch, ...], E_train[batch, ...], F_train[batch, ...]
    loss_data_test = data_test, E_test, F_test

    # perform one training step
    loss, opt_state = train_step(num_batches, opt_state, loss_data)

    # computing the test loss and energy predictions
    E_pred, test_loss = inference(loss_data_test, opt_state)
    running_loss.append([float(loss), float(test_loss)])

###################################
# Let us inspect the results. The following figure displays the training (in red) and testing (in blue) loss during the optimization. We observe that they are on top of each other, meaning that the model
# is training and generalising properly to the unseen test set.
#
history_loss = np.array(running_loss)

fontsize = 12
plt.figure(figsize=(4, 4))
plt.plot(history_loss[:, 0], "r-", label="training error")
plt.plot(history_loss[:, 1], "b-", label="testing error")

plt.yscale("log")
plt.xlabel("Optimization Steps", fontsize=fontsize)
plt.ylabel("Mean Squared Error", fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.tight_layout()
plt.show()
######################################################################
# Energy predictions
# ~~~~~~~~~~~~~~~~~~
#
# We first inspect the quality of the energy predictions. The exact test energy points are shown in black, while the predictions are in red.
# On the left, we see the exact data against the predicted ones (so the red points should be in the diagonal line), while the right plots show the
# energy as a scatter plot. The model is able to make fair predictions, especially near the equilibrium position. However, a few points in the higher energy range
# could be improved, e.g. by using a deeper model as in the original paper.
#

plt.figure(figsize=(4, 4))
plt.title("Energy predictions", fontsize=fontsize)
plt.plot(energy[indices_test], E_pred, "ro", label="Test predictions")
plt.plot(energy[indices_test], energy[indices_test], "k.-", lw=1, label="Exact")
plt.xlabel("Exact energy", fontsize=fontsize)
plt.ylabel("Predicted energy", fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.tight_layout()
plt.show()


######################################################################
# Force predictions
# ~~~~~~~~~~~~~~~~~
#
# As stated at the beginning, we are interested in obtaining the forces to drive MD simulations. Since
# we have access to the potential energy surface, the forces are directly available by taking the
# gradient
#
# .. math:: F_{i,j} = -\nabla_{\mathcal{X}_{ij}} E(\mathcal{X}, \Theta),
#
# where :math:`\mathcal{X}_{ij}` contains the :math:`j` coordinate of the :math:`i`-th atom, and :math:`\Theta`
# are the trainable parameters. In our framework, we can simply do the following. We note that we
# do not require the mixed terms of the Jacobian, which is why we select the diagonal part using ``numpy.einsum``.
#

opt_params = get_params(opt_state)  # Obtain the optimal parameters
gradient_coordinates = jax.jacobian(
    vec_vqlm, argnums=0
)  # Compute the gradient with respect to the Cartesian coordinates

pred_forces = gradient_coordinates(jnp.array(positions.real), opt_params)
pred_forces = -np.einsum(
    "iijk->ijk", np.array(pred_forces)
)  # We are only interested in the diagonal part of the Jacobian

fig, axs = plt.subplots(2, 3)

fig.suptitle("Force predictions", fontsize=fontsize)
for k in range(2):
    for l in range(3):

        axs[k, l].plot(forces[indices_test, k, l], forces[indices_test, k, l], "k.-", lw=1)
        axs[k, l].plot(forces[indices_test, k, l], pred_forces[indices_test, k, l], "r.")

axs[0, 0].set_ylabel("Hydrogen 1")
axs[1, 0].set_ylabel("Hydrogen 2")
for _, a in enumerate(["x", "y", "z"]):
    axs[1, _].set_xlabel("{}-axis".format(a))

plt.tight_layout()
plt.show()
######################################################################
# In this series of plots, we can see the predicted forces on the two Hydrogen atoms in the three :math:`x,` :math:`y` and :math:`z` directions. Again, the model
# does a fairly good job. The few points which are not on the diagonal can be improved using some tricks, such as incorporating the forces in the loss function.

######################################################################
# Conclusions
# -----------
#
# In this demo, we saw how to implement a symmetry-invariant VQLM to learn the energy and forces of
# small chemical systems and trained it for the specific example of water. The strong points with
# respect to symmetry-agnostic techniques are better generalization, more accurate force predictions,
# resilience to small data corruption, and reduction in classical pre- and postprocessing, as
# supported by the original paper.
#
# Further work could be devoted to studying larger systems by adopting a more systematic fragmentation as
# discussed in the original paper. As an alternative to building symmetry-invariant quantum
# architectures, the symmetries could instead be incorporated into the training routine, such as
# recently proposed by [#Wierichs23]_. Finally, symmetry-aware
# models could be used to design quantum symmetry functions, which in turn could serve as
# symmetry-invariant descriptors of the chemical systems within classical deep learning architectures,
# which can be easily operated and trained at scale.
#

######################################################################
# Bibliography
# ------------
#
# .. [#Le23]
#
#    Isabel Nha Minh Le, Oriel Kiss, Julian Schuhmacher, Ivano Tavernelli, Francesco Tacchino,
#    "Symmetry-invariant quantum machine learning force fields",
#    `arXiv:2311.11362 <https://arxiv.org/abs/2311.11362>`__, 2023.
#
#
# .. [#Kiss22]
#
#    Oriel Kiss, Francesco Tacchino, Sofia Vallecorsa, Ivano Tavernelli,
#    "Quantum neural networks force fields generation",
#    `Mach.Learn.: Sci. Technol. 3 035004 <https://iopscience.iop.org/article/10.1088/2632-2153/ac7d3c>`__, 2022.
#
#
# .. [#Meyer23]
#
#    Johannes Jakob Meyer, Marian Mularski, Elies Gil-Fuster, Antonio Anna Mele, Francesco Arzani, Alissa Wilms, Jens Eisert,
#    "Exploiting Symmetry in Variational Quantum Machine Learning",
#    `PRX Quantum 4,010328 <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.4.010328>`__, 2023.
#
#
# .. [#Wierichs23]
#
#    David Wierichs, Richard D. P. East, Martín Larocca, M. Cerezo, Nathan Killoran,
#    "Symmetric derivatives of parametrized quantum circuits",
#    `arXiv:2312.06752 <https://arxiv.org/abs/2312.06752>`__, 2023.
#
#
