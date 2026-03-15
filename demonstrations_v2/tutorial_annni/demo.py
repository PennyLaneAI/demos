r"""Supervised and Unsupervised Quantum Machine Learning Models for the Phase Detection of the ANNNI Spin Model
==========================================================

Quantum Machine Learning (QML) models provide a natural framework for analyzing quantum many-body systems due to the direct mapping between spins and qubits.

A key property of these systems is their phase diagram, which characterizes different physical phases based on parameters such as magnetization or the strength of interactions. 
These diagrams help identify transitions where the system undergoes sudden changes in behaviour, which is essential for understanding the fundamental properties of quantum materials and optimizing quantum technologies.

Following the same approach as in the paper on `Quantum phase detection generalization from marginal quantum neural network models <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.L081105>`_, 
we explore the phase diagram of the Axial Next-Nearest-Neighbor Ising (ANNNI) model using both supervised and unsupervised learning methods:

* **Supervised model**: The Quantum Convolutional Neural Network (QCNN) is trained on a small subset of analytically known phase points, demonstrating its ability to generalize beyond the training data.

* **Unsupervised model**: Quantum Anomaly Detection (QAD) requires minimal prior knowledge and efficiently identifies potential phase boundaries, making it a valuable tool for systems with unknown structures.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_CERN_ANNNI.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

    Figure 1. Adding an ANNNI Ising to the cake

The ANNNI model
--------------------------------------------------------------------------------

The ANNNI model describes a spin system with three types of competing interactions. Its Hamiltonian is given by

.. math::  H =   -J \sum_{i=1}^{N} \sigma_x^i\sigma_x^{i+1} - \kappa \sigma_x^{i}\sigma_x^{i+2} + h \sigma_z^i, \tag{1}

where

* :math:`\sigma_a^i` are the Pauli matrices acting on the :math:`i`-th spin (:math:`a \in \{x, y, z\}`),

* :math:`J` is the nearest-neighbor coupling constant, which we set to :math:`1` without any loss of generality,

* :math:`\kappa` controls the strength next-nearest-neighbor interaction,

* and :math:`h` represents the the transverse magnetic field strength.

Without loss of generality, we set :math:`J = 1` and consider open boundary conditions for positive :math:`\kappa` and :math:`h`.

We start by importing the necessary libraries for simulation, optimization, and plotting our results, 
as well as setting some important constants:
"""
import pennylane as qml
qml.capture.enable()
import numpy as np
from jax import jit, vmap, value_and_grad, random, config
from jax import numpy as jnp
import optax

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

config.update("jax_enable_x64", True)

seed = 123456

# Setting our constants
num_qubits = 8 # Number of spins in the Hamiltonian (= number of qubits)
side = 20      # Discretization of the Phase Diagram

######################################################################
# Implementing the Hamiltonian
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In PennyLane, we can easily build the ANNNI's spin Hamiltonian following the same approach as the 
# :doc:`demo on spin Hamiltonians <demos/tutorial_how_to_build_spin_hamiltonians>`:

def get_H(num_spins, k, h):
    """Construction function the ANNNI Hamiltonian (J=1)"""

    # Interaction between spins (neighbouring):
    H = -1 * (qml.PauliX(0) @ qml.PauliX(1))
    for i in range(1, num_spins - 1):
        H = H  - (qml.PauliX(i) @ qml.PauliX(i + 1))

    # Interaction between spins (next-neighbouring):
    for i in range(0, num_spins - 2):
        H = H + k * (qml.PauliX(i) @ qml.PauliX(i + 2))

    # Interaction of the spins with the magnetic field
    for i in range(0, num_spins):
        H = H - h * qml.PauliZ(i)

    return H

######################################################################
# Defining phase transition lines
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Due to the competition between the three types of interactions, the ANNNI model exhibits a rich and complex phase diagram.
#
# * The **Ising transition line** occurs at  
#   
#   .. math::  h_I(\kappa) \approx \frac{1 - \kappa}{\kappa} \left(1 - \sqrt{\frac{1 - 3 \kappa + 4 \kappa^2 }{1 - \kappa}} \right),\tag{2} 
#   
#   which separates the *ferromagnetic* phase from the *paramagnetic* phase.
#
# * The **Kosterlitz-Thouless (KT) transition line** occurs at  
#   
#   .. math::  h_C(\kappa) \approx 1.05 \sqrt{(\kappa - 0.5) (\kappa - 0.1)}, \tag{3}  
#   
#   which separates the *paramagnetic* phase from the *antiphase*.
#
#
# * Additionally, another phase transition has been numerically addressed but not yet confirmed. The **Berezinskii-Kosterlitz-Thouless (BKT) transition line** occurs at  
#   
#   .. math::  h_{BKT} \approx 1.05 (\kappa - 0.5), \tag{4}
#   
#   which entirely lies within the *antiphase* region.
#

def kt_transition(k):
    """Kosterlitz-Thouless transition line"""
    return 1.05 * np.sqrt((k - 0.5) * (k - 0.1))

def ising_transition(k):
    """Ising transition line"""
    return np.where(k == 0, 1, (1 - k) * (1 - np.sqrt((1 - 3 * k + 4 * k**2) / (1 - k))) / np.maximum(k, 1e-9))

def bkt_transition(k):
    """Floating Phase transition line"""
    return 1.05 * (k - 0.5)
    
def get_phase(k, h):
    """Get the phase from the DMRG transition lines"""
    # If under the Ising Transition Line (Left side)
    if k < .5 and h < ising_transition(k):
        return 0 # Ferromagnetic
    # If under the Kosterlitz-Thouless Transition Line (Right side)

    elif k > .5 and h < kt_transition(k):
        return 1 # Antiphase
    return 2 # else it is Paramagnetic

######################################################################
# .. figure:: ../_static/demonstration_assets/annni/annni_pd_L.png
#     :align: center
#     :width: 50%
#     :target: javascript:void(0)
# 
# State preparation
# -----------------
#
# In this section, we prepare the ground states of the system, which will serve as inputs for both QML models. Several methods can be used, including **Variational Quantum Eigensolver (VQE)**, introduced in [#Peruzzo]_ and demonstrated in the :doc:`demo on VQE <demos/tutorial_vqe>`, and **Matrix Product States (MPS)**, illustrated in the :doc:`demo on Constant-depth preparation of MPS with dynamic circuits <demos/tutorial_constant_depth_mps_prep>`.  
# For simplicity, in this demo, we compute the ground state directly by finding the *eigenvector* corresponding to the lowest eigenvalue of the Hamiltonian. The resulting states are then loaded into the quantum circuits using PennyLane’s :class:`~pennylane.StatePrep`.  
#
# It is important to note that this approach is only feasible within the classically simulable regime, as it becomes quickly intractable for larger system sizes.
# 
# Computing the ground states
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We implement a function to compute the ground state by diagonalizing the Hamiltonian.

def diagonalize_H(H_matrix):
    """Returns the lowest eigenvector of the Hamiltonian matrix."""
    _, psi = jnp.linalg.eigh(H_matrix)  # Compute eigenvalues and eigenvectors
    return jnp.array(psi[:, 0], dtype=jnp.complex64)  # Return the ground state

# Create meshgrid of the parameter space
ks = np.linspace(0, 1, side)
hs = np.linspace(0, 2, side)
K, H = np.meshgrid(ks, hs)

# Preallocate arrays for Hamiltonian matrices and phase labels.
H_matrices = np.empty((len(ks), len(hs), 2**num_qubits, 2**num_qubits))
phases = np.empty((len(ks), len(hs)), dtype=int)

for x, k in enumerate(ks):
    for y, h in enumerate(hs):
        H_matrices[y, x] = np.real(qml.matrix(get_H(num_qubits, k, h))) # Get Hamiltonian matrix
        phases[y, x] = get_phase(k, h)  # Get the respective phase given k and h

# Vectorized diagonalization
psis = vmap(vmap(diagonalize_H))(H_matrices)

######################################################################
# Supervised learning of phases: QCNN
# -----------------------------------
#
# QCNNs are a class of quantum circuits first introduced in [#Cong]_, inspired by their classical counterparts, Convolutional Neural Networks (CNNs). Like CNNs, QCNNs aim to learn representations from input data by leveraging its local properties. In this implementation, these local properties correspond to the interactions between neighbouring spins.
#
# A QCNN consists of two main components:
#
# * **Convolution layers**: alternating unitaries are applied to pairs of neighbouring spins.
#
# * **Pooling layers**: half of the qubits are measured, and based on the measurement outcome, different rotations are applied to the remaining qubits.
#
# For the output, we consider the model’s probability vector :math:`P(\kappa, h)` over the four computational basis states of the final two-qubit system, obtained using :func:`~pennylane.probs`. Each computational basis state is mapped to a specific phase as follows.
#
# * :math:`\vert 00 \rangle`: Ferromagnetic.
# * :math:`\vert 01 \rangle`: Antiphase.
# * :math:`\vert 10 \rangle`: Paramagnetic.
# * :math:`\vert 11 \rangle`: Trash class.
#
# Circuit definition
# ^^^^^^^^^^^^^^^^^^
#
# We now define and implement the QCNN circuit. The `qcnn_ansatz` function builds the QCNN architecture by alternating convolution and pooling layers 
# until only two qubits remain. The `qcnn_circuit` function embeds an input quantum state through :class:`~pennylane.StatePrep`, applies the QCNN ansatz, and returns 
# the probability distribution over the final two-qubit system. Finally, we vectorize the circuit for efficient evaluation.

def qcnn_ansatz(num_qubits, params):
    """Ansatz of the QCNN model
    Repetitions of the convolutional and pooling blocks
    until only 2 wires are left unmeasured
    """

    # Convolution block
    def conv(wires, params, index):
        if len(wires) % 2 == 0:
            groups = wires.reshape(-1, 2)
        else:
            groups = wires[:-1].reshape(-1, 2)
            qml.RY(params[index], wires=int(wires[-1]))
            index += 1

        for group in groups:
            qml.CNOT(wires=[int(group[0]), int(group[1])])
            for wire in group:
                qml.RY(params[index], wires=int(wire))
                index += 1

        return index

    # Pooiling block
    def pool(wires, params, index):
        # Process wires in pairs: measure one and conditionally rotate the other.
        for wire_pool, wire in zip(wires[0::2], wires[1::2]):
            m_0 = qml.measure(int(wire_pool))
            qml.cond(m_0 == 0, qml.RX)(params[index],     wires=int(wire))
            qml.cond(m_0 == 1, qml.RX)(params[index + 1], wires=int(wire))
            index += 2
            # Remove the measured wire from active wires.
            wires = np.delete(wires, np.where(wires == wire_pool))

        # If an odd wire remains, apply a RX rotation.
        if len(wires) % 2 != 0:
            qml.RX(params[index], wires=int(wires[-1]))
            index += 1

        return index, wires

    # Initialize active wires and parameter index.
    active_wires = np.arange(num_qubits)
    index = 0

    # Initial layer: apply RY to all wires.
    for wire in active_wires:
        qml.RY(params[index], wires=int(wire))
        index += 1

    # Repeatedly apply convolution and pooling until there are 2 unmeasured wires
    while len(active_wires) > 2:
        # Convolution
        index = conv(active_wires, params, index)
        # Pooling
        index, active_wires = pool(active_wires, params, index)  
        qml.Barrier()

    # Final layer: apply RY to the remaining active wires.
    for wire in active_wires:
        qml.RY(params[index], wires=int(wire))
        index += 1

    return index, active_wires

num_params, output_wires = qcnn_ansatz(num_qubits, [0]*100)

@qml.qnode(qml.device("default.qubit", wires=num_qubits))
def qcnn_circuit(params, state):
    """QNode with QCNN ansatz and probabilities of unmeasured qubits as output"""
    # Input ground state from diagonalization
    qml.StatePrep(state, wires=range(num_qubits), normalize = True)
    # QCNN
    _, output_wires = qcnn_ansatz(num_qubits, params)

    return qml.probs([int(k) for k in output_wires])

# Vectorized circuit through vmap
vectorized_qcnn_circuit = vmap(jit(qcnn_circuit), in_axes=(None, 0))

# Draw the QCNN Architecture
fig,ax = qml.draw_mpl(qcnn_circuit)(np.arange(num_params), psis[0,0])

######################################################################
# Training of the QCNN
# ^^^^^^^^^^^^^^^^^^^^
#
# The training is performed by minimizing the **Cross Entropy loss** on the output probabilities
#
# .. math::  \mathcal{L} = -\frac{1}{|S|} \sum_{(\kappa, h) \in S} \sum_{j} y_j^{\frac1T}(\kappa, h) \log(p_j(\kappa, h))^\frac1T,   \tag{5}
#
# where
#
# * :math:`S` is the training set,
#
# * :math:`p_j(\kappa, h)` is the model’s predicted probability of the system at :math:`(\kappa, h)` being in the :math:`j`-th phase,
#
# * :math:`y_j(\kappa, h)` represents the one-hot encoded labels for the three phases,
# 
# * and :math:`T` is a temperature factor that controls the sharpness of the predicted probability distribution.

def cross_entropy(pred, Y, T):
    """Multi-class cross entropy loss function"""
    epsilon = 1e-9  # Small value for numerical stability
    pred = jnp.clip(pred, epsilon, 1 - epsilon)  # Prevent log(0)
    
    # Apply sharpening (raise probabilities to the power of 1/T)
    pred_sharpened = pred ** (1 / T)
    pred_sharpened /= jnp.sum(pred_sharpened, axis=1, keepdims=True)  # Re-normalize
    
    loss = -jnp.sum(Y * jnp.log(pred_sharpened), axis=1)
    return jnp.mean(loss)

######################################################################
# The analytical points of the ANNNI model correspond to specific regions of the phase diagram where the system simplifies into two well-understood limits:
#
# * **Transverse-field Ising model** at :math:`\kappa = 0` in which we only have the magnetic field and the nearest-neighbor interactions.
# 
# * **Quasi classical model**  at :math:`h=0` in which we only have the nearest and next-nearest-neighbor interactions.
#
# For these points, we can derive the labels analytically which will then be used for the training of the QCNNs. 

# Mask for the analytical points
analytical_mask = (K == 0) | (H == 0)

######################################################################
# .. figure:: ../_static/demonstration_assets/annni/annni_pd_analytical_L.png
#     :align: center
#     :width: 50%
#     :target: javascript:void(0)

def train_qcnn(num_epochs, lr, T, seed):
    """Training function of the QCNN architecture"""

    # Initialize PRNG key
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    
    # Define the loss function
    def loss_fun(params, X, Y):
        preds = vectorized_qcnn_circuit(params, X)
        return cross_entropy(preds, Y, T)

    # Consider only analytical points for the training
    X_train, Y_train = psis[analytical_mask], phases[analytical_mask]
    
    # Convert labels to one-hot encoding
    Y_train_onehot = jnp.eye(4)[Y_train]

    # Randomly initialize the parameters
    params = random.normal(subkey, (num_params,))

    # Initialize Adam optimizer
    optimizer = optax.adam(learning_rate=lr)
    optimizer_state = optimizer.init(params)

    loss_curve = []
    for epoch in range(num_epochs):
        # Compute loss and gradients
        loss, grads = value_and_grad(loss_fun)(params, X_train, Y_train_onehot)
        
        # Update parameters
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, updates)
        
        loss_curve.append(loss)
            
    return params, loss_curve

trained_params, loss_curve = train_qcnn(num_epochs=100, lr=5e-2, T=.5, seed=seed) 

# Plot the loss curve
plt.plot(loss_curve, label="Loss", color="blue", linewidth=2)
plt.xlabel("Epochs"), plt.ylabel("Cross-Entropy Loss")
plt.title("Figure 4. QCNN Training Cross-Entropy Loss Curve")
plt.legend()
plt.grid()
plt.show()

######################################################################
# After the training, much alike in [#Monaco]_ and [#Caro]_, we can inspect the model's generalization capabilities, 
# by obtaining the predicted phase for every point across the 2D phase diagram and compare these predictions 
# with the phase boundaries identified through density matrix renormalization group (DMRG) methods.

# Take the predicted classes for each point in the phase diagram
predicted_classes = np.argmax(
    vectorized_qcnn_circuit(trained_params, psis.reshape(-1, 2**num_qubits)),
    axis=1
)

colors = ['#80bfff', '#fff2a8',  '#80f090', '#da8080',]
phase_labels = ["Ferromagnetic", "Antiphase", "Paramagnetic", "Trash Class",]
cmap = ListedColormap(colors)

bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = BoundaryNorm(bounds, cmap.N)

# Plot the predictions over the phase diagram
plt.figure(figsize=(4,4), constrained_layout=True)
plt.imshow(
    predicted_classes.reshape(side, side),
    cmap=cmap,
    norm=norm,
    aspect="auto",
    origin="lower",
    extent=[0, 1, 0, 2]
)

# Plot the transition lines (Ising and KT) for reference.
k_vals1 = np.linspace(0.0, 0.5, 50)
k_vals2 = np.linspace(0.5, 1.0, 50)
plt.plot(k_vals1, ising_transition(k_vals1), 'k')
plt.plot(k_vals2, kt_transition(k_vals2), 'k')
plt.plot(k_vals2, bkt_transition(k_vals2), 'k', ls = '--')

for color, phase in zip(colors, phase_labels[:-1]):
    plt.scatter([], [], color=color, label=phase, edgecolors='black')
plt.plot([], [], 'k', label='Transition lines')

plt.xlabel("k"), plt.ylabel("h")
plt.title("Figure 5. QCNN Classification")
plt.legend()
plt.show()

######################################################################
# Despite only being trained on the left and bottom borders of the phase diagram, the QCNN successfully generalizes across the entire diagram, aligning with the phase boundaries predicted by DMRG.
#
# In addition, [#Cea]_ presents an analysis of the QCNN’s performance as the number of qubits (hence the system's size) increases, showing that the overlap between trash class and the expected floating phase becomes more accurate.  
#
# Unsupervised learning of phases: Quantum Anomaly Detection
# ----------------------------------------------------------
# 
# Quantum Anomaly Detection (QAD), introduced in [#Kottmann]_, is the quantum version of an autoencoder. However, unlike classical autoencoders, only the encoding (forward) process is trained here. This is because quantum operations are invertible, making a separate decoder unnecessary.
# 
# In this method, we start with a single quantum state :math:`|\psi(\kappa, h)\rangle` taken from the ANNNI model. The goal is to optimize the parameters :math:`\theta` of a quantum circuit :math:`V(\theta)` so that it transforms the chosen input state into the following form:
# 
# .. math::  V(\theta)|\psi(\kappa, h)\rangle = |\phi\rangle^{N-K} \otimes |0\rangle^{\otimes K}\tag{6}
# 
# The equation above means that we are trying to find a unitary transformation that "compresses" the important information of the original state into a smaller number of qubits :math:`(N-K)`, while disentangling and resetting the remaining :math:`K` qubits to the trivial state :math:`|0\rangle`. In other words, the circuit learns to isolate the relevant features of the quantum state into :math:`|\phi\rangle`.
#
# Circuit definition
# ^^^^^^^^^^^^^^^^^^
#
# We now define and implement the QAD circuit. The `anomaly_ansatz` function builds the QAD architecture. The `anomaly_circuit` function embeds an input quantum state through :class:`~pennylane.StatePrep`, applies the QAD ansatz, and returns the expectation values of the *trash qubits* used to evaluate the compression score.

def anomaly_ansatz(n_qubit, params):
    """Ansatz of the QAD model
    Apply multi-qubit gates between trash and non-trash wires
    """

    # Block of gates connecting trash and non-trash wires
    def block(nontrash, trash, shift):
        # Connect trash wires
        for i, wire in enumerate(trash):
            target = trash[(i + 1 + shift) % len(trash)]
            qml.CZ(wires=[int(wire), int(target)])
        # Connect each nontrash wire to a trash wire
        for i, wire in enumerate(nontrash):
            trash_idx = (i + shift) % len(trash)
            qml.CNOT(wires=[int(wire), int(trash[trash_idx])])

    depth = 2  # Number of repeated block layers
    n_trashwire = n_qubit // 2

    # Define trash wires as a contiguous block in the middle.
    trash = np.arange(n_trashwire // 2, n_trashwire // 2 + n_trashwire)
    nontrash = np.setdiff1d(np.arange(n_qubit), trash)

    index = 0

    # Initial layer: apply RY rotations on all wires.
    for wire in np.arange(n_qubit):
        qml.RY(params[index], wires=int(wire))
        index += 1

    # Repeatedly apply blocks of entangling gates and additional rotations.
    for shift in range(depth):
        block(nontrash, trash, shift)
        qml.Barrier()
        # In the final layer, only apply rotations on trash wires.
        wires_to_rot = np.arange(n_qubit) if shift < depth - 1 else trash
        for wire in wires_to_rot:
            qml.RY(params[index], wires=int(wire))
            index += 1

    return index, list(trash)

num_anomaly_params, trash_wires = qcnn_ansatz(num_qubits, [0]*100)

@qml.qnode(qml.device("default.qubit", wires=num_qubits))
def anomaly_circuit(params, state):
    """QNode with QAD ansatz and expectation values of the trash wires as output"""
    # Input ground state from diagonalization
    qml.StatePrep(state, wires=range(num_qubits), normalize = True)
    # Quantum Anomaly Circuit
    _, trash_wires = anomaly_ansatz(num_qubits, params)

    return [qml.expval(qml.PauliZ(int(k))) for k in trash_wires]

# Vectorize the circuit using vmap
jitted_anomaly_circuit = jit(anomaly_circuit)
vectorized_anomaly_circuit = vmap(jitted_anomaly_circuit, in_axes=(None, 0))

# Draw the QAD Architecture
fig,ax = qml.draw_mpl(anomaly_circuit)(np.arange(num_anomaly_params), psis[0,0])

######################################################################
# Training of the QAD
# ^^^^^^^^^^^^^^^^^^^
#
# The training process for this architecture follows these steps:
#
# 1. **Selection of Training Event:** a single quantum state is selected as the training event.
#
# 2. **Compression Objective:** the training is performed to achieve the compression of the selected quantum state. This is done by minimizing the following loss function, known as the *compression score*:
#    
#    .. math::  \mathcal{C} = \frac{1}{2}\sum_{j\in q_T} (1-\left<Z_j\right>),\tag{7}
#    
#    where :math:`q_T` refers to the set of trash qubits, which make up :math:`N/2` of the total.
#    By doing so, all the information of the input quantum state is compressed in the remaining non-measured qubits.
#   
# In this case, the selected quantum state corresponds to the trivial case with :math:`\kappa = 0` and :math:`h = 0`.

def train_anomaly(num_epochs, lr, seed):
    """Training function of the QAD architecture"""

    # Initialize PRNG key
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    
    # Define the loss function
    def loss_fun(params, X):
        # Output expectation values of the qubits
        score = 1 - jnp.array(jitted_anomaly_circuit(params, X))
        loss_value = jnp.mean(score)

        return loss_value

    # Training set consists only of the k = 0 and h = 0 state
    X_train = jnp.array(psis[0, 0])
    
    # Randomly initialize parameters
    params = random.normal(subkey, (num_anomaly_params,))

    optimizer = optax.adam(learning_rate=lr)
    optimizer_state = optimizer.init(params)
    
    loss_curve = []
    for epoch in range(num_epochs):
        # Get random indices for a batch
        key, subkey = random.split(key)

        # Compute loss and gradients
        loss, grads = value_and_grad(loss_fun)(params, X_train)
        
        # Update parameters
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, updates)

        loss_curve.append(loss)
    
    return params, loss_curve

trained_anomaly_params, anomaly_loss_curve = train_anomaly(num_epochs=100, lr=1e-1, seed=seed) 

# Plot the loss curve

plt.plot(anomaly_loss_curve, label="Loss", color="blue", linewidth=2)
plt.xlabel("Epochs"), plt.ylabel("Compression Loss")
plt.title("Figure 6. Anomaly training compression loss curve")
plt.legend()
plt.grid()
plt.show()


######################################################################
# After training the circuit to optimally compress the (0,0) state, we evaluate the compression score for all other input states using the learned parameters.
# 
# The model is expected to achieve near-optimal compression for states similar to the training state (namely those belonging to the same phase) resulting in a low compression score. Conversely, states from different phases should exhibit poorer compression, leading to a higher compression score.

# Evaluate the compression score for each state in the phase diagram
compressions = vectorized_anomaly_circuit(trained_anomaly_params, psis.reshape(-1, 2**num_qubits))
compressions = jnp.mean(1 - jnp.array(compressions), axis = 0)

im = plt.imshow(compressions.reshape(side, side), aspect="auto", origin="lower", extent=[0, 1, 0, 2])

# Plot transition lines (assuming ising_transition and kt_transition are defined)
plt.plot(np.linspace(0.0, 0.5, 50), ising_transition(np.linspace(0.0, 0.5, 50)), 'k')
plt.plot(np.linspace(0.5, 1.0, 50), kt_transition(np.linspace(0.5, 1.0, 50)), 'k')

plt.plot([], [], 'k', label='Transition Lines')
plt.scatter([0 +.3/len(ks)], [0 + .5/len(hs)], color='r', marker = 'x', label="Training point", s=50)

plt.legend(), plt.xlabel("k"), plt.ylabel("h"), plt.title("Figure 7. Phase diagram with QAD")
cbar = plt.colorbar(im)
cbar.set_label(r"Compression Score  $\mathcal{C}$")
plt.show()

######################################################################
# 
# As expected, the compression score is nearly zero within the ferromagnetic phase, while higher scores are observed in the other regions. Surprisingly, the other regions display distinct compression scores that remain consistent within their respective areas.
# 
# Using this model, we can clearly identify the three phases and their locations. Furthermore, as observed in [#Cea]_, by increasing the system size (around ~20 spins), a fourth phase emerges in the anticipated floating phase region. These regions are expected to become more sharply defined at larger system sizes, with more pronounced transitions between phases.
# 
# Conclusion
# -----------
#
# Quantum many-body systems present a compelling use case for QML models. In this tutorial, we explored two different approaches:
#
# * **Supervised learning**: the QCNN effectively generalizes phase classification beyond its training data, aligning with phase boundaries identified by classical methods.
#
# * **Unsupervised learning**: QAD successfully distinguishes different quantum phases without requiring labelled training data, making it ideal for systems with unknown structures.
#
# These techniques could be valuable for studying spin systems beyond classical simulability, particularly when non-local interactions are present and tensor network methods become inadequate.
# 
# Acknowledgements
# ----------------
# 
# The author would like to acknowledge the contributions and support of Michele Grossi, Sofia Vallecorsa, Oriel Kiss and Enrique Rico Ortega from the CERN Quantum Technology Initiative, and Antonio Mandarino, and María Cea Fernández. The author also gratefully acknowledges the support of DESY (Deutsches Elektronen-Synchrotron).
# 
# References
# -----------
#
# .. [#Monaco]
#
#     Saverio Monaco, Oriel Kiss, Antonio Mandarino, Sofia Vallecorsa, Michele Grossi,
#     "Quantum phase detection generalization from marginal quantum neural network models",
#     `PhysRevB.107.L081105 <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.L081105>`__, 2023.
#
# .. [#Cea]
#
#     Maria Cea, Michele Grossi, Saverio Monaco, Enrique Rico, Luca Tagliacozzo, Sofia Vallecorsa,
#     "Exploring the Phase Diagram of the quantum one-dimensional ANNNI model",
#     `arxiv:2402.11022 <https://arxiv.org/abs/2402.11022>`__, 2023.
#
# .. [#Peruzzo]
#
#     Alberto Peruzzo, Jarrod McClean, Peter Shadbolt, Man-Hong Yung, Xiao-Qi Zhou, Peter J. Love, Alán Aspuru-Guzik, Jeremy L. O'Brien,
#     "A variational eigenvalue solver on a quantum processor",
#     `Nat. Commun. 5, 4213 <https://www.nature.com/articles/ncomms5213>`__, 2013.
#
# .. [#Kottmann]
#
#     Korbinian Kottmann, Friederike Metz, Joana Fraxanet, Niccolo Baldelli,
#     "Variational Quantum Anomaly Detection: Unsupervised mapping of phase diagrams on a physical quantum computer",
#     `PhysRevResearch.3.043184 <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.043184>`__, 2022.
#
# .. [#Cong]
#
#     Iris Cong, Soonwon Choi, Mikhail D. Lukin,
#     "Quantum Convolutional Neural Networks",
#     `Nat. Phys. 15, 1273–1278 <https://www.nature.com/articles/s41567-019-0648-8>`__, 2019.
#
# .. [#Caro]
#
#     Matthias C. Caro, Hsin-Yuan Huang, Marco Cerezo, Kunal Sharma, Andrew Sornborger, Lukasz Cincio, Patrick J. Coles,
#     "Generalization in quantum machine learning from few training data",
#     `Nat. Commun. 13, 4919 <https://www.nature.com/articles/s41467-022-32550-3>`__, 2022.
#
