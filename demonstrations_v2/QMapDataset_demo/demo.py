r"""Solving the qubit mapping problem with machine learning
=============================================================

The demo introduces the qubit mapping problem, an NP-hard challenge in quantum computing systems
with limited hardware connectivity, and explores machine learning-based approaches for solving it.
It presents the QMapDataset available in PennyLane and shows how it can be used to train a Graph
Neural Network to tackle the qubit mapping problem.
"""

######################################################################
# **1. What is the qubit mapping problem?**
# -----------------------------------------
# 
# When we write quantum circuits (with Pennylane, of course!), we usually assume that all the qubits
# are connected to each other, or more precisely, that two-qubit gates can be applied between any
# qubits. While some infrastructures have all-to-all connectivity (like the photonic devices developed
# by Xanadu, the ion or the neutral atom platforms), superconducting quantum processors have fixed
# connectivity between qubits. Only physically connected qubits can directly interact. For example,
# the first generation of superconducting devices had 5 qubits with a simple linear connectivity:
# 
# :math:`pq_0-pq_1-pq_2-pq_3-pq_4`
# 
# Physical qubit 0 and physical qubit 3 are not directly connected. If a two-qubit gate is required in
# your code between qubits 0 and 3, it is necessary to apply SWAP gates. SWAP gates exchange the
# states of two neighboring qubits for all basis states in the superposition (more details here `What
# is a SWAP gate? \| PennyLane <https://pennylane.ai/qml/glossary/what-is-a-swap-gate>`__).In our
# example, we would need to apply two SWAP gates: one between :math:`pq_0` and :math:`pq_1`, and one
# between :math:`pq_1` and :math:`pq_2`. If the state was
# :math:`\frac{1}{\sqrt{2}}(|10000\rangle + |00001\rangle)` before the SWAP gates, it becomes
# :math:`\frac{1}{\sqrt{2}}(|00100\rangle + |00001\rangle)`. Then, we can apply the CNOT gate between
# :math:`pq_2` and :math:`pq_3`.
# 
# The lack of full connectivity between physical qubits does not alter the universality of quantum
# computation in these devices but leads to an increase in the depth of the effective quantum circuit.
# A SWAP gate corresponds to 3 CNOT gates. Therefore, instead of 1 CNOT gate, we have 7 CNOT gates.
# This increase can be fatal for your quantum computation, especially on a NISQ device.
# 
# Indeed, NISQ devices are sensitive to noise. In particular, two-qubit gates are noisy. Therefore,
# when a large number of two-qubit gates is applied on a NISQ device, the quantum information is
# progressively lost and randomized. Therefore, when running a quantum circuit on a NISQ device, we
# try to minimize the number of two-qubit gates. Obviously, the increase by a factor of 7 in our
# example is terrible for the noise robustness of our quantum computation. Even for fault-tolerant
# quantum computers, reducing the number of CNOT gates would help to reduce the time of computation
# and therefore the computational cost.
# 
# It is here that qubit mapping can help mitigate the number of SWAP gates. There is no rule that
# constrains the first qubit of your quantum circuit to be the first physical qubit, the second qubit
# of your quantum circuit to be the second physical qubit, etc. We can separate the two sets of
# qubits: on one side, we have the qubits of our quantum circuit, called logical qubits and denoted
# :math:`lq_0`, :math:`lq_1`, etc., and on the other side, the physical qubits of the hardware denoted
# :math:`pq_0`, :math:`pq_1` etc.. There does not necessarily have to be an identity mapping between
# the two. Instead, we can define any mapping: :math:`\{lq\}\rightarrow \{pq\}`. The qubit mapping
# problem corresponds to finding the initial mapping between the logical qubits and the physical
# qubits that minimizes the number of SWAP gates during the execution of the quantum circuit by the
# hardware. In our simple example, if we assign :math:`lq_0\rightarrow pq_0` and
# :math:`lq_3\rightarrow pq_1`, a CNOT gate between physical qubits 0 and 1 corresponds to a CNOT gate
# between logical qubits 0 and 3, as desired.
# 
# Let's consider a slightly more complex quantum circuit:

import pennylane as qp
dev = qp.device("default.qubit", wires=5)
@qp.qnode(dev)
def  circuit():
    qp.CNOT(wires=[0,4])
    qp.CNOT(wires=[1,3])
    qp.CNOT(wires=[0,2])

######################################################################
# So, there are 3 CNOT gates: one between :math:`lq_0` and :math:`lq_4`, one between :math:`lq_1` and
# :math:`lq_3`, and one between :math:`lq_0` and :math:`lq_2`. Let’s first examine the number of
# SWAP gates with the identity mapping (:math:`lq_0\rightarrow pq_0`, :math:`lq_1\rightarrow pq_1`,
# etc.):
# 
# :math:`pq_0(lq_0)-pq_1(lq_1)-pq_2(lq_2)-pq_3(lq_3)-pq_4(lq_4)`
# 
# Here, in parentheses, the logical qubit mapped to the corresponding physical qubit is shown.
# 
# For the first CNOT gate between :math:`lq_0` and :math:`lq_4`, we need to apply 3 SWAP gates
# (SWAP(:math:`pq_0`, :math:`pq_1`), SWAP(:math:`pq_1`, :math:`pq_2`), and SWAP(:math:`pq_2`,
# :math:`pq_3`)). The mapping becomes:
# 
# :math:`pq_0(lq_1)-pq_1(lq_2)-pq_2(lq_3)-pq_3(lq_0)-pq_4(lq_4)`
# 
# Afterward, for the CNOT gate between :math:`lq_1` and :math:`lq_3`, we need a SWAP gate between
# :math:`pq_1` and :math:`pq_2`:
# 
# :math:`pq_0(lq_1)-pq_1(lq_3)-pq_2(lq_2)-pq_4(lq_0)-pq_4(lq_4)`
# 
# Finally, for the CNOT gate between :math:`lq_0` and :math:`lq_2`, we do not need any additional SWAP
# gates. Therefore, in total, there are 4 SWAP gates.
# 
# Now, let’s consider the following initial mapping:
# 
# :math:`pq_0(lq_1)-pq_1(lq_3)-pq_2(lq_2)-pq_3(lq_0)-pq_4(lq_4)`
# 
# In this case, there is no need for SWAP gates. The three CNOT gates can be applied directly. We go
# from 15 CNOT gates to 3 CNOT gates, a reduction by a factor of 5 just by changing the initial
# mapping!
# 
# Now that the qubit mapping problem is clear from these 5-qubit examples, let’s give a generalized
# definition. We start with:
# 
# -  A quantum circuit C defined on n logical qubits :math:`lq_0, lq_1, \dots, lq_{n-1}`
# -  A quantum device with m physical qubits :math:`pq_0, pq_1, \dots, pq_{m-1}` connected by some
#    graph :math:`G_D`.
# 
# The goal of the qubit mapping problem is to find the initial mapping
# :math:`\pi_0: \{lq\}\rightarrow \{pq\}` in order to minimize the number of SWAP gates during the
# execution of the quantum circuit C on the quantum device. Note that in general, the number of
# logical qubits :math:`n` is smaller than the number of physical qubits :math:`m` (and cannot be
# bigger).
# 
# The initial qubit mapping problem can be represented as an assignment matrix A of dimension
# :math:`n\times m`:
# 
# -  :math:`A_{ij}=1` if logical qubit :math:`lq_i` is mapped to the physical qubit :math:`pq_j`.
# -  :math:`A_{ij}=0` otherwise
# 
# There are two constraints on the assignment matrix A : \* Each logical qubit is assigned exactly one
# physical qubit :math:`\sum_{j=1}^m A_{ij} = 1, ~\forall i`
# 
# -  Each physical qubit holds at most one logical qubit :math:`\sum_{i=1}^n A_{ij}\leq 1, ~\forall j`
# 
# An important quantity is the distance between two physical qubits :math:`D_{ij}=d(pq_i, pq_j)`. The
# distance is the shortest-path length between two physical qubits in the hardware connectivity graph,
# and it directly estimates how many SWAPs are needed to make them interact. Of course, the distance
# would depend on the hardware connectivity graph.
# 
# We will explore in the next section the actual methods used to find the optimal assignment matrix A
# and therefore to solve the qubit mapping problem.
# 
# **2. How to solve the qubit mapping problem?**
# ---------------------------------------------
# 
# The qubit mapping problem is unfortunately a NP problem and is exponentially hard to find the
# optimal solution when the number of logical and physical qubits increases. However, there are some
# heuristic methods that allow us to solve it with a reasonable computational time. The most famous
# heuristic algorithm is the SABRE algorithm, used for example by Qiskit in its compilation.
# 
# The SABRE algorithm works by iteratively updating the qubit mapping guided by a cost function based
# on the total distance.
# 
# **Forward propagation** 1. **Initialize the mapping**: The algorithm starts by initializing a single
# mapping between logical and physical qubits (typically random or identity). Example (5-qubit
# device): :math:`pq_0(lq_0)-pq_1(lq_2)-pq_2(lq_3)-pq_3(lq_4)-pq_4(lq_1)`
# 
# 2. **Define the front layer**: The front layer is the set of gates that are ready to be executed,
#    meaning all their dependencies in the circuit have been satisfied. For the quantum circuit
#    (CNOT(0,4), CNOT(1,3) and CNOT(0,2), the front layer is {CNOT(0,4), CNOT(1,3)}. CNOT(0,2) is not
#    included because it depends on the execution of CNOT(0,4).
# 
# 3. **Check gate executability**: For each gate in the front layer, the algorithm checks whether the
#    mapped physical qubits are connected in the hardware graph: If they are adjacent, the gate "is
#    executed" and is withdrawn from the front layer. If not, SWAP gate is required and step 4 is
#    applied. In our case, the CNOT(0,4) and CNOT(1,3) cannot be executed.
# 
# 4. **Compute routing cost**: The algorithm calculates the sum of the distance of the operations in
#    the front layer: :math:`H = \sum_{(a,b)} d(\pi(a), \pi(b))`
# 
# 5. **Evaluate SWAP candidates**: The algorithm considers SWAPs on edges of the hardware graph. For
#    each candidate SWAP, they recompute the cost H and select the SWAP that minimizes the cost. In
#    our case, the SWAP(2,3) gives the distance :math:`H=2+1=3`, which is better than other options.
#    So this SWAP is applied.
# 
# 6. **Iterate forward pass**: Steps 2–5 are repeated. The front layer is updated dynamically as
#    gates become executable. Gates are executed whenever possible. SWAPs are inserted when necessary.
#    At the end of this process, the algorithm produces a final mapping for the forward pass.
# 
# **Backward propagation**
# 
# 7. **Reverse circuit execution**: The algorithm then repeats the same procedure, but on the reversed
#    circuit (executing gates in reverse order), starting from the final mapping obtained in the
#    forward pass.
# 
# 8. **Improved initial mapping**: At the end of the backward pass, a new initial mapping
#    :math:`\pi_0` is obtained. This initial mapping is typically better than the original random
#    initialization.
# 
# In practice, SABRE may repeat multiple forward and backward passes (usually around 3 iterations),
# gradually improving the mapping until no significant improvement is observed.
# 
# The SABRE algorithm has a polynomial computational complexity of approximately
# :math:`\mathcal{O}(Tm^2 n)` where :math:`T` is the total number of two-qubit gates in the quantum
# circuit. This is significantly better than the exponential scaling of an exact solution. However, as
# the number of qubits in superconducting processors continues to increase, the runtime of the SABRE
# algorithm also grows accordingly.
# 
# Machine learning models could offer better scalability while maintaining high accuracy. In this
# demo, we consider supervised learning. Supervised learning requires a dataset with input data and
# corresponding labels to train the model. This is the purpose of QMapDataset, which provides datasets
# and tools to generate them for solving the qubit mapping problem using supervised machine learning
# models. QMapDataset is available in PennyLane. We will explore its features in the next section.
# 
# **3. QMapDataset**
# ------------------
# 
# QMapDataset is a dataset generator for the qubit mapping problem, designed for IBM quantum devices.
# The dataset is built from three main components: the hardware, the circuits, and the mapping.
# 
# **For the hardware**, First, we choose an IBM quantum processor. To make the dataset more diverse,
# the code can randomly permute the properties of the qubits and gates (data augmentation). We can
# control how often these permutations are applied.
# 
# Then, the code collects all the relevant hardware information:
# 
# -  physical qubit properties (:math:`T_1`, :math:`T_2`, readout error)
# -  single-qubit gate errors for each qubit
# -  two-qubit gate errors for each connected pair of qubits
# 
# **For the circuits**, These are sampled either from:
# 
# -  random quantum circuits
# -  well-known circuits (QPE, Grover, QFT, GHZ state preparation etc.)
# 
# By default, the choice is split 50–50 between random and well-known circuits.
# 
# For random quantum circuits, the code also sample: \* the number of logical qubits \* the circuit
# depth
# 
# The number of logical qubits is chosen between 3 and the number of physical qubits available on the
# hardware. The depth follows this distribution: 25% shallow (5–15 layers), 40% medium (16–50),
# 30% deep (51–150), and 5% very deep circuits (151–400).
# 
# To further enrich the dataset, we apply random permutations of qubits to the well-known circuits
# (since there are only a few of them). After that, the code collects:
# 
# -  Single-qubit gate counts for each logical qubit
# -  Two-qubit gate counts for each pair of logical qubits
# 
# Finally, **for the mapping**, the code uses the generated hardware and circuits to compute the qubit
# mapping with an advanced version of the SABRE algorithm (optimization_level = 3 in Qiskit).
# 
# If you want more details, you can have a look at the GitHub repo: `rscadrien/QMapDataset: Generator
# of dataset for qubit mapping <https://github.com/rscadrien/QMapDataset>`__.
# 
# The good news is that you do not need to generate this dataset using QMapDataset yourself! We have
# already prepared datasets for three different types of IBM quantum processors: Eagle 3, Heron 1, and
# Heron 2. Each dataset contains 10,000 samples, where each sample consists of a circuit, hardware
# description, and mapping.
# 
# Let’s see how to use these datasets with PennyLane and extract their information.
# 
# The first step is to load a dataset from the Xanadu server. For example, to load the dataset for the
# Eagle 3 architecture:

import pennylane as qp
[ds] = qp.data.load("other", name="qubit-mapping-eagle3")

######################################################################
# For the Heron 1 and Heron 2 architectures, you only need to replace the name argument with
# "qubit-mapping-heron1" or "qubit-mapping-heron2", respectively.
# 
# Each dataset is divided into three attributes: circuits, hardware, and mappings, following the
# structure discussed earlier. For example, to access the first sample:
# 

sample0_circuit = ds.circuits[0]
sample0_hardware = ds.hardware[0]
sample0_mapping = ds.mappings[0]

######################################################################
# Each of these is a dictionary containing the features of the circuit, hardware, or mapping. For
# example, to access the single-qubit and two-qubit gate counts of the quantum circuit:
# 

Single_qubit_count = sample0_circuit["single_qubit_counts"]
Two_qubit_count = sample0_circuit["two_qubit_counts"]

######################################################################
# To retrieve :math:`T_1`, :math:`T_2`, and the two-qubit gate error from the hardware data:
# 

T_1 = sample0_hardware["T1"]
T_2 = sample0_hardware["T2"]
Two_qubit_error = sample0_hardware["ecr"]

######################################################################
# Finally, to obtain the mapping (which can serve as the label in a supervised learning model):
# 

mapping = sample0_mapping["final_mapping"]

######################################################################
# Detailed information about all available features and how to access them can be found in the "Data"
# section of the Qubit Mapping Eagle 3 page on the PennyLane website. (`Qubit Mapping Eagle
# 3 <https://pennylane.ai/datasets/qubit-mapping-eagle3>`__ ).
# 
# **4. Building a machine learning model to solve the qubit mapping problem**
# --------------------------------------------------------------------------
# 
# In this session, we will show an example of how to preprocess data from QMapDataset and how to build
# a machine learning model that can solve the qubit mapping problem.
# 
# **4.1 Preprocessing**
# ~~~~~~~~~~~~~~~~~~~~~
# 
# The qubit mapping problem strongly depends on the connectivity between the physical qubits in the
# given hardware, as well as the connectivity between the logical qubits in the circuits (represented
# by two-qubit gate counts). Therefore, a graph is a natural representation for solving the qubit
# mapping problem. We will see how to construct graphs for both the circuits and the hardware using
# the QMapDataset available in PennyLane.
# 
# A graph is a way of showing relationships. It is made up of points, called nodes (or vertices), that
# are connected by lines, called edges. Both nodes and edges can also have extra information attached
# to them, called features.
# 
# For the **circuit**, the nodes represent the logical qubits, and edges connect pairs of logical
# qubits between which two-qubit gates are applied in the quantum circuit. The features of each node
# are the counts of different single-qubit gates applied to the corresponding logical qubit. The
# features of each edge are the counts of different two-qubit gates applied between the two logical
# qubits. Since the relative values of these features are what matter, we standardize them. This is
# the code to build the graph for the quantum circuit of the QMapDataset:
# 
import numpy as np
import torch
from torch_geometric.data import Data
def build_circuit_graph (circuit_data):
    # -----------------------------
    # STEP 1: Extract and scale single-qubit gate counts
    # Each function returns a vector of size (n_qubits,)
    # representing how often each gate was applied per qubit
    # -----------------------------
    rz_counts_scaled = get_single_qubit_counts(circuit_data, "rz")
    sx_counts_scaled = get_single_qubit_counts(circuit_data, "sx")
    x_counts_scaled = get_single_qubit_counts(circuit_data, "x")
    # -----------------------------
    # STEP 2: Extract and scale two-qubit gate counts
    # These describe interactions between pairs of qubits
    # -----------------------------
    ecr_scaled = get_two_qubit_counts(circuit_data,'ecr')
    # -----------------------------
    # STEP 3: Build node feature matrix
    # Each node = a logical qubit
    # Features = [rz count, sx count, x count]
    # Shape: (n_qubits, 3)
    # -----------------------------
    X =torch.tensor(
    list(zip(rz_counts_scaled, sx_counts_scaled, x_counts_scaled)),
    dtype=torch.float
    )
    # -----------------------------
    # STEP 4: Build graph connectivity (edges)
    # edge_index: which qubits are connected
    # edge_attr: weight/feature of each edge (scaled gate counts)
    # -----------------------------
    ecr_index, ecr_attr = edge_info(ecr_scaled)
    # -----------------------------
    # STEP 5: Create PyTorch Geometric graph object
    # x -> node features
    # edge_index -> connectivity (COO format)
    # edge_attr -> edge features
    # -----------------------------
    G_circuit = Data(
    x=X,
    edge_index=ecr_index,
    edge_attr=ecr_attr,
    )
    return G_circuit
def get_single_qubit_counts(data, gate_name):
    # Number of logical qubits in the circuit
    n = data["n_logical_qubits"]
    # Try to retrieve counts for the given gate type
    counts = data.get("single_qubit_counts", {}).get(gate_name, None)
    # If no data exists, return a zero vector
    if counts is None:
        return np.zeros(n)
    else:
    # Otherwise normalize counts across qubits (z-score normalization)
        return standard_scaling_across_qubits(np.array(counts))
def get_two_qubit_counts(data, gate_name):
    # Safely retrieve list of two-qubit gate interactions
    counts = data.get("two_qubit_counts", {}).get(gate_name, [])
    # If no interactions exist, return empty list
    if not counts:
        return counts
    else:
    # Otherwise scale the edge weights
      return scaled_two_qubit_list(counts)
def standard_scaling_across_qubits(vector):
    # Compute mean and standard deviation for normalization
    mean = vector.mean()
    std = vector.std()
    # Avoid division by zero if all values are identical
    if std == 0 :
        vector_scaled = vector
    else:
        vector_scaled = (vector - mean)/std
    return vector_scaled
def scaled_two_qubit_list(two_data):
    # Extract raw edge weights from dictionary list
    values = np.array([d['value']for d in two_data], dtype=float)
    # Normalize weights
    scaled_values = standard_scaling_across_qubits(values)
    # Reattach scaled values back into original edge dictionaries
    two_gate_scaled = [
    {**d, 'value':v_scaled}
    for d, v_scaled in zip(two_data, scaled_values)
    ]
    return two_gate_scaled
def edge_info(gate_dict):
    # Lists to store graph connectivity
    edge_index = []
    edge_attr = []
    # Convert list of edge dictionaries into tensor format
    for gate in gate_dict:
        i,j,w = gate["row"], gate["col"], gate["value"]
        edge_index.append([i,j])
        edge_attr.append([w])
    # Convert to PyTorch tensors
    # edge_index shape: [2, num_edges]
    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return edge_index, edge_attr

######################################################################
# For example, if this code is applied for the circuit of the 27th sample, we obtain:
# 

Graph_circuit = build_circuit_graph(ds.circuits[26])

######################################################################
# Using the networkx package, we can visualize the graph using the following graph:
# 

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
# Convert PyG Data object to NetworkX graph
G = to_networkx(
    Graph_circuit,
    edge_attrs=["edge_attr"],  # keep CX counts
    node_attrs=["x"],          # keep node features
    to_undirected=False
)
edge_labels = {
    (u, v): f"{d['edge_attr'][0]:.2f}"
    for u, v, d in G.edges(data=True)
}
nx.draw(
    G,
    nx.spring_layout(G, seed=42),
    with_labels=True,
    node_size=800,
)
nx.draw_networkx_edge_labels(
    G,
    nx.spring_layout(G, seed=42),
    edge_labels=edge_labels
)
plt.show()

######################################################################
# We obtain the following graph: |My graph|
# 
# .. figure:: ../_static/demonstration_assets/qubit-mapping/graph_circuit.png
#   :alt: My graph
#   :width: 95%
#   :align: center
# 
# For the **hardware**, the nodes represent the physical qubits, and edges connect pairs of physical
# qubits that are directly connected in the device architecture. The features of each node are the
# :math:`T_1` and :math:`T_2` coherence times, readout error, and single-qubit gate errors for each
# physical qubit. The features of each edge are the errors of the different two-qubit gates applied
# between connected physical qubits. As with the circuits, the feature values are standardized. The
# code to build the graph for the hardware is pretty similar to the one for the circuit: 

def build_hardware_graph(hardware_data):
    # Scaling single qubit properties and gate
    T1_scaled = standard_scaling_across_qubits(np.array(hardware_data['T1']))
    T2_scaled = standard_scaling_across_qubits(np.array(hardware_data['T2']))
    readout_error_scaled = standard_scaling_across_qubits(np.array(hardware_data['readout_error']))
    sx_error_scaled = standard_scaling_across_qubits(np.array(hardware_data['sx_error']))
    x_error_scaled = standard_scaling_across_qubits(np.array(hardware_data['x_error']))
    #Scaling two qubit gate properties
    ecr_error_scaled = scaled_two_qubit_list(hardware_data['ecr'])
    #Build the node feature matrix
    X = torch.tensor(
    list(zip(T1_scaled, T2_scaled, readout_error_scaled, sx_error_scaled, x_error_scaled)),
    dtype=torch.float
    )
    #Build the edge index and attributes matrix
    cx_index, cx_attr = edge_info(ecr_error_scaled)
    #Build the graph
    G_hardware = Data(
    x=X,
    edge_index=cx_index,
    edge_attr=cx_attr,
    )
    return G_hardware

Graph_hardware = build_hardware_graph(ds.hardware[26])

######################################################################
# We can again visualize the graph using networkx:
# 
# Draw graph
G_hardware = to_networkx(
    Graph_hardware,
    edge_attrs=["edge_attr"],  # keep CX counts
    node_attrs=["x"],          # keep node features
    to_undirected=False
)

plt.figure(figsize=(10, 10))
pos = nx.spring_layout(
    G_hardware,
    seed=42,
    k=1 / np.sqrt(len(G_hardware)),
    iterations=200
)
nx.draw_networkx_edges(G_hardware, pos, alpha=0.2, width=0.5)
nx.draw_networkx_nodes(G_hardware, pos, node_size=40)
plt.axis("off")
plt.savefig("graph_hardware.png", dpi=300, bbox_inches="tight")
plt.show()

######################################################################
# 
# .. figure:: ../_static/demonstration_assets/qubit-mapping/graph_hardware.png
#   :alt: My graph
#   :width: 95%
#   :align: center
# 
#    My graph
# 
# **4.2 Machine learning model**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# We will show how to build a model that can learn qubit mapping. For this model, we will use an
# architecture adapted to the graph structure of the input: a Graph Neural Network (GNN). The key
# principle behind GNNs is that a node learns by aggregating information from its neighbors. A GNN
# typically works in layers, and each layer is mostly composed of three steps:
# 
# -  Message passing : Each node receives information from its neighbors
# -  Aggregation: Combine neighbor information (e.g., sum, mean, max)
# -  Update: Use a neural network to update the node’s embedding.
# 
# In our case, we will use two separate GNNs: one for the quantum circuit and one for the hardware.
# After several GNN layers, we obtain updated embedding vectors for the nodes of both graphs,
# :math:`h_L^{(i)}` and :math:`h_P^{(j)}`. (:math:`h_L^{(i)}` is the vector of the i th node of the
# quantum circuit graph while :math:`h_P^{(j)}` is the vector of the jth node of the hardware graph).
# 
# We then compare these embeddings using a dot product:
# 
# :math:`S_{ij}=h_L^{(i)}\cdot h_P^{(j)}`
# 
# The matrix element :math:`S_{ij}` represents the logit that the logical qubit :math:`lq_i` is mapped
# to the physical qubit :math:`pq_j`. The probability that logical qubit k maps to physical qubit j is
# calculated from the logit:
# 
# :math:`p_{ij}=\frac{\text{exp}(S_{ij})}{\sum_{j^\prime} \text{exp}(S_{ij}^\prime)}`
# 
# During training, the matrix :math:`p_{ij}` should ideally converge to the assignment matrix
# :math:`A_{ij}` presented in Section 2, where each row contains exactly one nonzero element equal to
# 1. After training, during inference, the strict assignment constraints are enforced by applying the
# Hungarian algorithm. Unfortunately, this algorithm is not differentiable and therefore cannot be
# used during training.The code for this architecture is as follows :
# 

import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import NNConv
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.optimize import linear_sum_assignment

class QubitMapping_dot_Model(nn.Module):
    def __init__(self, node_c_dim, node_h_dim, edge_c_dim, edge_h_dim, hidden_dim):
        super().__init__()
        # === GNN for the LOGICAL CIRCUIT graph ===
        # These layers learn embeddings for logical qubits based on circuit structure
        # First layer: edge-conditioned convolution
        # Edge features (e.g., gate types) generate weights dynamically
        self.nn_c1 = make_edge_mlp(edge_c_dim, node_c_dim * hidden_dim)
        self.gnn_c1 = NNConv(node_c_dim, hidden_dim, nn=self.nn_c1, aggr='add')
        # Second layer: refines embeddings
        self.nn_c2 = make_edge_mlp(edge_c_dim, hidden_dim * hidden_dim)
        self.gnn_c2 = NNConv(hidden_dim, hidden_dim, nn=self.nn_c2, aggr='add')
        # === GNN for the HARDWARE graph ===
        # These layers learn embeddings for physical qubits based on hardware connectivity
        self.nn_h1 = make_edge_mlp(edge_h_dim, node_h_dim * hidden_dim)
        self.gnn_h1 = NNConv(node_h_dim, hidden_dim, nn=self.nn_h1, aggr='add')
        self.nn_h2 = make_edge_mlp(edge_h_dim, hidden_dim * hidden_dim)
        self.gnn_h2 = NNConv(hidden_dim, hidden_dim, nn=self.nn_h2, aggr='add')
    def forward(self, G_circuit, G_hardware):
        # === Step 1: Compute embeddings for LOGICAL qubits ===
        # Each node becomes a vector summarizing its role in the circuit
        x_c = F.relu(self.gnn_c1(
            G_circuit.x,              # node features
            G_circuit.edge_index,    # connectivity
            G_circuit.edge_attr      # edge features (e.g., gate info)
        ))
        h_L = F.relu(self.gnn_c2(
            x_c,
            G_circuit.edge_index,
            G_circuit.edge_attr
        ))
        # === Step 2: Compute embeddings for PHYSICAL qubits ===
        # Each node becomes a vector describing hardware properties
        x_h = F.relu(self.gnn_h1(
            G_hardware.x,
            G_hardware.edge_index,
            G_hardware.edge_attr
        ))
        h_P = F.relu(self.gnn_h2(
            x_h,
            G_hardware.edge_index,
            G_hardware.edge_attr
        ))
        # === Step 3: Compute compatibility between logical and physical qubits ===
        prob_list = []
        # Number of graphs in the batch
        num_graphs = G_circuit.batch.max().item() + 1
        # Loop over each graph in the batch
        for i in range(num_graphs):
            # --- Select nodes belonging to graph i ---
            # (because batching mixes all graphs together)
            mask_c = (G_circuit.batch == i)
            mask_h = (G_hardware.batch == i)
            # Extract embeddings for this specific graph
            h_L_i = h_L[mask_c]  # logical qubits
            h_P_i = h_P[mask_h]  # physical qubits
            # --- Compute compatibility matrix ---
            # Dot product measures how well each pair matches
            # Shape: [num_logical_qubits, num_physical_qubits]
            S_i = torch.matmul(h_L_i, h_P_i.T)
            if self.training:
                # During training:
                # return raw scores (used in loss function)
                prob_i = S_i
                prob_list.append(prob_i)
            else:
                # During inference:
                # find the BEST one-to-one assignment
                # Move to CPU + numpy for Hungarian algorithm
                S_cpu = S_i.detach().cpu().numpy()
                # Hungarian algorithm finds assignment maximizing total score
                row_ind, col_ind = linear_sum_assignment(-S_cpu)
                # Build mapping: logical qubit -> physical qubit
                mapping_i = torch.zeros(h_L_i.size(0), dtype=torch.long)
                mapping_i[row_ind] = torch.tensor(col_ind, dtype=torch.long)
                prob_list.append(mapping_i)
        # === Step 4: Merge results from all graphs ===
        if self.training:
            # Concatenate score matrices
            # Shape: [total_logical_nodes_in_batch, num_physical_nodes_per_graph]
            prob = torch.cat(prob_list, dim=0)
            return prob
        else:
            # Concatenate mappings
            # Shape: [total_logical_nodes_in_batch]
            mapping = torch.cat(prob_list, dim=0)
            return mapping

def make_edge_mlp(in_dim, out_dim, hid_dim=32):
    """
    Create a small multilayer perceptron (MLP) used to process edge features.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hid_dim),
        nn.ReLU(),
        nn.Linear(hid_dim, out_dim)
    )


######################################################################
# **4.3 Loss function and metric**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# We need to have a loss function to minimize for the training of the model.
# 
# For each logical qubit k, we know the correct assignment (from the training data), which we denote
# as :math:`k_c`. This is the physical qubit that the logical qubit should be mapped to.
# 
# The loss for the logical qubit k is then defined as:
# 
# :math:`L_k=\log(p_{k,k_c})`
# 
# Finally, the total loss is computed by averaging over all logical qubits :
# 
# :math:`L = \frac{1}{n} \sum_{k=1}^n -\log(p_{k,k_c})`
# 
# This is known as the cross-entropy loss. With Pytorch, the cross entropy loss can be calculated by
# using nn.CrossEntropyLoss():
# 

loss_fn = nn.CrossEntropyLoss()

######################################################################
# This loss encourages the model to assign high probability to the correct qubit mapping and to assign
# low probability to incorrect mappings. However, it does not enforce one-to-one mapping and does not
# prevent collisions (two logical qubits choosing the same physical qubit). It is the reason that it
# is better to use the Hungarian algorithm during inference.
# 
# In addition to the loss function, there are some metrics that help to evaluate the quality of the
# model and are easier to interpret than the loss function. We consider two different metrics: \*
# logical qubit-level accuracy \* full mapping accuracy
# 
# To understand the difference between them, let us consider an example with 5 logical qubits and 5
# physical qubits. The correct mapping (from the training data) is [2,1,4,3,0] and the model predicts
# [2,0,4,3,1].
# 
# For the logical qubit-level accuracy, we simply count how many individual assignments are correct.
# In this example, 3 out of 5 qubits are correctly mapped. So the accuracy is 60%.
# 
# For the full mapping accuracy, the prediction is considered correct only if the entire assignment is
# exactly correct. In this example, the predicted mapping is not identical to the ground truth. So the
# full mapping is incorrect. This metric is therefore much stricter and harder to improve.
# 
# You could ask: Why use both metrics? These two metrics serve different goals. The logical
# qubit-level accuracy is useful to monitor progress during training. The full mapping accuracy
# reflects the actual goal of the problem: producing a correct global qubit mapping. During training,
# the logical-level accuracy improves first and helps track learning progress while the full-mapping
# accuracy becomes meaningful only later, when the model is already quite accurate. Note that the
# Hungarian algorithm significantly improves the full-mapping accuracy. As a result, the full-mapping
# accuracy is often higher on the validation set than on the training set.
# 
# These metrics are calculated with the following code for the training and the validation dataset:
# 

def batch_graph_metrics_training(logit, qu_map, batch):
    """
    Compute per-graph metrics in a single pass:
      - circuit-averaged node accuracy
      - exact-mapping accuracy
    Args:
        logit: [N_total_nodes, N_P] predicted logit
        qu_map: [N_total_nodes] ground-truth physical qubits
        batch: [N_total_nodes] graph indices
    Returns:
        node_acc_sum: sum of node-level accuracies per graph
        exact_sum: number of perfectly mapped graphs
        num_graphs: number of graphs in the batch
    """
    pred = logit.argmax(dim=1)
    num_graphs = batch.max().item() + 1
    node_acc_sum = 0.0
    exact_sum = 0
    for g in range(num_graphs):
        mask = (batch == g)
        correct_nodes = (pred[mask] == qu_map[mask]).sum().item()
        n_nodes = mask.sum().item()
        node_acc_sum += correct_nodes / n_nodes
        if correct_nodes == n_nodes:
            exact_sum += 1
    return node_acc_sum, exact_sum, num_graphs
def batch_graph_metrics_val(pred, qu_map, batch):
    """
    Compute per-graph metrics in a single pass:
      - circuit-averaged node accuracy
      - exact-mapping accuracy
    Args:
        pred: [N_total_nodes] predicted mapping
        qu_map: [N_total_nodes] ground-truth physical qubits
        batch: [N_total_nodes] graph indices
    Returns:
        node_acc_sum: sum of node-level accuracies per graph
        exact_sum: number of perfectly mapped graphs
        num_graphs: number of graphs in the batch
    """
    num_graphs = batch.max().item() + 1
    node_acc_sum = 0.0
    exact_sum = 0
    for g in range(num_graphs):
        mask = (batch == g)
        correct_nodes = (pred[mask] == qu_map[mask]).sum().item()
        n_nodes = mask.sum().item()
        node_acc_sum += correct_nodes / n_nodes
        if correct_nodes == n_nodes:
            exact_sum += 1
    return node_acc_sum, exact_sum, num_graphs

######################################################################
# **4.3 Training Loop**
# ~~~~~~~~~~~~~~~~~~~~~
# 
# We now have all the elements to build our training loop! Here it is the code:
# 

def train_loop(model, optimizer, loss_fn, loader_train, loader_valid, n_epoch, device):
    """
    Training and validation loop for qubit mapping model.
    Returns:
        loss_hist_train, accuracy_hist_train, exact_accuracy_hist_train,
        accuracy_hist_valid, exact_accuracy_hist_valid
    """
    loss_hist_train = [0.0] * n_epoch
    accuracy_hist_train = [0.0] * n_epoch
    exact_accuracy_hist_train = [0.0] * n_epoch
    accuracy_hist_valid = [0.0] * n_epoch
    exact_accuracy_hist_valid = [0.0] * n_epoch
    for epoch in range(n_epoch):
        # ------------ TRAIN ------------------
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_exact = 0
        total_graphs = 0
        for G_c, G_h,  qu_map in loader_train:
            G_c = G_c.to(device)
            G_h = G_h.to(device)
            qu_map = qu_map.to(device)
            optimizer.zero_grad()
            logit = model(G_c, G_h)
            loss = loss_fn(logit, qu_map)
            loss.backward()
            optimizer.step()
            #Metrics
            acc_sum, exact_sum, n_graphs = batch_graph_metrics_training(
                logit, qu_map, G_c.batch
            )
            total_loss += loss.item() * n_graphs
            total_acc += acc_sum
            total_exact += exact_sum
            total_graphs += n_graphs
        #Store epoch metrics
        loss_hist_train[epoch] = total_loss  / total_graphs
        accuracy_hist_train[epoch] = total_acc / total_graphs
        exact_accuracy_hist_train[epoch] = total_exact / total_graphs
        #VALIDATION
        model.eval()
        total_acc = 0.0
        total_exact = 0
        total_graphs = 0
        with torch.no_grad():
            for G_c_val, G_h_val, qu_map_val in loader_valid:
                G_c_val = G_c_val.to(device)
                G_h_val = G_h_val.to(device)
                qu_map_val = qu_map_val.to(device)
                mapping = model(G_c_val, G_h_val)
                acc_sum, exact_sum, n_graphs = batch_graph_metrics_val(
                mapping, qu_map_val, G_c_val.batch
                )
                total_acc += acc_sum
                total_exact += exact_sum
                total_graphs += n_graphs
        accuracy_hist_valid[epoch] = total_acc / total_graphs
        exact_accuracy_hist_valid[epoch] = total_exact / total_graphs
        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {loss_hist_train[epoch]:.4f} | "
            f"Train Acc: {accuracy_hist_train[epoch]:.4f} | "
            f"Train Exact Acc: {exact_accuracy_hist_train[epoch]:.4f} | "
            f"Val Acc: {accuracy_hist_valid[epoch]:.4f} | "
            f"Val Exact Acc: {exact_accuracy_hist_valid[epoch]:.4f}"
        )
    return (loss_hist_train,
            accuracy_hist_train, exact_accuracy_hist_train,
            accuracy_hist_valid, exact_accuracy_hist_valid)

######################################################################
# The training can be run by defining the model, the optimizer, the loss function, the sample loaders,
# the number of epochs and the device.
# 
# For the optimizer, we can for example using an Adam optimizer which is an efficient optimizer which
# adapts the learning rate for each parameter based on estimates of the first and second moments (mean
# and variance) of gradients.
# 

######################################################################
# With this architecture applied to a qubit mapping problem on hardware with 5 physical qubits (IBM
# Athens), a full-mapping accuracy of 90% can be obtained. This is not yet perfect, but it shows that
# a machine learning model can learn and solve the qubit mapping problem.
# 
# **5. Conclusion**
# -----------------
# 
# In this demo, we introduced the qubit mapping problem. Qubit mapping is a fundamental problem for
# hardwares with limited connectivity, such as superconducting devices. Unfortunately, the qubit
# mapping problem is NP-hard. One way to find approximate solutions is to use heuristic methods, such
# as the SABRE algorithm presented in this demo.
# 
# It remains an open question whether a machine learning model can efficiently solve this problem and
# have better scaling than the SABRE algorithm. To explore this idea, we need a dataset, which is the
# purpose of QMapDataset. Three datasets for the IBM architectures Eagle R3, Heron r1, and Heron r2,
# respectively, are now available in PennyLane, and we showed how to access them.
# 
# Finally, we presented a Graph Neural Network model to solve the qubit mapping problem. This model
# shows encouraging results on small hardware. The next challenge is how to scale it to larger
# hardware, such as Eagle R3 (127 qubits), Heron r1 (133 qubits), and Heron r2 (156 qubits). Maybe the
# person who finds the solution will be you, using the dataset available in PennyLane!
# 
