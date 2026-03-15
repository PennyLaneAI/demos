import pennylane as qml
qml.capture.enable()
from pennylane import numpy as np

""" Based on the SimplifiedTwoDesign template from pennylane
https://docs.pennylane.ai/en/latest/code/api/pennylane.SimplifiedTwoDesign.html
as proposed in `Cerezo et al. (2021) <https://doi.org/10.1038/s41467-021-21728-w>`_.
but changin the Y-rotations for a random choice of {X, Y, Z}-rotations.
"""

def build_ansatz(initial_layer_weights, weights, wires, gate_sequence=None):

    n_layers = qml.math.shape(weights)[0]
    op_list = []

    # initial rotations
    for i in range(len(wires)):
        op_list.append(qml.RY(initial_layer_weights[i], wires=wires[i]))

    # generating the rotation sequence
    if gate_sequence is None:
        gate_sequence = generate_random_gate_sequence(qml.math.shape(weights))

    # repeated layers
    for layer in range(n_layers):

        # even layer of entanglers
        even_wires = [wires[i : i + 2] for i in range(0, len(wires) - 1, 2)]
        for i, wire_pair in enumerate(even_wires):
            op_list.append(qml.CZ(wires=wire_pair))
            op_list.append(gate_sequence[layer, i, 0].item()(weights[layer, i, 0], wires=wire_pair[0]))
            op_list.append(gate_sequence[layer, i, 1].item()(weights[layer, i, 1], wires=wire_pair[1]))
            # op_list.append(qml.RX(weights[layer, i, 0], wires=wire_pair[0]))
            # op_list.append(qml.RX(weights[layer, i, 1], wires=wire_pair[1]))

        # odd layer of entanglers
        odd_wires = [wires[i : i + 2] for i in range(1, len(wires) - 1, 2)]
        for i, wire_pair in enumerate(odd_wires):
            op_list.append(qml.CZ(wires=wire_pair))
            op_list.append(gate_sequence[layer, len(wires) // 2 + i, 0].item()(weights[layer, len(wires) // 2 + i, 0], wires=wire_pair[0]))
            op_list.append(gate_sequence[layer, len(wires) // 2 + i, 1].item()(weights[layer, len(wires) // 2 + i, 1], wires=wire_pair[1]))
            # op_list.append(qml.RX(weights[layer, len(wires) // 2 + i, 0], wires=wire_pair[0]))
            # op_list.append(qml.RX(weights[layer, len(wires) // 2 + i, 1], wires=wire_pair[1]))

    return op_list

def generate_random_gate_sequence(shape):
    gate_set = [qml.RX, qml.RY, qml.RZ]
    return np.random.choice(gate_set, size=shape)

def get_parameter_shape(n_layers, n_wires):
        if n_wires == 1:
            return [(n_wires,), (n_layers,)]
        return [(n_wires,), (n_layers, n_wires - 1, 2)]

