r"""

.. _adjoint_differentiation_benchmarking:

Adjoint Differentiation
=======================

.. meta::
    :property="og:description": Benchmarking file for adjoint diff demonstration.
    :property="og:image": https://pennylane.ai/qml/_static/demo_thumbnails/opengraph_demo_thumbnails/code.png

"""

##############################################################################
# This page is supplementary material to the
# `Adjoint Differentiation <https://pennylane.ai/qml/demos/tutorial_adjoint_diff>`__
# demonstration.  The below script produces the benchmarking images used.

import timeit
import matplotlib.pyplot as plt
import pennylane as qml
qml.capture.enable()
import pennylane.numpy as pnp

plt.style.use("bmh")

n_samples = 5


def get_time(qnode, params):
    globals_dict = {"grad": qml.grad, "circuit": qnode, "params": params}
    return timeit.timeit("grad(circuit)(params)", globals=globals_dict, number=n_samples)


def wires_scaling(n_wires, n_layers):
    rng = pnp.random.default_rng(12345)

    t_adjoint = []
    t_ps = []
    t_backprop = []

    def circuit(params, wires):
        qml.StronglyEntanglingLayers(params, wires=range(wires))
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

    for i_wires in n_wires:
        dev = qml.device("lightning.qubit", wires=i_wires)
        dev_python = qml.device("default.qubit", wires=i_wires)

        circuit_adjoint = qml.QNode(lambda x: circuit(x, wires=i_wires), dev, diff_method="adjoint")
        circuit_ps = qml.QNode(lambda x: circuit(x, wires=i_wires), dev, diff_method="parameter-shift")
        circuit_backprop = qml.QNode(lambda x: circuit(x, wires=i_wires), dev_python, diff_method="backprop")

        # set up the parameters
        param_shape = qml.StronglyEntanglingLayers.shape(n_wires=i_wires, n_layers=n_layers)
        params = rng.normal(size=pnp.prod(param_shape), requires_grad=True).reshape(param_shape)

        t_adjoint.append(get_time(circuit_adjoint, params))
        t_backprop.append(get_time(circuit_backprop, params))
        t_ps.append(get_time(circuit_ps, params))

    return t_adjoint, t_backprop, t_ps


def layers_scaling(n_wires, n_layers):
    rng = pnp.random.default_rng(12345)

    dev = qml.device("lightning.qubit", wires=n_wires)
    dev_python = qml.device("default.qubit", wires=n_wires)

    t_adjoint = []
    t_ps = []
    t_backprop = []

    def circuit(params):
        qml.StronglyEntanglingLayers(params, wires=range(n_wires))
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

    circuit_adjoint = qml.QNode(circuit, dev, diff_method="adjoint")
    circuit_ps = qml.QNode(circuit, dev, diff_method="parameter-shift")
    circuit_backprop = qml.QNode(circuit, dev_python, diff_method="backprop")

    for i_layers in n_layers:
        # set up the parameters
        param_shape = qml.StronglyEntanglingLayers.shape(n_wires=n_wires, n_layers=i_layers)
        params = rng.normal(size=pnp.prod(param_shape), requires_grad=True).reshape(param_shape)

        t_adjoint.append(get_time(circuit_adjoint, params))
        t_backprop.append(get_time(circuit_backprop, params))
        t_ps.append(get_time(circuit_ps, params))

    return t_adjoint, t_backprop, t_ps


if __name__ == "__main__":

    wires_list = [3, 6, 9, 12, 15]
    n_layers = 6
    adjoint_wires, backprop_wires, ps_wires = wires_scaling(wires_list, n_layers)

    layers_list = [3, 9, 15, 21, 27]
    n_wires = 12
    adjoint_layers, backprop_layers, ps_layers = layers_scaling(n_wires, layers_list)

    # Generating the graphic
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(wires_list, adjoint_wires, ".-", label="adjoint")
    ax1.plot(wires_list, ps_wires, ".-", label="parameter-shift")
    ax1.plot(wires_list, backprop_wires, ".-", label="backprop")

    ax1.legend()

    ax1.set_xlabel("Number of wires")
    ax1.set_xticks(wires_list)
    ax1.set_ylabel("Log Time")
    ax1.set_yscale("log")
    ax1.set_title("Scaling with wires")

    ax2.plot(layers_list, adjoint_layers, ".-", label="adjoint")
    ax2.plot(layers_list, ps_layers, ".-", label="parameter-shift")
    ax2.plot(layers_list, backprop_layers, ".-", label="backprop")

    ax2.legend()

    ax2.set_xlabel("Number of layers")
    ax2.set_xticks(layers_list)
    ax2.set_ylabel("Log Time")
    ax2.set_yscale("log")
    ax2.set_title("Scaling with layers")

    plt.savefig("scaling.png")

##############################################################################
#
# .. figure:: ../_static/demonstration_assets/adjoint_diff/scaling.png
#     :width: 80%
#     :align: center
#
