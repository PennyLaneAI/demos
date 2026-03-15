r"""Dropout for Quantum Neural Networks
===================================
"""

######################################################################
# Are you struggling with overfitting while training Quantum Neural Networks (QNNs)?
# 
# In this demo, we show how to exploit the quantum version of the dropout technique to avoid the problem of
# overfitting in overparametrized QNNs. What follows is based on the paper “A General
# Approach to Dropout in Quantum Neural Networks” by F. Scala, et al. [#dropout]_.
#
# .. figure:: ../_static/demonstration_assets/quantum_dropout/socialthumbnail_large_QuantumDropout_2024-03-07.png
#    :align: center
#    :width: 60%
#    :target: javascript:void(0)
#
#
# What is overfitting and dropout?
# ---------------------------------
#
# Neural Networks (NNs) usually require highly flexible models with lots of trainable parameters in
# order to *learn* a certain underlying function (or data distribution).
# However, being able to learn with low in-sample error is not enough; *generalization* — the ability to provide
# good predictions on previously unseen data — is also desirable.
#
# Highly expressive models may suffer from **overfitting**, which means that
# they are trained too well on the training data, and as a result perform poorly on new, unseen
# data. This happens because the model has learned the noise in the training data, rather than the
# underlying pattern that is generalizable to new data.
#
# **Dropout** is a common technique for classical Deep Neural Networks (DNNs) preventing computational units
# from becoming too specialized and reducing the risk of overfitting [#Hinton2012]_, [#Srivastava2014]_. It consists of randomly removing
# neurons or connections *only during training* to block the flow of information. Once the
# model is trained, the DNN is employed in its original form.
#
# Why dropout for Quantum Neural Networks?
# ----------------------------------------
#
# Recently, it has been shown that the use of overparametrized QNN models
# changes the optimization landscape by removing lots of local minima [#Kiani2020]_, [#Larocca2023]_. On the one hand, this increased number of
# parameters leads to faster and easier training, but on the other hand, it may drive
# the model to overfit the data. This is also strictly related to the `repeated encoding <https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series/>`__ of classical
# data to achieve nonlinearity in the computation. This is why, inspired from classical DNNs, one
# can think of applying some sort of dropout to QNNs. This would correspond to randomly dropping some
# (groups of) parameterized gates during training to achieve better generalization.
#
# Quantum dropout of rotations in a sine regression
# --------------------------------------------------
#
# In this demo we will exploit quantum dropout to avoid overfitting during the regression of noisy
# data originally coming from the *sinusoidal* function. In particular, we will randomly “drop”
# rotations during the training phase. In practice, this will correspond to temporarily setting parameters to a value of 0.
#
# Let’s start by importing Pennylane and ``numpy`` and fixing the random seed for reproducibility:
#

import numpy as np
import pennylane as qml
qml.capture.enable()

seed = 12345
np.random.seed(seed=seed)

######################################################################
# The circuit
# ~~~~~~~~~~~
#
# Now we define the embedding of classical data and the variational ansatz that will then be combined
# to construct our QNN. Dropout will happen inside the variational ansatz. Obtaining dropout with standard
# Pennylane would be quite straightforward by means of some "if statements", but the training procedure
# will take ages. Here we will leverage JAX in order to speed up the training process with
# Just In Time (JIT) compilation. The drawback is that the definition of the variational ansatz becomes a
# little elaborated, since JAX has its own language for conditional statements. For this purpose we
# define two functions ``true_cond`` and ``false_cond`` to work with ``jax.lax.cond```, which is the JAX
# conditional statement. See this `demo <https://pennylane.ai/qml/demos/tutorial_How_to_optimize_QML_model_using_JAX_and_JAXopt/>`__
# for additional insights on how to optimize QNNs with JAX.
#
# Practically speaking, rotation dropout will be performed by passing a list to the ansatz.
# The single qubit rotations are applied depending on the values stored in this list:
# if the value is negative the rotation is dropped (rotation dropout), otherwise it is applied.
# How to produce this list will be explained later in this demo (see the ``make_dropout`` function).

import jax  # require for Just In Time (JIT) compilation
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def embedding(x, wires):
    # Encodes the datum multiple times in the register,
    # employing also nonlinear functions
    assert len(x) == 1  # check feature is 1-D
    for i in wires:
        qml.RY(jnp.arcsin(x), wires=i)
    for i in wires:
        qml.RZ(jnp.arccos(x ** 2), wires=i)


def true_cond(angle):
    # necessary for using an if statement within jitted function
    # exploiting jax.lax.cond
    # if this function is assessed the rotation is dropped
    return 0.0


def false_cond(angle):
    # necessary for using an if statement within jitted function
    # exploiting jax.lax.cond
    # if this function is assessed the rotation is kept
    return angle


def var_ansatz(
    theta, wires, rotations=[qml.RX, qml.RZ, qml.RX], entangler=qml.CNOT, keep_rotation=None
):

    """Single layer of the variational ansatz for our QNN. 
    We have a single qubit rotation per each qubit (wire) followed by 
    a linear chain of entangling gates (entangler). This structure is repeated per each rotation in `rotations` 
    (defining `inner_layers`).
    The single qubit rotations are applied depending on the values stored in `keep_rotation`:
    if the value is negative the rotation is dropped (rotation dropout), otherwise it is applied.

    Params:
    - theta: variational angles that will undergo optimization
    - wires: list of qubits (wires)
    - rotations: list of rotation kind per each `inner_layer`
    - entangler: entangling gate
    - keep_rotation: list of lists. There is one list per each `inner_layer`. 
                    In each list there are indexes of the rotations that we want to apply. 
                    Some of these values may be substituted by -1 value 
                    which means that the rotation gate wont be applied (dropout). 
    """

    # the length of `rotations` defines the number of inner layers
    N = len(wires)
    assert len(theta) == 3 * N
    wires = list(wires)

    counter = 0
    # keep_rotations contains a list per each inner_layer
    for rots in keep_rotation:
        # we cicle over the elements of the lists inside keep_rotation
        for qb, keep_or_drop in enumerate(rots):
            rot = rotations[counter]  # each inner layer can have a different rotation

            angle = theta[counter * N + qb]
            # conditional statement implementing dropout
            # if `keep_or_drop` is negative the rotation is dropped
            angle_drop = jax.lax.cond(keep_or_drop < 0, true_cond, false_cond, angle)
            rot(angle_drop, wires=wires[qb])
        for qb in wires[:-1]:
            entangler(wires=[wires[qb], wires[qb + 1]])
        counter += 1


######################################################################
# And then we define the hyperparameters of our QNN, namely the number of qubits,
# the number of sublayers in the variational ansatz (``inner_layers``) and the resulting
# number of parameters per layer:
#

n_qubits = 5
inner_layers = 3
params_per_layer = n_qubits * inner_layers

######################################################################
# Now we actually build the QNN:
#


def create_circuit(n_qubits, layers):
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device)
    def circuit(x, theta, keep_rot):
        # print(x)
        # print(theta)

        for i in range(layers):
            embedding(x, wires=range(n_qubits))

            keep_rotation = keep_rot[i]

            var_ansatz(
                theta[i * params_per_layer : (i + 1) * params_per_layer],
                wires=range(n_qubits),
                entangler=qml.CNOT,
                keep_rotation=keep_rotation,
            )

        return qml.expval(qml.PauliZ(wires=0))  # we measure only the first qubit

    return circuit


######################################################################
# Let’s have a look at a single layer of our QNN:
#
import matplotlib.pyplot as plt


plt.style.use("pennylane.drawer.plot")  # set pennylane theme, which is nice to see

# create the circuit with given number of qubits and layers
layers = 1
circ = create_circuit(n_qubits, layers=layers)

# for the moment let's keep all the rotations in all sublayers
keep_all_rot = [
    [list(range((n_qubits))) for j in range(1, inner_layers + 1)],
]
# we count the parameters
numbered_params = np.array(range(params_per_layer * layers), dtype=float)
# we encode a single coordinate
single_sample = np.array([0])

qml.draw_mpl(circ, decimals=2,)(single_sample, numbered_params, keep_all_rot)

plt.show()

######################################################################
# We now build the model that we will employ for the regression task.
# Since we want to have an overparametrized QNN, we will add 10 layers and we exploit
# ``JAX`` to speed the training up:
#

layers = 10
qnn_tmp = create_circuit(n_qubits, layers)
qnn_tmp = jax.jit(qnn_tmp)
qnn_batched = jax.vmap(
    qnn_tmp, (0, None, None)
)  # we want to vmap on 0-axis of the first circuit param
# in this way we process in parallel all the inputs
# We jit for faster execution
qnn = jax.jit(qnn_batched)


######################################################################
# Dropping rotations
# ~~~~~~~~~~~~~~~~~~
#
# As anticipated, we need to set some random parameters to 0 at each optimization step. Given a layer
# dropout rate :math:`p_L` (this will be called ``layer_drop_rate``) and the gate dropout rate :math:`p_G`
# (this will be called ``rot_drop_rate``), the probability :math:`p` that a
# gate is dropped in a layer can be calculated with the conditioned probability law:
#
# .. math::
#
#
#    p=p(A\cap B)=p(A|B)p(B)=p_Gp_L
#
# where :math:`B` represents the selection of a specific layer and
# :math:`A` the selection of a specific gate within the chosen layer.
#
# In the following cell we define a function that produces the list of the indices of rotation gates that
# are kept. For gates which are dropped, the value ``-1`` is assigned to the corresponding index. The structure of the list
# is nested; we have one list per ``inner_layer`` inside one list per each layer, all contained in another list.
# This function will be called at each iteration.
#


def make_dropout(key):
    drop_layers = []

    for lay in range(layers):
        # each layer has prob p_L=layer_drop_rate of being dropped
        # according to that for every layer we sample
        # if we have to appy dropout in it or not
        out = jax.random.choice(
            key, jnp.array(range(2)), p=jnp.array([1 - layer_drop_rate, layer_drop_rate])
        )
        key = jax.random.split(key)[0]  # update the random key

        if out == 1:  # if it has to be dropped
            drop_layers.append(lay)

    keep_rot = []
    # we make list of indexes corresponding to the rotations gates
    # that are kept in the computation during a single train step
    for i in range(layers):
        # each list is divded in layers and then in "inner layers"
        # this is strictly related to the QNN architecture that we use
        keep_rot_layer = [list(range((n_qubits))) for j in range(1, inner_layers + 1)]

        if i in drop_layers:  # if dropout has to be applied in this layer
            keep_rot_layer = []  # list of indexes for a single layer
            inner_keep_r = []  # list of indexes for a single inner layer
            for param in range(params_per_layer):
                # each rotation within the layer has prob p=rot_drop_rate of being dropped
                # according to that for every parameter (rotation) we sample
                # if we have to drop it or not
                out = jax.random.choice(
                    key, jnp.array(range(2)), p=jnp.array([1 - rot_drop_rate, rot_drop_rate])
                )
                key = jax.random.split(key)[0]  # update the random key

                if out == 0:  # if we have to keep it
                    inner_keep_r.append(param % n_qubits)  # % is required because we work
                    # inner layer by inner layer
                else:  # if the rotation has to be dropped
                    inner_keep_r.append(-1)  # we assign the value -1

                if param % n_qubits == n_qubits - 1:  # if it's the last qubit of the register
                    # append the inner layer list
                    keep_rot_layer.append(inner_keep_r)
                    # and reset it
                    inner_keep_r = []

        keep_rot.append(keep_rot_layer)

    return jnp.array(keep_rot)


######################################################################
# We can check the output of the ``make_dropout`` function:
#

# setting the drop probability
layer_drop_rate, rot_drop_rate = 0.5, 0.3  # 15% probability of dropping a gate

# JAX random key
key = jax.random.PRNGKey(12345)
# create the list of indexes,
# -1 implies we are dropping a gate
keep_rot = make_dropout(key)

# let's just print the list for first layer
print(keep_rot[0])

######################################################################
# Noisy sinusoidal function
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To test the effectiveness of the dropout technique, we will use a prototypical dataset
# with which it is very easy to overfit: the sinusoidal function. We produce some
# points according to the :math:`\sin` function and then we add some white Gaussian noise
# (noise that follows a normal distribution) :math:`\epsilon.` The noise is essential to obtain overfitting;
# when our model is extremely expressive, it is capable of exactly fit each point and some parameters
# become hyper-specialized in recognizing the noisy features. This makes predictions on new unseen
# data difficult, since the overfitting model did not learn the true underlying data distribution.
# The dropout technique will help in avoiding co-adaptation and hyper-specialization,
# effectively reducing overfitting.
#

from sklearn.model_selection import train_test_split


def make_sin_dataset(dataset_size=100, test_size=0.4, noise_value=0.4, plot=False):
    """1D regression problem y=sin(x*\pi)"""
    # x-axis
    x_ax = np.linspace(-1, 1, dataset_size)
    y = [[np.sin(x * np.pi)] for x in x_ax]
    np.random.seed(123)
    # noise vector
    noise = np.array([np.random.normal(0, 0.5, 1) for i in y]) * noise_value
    X = np.array(x_ax)
    y = np.array(y + noise)  # apply noise

    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=40, shuffle=True
    )

    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return X_train, X_test, y_train, y_test


from matplotlib import ticker

X, X_test, y, y_test = make_sin_dataset(dataset_size=20, test_size=0.25)


fig, ax = plt.subplots()
plt.plot(X, y, "o", label="Training")
plt.plot(X_test, y_test, "o", label="Test")

plt.plot(
    np.linspace(-1, 1, 100),
    [np.sin(x * np.pi) for x in np.linspace(-1, 1, 100)],
    linestyle="dotted",
    label=r"$\sin(x)$",
)
plt.ylabel(r"$y = \sin(\pi\cdot x) + \epsilon$")
plt.xlabel(r"$x$")
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
plt.legend()

plt.show()

######################################################################
# Since our circuit is only able to provide outputs in the range :math:`[-1,1],` we rescale all the
# noisy data within this range. To do this we leverage the `MinMaxScaler` from `sklearn`.
# It is common practice to fit the scaler only from training data and then apply it also to the
# test. The reason behind this is that in general one only has knowledge about the training dataset.
# (If the training dataset is not exhaustively representative of the underlying distribution,
# this preprocessing may lead to some outliers in the test set to be scaled out of the desired range.)
#

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
y = scaler.fit_transform(y)
y_test = scaler.transform(y_test)

# reshaping for computation
y = y.reshape(-1,)
y_test = y_test.reshape(-1,)

######################################################################
# Optimization
# ~~~~~~~~~~~~
#
# At this point we have to set the hyperparameters of the optimization, namely the number of epochs, the
# learning rate, and the optimizer:
#

import optax  # optimization using jax

epochs = 700
optimizer = optax.adam(learning_rate=0.01)

######################################################################
# We define the cost function as the Mean Square Error:
#


@jax.jit
def calculate_mse_cost(X, y, theta, keep_rot):
    yp = qnn(X, theta, keep_rot)
    # depending on your version of Pennylane you may require the following line
    #####
    yp = jnp.array(yp).T
    #####
    cost = jnp.mean((yp - y) ** 2)

    return cost


# Optimization update step
@jax.jit
def optimizer_update(opt_state, params, x, y, keep_rot):
    loss, grads = jax.value_and_grad(lambda theta: calculate_mse_cost(x, y, theta, keep_rot))(
        params
    )
    updates, opt_state = optimizer.update(grads, opt_state)

    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


######################################################################
# Training the model
# ------------------
#
# And now we can try to train the model. We execute different runs of the training to understand the
# average behaviour of quantum dropout. To see the effect of dropout we can set different values of
# ``layer_drop_rate`` and ``rot_drop_rate``:
#

n_run = 3
drop_rates = [(0.0, 0.0), (0.3, 0.2), (0.7, 0.7)]

train_history = {}
test_history = {}
opt_params = {}


for layer_drop_rate, rot_drop_rate in drop_rates:
    # initialization of some lists to store data
    costs_per_comb = []
    test_costs_per_comb = []
    opt_params_per_comb = []
    # we execute multiple runs in order to see the average behaviour
    for tmp_seed in range(seed, seed + n_run):
        key = jax.random.PRNGKey(tmp_seed)
        assert len(X.shape) == 2  # X must be a matrix
        assert len(y.shape) == 1  # y must be an array
        assert X.shape[0] == y.shape[0]  # compatibility check

        # parameters initialization with gaussian ditribution
        initial_params = jax.random.normal(key, shape=(layers * params_per_layer,))
        # update the random key
        key = jax.random.split(key)[0]

        params = jnp.copy(initial_params)

        # optimizer initialization
        opt_state = optimizer.init(initial_params)

        # lists for saving single run training and test cost trend
        costs = []
        test_costs = []

        for epoch in range(epochs):
            # generate the list for dropout
            keep_rot = make_dropout(key)
            # update the random key
            key = jax.random.split(key)[0]

            # optimization step
            params, opt_state, cost = optimizer_update(opt_state, params, X, y, keep_rot)

            ############## performance evaluation #############
            # inference is done with the original model
            # with all the gates
            keep_rot = jnp.array(
                [
                    [list(range((n_qubits))) for j in range(1, inner_layers + 1)]
                    for i in range(layers)
                ]
            )
            # inference on train set
            cost = calculate_mse_cost(X, y, params, keep_rot)

            costs.append(cost)

            # inference on test set
            test_cost = calculate_mse_cost(X_test, y_test, params, keep_rot)
            test_costs.append(test_cost)

            # we print updates every 5 iterations
            if epoch % 5 == 0:
                print(
                    f"{layer_drop_rate:.1f}-{rot_drop_rate:.1f}",
                    f"run {tmp_seed-seed} - epoch {epoch}/{epochs}",
                    f"--- Train cost:{cost:.5f}",
                    f"--- Test cost:{test_cost:.5f}",
                    end="\r",
                )

        costs_per_comb.append(costs)
        test_costs_per_comb.append(test_costs)
        opt_params_per_comb.append(params)
        print()
    costs_per_comb = np.array(costs_per_comb)
    test_costs_per_comb = np.array(test_costs_per_comb)
    opt_params_per_comb = np.array(opt_params_per_comb)

    train_history[(layer_drop_rate, rot_drop_rate)] = costs_per_comb
    test_history[(layer_drop_rate, rot_drop_rate)] = test_costs_per_comb
    opt_params[(layer_drop_rate, rot_drop_rate)] = opt_params_per_comb

######################################################################
# Performance evaluation
# ----------------------
#
# Let’s compare the difference in performance with a plot:
#

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
plt.subplots_adjust(wspace=0.05)
axs[0].set_title("MSE train")
for k, v in train_history.items():
    train_losses = np.array(v)
    mean_train_history = np.mean(train_losses, axis=0)
    std_train_history = np.std(train_losses, axis=0,)

    mean_train_history = mean_train_history.reshape((epochs,))
    std_train_history = std_train_history.reshape((epochs,))

    # shadow standard deviation
    axs[0].fill_between(
        range(epochs),
        mean_train_history - std_train_history,
        mean_train_history + std_train_history,
        alpha=0.2,
    )
    # average trend
    axs[0].plot(range(epochs), mean_train_history, label=f"{k}")  # Avg Loss

axs[1].set_title("MSE test")
for k, v in test_history.items():
    test_losses = np.array(v)
    mean_test_history = np.mean(test_losses, axis=0)
    std_test_history = np.std(test_losses, axis=0,)

    mean_test_history = mean_test_history.reshape((epochs,))
    std_test_history = std_test_history.reshape((epochs,))

    # shadow standard deviation
    axs[1].fill_between(
        range(epochs),
        mean_test_history - std_test_history,
        mean_test_history + std_test_history,
        alpha=0.2,
    )
    # averange trend
    axs[1].plot(range(epochs), mean_test_history, label=f"{k}")  # Avg Loss

axs[0].legend(loc="upper center", bbox_to_anchor=(1.01, 1.25), ncol=4, fancybox=True, shadow=True)

for ax in axs.flat:
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_ylim([1e-3, 0.6])
    ax.label_outer()

plt.subplots_adjust(bottom=0.3)

plt.show()

######################################################################
# On the left you can see that without dropout there is a deep minimization of the training loss,
# moderate values of dropout converge, whereas high drop probabilities impede any learning. On
# the right, we can see the difference in generalization during the optimization process. Standard
# training without dropout initially reaches a low value of generalization error, but as the
# model starts to learn the noise in the training data (overfitting), the generalization error grows
# back. Oppositely, moderate values of dropout enable generalization errors comparable to the respective
# training ones. As the learning is not successful for elevated drop probabilities, the generalization
# error is huge. It is interesting to notice that the “not-learning” error is very close to the final
# error of the QNN trained without dropout.
#
# Hence, one can conclude that low values of dropout greatly improve the generalization performance of
# the model and remove overfitting, even if the randomness of the technique inevitably makes the
# training a little noisy. On the other hand, high drop probabilities only hinder the training
# process.
#
# Validation
# ~~~~~~~~~~
#
# To validate the technique we can also check how the model predicts in the whole :math:`[-1,1]` range
# with and without quantum dropout.
#

X, X_test, y, y_test = make_sin_dataset(dataset_size=20, test_size=0.25)

# spanning the whole range
x_ax = jnp.linspace(-1, 1, 100).reshape(100, 1)

# selecting which run we want to plot
run = 1

fig, ax = plt.subplots()
styles = ["dashed", "-.", "solid", "-."]
for i, k in enumerate(train_history.keys()):
    if k[0] == 0.3:
        alpha = 1
    else:
        alpha = 0.5
    # predicting and rescaling
    yp = scaler.inverse_transform(qnn(x_ax, opt_params[k][run], keep_rot).reshape(-1, 1))
    plt.plot([[i] for i in np.linspace(-1, 1, 100)], yp, label=k, alpha=alpha, linestyle=styles[i])

plt.scatter(X, y, label="Training", zorder=10)
plt.scatter(X_test, y_test, label="Test", zorder=10)

ylabel = r"$y = \sin(\pi\cdot x) + \epsilon$"
plt.xlabel("x", fontsize="medium")
plt.ylabel(ylabel, fontsize="medium")
plt.legend()
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

plt.show()

######################################################################
# The model without dropout overfits the noisy data by trying to exactly predict each of them,
# whereas dropout actually mitigates overfitting and makes the approximation of the underlying sinusoidal
# function way smoother.
#
# Conclusion
# ----------------------
# In this demo, we explained the basic idea behind quantum dropout and
# how to avoid overfitting by randomly "dropping" some rotation gates
# of a QNN during the training phase. We invite you to check out the paper [#dropout]_
# for more dropout techniques and additional analysis. Try it yourself and develop new
# dropout strategies.
#
#
# References
# ----------
#
# .. [#dropout] Scala, F., Ceschini, A., Panella, M., & Gerace, D. (2023).
#    *A General Approach to Dropout in Quantum Neural Networks*.
#    `Adv. Quantum Technol., 2300220 <https://onlinelibrary.wiley.com/doi/full/10.1002/qute.202300220>`__.
#
# .. [#Hinton2012] Hinton, G., Srivastava, N., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2012).
#    *Improving neural networks by preventing co-adaptation of feature detectors*.
#    `arXiv:1207.0580. <https://arxiv.org/abs/1207.0580>`__.
#
# .. [#Srivastava2014] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).
#    *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*.
#    `Journal of Machine Learning Research, 15(56):1929−1958. <http://jmlr.org/papers/v15/srivastava14a.html>`__.
#
# .. [#Kiani2020] Kiani,B. T., Lloyd, S., & Maity, R. (2020).
#    *Learning Unitaries by Gradient Descent*.
#    `arXiv: 2001.11897. <https://arxiv.org/abs/2001.11897>`__.
#
# .. [#Larocca2023] Larocca, M., Ju, N., García-Martín, D., Coles, P. J., & Cerezo, M. (2023).
#    *Theory of overparametrization in quantum neural networks*.
#    `Nat. Comp. Science, 3, 542–551. <http://dx.doi.org/10.1038/s43588-023-00467-6>`__.
#
