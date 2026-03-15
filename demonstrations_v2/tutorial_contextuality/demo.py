r"""
Contextuality and inductive bias in QML
============================================================

.. meta::
    :property="og:description": Train a tailored quantum model on a contextuality-inspired dataset
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/contextuality_thumbnail.png

.. related::
    tutorial_geometric_qml

"""


######################################################################
# What machine learning problems are quantum computers likely to excel
# at?
#
# In the article *Contextuality and inductive bias in quantum machine
# learning* [#paper]_ by Joseph Bowles,
# Victoria J Wright, Máté Farkas, Nathan Killoran and Maria Schuld, we
# look to contextuality for answers to this question.
#
# Contextuality is a nonclassical phenomenon exhibited by quantum
# systems, and it is necessary for computational advantage relative to
# classical machines. To be a little more specific, we focus on the
# framework of *generalized
# contextuality* [#contextuality]_, 
# which was introduced by Robert Spekkens in 2004. We find learning
# problems for which contextuality plays a key role, and these problems
# may therefore be good areas where quantum machine learning algorithms
# shine. In this demo we will:
#
# -  Describe a specific example of a contextuality-relevant problem that is based on the
#    well-known rock-paper-scissors game, and
# -  Construct and train a quantum model that is tailored to the
#    symmetries of the problem.
#
# Throughout the demo we will make use of JAX to vectorise and just-in-time compile
# certain functions, which will speed things up. For more information on how to
# combine JAX and PennyLane, see the PennyLane
# `documentation <https://docs.pennylane.ai/en/stable/introduction/interfaces/jax.html>`__.
#
# .. figure:: ../_static/demonstration_assets/contextuality/socialthumbnail_large_Contextuality.png
#    :align: center
#    :width: 50%
#

######################################################################
# Generalized contextuality
# -------------------------
#


######################################################################
# Suppose we want to prepare the maximally mixed state of a single qubit, 
# with :math:`\rho = \frac{1}{2}\mathbb{I}.` Although this corresponds to a
# single density matrix, there are many ways we could prepare the state.
# For example, we could mix the states :math:`\vert 0 \rangle` or
# :math:`\vert 1 \rangle` with equal probability. Alternatively, we could
# use the :math:`X` basis, and mix the states :math:`\vert + \rangle` or
# :math:`\vert - \rangle.` Even though this may not strike us as particularly
# strange, a remarkable coincidence is in fact going on here: an
# experimentalist can perform two physically distinct procedures (namely,
# preparing :math:`\rho` in the :math:`Z` or :math:`X` basis), however it
# is impossible to distinguish which procedure was performed, since they
# both result in the same density matrix and therefore give identical
# predictions for all future measurements.
#
# Such a coincidence demands an explanation. Something that one might expect
# is the following: the description of the experiment in terms of quantum
# states is not the most fundamental, and there are in fact other states
# (we’ll write them as :math:`\lambda`), that comprise our quantum states.
# In contextuality these are called *ontic states*, although they also go
# by the name of *hidden variables*. When we prepare a state
# :math:`\vert 0 \rangle`, :math:`\vert 1 \rangle,`
# :math:`\vert + \rangle`, :math:`\vert - \rangle,` what is really going
# on is that we prepare a mixture :math:`P_{\vert 0 \rangle}(\lambda),`
# :math:`P_{\vert 1 \rangle}(\lambda),`
# :math:`P_{\vert + \rangle}(\lambda),`
# :math:`P_{\vert - \rangle}(\lambda)` over the true ontic states. One may
# imagine that the corresponding mixtures over the :math:`\lambda` s are
# the same for the :math:`Z` and :math:`X` basis preparation:
#
# .. math:: \frac{1}{2}P_{\vert 0 \rangle}(\lambda)+\frac{1}{2}P_{\vert 1 \rangle}(\lambda)=\frac{1}{2}P_{\vert + \rangle}(\lambda)+\frac{1}{2}P_{\vert - \rangle}(\lambda).
#
# This is a rather natural explanation of our coincidence: the two
# procedures are indistinguishable because they actually correspond to the
# same mixture over the fundamental states :math:`\lambda.` This sort of
# explanation is called *non-contextual*, since the two mixtures do not
# depend on the basis (that is, the context) in which the state is
# prepared. It turns out that if one tries to apply this logic to all the
# indistinguishabilities in quantum theory, one arrives at contradictions:
# it simply cannot be done. For this reason we say that quantum theory is
# a *contextual* theory.
#
# In the paper we frame generalized contextuality in the machine learning
# setting, which allows us to define what we mean by a contextual learning
# model. In a nutshell, this definition demands that if a learning model
# is non-contextual, then any indistinguishabilities in the model should
# be explained in a non-contextual way similar to the above. This results
# in constraints on the learning model, which limits their expressivity.
# Since quantum models are contextual, they can of course go beyond these
# constraints, and understanding when and how they do this may shed light
# on the non-classical features that separate quantum models from
# classical ones.
#
# Below we describe a specific learning problem that demonstrates this
# approach. As we will see, the corresponding indistinguishability relates
# to an *inductive bias* of the learning model.
#


######################################################################
# The rock-paper-scissors game
# ------------------------------
#


######################################################################
# The learning problem we will consider involves three players
# (we'll call them players 0, 1 and 2) playing a
# variant of the rock-paper-scissors game with a referee.
# The game goes as follows. In each round, a player can choose to play
# either rock (R), paper (P) or scissors (S). Each player also has a
# ‘special’ action. For player 0 it is R, for player 1 it is P and for
# player 2 it is S. The actions of the players are then compared pairwise,
# with the following rules:
#
# -  If two players play different actions, then one player beats the
#    other following the usual rule (rock beats scissors, scissors beats
#    paper, paper beats rock).
# -  If two players play the same action, the one who plays their special
#    action beats the other. If neither plays their special action, it is
#    a draw.
#
# A referee then decides the winners and the losers of that round: the
# winners receive :math:`\$1` and the losers lose :math:`\$1` (we will
# call this their *payoff* for that round).
#

##############################################################################
# .. figure:: ../_static/demonstration_assets/contextuality/rps.png
#    :align: center
#    :width: 50%

######################################################################
# Naturally, the more players a given player beats, the higher the
# probability that they get a positive payoff. In particular, if we denote
# the payoff of player :math:`k` by :math:`y_k=\pm1` then
#
# .. math:: \mathbb{E}(y_k) = \frac{n^k_{\text{win}}-n^k_{\text{lose}}}{2},
#
# where :math:`n^k_{\text{win}},` :math:`n^k_{\text{lose}}` is the number
# of players that player :math:`k` beats or loses to in that round. This
# ensures that a player is certain to get a positive (or negative) payoff
# if they beat (or lose) to everyone.
#
# To make this concrete, we will construct three 3x3 matrices ``A01``,
# ``A02``, ``A12`` which determine the rules for each pair of players.
# ``A01`` contains the expected payoff values of player 0 when playing
# against player 1. Using the rules of the game it looks as follows.
#

##############################################################################
# .. figure:: ../_static/demonstration_assets/contextuality/rpstable.png
#    :align: center
#    :width: 50%


######################################################################
# The matrices ``A02`` and ``A12`` are defined similarly.
#

import jax
import jax.numpy as jnp
import pennylane as qml
qml.capture.enable()
import numpy as np
jax.config.update("jax_platform_name", "cpu")
np.random.seed(666) # seed used for random functions

A01 = np.array([[1, -1, 1], [1, -1, -1], [-1, 1, 0]])  # rules for player 0 vs player 1
A02 = np.array([[1, -1, 1], [1, 0, -1], [-1, 1, -1]])
A12 = np.array([[0, -1, 1], [1, 1, -1], [-1, 1, -1]])


######################################################################
# We can also define the matrices ``A10``, ``A20``, ``A21``. Since
# switching the players corresponds to taking the transpose matrix and
# a positive payoff for one player implies a negative for the other,
# these matrices are given by
# the negative of the transposed matrix:
#

A10 = -A01.T  # rules for player 1 vs player 0
A20 = -A02.T
A21 = -A12.T

######################################################################
# Note that the above game is an example of a *zero-sum game*: if player 1 beats
# player 2 then necessarily player 2 loses to player 1. This implies
# :math:`\sum_k n^k_{\text{wins}}=\sum_kn^k_{\text{lose}}` and so in every
# round we have
#
# .. math:: \mathbb{E}(y_1)+\mathbb{E}(y_2)+\mathbb{E}(y_3)=0.
#


######################################################################
# Constructing the dataset
# ------------
#


######################################################################
# Here we construct a dataset based on the above game. Our data points
# correspond to probability
# distributions over possible actions: in the  zero-sum game literature
# these are called *strategies*.
# For example, a strategy for player k is a
# vector
#
# .. math:: x_k=(P(a_k=R), P(a_k=P), P(a_k=S))
#
# where :math:`a_k` denotes player :math:`k`\ ’s action. We collect these
# into a strategy matrix X
#
# .. math::
#
#    X = \begin{pmatrix}
#        P(a_0=R) & P(a_0=P) & P(a_0=S) \\
#        P(a_1=R) & P(a_1=P) & P(a_1=S) \\
#        P(a_2=R) & P(a_2=P) & P(a_2=S) .
#        \end{pmatrix}
#
#
#


######################################################################
# Let’s write a function
# to generate a set of strategy matrices.
#


def get_strategy_matrices(N):
    """
    Generates N strategy matrices, normalised by row
    """
    X = np.random.rand(N, 3, 3)
    for i in range(N):
        norm = np.array(X[i].sum(axis=1))
        for k in range(3):
            X[i, k, :] = X[i, k, :] / norm[k]
    return X


######################################################################
# The labels in our dataset correspond to payoff values :math:`y_k` of the
# three players. Following the rules of probability we find that if the
# players use strategies :math:`x_0, x_1, x_2` the expected values of
# :math:`n_{\text{wins}}^k - n_{\text{lose}}^k` are given
# by
#
# .. math:: \mathbb{E}[n_{\text{wins}}^0 - n_{\text{lose}}^0]  = x_0 \cdot A_{01}\cdot x_1^T+x_0 \cdot A_{02}\cdot x_2^T
#
# .. math:: \mathbb{E}[n_{\text{wins}}^1 - n_{\text{lose}}^1] = x_1 \cdot A_{10}\cdot x_0^T+x_1 \cdot A_{12}\cdot x_2^T
#
# .. math:: \mathbb{E}[n_{\text{wins}}^2 - n_{\text{lose}}^2] = x_2 \cdot A_{20}\cdot x_0^T+x_2 \cdot A_{21}\cdot x_1^T
#
# Since we have seen that
# :math:`\mathbb{E}(y_k) = \frac{n^k_{\text{win}}-n^k_{\text{lose}}}{2}`
# it follows that the probability for player :math:`k` to receive a
# positive payoff given strategies :math:`X` is
#
# .. math:: P(y_k=+1\vert X) = \frac{\mathbb{E}(y_k\vert X)+1}{2} =  \frac{(\mathbb{E}[n_{\text{wins}}^k - n_{\text{lose}}^k])/2+1}{2}
#
# Putting all this together we can write some code to generate the labels
# for our data set.
#


def payoff_probs(X):
    """
    get the payoff probabilities for each player given a strategy matrix X
    """
    n0 = X[0] @ A01 @ X[1] + X[0] @ A02 @ X[2]  # n0 = <n0_wins - n0_lose>
    n1 = X[1] @ A10 @ X[0] + X[1] @ A12 @ X[2]
    n2 = X[2] @ A20 @ X[0] + X[2] @ A21 @ X[1]
    probs = (jnp.array([n0, n1, n2]) / 2 + 1) / 2
    return probs


# JAX vectorisation
vpayoff_probs = jax.vmap(payoff_probs)


def generate_data(N):
    X = get_strategy_matrices(N)  # strategies
    P = vpayoff_probs(X)  # payoff probabilities
    r = np.random.rand(*P.shape)
    Y = np.where(P > r, 1, -1)  # sampled payoffs for data labels
    return X, Y, P


X, Y, P = generate_data(2000)

print(X[0])  # the first strategy matrix in our dataset
print(Y[0])  # the corresponding sampled payoff values

######################################################################
# Note that since strategies are probabilistic mixtures of actions, our
# data labels satisfy a zero-sum condition
#
# .. math:: \mathbb{E}(y_1\vert X_i)+\mathbb{E}(y_2\vert X_i)+\mathbb{E}(y_3\vert X_i)=0.
#
# We can verify this using the payoff probability matrix ``P`` that we
# used to sample the labels:
#

expvals = 2 * P - 1  # convert probs to expvals
expvals[:10].sum(axis=1)  # check first 10 entries


######################################################################
# The learning problem
# --------------------
#


######################################################################
# Suppose we are given a data set :math:`\{X_i,\vec{y}_i\}` consisting of
# strategy matrices and payoff values, however we don’t know what the
# underlying game is (that is, we don’t know the players were playing the
# rock, paper scissors game described above). We do have one piece of
# information though: we know the game is zero-sum so that the data
# generation process satisfies
#
# .. math:: \mathbb{E}(y_0\vert X_i)+\mathbb{E}(y_1\vert X_i)+\mathbb{E}(y_2\vert X_i)=0.
#
# Can we learn the rock, paper scissors game from this data? More
# precisely, if we are given an unseen strategy matrix
# :math:`X_{\text{test}}` our task is to sample from the three
# distributions
#
# .. math:: P(y_0\vert X_{\text{test}}), P(y_1\vert X_{\text{test}}), P(y_2\vert X_{\text{test}}).
#
# Note we are not asking to sample from the joint distribution
# :math:`P(\vec{y}\vert X_{\text{test}})` but the three marginal
# distributions only. This can be seen as an instance of multi-task
# learning, where a single task corresponds to sampling the payoff for one
# of the three players.
#


######################################################################
# Building inductive bias into a quantum model
# --------------------------------------------
#


######################################################################
# Here we describe a simple three qubit model to tackle this problem.
# Since we know that the data satisfies the zero-sum condition, we aim to
# create a quantum model that encodes this knowledge. That is, like
# the data we want our model to satisfy
#
# .. math:: \mathbb{E}(y_0\vert X_i)+\mathbb{E}(y_1\vert X_i)+\mathbb{E}(y_2\vert X_i)=0.
#
# In machine learning, this is called encoding an *inductive
# bias* into the model, and considerations like this are often crucial for
# good generalisation performance.
#
# .. note::
#   Since the above holds for all :math:`X_i,` it implies an
#   indistinguishability of the model: if we look at one of the labels at
#   random, we are equally likely to see a positive or negative payoff
#   regardless of :math:`X_i,` and so the :math:`X_i` are indistinguishable
#   with respect to this observation. This implies a corresponding constraint
#   on non-contextual learning models, which limits their expressivity and
#   may therefore hinder their performance: see the paper for more details
#   on how this looks in practice. Luckily for us quantum theory is a
#   contextual theory, so these limitations don’t apply to our model!
#
# The quantum model we consider has the following structure:
#

##############################################################################
# .. figure:: ../_static/demonstration_assets/contextuality/model.png
#    :align: center
#    :width: 50%


######################################################################
# The parameters :math:`\theta` and :math:`\alpha` are trainable
# parameters of the model, and we will use the three :math:`Z`
# measurements at the end of the circuit to sample the three labels.
# Therefore, if we write the entire circuit as
# :math:`\vert \psi(\alpha,\theta,X)\rangle` the zero sum condition will
# be satisfied if
#
# .. math:: \langle \psi(\alpha,\theta,X) \vert (Z_0+Z_1+Z_2) \vert \psi(\alpha,\theta,X) \rangle = 0.
#
# Let’s see how we can create a model class that satisfies this. For
# precise details on the structure of the model, check out Figure 6 in the
# paper. We’ll first look at the parameterised unitary :math:`V_{\alpha},`
# that we call the *input preparation unitary*. This prepares a state
# :math:`V_\alpha\vert 0 \rangle` such that
#
# .. math:: \langle 0 \vert V^\dagger_\alpha (Z_0+Z_1+Z_2) V_\alpha\vert 0 \rangle = 0.
#
# An example of such a circuit is the following.
#


def input_prep(alpha):
    # This ensures the prepared state has <Z_0+Z_1+Z_2>=0
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    qml.RY(alpha[0], wires=0)
    qml.RY(alpha[0] + np.pi, wires=1)


######################################################################
# The second unitary is a *bias invariant layer*: it preserves the value
# of :math:`\langle Z_0+Z_1+Z_2 \rangle` for all input states into the
# layer. To achieve this, the generators of the unitaries in this layer
# must commute with the operator :math:`Z_0+Z_1+Z_2.` For example the
# operator :math:`X\otimes X + Y\otimes Y + Z\otimes Z` (on any pair of
# qubits) commutes with :math:`Z_0+Z_1+Z_2` and so a valid parameterised
# gate could be
#
# .. math:: e^{i\theta(X\otimes X\otimes\mathbb{I} + Y\otimes Y\otimes\mathbb{I} + Z\otimes Z\otimes\mathbb{I})}.
#
# This kind of reasoning is an example of geometric quantum machine
# learning (check out [#reptheory]_ and [#equivariant]_ or our own
# :doc:`demo <demos/tutorial_geometric_qml>` for an awesome introduction to the subject).
# Below we construct the
# bias invariant layer: note that all the generators commute with
# :math:`Z_0+Z_1+Z_2.` The variables ``blocks`` and ``layers`` are model
# hyperparameters that we will fix as ``blocks=1`` and ``layers=2``.
#

blocks = 1
layers = 2


def swap_rot(weights, wires):
    """
    bias-invariant unitary with swap matrix as generator
    """
    qml.PauliRot(weights, "XX", wires=wires)
    qml.PauliRot(weights, "YY", wires=wires)
    qml.PauliRot(weights, "ZZ", wires=wires)


def param_unitary(weights):
    """
    A bias-invariant unitary (U in the paper)
    """
    for b in range(blocks):
        for q in range(3):
            qml.RZ(weights[b, q], wires=q)
        qml.PauliRot(weights[b, 3], "ZZ", wires=[0, 1])
        qml.PauliRot(weights[b, 4], "ZZ", wires=[0, 2])
        qml.PauliRot(weights[b, 5], "ZZ", wires=[1, 2])
        swap_rot(weights[b, 6], wires=[0, 1])
        swap_rot(weights[b, 7], wires=[1, 2])
        swap_rot(weights[b, 8], wires=[0, 2])


def data_encoding(x):
    """
    S_x^1 in paper
    """
    for q in range(3):
        qml.RZ(x[q], wires=q)


def data_encoding_pairs(x):
    """
    S_x^2 in paper
    """
    qml.PauliRot(x[0] * x[1], "ZZ", wires=[0, 1])
    qml.PauliRot(x[1] * x[2], "ZZ", wires=[1, 2])
    qml.PauliRot(x[0] * x[2], "ZZ", wires=[0, 2])


def bias_inv_layer(weights, x):
    """
    The full bias invariant layer.
    """
    # data preprocessing
    x1 = jnp.array([x[0, 0], x[1, 1], x[2, 2]])
    x2 = jnp.array(([x[0, 1] - x[0, 2], x[1, 2] - x[1, 0], x[2, 0] - x[2, 1]]))
    for l in range(0, 2 * layers, 2):
        param_unitary(weights[l])
        data_encoding(x1)
        param_unitary(weights[l + 1])
        data_encoding_pairs(x2)
    param_unitary(weights[2 * layers])


######################################################################
# With our ``input_prep`` and ``bias_inv_layer`` functions we can now
# define our quantum model.
#

dev = qml.device("default.qubit", wires=3)


@qml.qnode(dev)
def model(weights, x):
    input_prep(weights[2 * layers + 1, 0])  # alpha is stored in the weights array
    bias_inv_layer(weights, x)
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))]


# jax vectorisation, we vectorise over the data input (the second argument)
vmodel = jax.vmap(model, (None, 0))
vmodel = jax.jit(vmodel)


######################################################################
# To investigate the effect of the encoded inductive bias, we will compare
# this model to a generic model with the same data encoding and similar
# number of parameters (46 vs 45 parameters).
#


def generic_layer(weights, x):
    # data preprocessing
    x1 = jnp.array([x[0, 0], x[1, 1], x[2, 2]])
    x2 = jnp.array(([x[0, 1] - x[0, 2], x[1, 2] - x[1, 0], x[2, 0] - x[2, 1]]))
    for l in range(0, 2 * layers, 2):
        qml.StronglyEntanglingLayers(weights[l], wires=range(3))
        data_encoding(x1)
        qml.StronglyEntanglingLayers(weights[l + 1], wires=range(3))
        data_encoding_pairs(x2)
    qml.StronglyEntanglingLayers(weights[2 * layers], wires=range(3))


dev = qml.device("default.qubit", wires=3)


@qml.qnode(dev)
def generic_model(weights, x):
    generic_layer(weights, x)
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))]


vmodel_generic = jax.vmap(generic_model, (None, 0))
vmodel_generic = jax.jit(vmodel_generic)


######################################################################
# **Warning**: Since we are using JAX it is important that our ``model``
# and ``generic model`` functions are functionally pure (read more
# `here <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`__).
# This means we cannot change the values of ``blocks`` or ``layers`` from
# hereon since these values have been cached for JIT compilation.
#


######################################################################
# Training and evaluation
# -----------------------
#


######################################################################
# To train the model we will minimise the negative log likelihood of the
# labels given the data
#
# .. math:: \mathcal{L} = -\frac{1}{3\vert N \vert}\sum_{(X_i,\vec{y}_i)} \log(\mathcal{P}_0(y_i^{(0)}\vert X_i))+\log(\mathcal{P}_1(y_i^{(1)}\vert X_i))+\log(\mathcal{P}_2(y_i^{(2)}\vert X_i))
#
# Here :math:`\mathcal{P}_k` is the probability distribution of the
# :math:`k` label from the model, :math:`y_i^{(k)}` is the kth element
# of the payoff vector :math:`\vec{y}_i` in the dataset, and :math:`N` is
# the size of the training dataset. We remark that
# training the negative log likelihood is in some sense cheating, since
# for large quantum circuits we don’t know how to estimate it efficiently.
# As generative modeling in QML progresses, we can hope however that
# scalable methods that approximate this type of training may appear.
#


def likelihood(weights, X, Y, model):
    """
    The cost function. Returns the negative log likelihood
    """
    expvals = jnp.array(model(weights, X)).T
    probs = (1 + Y * expvals) / 2  # get the relevant probabilites
    probs = jnp.log(probs)
    llh = jnp.sum(probs) / len(X) / 3
    return -llh


######################################################################
# For evaluation we will use the average KL divergence between the true
# data distribution and the model distribution
#
# .. math:: \mathbb{E}_{P^\text{data}(X)} \left[\frac{1}{3}\sum_{k=1}^{3} D_{\text{KL}}(P^\text{data}_k(y\vert X)\vert\vert \mathcal{P}_k(y\vert X)) \right].
#
# To estimate this we sample a test set of strategies, calculate their
# payoff probabilities, and estimate the above expectation via the sample
# mean.
#

N_test = 10000
X_test = get_strategy_matrices(N_test)

probs_test = np.zeros([N_test, 3, 2])
probs_test[:, :, 0] = vpayoff_probs(X_test)  # the true probabilities for the test set
probs_test[:, :, 1] = 1 - probs_test[:, :, 0]
probs_test = jnp.array(probs_test)


def kl_div(p, q):
    """
    Get the KL divergence between two probability distribtuions
    """
    p = jnp.vstack([p, jnp.ones(len(p)) * 10 ** (-8)])  # lower cutoff of prob values of 10e-8
    p = jnp.max(p, axis=0)
    return jnp.sum(q * jnp.log(q / p))  # forward kl div


def kl_marginals(probs, probs_test):
    """
    get the mean KL divergence of the three marginal distributions
    (the square brackets above)
    """
    kl = 0
    for t in range(3):
        kl = kl + kl_div(probs[t, :], probs_test[t, :])
    return kl / 3


# vectorise the kl_marginals function. Makes estimating the average KL diverence of a model faster.
vkl_marginals = jax.vmap(kl_marginals, (0, 0))


def get_av_test_kl(model, weights, probs_test, X_test):
    """
    returns the average KL divergence for a test set X_test.
    """
    N_test = len(X_test)
    probs = np.zeros(probs_test.shape)
    expvals = jnp.array(model(weights, X_test)).T
    for t in range(3):
        probs[:, t, 0] = (1 + expvals[:, t]) / 2
        probs[:, t, 1] = (1 - expvals[:, t]) / 2
    return np.sum(vkl_marginals(probs, probs_test)) / N_test


######################################################################
# To optimise the model we make use of the JAX optimization library optax.
# We will use the adam gradient descent optimizer.
#

import optax
from tqdm import tqdm


def optimise_model(model, nstep, lr, weights):
    plot = [[], [], []]
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(weights)
    steps = tqdm(range(nstep))
    for step in steps:
        #         use optax to update parameters
        llh, grads = jax.value_and_grad(likelihood)(weights, X, Y, model)
        updates, opt_state = optimizer.update(grads, opt_state, weights)
        weights = optax.apply_updates(weights, updates)

        kl = get_av_test_kl(model, weights, probs_test, X_test)
        steps.set_description(
            "Current divergence: %s" % str(kl) + " :::: " + "Current likelihood: %s" % str(llh)
        )
        plot[0].append(step)
        plot[1].append(float(llh))
        plot[2].append(float(kl))
    return weights, llh, kl, plot


######################################################################
# We are now ready to generate a data set and optimize our models!
#

# generate data
N = 2000  # number of data points
X, Y, P = generate_data(N)

nstep = 2000  # number of optimisation steps

lr = 0.001  # initial learning rate
weights_model = np.random.rand(2 * layers + 2, blocks, 9) * 2 * np.pi
weights_generic = np.random.rand(2 * layers + 1, blocks, 3, 3) * 2 * np.pi

# optimise the structured model
weights_model, llh, kl, plot_model = optimise_model(vmodel, nstep, lr, weights_model)
# optimise the generic model
weights_generic, llh, kl, plot_genereic = optimise_model(vmodel_generic, nstep, lr, weights_generic)


######################################################################
# Let’s plot the average KL divergence and the negative log likelihood for
# both models.
#

import matplotlib.pyplot as plt

plt.style.use('pennylane.drawer.plot')

# subplots
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
fig.tight_layout(pad=10.0)

# KL divergence
ax1.plot(plot_model[0], plot_model[2], label="biased model")
ax1.plot(plot_genereic[0], plot_genereic[2], label="generic model")

ax1.set_yscale("log")
ax1.set_ylabel("KL divergence (test)")
ax1.set_xlabel("training step")
ax1.legend()

# negative log likelihood
ax2.plot(plot_model[0], plot_model[1])
ax2.plot(plot_genereic[0], plot_genereic[1])

ax2.set_yscale("log")
ax2.set_ylabel("Negative log likelihood (train)")
ax2.set_xlabel("training step")

plt.show()

######################################################################
# We see that the model that encodes the inductive bias achieves both a
# lower training error and generalisation error, as can be expected.
# Incorporating knowledge about the data into the model design is
# generally a very good idea!
#


######################################################################
# Conclusion
# ----------
#

######################################################################
# In this demo we have constructed a dataset whose structure is
# connected to generalized contextuality, and have shown how to encode
# this structure as an inductive bias of a quantum model class. As is
# often the case, we saw that this approach outperforms a generic model
# class that does not take this knowledge into account. As a general rule,
# considerations like this should be at the front of one's mind when
# building a quantum model for a specific task.
#
# That is all for this demo. In our paper [#paper]_, it is also shown how models of
# this kind can perform better than classical surrogate
# models [#surrogates]_ at this specific task,
# which further strengthens the claim that the inductive bias of the
# quantum model is useful. For more information and to read more about the
# link between contextuality and QML, check out the full paper.
#
#
# References
# ----------
#
# .. [#paper]
#
#     J. Bowles, V. J. Wright, M. Farkas, N. Killoran, M. Schuld
#     "Contextuality and inductive bias in quantum machine learning."
#     `arXiv:2302.01365 <https://arxiv.org/abs/2302.01365>`__, 2023.
#
# .. [#contextuality]
#
#     R. W. Spekkens
#     "Contextuality for preparations, transformations, and unsharp measurements."
#     `Phys. Rev. A 71, 052108 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.71.052108>`__, 2005.
#
# .. [#reptheory]
#
#     M. Ragone, P. Braccia, Q. T. Nguyen, L. Schatzki, P. J. Coles, F. Sauvage, M. Larocca, M. Cerezo
#     "Representation Theory for Geometric Quantum Machine Learning."
#     `arXiv:2210.07980 <https://arxiv.org/abs/2210.07980>`__, 2023.
#
# .. [#equivariant]
#
#     Q. T. Nguyen, L. Schatzki, P. Braccia, M. Ragone, P. J. Coles, F. Sauvage, M. Larocca, M. Cerezo
#     "Theory for Equivariant Quantum Neural Networks."
#     `arXiv:2210.08566 <https://arxiv.org/abs/2210.08566>`__, 2022.
#
# .. [#surrogates]
#
#     F. J. Schreiber, J. Eiser, J. J. Meyer
#     "Classical surrogates for quantum learning models."
#     `arXiv:2206.11740 <https://arxiv.org/abs/2206.11740>`__, 2022.
#
