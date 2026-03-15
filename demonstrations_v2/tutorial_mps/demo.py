r"""Introducing matrix product states for quantum practitioners
===============================================================

Matrix product states remain the workhorse for a broad range of modern classical quantum simulation techniques,
still to this day. Their unique features (like offering a canonical form) make them an incredibly neat tool
in terms of simplicity and algorithmic complexity.
In this demo, we are going to cover all the essentials you need to know in order to handle matrix product states,
and show how to use them to simulate quantum circuits.

.. figure:: ../_static/demonstration_assets/how_to_simulate_quantum_circuits_with_tensor_networks/TN_MPS.gif
    :align: center
    :width: 90%
    
Introduction
------------

Matrix product states (MPS) are an efficient representation of quantum states in one spatial dimension.
However, due to their unique features like offering a canonical form, they are employed in a variety of tasks beyond just 1D systems.

The amount of entanglement the MPS can represent is user-controlled via a hyper-parameter, the so-called `bond dimension` :math:`\chi.`
If we allow :math:`\chi` to be of :math:`\mathcal{O}(2^{\frac{n}{2}})` for a system of :math:`n` qubits, we can write `any` state as an `exact` MPS.
To avoid exponentially large resources, however, one typically sets a finite bond dimension :math:`\chi` at the cost of introducing an approximation error.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_mps_simulation.png
    :align: center
    :width: 70%

For some specific classes of states, this is provably sufficient to have faithful representations (see :ref:`Area Law`). 
But because MPS come with a lot of powerful computational features that we are going to discuss later (in particular :ref:`canonical forms <Canonical Forms>`),
they are still used in much more complex systems where these requirements do not hold anymore, and still yield good results.
For example, state-of-the-art `quantum chemistry <https://pennylane.ai/qml/quantum-chemistry/>`__ simulations were performed using MPS [#Baiardi]_ 
and similar methods have been used to simulate experiments on the largest available quantum computers at the time [#Patra]_.

It is known that there are more suitable tensor network states like projected entangled pair states (PEPS) for 
more complex situations (see the :ref:`Area Law` section).
However, they all suffer from the need for significantly more complicated algorithms 
and higher costs, as a lot of the things that make MPS so attractive, 
like the availability of a canonical form, are not true anymore.
To put it plainly, it is often simply much easier to use readily available 
MPS code and throw a large amount of resources into the bond dimension than to develop
more advanced tensor network methods.

An exception to that are so-called `simple update`
methods [#Jiang]_, which use pseudo-canonical forms that allow a similarly simple
algorithmic complexity at the cost of not being optimal in its resources.
Reference [#Patra]_ is a good example of that.

More advanced tensor network methods are continually being developed and optimized, though the biggest hindrance to the widespread use of already known advanced tensor network methods is the lack of reliable open-source implementations.
While the current state of affairs in academia is giving little incentive to change that,
MPS continue to be the workhorse for a wide variety of quantum simulation techniques.

We are going to introduce the essentials of MPS in the first part of this demo. 
Afterwards, in the second part, we will look at the specific application of simulating quantum circuits using MPS.

Matrix product state essentials
-------------------------------

Compression using singular value decomposition (SVD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To understand MPS and the bond dimension, we first need to understand how one can generally use singular value decomposition (SVD) to do compression.
Any matrix :math:`M = \mathbb{C}^{M\times N}` can be singular-value-decomposed as

.. math:: M = U \Lambda V^\dagger,

where :math:`\Lambda` is the diagonal matrix of the :math:`r=\min(M, N)` real and non-negative singular values, 
:math:`U \in \mathbb{C}^{M\times r}` is left-unitary, :math:`U^\dagger U = \mathbb{I}_r,` and
:math:`V^\dagger \in \mathbb{C}^{r\times N}` is right-unitary, :math:`V^\dagger (V^\dagger)^\dagger = \mathbb{I}_r.`
We say the columns and rows of :math:`U` and :math:`V^\dagger` are the left- and right-orthonormal
singular vectors, respectively. In the case of square and normal matrices, the singular values and singular vectors
are just the eigenvalues and eigenvectors.

Small singular values and their corresponding singular vectors carry little information of the matrix. When the singular values have
a tail of small values, we can compress the matrix by throwing away both these numbers and the corresponding singular vectors.
The power of this approach is best seen by a small example. Let us load the image we have shown in the header of this demo and compress it.

"""

import numpy as np
import matplotlib.pyplot as plt

# import image
img = plt.imread("../_static/demo_thumbnails/regular_demo_thumbnails/thumbnail_mps_simulation.png")
# alternative: import image directly from url
# import urllib2
# img_url = urllib2.urlopen("https://pennylane.ai/_static/demo_thumbnails/regular_demo_thumbnails/thumbnail_mps_simulation.png")
# img = plt.imread(img_url)

# only look at one color channel for demonstration purposes
img = img[:, :, 0]

# Perform SVD
U, Lambda, Vd = np.linalg.svd(img)

# Keep only the 50 largest singular values and vectors
chi = 50
U_compressed = U[:, :chi]
Lambda_compressed = Lambda[:chi]
Vd_compressed = Vd[:chi]

# Reconstruct the compressed image
compressed_img = U_compressed @ np.diag(Lambda_compressed) @ Vd_compressed

fig, axs = plt.subplots(ncols=2)
ax = axs[0]
ax.imshow(img, vmin=0, vmax=1)
ax.set_title("Uncompressed image")

ax = axs[1]
ax.imshow(compressed_img, vmin=0, vmax=1)
ax.set_title("Compressed image")

plt.show()

size_original = np.prod(img.shape)
size_compressed = np.prod(U_compressed.shape) + np.prod(Lambda_compressed.shape) + np.prod(Vd_compressed.shape)

print(f"original image size: {size_original}, compressed image size: {size_compressed}, factor {size_original/size_compressed:.3f} saving")
##############################################################################
# 
#
# The original image is :math:`334 \times 542` pixels, that we compress as :math:`334 \times 50` pixels in
# :math:`U,` :math:`50` pixels in :math:`\Lambda` and :math:`50 \times 542` pixels in :math:`V^\dagger.`
# This is possible because the information density in the image is low, as seen by the distribution of singular values.
# Let us visualize this by re-computing the singular values and plotting their distribution (note that they are automatically ordered in descending order).


_, Lambda, _ = np.linalg.svd(img) # recompute the full spectrum
plt.plot(Lambda)
plt.xlabel("index $i$")
plt.ylabel("$\\Lambda_i$")
plt.show()

##############################################################################
# 
# We are later going to do the same trick with state vectors.
# Note that the compressed information is encoded in :math:`U,` :math:`\Lambda` and :math:`V^\dagger.`
# If we want to retrieve the actual image :math:`M` (or state vector), we still need to reconstruct the full :math:`334 \times 542` pixels.
# Luckily, as we will later see in the case of MPS, we can retrieve all relevant information efficiently from the compressed components without ever 
# having to reconstruct the full state vector.
#
# Turn any state into an MPS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Formally, any quantum state :math:`|\psi\rangle \in \mathbb{C}^{2^n}` on :math:`n` qubits can be written as a matrix product state (MPS).
# The goal of this section will be to write an arbitrary state :math:`|\psi\rangle = \sum_{\sigma_1, .., \sigma_n} \psi_{\sigma_1, .., \sigma_n} |\sigma_1 .. \sigma_n\rangle` in the form
#
# .. math:: |\psi \rangle = \sum_{\sigma_1, .., \sigma_n} U^{\sigma_1} .. U^{\sigma_n} |\sigma_1 .. \sigma_n\rangle,
#
# where we decomposed the rank :math:`n` tensor :math:`\psi_{\sigma_1, .., \sigma_n}` into a product of matrices :math:`U^{\sigma_j}`
# for each value of :math:`\sigma_j` (equal to :math:`0` or :math:`1` for qubits). This is why it is called a **matrix product** state.
# While it historically makes sense to treat these indeed as matrices, given a concrete value of :math:`\sigma_j,` I find it more convenient
# to forget about the notion of matrices and just treat them as the collection of rank-3 tensors :math:`\{ U_{\mu_{i-1} \sigma_i \mu_i} \}`
# that have two so-called virtual indices, :math:`\mu_{i-1}, \mu_i,` and one so-called physical index, :math:`\sigma_j.`
# Whenever we write :math:`U^{\sigma_j}` with a superscript, we mean a matrix given a concrete value of :math:`\sigma_j.` 
# In particular, we define :math:`\left(U^{\sigma_j}\right)_{\mu_{i-1} \mu_i} = U_{\mu_{i-1} \sigma_i \mu_i}` for notational convenience and to make it easier to follow along with the code.
#
# Graphically, this corresponds to splitting up the big rank-:math:`n` tensor into :math:`n` smaller tensors, 
# similarly to our approach in the example of compressing an image using SVD.
#
# .. figure:: ../_static/demonstration_assets/mps/psi_to_mps_0.png
#     :align: center
#     :width: 70%
# 
# .. note:: 
#     We are going to use combinations of tensor indices and treat them as one big index.
#     In particular, the two indices :math:`\sigma_1 = \{0, 1\}` and :math:`\sigma_2 = \{0, 1\}` will have the combined index
#     :math:`(\sigma_1 \sigma_2) = \{00, 01, 10, 11\}.` The actual order is a choice and does not matter for the analytic descriptions,
#     but in practice, we just choose to do it in the same way as ``numpy`` arrays are reshaped for convenience. 
#     That way we don't have to worry about it and we also save on transpositions.
#
# The horizontal connections between the :math:`U` tensors are the matrix multiplications in the equation above.
# They are contractions over the virtual indices. The dangling vertical lines are the
# `physical` indices of the original state, :math:`\sigma_i.`
#
# Let us look at a concrete state vector with :math:`n=3` sites, so :math:`\psi_{\sigma_1 \sigma_2 \sigma_3},` and decompose it as an MPS.

n = 3 # three sites = three legs
psi = np.random.rand(2**3)
psi = psi / np.linalg.norm(psi)  # random, normalized state vector
psi = np.reshape(psi, (2, 2, 2)) # rewrite psi as rank-n tensor

##############################################################################
#
#
# We rewrite the tensor as a matrix with indices of the first site :math:`\sigma_1` and the combined indices of all remaining sites, :math:`(\sigma_2 \sigma_3).`
# Now that we have a matrix, we can perform SVD to split off the first site. Mathematically, this is
#
# .. math:: \psi_{\sigma_1 \sigma_2 \sigma_3} \stackrel{\text{reshape}}{=} \psi_{\sigma_1, (\sigma_2 \sigma_3)} \stackrel{\text{SVD}}{=} \sum_{\mu_1} U_{\sigma_1 \mu_1} \Lambda_{\mu_1} V^\dagger_{\mu_1 (\sigma_2 \sigma_3)}.
#

# reshape vector to matrix
psi = np.reshape(psi, (2, 2**(n-1)))
# SVD to split off first site
U, Lambda, Vd = np.linalg.svd(psi, full_matrices=False)

##############################################################################
#
# We multiply the singular values onto :math:`V^\dagger` and call this the remainder state, :math:`\psi'_{\mu_1, (\sigma_2 \sigma_3)},`
# so overall we have
# 
# .. math:: \psi_{\sigma_1 \sigma_2 \sigma_3} = \sum_{\mu_1} U_{\sigma_1 \mu_1} \psi'_{\mu_1, (\sigma_2 \sigma_3)}.
#
# Graphically, this corresponds to the following.
#
# .. figure:: ../_static/demonstration_assets/mps/psi_to_mps_1.png
#     :align: center
#     :width: 70%
#
# Note that because :math:`\Lambda` is diagonal, it has the same virtual index on either side.
#
# We keep the :math:`U` tensors. We want to maintain the convention that they are of the shape ``(virtual_left, physical, virtual_right).``
# Because there is no virtual index on the left for the first site, we introduce a dummy index of size ``1``.
# This is just to make the bookkeeping of the final MPS a bit simpler, as all tensors have the same shape structure.

Us = []
U = np.reshape(U, (1, 2, 2)) # mu1, s2, mu2
Us.append(U)

##############################################################################
#
# This procedure is repeated through all sites. The first step was special in that :math:`U_{\sigma_1 \mu_1}` is a vector for each value of :math:`\sigma_1.`
# When splitting up :math:`\psi'_{\mu_1, (\sigma_2 \sigma_3)}` we combine the virtual bond with the current site, and have all remaining sites be the other leg of the matrix we create for SVD.
# In particular, we do the following.
# 
# .. math:: \psi'_{\mu_1, (\sigma_2 \sigma_3)} \stackrel{\text{reshape}}{=} \psi'_{(\mu_1 \sigma_2), (\sigma_3)} \stackrel{\text{SVD}}{=} \sum_{\mu_2} U_{\mu_1 \sigma_2 \mu_2} \Lambda_{\mu_2} V_{\mu_2 \sigma_3}
#

psi_remainder = np.diag(Lambda) @ Vd                 # mu1 (s2 s3)
psi_remainder = np.reshape(psi_remainder, (2*2, 2))  # (mu1 s2), s3
U, Lambda, Vd = np.linalg.svd(psi_remainder, full_matrices=False)

U = np.reshape(U, (2, 2, 2)) # mu1, s2, mu2
Us.append(U)

U.shape, Lambda.shape, Vd.shape

##############################################################################
# We again multiply the singular values onto the new :math:`V^\dagger` and take that as the remainder state :math:`\psi''.`
# The state overall now reads
#
# .. math:: \psi_{\sigma_1 \sigma_2 \sigma_3} = \sum_{\mu_1 \mu_2} U_{\sigma_1\mu_1} U_{\mu_1 \sigma_2 \mu_2} \psi''_{\mu_2 \sigma_3}.
#
# Graphically, this corresponds to 
#
# .. figure:: ../_static/demonstration_assets/mps/psi_to_mps_2.png
#     :align: center
#     :width: 70%
#
# When the state is normalized, we are done. Otherwise, we can do the procedure one more time, again with a virtual dummy dimension on the right-most site.

psi_remainder = np.diag(Lambda) @ Vd                 # mu1 (s2 s3)
psi_remainder = np.reshape(psi_remainder, (2*2, 1))  # (mu1 s2), s3
U, Lambda, Vd = np.linalg.svd(psi_remainder, full_matrices=False)

U = np.reshape(U, (2, 2, 1)) # mu1, s2, mu2
Us.append(U)

U.shape, Lambda.shape, Vd.shape

##############################################################################
# Because our state vector was already normalized, the singular value in this last SVD is just ``1``, else it would yield the norm of ``psi``
# (a good exercise to confirm by skipping the normalization step in the definition of ``psi`` above).
#
# The collected tensors :math:`U_{\mu_{i-1} \sigma_i \mu_i}` now make up the Matrix Product State and describe the original state :math:`|\psi\rangle`
# by appropriately contracting the virtual indices :math:`\mu_i.` We can briefly confirm this by reverse engineering the original state. 
# 
#
# .. figure:: ../_static/demonstration_assets/mps/psi_to_mps_3.png
#     :align: center
#     :width: 70%
#
# 
# Due to the convention of
# the indices as ``(virtual_left, physical, virtual_right)``, the contraction is simple and we can use 
# `np.tensordot <https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html>`_ with ``axes=1``, indicating matrix-product-like contraction of the left-most and right-most index.
# This is another way of thinking of the obtained state as a **matrix product** state.

print(f"Shapes of Us: {[_.shape for _ in Us]}")

psi_reconstruct = Us[0]

for i in range(1, len(Us)):
    # contract the rightmost with the left most index
    psi_reconstruct = np.tensordot(psi_reconstruct, Us[i], axes=1)

print(f"Shape of reconstructed psi: {psi_reconstruct.shape}")
# remove dummy dimensions
psi_reconstruct = np.reshape(psi_reconstruct, (2, 2, 2))
# original shape of original psi
psi = np.reshape(psi, (2, 2, 2))

np.allclose(psi, psi_reconstruct)


##############################################################################
# 
# Up to this point, the description of the original state in terms of the MPS, made up by the three matrices :math:`U_{\mu_{i-1} \sigma_i \mu_i},` is exact.
# With this construction, the sizes of the virtual bonds grow exponentially from :math:`2` to :math:`2^{n/2}` until the middle of the chain (to be confirmed here below).
#
# Just like in the example with images before, we can compress the state by only keeping 
# the :math:`\chi` largest singular values with their respective singular vectors.
# This hyper-parameter :math:`\chi` is the bond dimension we mentioned earlier. It allows us 
# to control the amount of entanglement the state can represent between everything that 
# is left and right of the bond (more on that later).
#
# A full subroutine from :math:`|\psi\rangle` to its compressed MPS description is given by the following function ``dense_to_mps`.`
# It is convenient to also keep the singular values for each bond to easily change the orthonormality of the tensors, but more on that in the next section on canonical forms.

def split(M, bond_dim):
    """Split a matrix M via SVD and keep only the ``bond_dim`` largest entries."""
    U, S, Vd = np.linalg.svd(M, full_matrices=False)
    bonds = len(S)
    Vd = Vd.reshape(bonds, 2, -1)
    U = U.reshape((-1, 2, bonds))
    
    # keep only chi bonds
    chi = np.min([bonds, bond_dim])
    U, S, Vd = U[:, :, :chi], S[:chi], Vd[:chi]
    return U, S, Vd

def dense_to_mps(psi, bond_dim):
    """Turn a state vector ``psi`` into an MPS with bond dimension ``bond_dim``."""
    Ms = []
    Ss = []

    psi = np.reshape(psi, (2, -1))   # split psi[2, 2, 2, 2..] = psi[2, (2x2x2...)]
    U, S, Vd = split(psi, bond_dim)  # psi[2, (2x2x..)] = U[2, mu] S[mu] Vd[mu, (2x2x2x..)]

    Ms.append(U)
    Ss.append(S)
    bondL = Vd.shape[0]
    psi = np.tensordot(np.diag(S), Vd, 1)

    for _ in range(n-2):
        psi = np.reshape(psi, (2*bondL, -1)) # reshape psi[2 * bondL, (2x2x2...)]
        U, S, Vd = split(psi, bond_dim) # psi[2, (2x2x..)] = U[2, mu] S[mu] Vd[mu, (2x2x2x..)]
        Ms.append(U)
        Ss.append(S)

        psi = np.tensordot(np.diag(S), Vd, 1)
        bondL = Vd.shape[0]

    # dummy step on last site
    psi = np.reshape(psi, (-1, 1))
    U, _, _ = np.linalg.svd(psi, full_matrices=False)

    U = np.reshape(U, (-1, 2, 1))
    Ms.append(U)
    
    return Ms, Ss

##############################################################################
# Let us look at a larger state. First, let us observe how the dimension scales exponentially when we don't truncate the bonds.

n = 12
bond_dim = 10000

psi = np.random.rand(*[2]*n)
psi = psi/np.linalg.norm(psi)
Ms, Ss = dense_to_mps(psi, bond_dim)

[M.shape for M in Ms]

##############################################################################
# When setting a finite bond dimension :math:`\chi \leq 2^{n/2},` 
# we see the virtual bonds grow at the boundaries until reaching 
# the maximal bond dimension and staying constant thereafter until 
# reaching the other side.

Ms, Ss = dense_to_mps(psi, 5)

[M.shape for M in Ms]

##############################################################################
#
# This was all to conceptually understand the relationship between dense vectors and a compressed matrix product state.
# We want to use MPS for many sites, where it is often not possible to write down the exponentially large state vector in the first place.
# In that case, we would simply start from an MPS description in terms of :math:`n` :math:`\chi \times 2 \times \chi` tensors.
# Luckily, we can obtain all relevant information without ever reconstructing the full state vector.
# 
# .. _Canonical Forms:
#
# Canonical forms
# ~~~~~~~~~~~~~~~
#
# In the above construction, we unknowingly already baked in a very useful feature of our MPS because all the :math:`U` matrices from the SVD
# are left-orthonormal (highlighted by the pink color in the illustrations of the left-orthonormal :math:`U` tensors). In particular, they satisfy
#
# .. math:: \sum_{\sigma_i} \left(U^{\sigma_i} \right)^\dagger U^{\sigma_i} = \mathbb{I},
#
# which is a compact matrix notation treating :math:`U^{\sigma_i}` as a matrix for a concrete value of :math:`\sigma_i.`
# Making the coefficient in the matrix multiplication explicit we have
# 
# .. math:: \sum_{\sigma_i \mu_{i-1}} U^{*}_{\mu_{i-1} \sigma_i \mu'_i} U_{\mu_{i-1} \sigma_i \mu_i} = \mathbb{I}_{\mu'_i \mu_i}.
#
# Note that we only use the complex conjugation :math:`U^{*}` instead of Hermitian conjugate :math:`U^\dagger`
# because we can just choose the indices to contract over accordingly.
#
# Let us briefly confirm that:

for i in range(len(Ms)):
    id_ = np.tensordot(Ms[i].conj(), Ms[i], axes=([0, 1], [0, 1]))
    is_id = np.allclose(id_, np.eye(len(id_)))
    print(f"U[{i}] is left-orthonormal: {is_id}")

##############################################################################
# This is a very powerful identity as it tells us that contracting a site of the MPS from the left is just the identity.
#
# .. figure:: ../_static/demonstration_assets/mps/left_orthonormal.png
#     :align: center
#     :width: 30%
#
# This means that computing the norm, which is just contracting the MPS with itself, becomes trivial.
#
# .. figure:: ../_static/demonstration_assets/mps/norm_trivial.png
#     :align: center
#     :width: 70%
#
# The fact that we went through the MPS from left to right was a choice.
# We could have equivalently gone through the MPS from right to left and obtained a right-canonical state by keeping the right-orthonormal :math:`V^\dagger` of the decompositions.
#
# When computing expectation values, it is convenient to have the MPS in a mixed canonical form.
# Take some single-site observable :math:`O_i` for which we want to compute the expectation value.
# The best way to do this is to have the MPS such that all sites left of site :math:`i` are left-canonical and all sites right of it are right-canonical.
# That way, the contraction :math:`\langle \psi | O | \psi \rangle` for local expectation values reduces to contractions on just a single site,
# because all other contractions are just the identity.
# We call that single tensor :math:`\Theta,` it is all we need to compute the expectation value and we shall see how we can easily obtain it while not having to care about all other sites.
#
# .. figure:: ../_static/demonstration_assets/mps/mixed_canonical_observable.png
#     :align: center
#     :width: 70%
#
# We can obtain such a mixed canonical form by starting from our left-canonical MPS and going through the sites from right to left, thereby right-canonizing all sites until the observable.
# However, if we keep track of the singular values at all times, we can switch any site tensor from left- to right-orthonormal by just multiplying with the singular values.
# This is the so-called Vidal form introduced in [#Vidal]_ and it works like this: Even though we are not going to use them in our representation, it makes sense to introduce the "bare" local :math:`\Gamma`-tensors :math:`\{\Gamma^{\sigma_i}\}` in terms of
#
# .. math:: \Gamma^{\sigma_i} = \left(\Lambda^{[i-1]}\right)^{-1} U^{\sigma_i} = \left(V^\dagger\right)^{\sigma_i} \left(\Lambda^{[i]}\right)^{-1}
#
# and write the MPS in terms of those bare :math:`\Gamma`-tensors with the singular values connecting them. We can then just sub-select and recombine parts to get either right- or left-canonical tensors in the following manner.
#
# .. figure:: ../_static/demonstration_assets/mps/vidal.png
#     :align: center
#     :width: 70%
#
# In particular, to compute the expectation value described just above, we need the central tensor that we named :math:`\Theta.`
# We are not going to actually store the :math:`\Gamma`-tensors but continue to use the left-orthonormal :math:`U`-tensors.
# So all we need to do is construct :math:`\Theta_{\mu_{i-1} \sigma_i \mu_i} = \sum_{\tilde{\mu}_i} U_{\mu_{i-1} \sigma_i \tilde{\mu}_i} \Lambda^{[i]}_{\tilde{\mu}_i \mu_i}.` This has two advantages: 1) 
# we only need to perform one contraction with the singular values from the right and 2) we avoid having to compute any 
# inverses of the singular values, which numerically can become messy for very small singular values.
#
# Finally, the local observable expectation value simply becomes
#
# .. math:: \langle \psi | O | \psi \rangle = \text{tr}\left[ \sum_{\sigma_i \tilde{\sigma}_i} \Theta^{\sigma_i} O^{\sigma_i \tilde{\sigma}_i} \Theta^{*\tilde{\sigma}_i} \right],
#
# or, graphically, the following.
#
# .. figure:: ../_static/demonstration_assets/mps/final_expval.png
#     :align: center
#     :width: 50%
#
# The canonical form essentially allows us to treat local operations locally and remove all redundancy on other sites. This will come in handy later when we look at simulating quantum circuits with MPS.
# It also enables the very powerful density matrix renormalization group (DMRG) algorithm. Here, one constructs the ground state of a Hamiltonian by iteratively sweeping through the MPS back and forth, solving
# the eigenvalue problem locally at each site (with all other sites "frozen"). This works extremely well in practice and is hence still one of the workhorses of classical quantum simulation to this day. For a very good review
# on DMRG with MPS, see [#Schollwoeck]_.
#
# .. _Area Law:
#
# Entanglement and area laws
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Entanglement is best quantified via bipartitions of states. In particular, separating a full system :math:`|\psi\rangle` and a sub-system :math:`\rho_\text{sub},` the von Neumann entanglement entropy is given by
#
# .. math:: S(\rho_\text{sub}) = -\text{tr}\left[\rho_\text{sub} \log\left(\rho_\text{sub}\right) \right].
#
# In an MPS, the singular values of the bonds naturally encode the entanglement of bipartitions between all sites left vs all sites right of the bond.
# In particular, the von Neumann entanglement entropy at bond :math:`i` is given by
#
# .. math:: S(\rho_{1:i}) = S(\rho_{i+1:n}) = - \sum_{\mu_i=1}^{\chi} \Lambda^2_{\mu_i} \log\left( \Lambda_{\mu_i}^2 \right).
#
# Given a bond dimension :math:`\chi,` the maximal entanglement entropy we can obtain is for the all-equal distribution of singular values, :math:`\Lambda_i^2 \equiv 1/\chi.`
# The entanglement entropy is thus bounded by
#
# .. math:: S(\rho_{1:i}) \leq \log(\chi) = \text{const}.
#
# This is the area law of entanglement for one-dimensional systems. A "volume" in one spatial dimension is a line, and its surface area, :math:`\partial V,` two points, so constant in the system size (see note below).
#
# .. note::
# 
#     Ground states of local and gapped Hamiltonians are known to satisfy the area law of entanglement.
#     This law states that the entanglement entropy of a sub-system grows with its surface area :math:`\partial V` instead of its volume :math:`V.`
#     For one-dimensional systems, the volume is just a line and its surface area just a constant. The entanglement
#     between any such sub-system in an MPS with a finite bond dimension :math:`\chi` is naturally bounded by :math:`\log(\chi)=\text{const.},`
#     so MPS satisfy the area law of entanglement for one-dimensional systems.
# 
#     Projected entangled pair states (PEPS) are the natural generalization of MPS to regular 2D or 3D grids as well as more general graph connectivities and are known to fulfill the
#     respective area laws of entanglement, making them the correct ansätze for local and gapped Hamiltonians in those cases. For example, for a 2D
#     PEPS with a square subsystem of volume :math:`L \times L` and bond dimension :math:`\chi,` the entanglement entropy between the square and the rest of the system is bounded by
#     :math:`\log(\chi^L) = L \log(\chi)`, which is proportional to the circumference :math:`\propto L,` and not its area.
#
#     .. figure:: ../_static/demonstration_assets/mps/area_law.png
#         :align: center
#         :width: 50%



##############################################################################
# 
# 
# Quantum simulation with MPS
# ---------------------------
#
# We can use MPS to classically simulate quantum algorithms. This is a very useful tool for as long as
# real quantum devices are noisy and costly to use.
# An MPS simulator works very similarly to a state vector simulator like :class:`~DefaultQubit`, the only difference is that the underlying state
# is encoded in an MPS.
#
# Applying local gates
# ~~~~~~~~~~~~~~~~~~~~
#
# Due to its canonical form, applying local gates onto an MPS is very straightforward and little work.
# Let us walk through the example of applying a :math:`\text{CNOT}` gate on neighboring sites on the underlying MPS state of the simulator.
#
# Graphically, this is happening in three steps as illustrated here.
#
# .. figure:: ../_static/demonstration_assets/mps/apply_gate.png
#     :align: center
#     :width: 65%
#
# In the first step, we simply contract the appropriate physical indices of the MPS with those of the :math:`\text{CNOT}` matrix (which we reshape to :math:`2\times 2\times 2\times 2`).
# Note that we also take the bond singular values into account. The result is a big blob with two virtual indices and two physical indices. 
# We then just split this blob in the same way we split up the dense state vector and keep track of its singular values to restore the canonical form.
#
# Applying non-local gates
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# The situation gets a bit more complicated when we apply a :math:`\text{CNOT}` (or any multi-site) gate on non-neighboring sites. We have two possibilities to handle this.
# We can do what a quantum computer would do, which is swap out sites until the appropriate indices are neighboring, perform the operation, and then un-swap the sites.
# Alternatively, we can construct a so-called matrix product operator (MPO) that acts on all sites in between with an identity. This is done in the following way.
#
# First, we split the matrix of the gate into tensors that act locally. This is done again using SVD. In general, the MPO bond dimension for a two-qubit gate is maximally 4,
# but in some cases like :math:`\text{CNOT}` it is :math:`2,` so we can do a lossless compression by only keeping the two non-zero singular values and tossing the zeros. After we have done that, we can multiply 
# the singular values onto either site as we do not have a use for them in the MPO other than for doing the compression.
#
# .. figure:: ../_static/demonstration_assets/mps/cnot_split.png
#     :align: center
#     :width: 50%
#
# Now if we want to apply the CNOT gate on non-neighboring sites, we fill the intermediate sites with identities :math:`\mathbb{I} = \delta_{\sigma_i \sigma'_i} \delta_{\mu_{i-1} \mu_i},` and contract that larger
# MPO with the MPS.
#
# .. figure:: ../_static/demonstration_assets/mps/non_local_cnot.png
#     :align: center
#     :width: 80%
#
# Here we just need to be careful to contract in the right order, otherwise we might end up with unnecessarily large tensors in intermediate steps. For example, if we first contract all physical indices we get a big blob
# that is exponentially large in the number of intermediate sites. While it is in general NP-hard to find the optimal contraction path,
# for MPS the optimal path is known. The way to do it is by alternating between the physical index and the corresponding two virtual indices, going either from left to right or, equivalently, from right to left (see Figure 21 in [#Schollwoeck]_).
#
#
# Running simulations and setting the bond dimension
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# While we focussed on the specific case of a :math:`\text{CNOT}` gate, this concept is readily generalized to arbitrary two- or multi-qubit gates.
# With that, we are now ready to run some quantum circuit simulations. We don't have to code up all contractions by hand. Instead, we can use
# PennyLane's :class:`~pennylane.devices.default_tensor.DefaultTensor` device that takes care of all of this under the hood. All we need to do is set the bond dimension and tell the device whether
# it should use the swap-unswap or MPO method for applying non-local gates. This is done via the keyword argument ``contract``, where we can choose between ``"swap+split"`` (what I called swap-unswap),
# ``"nonlocal"`` (what I called the MPO method), and ``"auto-mps"``, which uses swap-unswap for 2-qubit gates and the MPO method for 3 and more qubits.
#
# Aside from that, we can basically use the device like any other state vector simulator device.
# Let us run a VQE example from using the `PennyLane Datasets <https://pennylane.ai/datasets/>`__ data for the :math:`H_6` molecule. What you will see here is mostly boilerplate code in PennyLane for using ``default.tensor`,` and you can
# see our :doc:`demo on how to use default.tensor <demos/tutorial_How_to_simulate_quantum_circuits_with_tensor_networks/>` for more details.

import pennylane as qml
qml.capture.enable()

[dataset] = qml.data.load("qchem", molname="H6", bondlength=1.3, basis="STO-3G")

H = dataset.hamiltonian # molecular Hamiltonian in qubit basis
n_wires = len(H.wires)  # number of qubits

def circuit():
    qml.BasisState(dataset.hf_state, wires=H.wires) # Hartree–Fock initial state
    for op in dataset.vqe_gates:                    # Applying all pre-optimized VQE gates
        qml.apply(op)               
    return qml.expval(H)                            # expectation value of molecular Hamiltonian

# set up device with hyper-parameters and kwargs
mps = qml.device("default.tensor", wires=n_wires, method="mps", max_bond_dim=30, contract="auto-mps")

# Create the QNode to execute the circuit on the device, and call it (w/o arguments)
res = qml.QNode(circuit, mps)()

# Compare MPS simulation result with pre-optimized state-vector result
res, dataset.vqe_energy

##############################################################################
# We've set the bond dimension to ``30`` kind of arbitrarily, and saw that we are pretty close to the exact result
# (note that exact here refers to the simulation method, not the ground state energy of the VQE result itself).
# But when dealing with systems of sizes where we don't have the means to compare to an exact result, how do know
# that our simulation results make sense, and are not scrambled by the errors introduced by a small bond dimension?
#
# The answer is **finite-size scaling** (sometimes also called bond dimension scaling or just extrapolation). This is a standard method 
# in tensor network simulations and originates from condensed matter physics and quantum phase transitions. 
# The idea is to run the same simulation with an increasing bond dimension and check that it saturates and converges to an extrapolated value.
# In spirit, this is similar to :doc:`zero noise extrapolation <demos/tutorial_diffable-mitigation>`.
#
# We choose a range of bond dimensions and plot the results for the simulation against them, keeping in mind that
# the maximum bond dimension of a system of :math:`n` qubits is :math:`2^{\frac{n}{2}}.`

bond_dims = 2**np.arange(2, (n_wires//2)+1) # maximum required bond dimension is 2**(n_wires//2) = 64
ress = []

for bond_dim in bond_dims:
    mps = qml.device("default.tensor", wires=n_wires, method="mps", max_bond_dim=bond_dim, contract="auto-mps")
    res = qml.QNode(circuit, mps)()
    ress.append(res)


plt.plot(bond_dims, ress, "x:", label="mps sim")
plt.hlines(dataset.vqe_energy, bond_dims[0], bond_dims[-1], label="exact result")
plt.xscale("log")
plt.xlabel("bond dim $\\chi$")
plt.xticks(bond_dims, bond_dims)
plt.legend()
plt.show()


##############################################################################
# We see that already for :math:`\chi = 32` we have pretty accurate results.
# We might even get away with :math:`\chi = 16` for some qualitative simulations at the cost of some approximation error.
#
# Setting the bond dimension is an important part of performing MPS simulations. Finite-size scaling is a
# quantitative tool to orient ourselves and choose a suitable bond dimension for our simulations.
# We can extrapolate the exact result for every simulation run,
# but this is of course more expensive as it requires multiple executions at different bond dimensions, and is not guaranteed to converge.
# Sometimes we are also just happy to get a cheap qualitative result for large systems that may or may not be 100% accurate.
#
# .. note:: 
#     ``default.tensor`` is based on `quimb <https://quimb.readthedocs.io/en/latest/index.html>`_. Both 
#     ``quimb`` and ``default.tensor`` are under active development, so in case you encounter some rough edges, please
#     submit a `GitHub issue <https://github.com/PennyLaneAI/pennylane/issues/new/choose>`_ in the PennyLane repository.


##############################################################################
# 
# Conclusion
# ----------
#
# In this demo we introduced the basics of matrix product states (MPS) and saw how the existence of a canonical form simplifies a lot of the contractions.
# This fact can also be used for simulations of quantum circuits with local and non-local gates.
# We showed how to run quantum circuits using PennyLane's :class:`~pennylane.devices.default_tensor.DefaultTensor` device and how to systematically find an appropriate bond dimension.
#
# While MPS are mathematically known to be well-suited to describe a particular class of states (those that fulfill the area law of entanglement in 1D), we can
# also simulate more complex systems by throwing some extra resources into the bond dimension. 



##############################################################################
# 
# References
# ----------
#
# .. [#Baiardi]
#
#     Alberto Baiardi, Markus Reiher
#     "The Density Matrix Renormalization Group in Chemistry and Molecular Physics: Recent Developments and New Challenges"
#     `arXiv:1910.00137 <https://arxiv.org/abs/1910.00137>`__, 2019.
#
# .. [#Patra]
#
#     Siddhartha Patra, Saeed S. Jahromi, Sukhbinder Singh, Roman Orus
#     "Efficient tensor network simulation of IBM's largest quantum processors"
#     `arXiv:2309.15642 <https://arxiv.org/abs/2309.15642>`__, 2023.
#
# .. [#Jiang]
#
#     H. C. Jiang, Z. Y. Weng, T. Xiang
#     "Accurate determination of tensor network state of quantum lattice models in two dimensions"
#     `arXiv:0806.3719 <https://arxiv.org/abs/0806.3719>`__, 2008.
#
# .. [#Vidal]
#
#     Guifre Vidal
#     "Efficient classical simulation of slightly entangled quantum computations"
#     `arXiv:quant-ph/0301063 <https://arxiv.org/abs/quant-ph/0301063>`__, 2003.
#
# .. [#Schollwoeck]
#
#     Ulrich Schollwoeck
#     "The density-matrix renormalization group in the age of Matrix Product States"
#     `arXiv:1008.3477 <https://arxiv.org/abs/1008.3477>`__, 2010.
#
