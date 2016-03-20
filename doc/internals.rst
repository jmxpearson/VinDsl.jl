.. _internals:

Internals
=========

Overview
--------
At its core, VinDsl, like `Stan <http://mc-stan.org>`_, is nothing more than a set of tools for constructing objective functions to be optimized. But whereas Stan prioritizes stability and a broad non-statistical user base, VinDsl targets machine learning researchers attempting to prototype new algorithms. Its inference engine and domain-specific language are all in Julia, making it easily extensible. If it doesn't exist, you should be able to hack it yourself in Julia.

Mathematical framework
-----------------------
While nothing in the Variational Inference framework require that the underlying model be a graphical model, many models are. So VinDsl aims to make it easy to construct such models by organizing many of its underlying data structures around `factor graphs <https://en.wikipedia.org/wiki/Factor_graph>`_, as suggested by `this talk <http://people.csail.mit.edu/dhlin/jubayes/julia_bayes_inference.pdf>`_.

A factor graph can be thought of as a bipartite graph in which random variables form nodes and nodes connect to factors. In the case of variational inference, factors then represent terms in the optimization objective, also known as the evidence lower bound (ELBO).

Nodes
-------
A ``Node`` represents a variable in the factor graph defining the model. ``Node`` is an abstract type, with subtypes ``RandomNode`` (random variables), ``ConstantNode`` (constants and data), and ``ExprNode`` (see `Expression Nodes`_ below). Nodes can be created using the ``~`` macro:

.. code-block:: julia

    x[i] ~ Normal(ones(5), rand(5))

which is translated to

.. code-block:: julia

    x = RandomNode(:x,[:i],Normal,ones(5),rand(5))

In this case, VinDsl infers that the random variable ``x`` is indexed by ``i`` and checks that the two arguments to ``Normal`` have the same dimension. The resulting ``data`` field of ``x`` is an ``Array{Normal, 1}`` — an array of variables of type ``Normal`` as defined by `Distributions.jl <https://github.com/JuliaStats/Distributions.jl>`_.

More complicated cases are handled similarly:

.. code-block:: julia

    λ[i, j, k] ~ Poisson(rand(5, 3, 7))
    y[p, q] ~ MvNormalCanon([rand(3) for _ in 1:5], [eye(3) for _ in 1:5])

Once again, the dimensions of each index are inferred and checked for consistency. In the second case, because the entries in the final data array are ``MvNormalCanon`` (`multivariate normals with natural parameters <http://distributionsjl.readthedocs.org/en/latest/multivariate.html#multivariate-normal-distribution>`_), the entries in the arguments for μ and Σ must be vectors and matrices, respectively. That is, the ``data`` field of ``y`` is ``Array{MvNormalCanon, 5}``, each entry of which is a 3-vector-valued distribution. The implied distinction between indices ``p`` and ``q``, which index the entries of the random variable and the entries of the containing array, are further explored in `Factor Structure and Indices`_.

Factors
-------
Factors are collections of variables, along with a value formula expression that can be used to calculate the appropriate term in the objective function. In future, these will be defined in an ``@pmodel`` block, but currently, they can be defined using lower-level macros. For instance, a term in the generative (p) model of the form

.. math::
    y_{ij} \sim \mathrm{Normal}(\mu_j, \tau_j)

can be captured by defining a factor:

.. code-block:: julia

    y[i, j] ~ Const(rand(dims))
    μ[j] ~ Normal(zeros(dims[2]), ones(dims[2]))
    τ[j] ~ Gamma(1.1 * ones(dims[2]), ones(dims[2]))

    obs = @factor LogNormalFactor y μ τ

This last definition calls the constructor for the type ``LogNormalFactor <: Factor``, which calls ``get_structure`` on the provided list of nodes to create a ``FactorInds`` variable that can be used to define ``value(obs)``, the contribution of this factor to the ELBO.

VinDsl supports a number of predefined factors, but defining new ones is made simple by the ``@deffactor`` macro. For instance, the ``LogNormalFactor`` above is defined in VinDsl itself by

.. code-block:: julia

    @deffactor LogNormalFactor [x, μ, τ] begin
        -(1/2) * ((E(τ) * ( V(x) + V(μ) + (E(x) - E(μ))^2 ) + log(2π) + Elog(τ)))
    end

Note that defining a factor only requires three components:

1. A name for the factor

2. A list of canonical names for the nodes in the factor (these do *not* need to be the same as the nodes passed to creat the factor)

3. An expression (which can be put in a ``begin`` block) for the formula used to compute the value of the factor in terms of its nodes.

A few points to note about the value formula:

- It does not contain indices. The process of summing over indices is handled by VinDsl, which tracks and matches indices across nodes. Ultimately, the definition of ``value`` for each subtype of ``Factor`` uses Julia's `generated functions <http://docs.julialang.org/en/release-0.4/manual/metaprogramming/#generated-functions>`_ along with `Base.Cartesian <http://docs.julialang.org/en/release-0.4/devdocs/cartesian/>`_ to define an appropriate nested loop over all indices. In the final code, each node in the factor (``x``, ``μ``, and ``τ`` above) is fully indexed, requiring only that the relevant expression be defined on subtypes of ``Distribution`` (i.e., "atomic" random variables, not arrays of such variables).

- It makes use of a handful of specialized functions, ``E`` (expectation), ``V`` variance, ``Elog`` (expectation of :math:`\log x`). Most of these are aliased from ``mean``, ``var``, and the like from `Distributions.jl <https://github.com/JuliaStats/Distributions.jl>`_, while some, like ``Elog`` and ``Eloggamma`` are defined by VinDsl for those variables where the answer is known in closed form.

Factor Structure and Indices
----------------------------

Expression Nodes
--------------------------------
**EXPERIMENTAL!**

Models
------

Conjugate updates
-----------------

Automatic differentiation
-------------------------
