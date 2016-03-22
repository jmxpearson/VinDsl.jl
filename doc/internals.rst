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
VinDsl's handling of indices through ``FactorInds`` structure objects represents both one of its principal advantages (in facilitating model definitions) and one of its largest sources of complexity under the hood. This stems at least in part from the fact that not all distributions in the `Distributions.jl <https://github.com/JuliaStats/Distributions.jl>`_ package are univariate, and so there is an intrinsic difficulty in handling the distinct between indices *within* multivariate distributions and indices for *replicates* of distributions. In VinDsl, this is captured by the distinction between *inner* and *outer* indices:

inner indices
    Vector-valued distributions like the Dirichlet or multivariate Normal are treated as having a single inner index. Matrix-valued distributions like the Wishart are treated as having two inner indices. These indices **must** be listed first in the definitions of ``Node`` objects when constructed through the ``~`` macro.

    Two notes:

    - Inner indices are not strictly required, if they do not need to be matched across nodes. However, for clarity, they should be included.

    - Somewhat counterintuitively, the covariance/precision matrices for multivariate Normal distributions should have only a *single* index. That is, you want to write

        .. code-block:: julia

            Λ[i, i] ~ Wishart(...)

      so that *both* dimensions of the matrix are appropriately matched with other variables, as explained below.

outer indices
    Are everything else. These indices correspond to the dimensions of arrays containing the distribution variables. These indices are checked for consistent sizing across arguments to node definitions and across nodes within factors.

**Factor Structure**:

Put simply, the goal of determining the factor structure is to ensure that the ``value`` function defined on each factor correctly sums over all node indices to produce a scalar value. Specifically, this process specifies how to take the value formula from the definition of the factor and supply all the indices in a way that transforms it into legitimate Julia code to go inside a loop.

For the case of scalar variables only, this is trivial: just use `Base.Cartesian <http://docs.julialang.org/en/release-0.4/devdocs/cartesian/>`_ to define a nested loop over the union of all indices and use the VinDsl functions ``project`` and ``project_inds`` to transform the nodes in their elemental distributions. But this process is significantly complicated in the case of inner indices, where we would like to be able to define, as VinDsl does, factors like

.. code-block:: julia

    @deffactor LogMvNormalCanonFactor [x, μ, Λ] begin
        δ = E(x) - E(μ)
        EΛ = E(Λ)
        -(1/2) * (trace(EΛ * (C(x) .+ C(μ) .+ δ * δ')) + length(x) * log(2π) - Elogdet(Λ))
    end

which (implicitly) treats x and μ as vectors. But what if x is ``MvNormal`` and μ is ``Array{Normal, 1}``? This dilemma is solved by the inner constructor of the factor.

When a factor is defined, the ``get_structure`` function is called. It takes the list of nodes provided for the factor and

1. Figures out which indices are "fully outer." These indices are not inner for *any* node in the factor. In effect, these are all the indices we can trivially sum over.

2. Figure out the maximum values of every index and make sure these are consistent across nodes. This defines the limits of the sums over indices in ``value``.

3. Define a mapping (``inds_in_factor``) mapping the name of each node to the (integer) indices within the *factor's* total set that index it.

4. Define another mapping (``inds_in_node``) mapping the name of each node to the (integer) indices within *that node's* total set that are involved in the factor.

These last two mappings are then used by functions like ``project`` and ``project_inds`` to take a tuple of all fully outer indices and select from that the appropriate element of a node with fewer dimensions. That is, VinDsl takes a value formula like

.. code-block:: julia

    -(1/2) * (trace(EΛ * (C(x) .+ C(μ) .+ δ * δ')) + length(x) * log(2π) - Elogdet(Λ))

wraps each variable in a call to ``project``, and evaluates the (scalar) result. The final trick needed to understand all this is that functions like ``E`` and ``C`` (the covariance) transform distributions into scalars, vectors, and matrices (for scalar, vector-, and matrix-valued random variables, respectively) but also map over ``Arrays``, so that nodes that are not fully indexed still end up as multidimensional arrays in a way that makes sense.

More explicitly, in the model mentioned above with ``x[i]`` an ``MvNormal`` node and ``μ[i]`` an ``Array{Normal, 1}``, the end result is:

- ``i`` is an outer index for ``μ`` but an inner index for ``x``. It is thus not fully outer and treated as an inner index for all the nodes in the factor.

- As a result, ``i`` is not explicitly summed over. In the value formula, once nodes are projected down to their "atomic" distribution components, ``x`` is an ``MvNormal`` distribution so that ``E(x)`` is a vector and ``C(x)`` a matrix. However, ``μ`` is *not* a distribution, but a (vector) slice of an array of distributions. Yet the expectation functions also work elementwise on arrays so that ``E(μ)`` is a vector and ``C(μ)`` a diagonal matrix. As a result, the formula obviates the need to worry about all "trivial" (fully outer indices), requiring only that the programmer define the kernel of the computation.

Expression Nodes
--------------------------------
**EXPERIMENTAL!**

In many models, it is convenient to define new random variables as deterministic functions of other nodes in the model. For instance, we might want to define a new variable x as a linear transformation of variables z: :math:`x = a + B \cdot z`. In the language of factor graphs, we could think of this as a "Lagrange multiplier factor" that ties the variables x and z, enforcing the constraint, but VinDsl uses a hybrid "expression node" to define x in terms of z:

.. code-block:: julia

    x := a + B * z

Note that this doesn't currently work. Instead, one must use the ``@exprnode`` macro:

.. code-block:: julia

    @exprnode x (a + B * z)

which translates (in part) to the constructor call:

.. code-block:: julia

    x = ExprNode(:x, :(a + B * z), Node[a, B, z])

Given this code, VinDsl constructs an ``ExprNode``, which calls ``get_structure`` (just like a factor) to determine the appropriate relationships among the indices for the constituent nodes.

What's more important (and trickier) is how ``@exprnode`` uses the supplied expression to calculate various expectations (``E``, ``V``, etc.) of the node x. Automating this calculation involves several steps:

1. For every expression node, a new ``ExprDist{V <: Val} <: Distribution`` is defined.

2. The macro defines node-specific versions of ``E``, ``V``, etc. that dispatch on this distribution type. These versions call several other macros that:

    - Wrap the expression defining the node in the appropriate expectation call (e.g., ``E``).

    - Wrap each symbol in a call to ``nodeextract``, which translates the symbol to the node variable.

    - Call ``@simplify`` on the result and use the resulting formula expression to define the function.

Of these steps, the most difficult is the definition of ``@simplify``. The macro does know some things. For instance[1]_:

.. code-block:: julia

    @simplify E(x.data[1] + y.data[1])
    E(x.data[1]) + E(y.data[1])

    @simplify E(x.data[1] * y.data[1] + 5)
    E(x.data[1]) * E(y.data[1]) + 5

but providing an entire computer algebra system is beyond the scope of the project, and it's unclear at present how much functionality will be supported. The details are in ``dsl.jl`` and involve the ``_simplify*`` functions that manipulate the AST. As always the tests (``expressiontests.jl``) are currently the best documentation for what works and what doesn't.

.. [1] Note that ``@simplify`` assumes that nodes are independent, so that expectations of products are products of expectations.

Models
------
Models are currently pretty primitive. Models can be defined by

.. code-block:: julia

    m = VBModel(<list of nodes>, <list of factors>)

The ``VBModel`` constructor then constructs a factor graph (essentially a dictionary linking nodes to the factors that contain them) and performs some simple checks. Currently, the check is whether any given node is conjugate to all its factors, so that `conjugate updates`_ are possible. Each node in the graph is then supplied with an ``update_strategy``, which determines what algorithm is used to update the parameters of the node's posterior. The ``update!`` function then dispatches on the value of this strategy.

Update strategies are loaded in ``inference.jl``, which loads files from the ``inference`` folder.

Conjugate updates
-----------------
VinDsl does not currently have the power to determine conjugacy on its own. Rather, it relies on checking against possible conjugate updates provided with the ``@defnaturals`` macro:

.. code-block:: julia

    @defnaturals LogNormalFactor μ Normal begin
        Ex, Eτ = E(x), E(τ)
        (Ex * Eτ, -Eτ/2)
    end

This macro takes as its arguments a factor, a node within that factor (the name given to the variable in that factor's value formula, not the node), a distribution conjugate to that variable in that factor, and a formula specifying how to calculate the natural parameter updates for the given distribution from the factor. Much like the ``@deffactor`` macro, ``@defnaturals`` requires only that the formula defining the natural parameters be defined for a kernel of the calculation. VinDsl handles all the appropriate index summations through the ``naturals`` function in ``conjugacy.jl``. In addition, this machinery relies on definitions of natural parameters provided in the ``distributions`` folder for canonical exponential family forms. Conventions are as `here <https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions>`_.

When the ``update!`` function is called on a node that is conjugate to all factors connected with it, VinDsl calls ``naturals`` on each of these factors, which in return provide tuples of natural parameter "messages". These messages are then summed elementwise and used to update the node.

Automatic differentiation
-------------------------
Coming soon!

Automatic forward-mode differentiation will be handled through `ForwardDiff.jl <https://github.com/JuliaDiff/ForwardDiff.jl>`_. When the elbo is a sum over ``value(f)`` for all factors ``f``, the idea will be to create a wrapper function that takes as its lone argument an "unrolled" vector ``x``, "re-rolls" it into parameters for each of the nodes, and sums the value of each factor in the model. This ELBO function will then be differentiated as a function of ``x`` and the corresponding derivatives "re-rolled" and used to update the individual node parameters. 
