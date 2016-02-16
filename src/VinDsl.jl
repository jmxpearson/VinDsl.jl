"""
A Variational Bayesian modeling approach. Models are defined by
bipartite graphs in which factors defining terms in the variational
objective are connected to nodes defining variables.
"""
module VinDsl

using Distributions
using PDMats
using Optim
using ForwardDiff
import Base: convert, call, zero, getindex, setindex!, ndims
import Distributions: var, entropy, cov
using Base.Cartesian

# general helper functions
include("utils.jl")

# new distributions and extended functionality
include("distributions.jl")

# type system
include("types.jl")

# operations and structure on types
include("structure.jl")

# inference methods in models
include("inference.jl")

# tools for working with expressions
include("expressions.jl")

# random variable calculus
include("expectations.jl")

# language macros
include("dsl.jl")

# predefined factors
include("factors.jl")

# predefined conjugate updates
include("conjugate_relations.jl")

export VBModel,
    Factor, FactorInds, @factor, get_structure, project, project_inds, @wrapvars, value,
    EntropyFactor, LogNormalFactor, LogGammaFactor, LogMvNormalCanonFactor,
    LogMvNormalDiagCanonFactor, LogMarkovChainFactor, LogDirichletFactor,
    LogMarkovMatrixFactor,
    _expandE, _expand_wrapE, get_all_syms, @exprnode,
    @deffactor, @~,
    Node, RandomNode, ConstantNode, ExprNode, ExprDist,  @expandE, nodeextract,
    register, check_conjugate, update!, unroll_pars, update_pars!, reroll_pars,
    get_par_sizes, flatten,
    E, Elog, Eloggamma, Elogdet, V, H, C, value, naturals, @defnaturals,
    get_node_size, get_name_mapping, HMM, MarkovChain, MarkovMatrix,
    nstates, naturals, naturals_to_params, forwardbackward

end  # module