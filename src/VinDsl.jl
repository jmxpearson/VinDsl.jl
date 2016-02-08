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

include("utils.jl")
include("distributions.jl")
include("types.jl")
include("structure.jl")
include("expectations.jl")
include("expressions.jl")
include("dsl.jl")
include("conjugacy.jl")
include("factors.jl")
include("factornaturals.jl")

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
