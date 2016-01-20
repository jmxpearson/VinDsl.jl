"""
A Variational Bayesian modeling approach. Models are defined by
bipartite graphs in which factors defining terms in the variational
objective are connected to nodes defining variables.
"""
module VB

using Distributions
import Base: convert, call, zero
import Distributions: var, entropy, cov
using Base.Cartesian

zero_like(A::Array) = zeros(A)
zero_like(x::Number) = zero(x)

include("expfam.jl")
# data types, including VBModel, Factor, and Node
include("HMM.jl")
include("types.jl")

export VBModel,
    Factor, FactorInds, @factor, get_structure, project, project_inds, @wrapvars, value,
    EntropyFactor, LogNormalFactor, LogGammaFactor, LogMvNormalCanonFactor,
    LogMvNormalDiagCanonFactor,
    @deffactor,
    Node, RandomNode, ConstantNode, @~,
    register, check_conjugate, update!,
    E, Elog, Eloggamma, Elogdet, V, H, value, naturals, @defnaturals,
    get_node_size, get_name_mapping

end  # module
