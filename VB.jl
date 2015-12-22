"""
A Variational Bayesian modeling approach. Models are defined by
bipartite graphs in which factors defining terms in the variational
objective are connected to nodes defining variables.
"""
module VB

using Distributions
import Base: convert, call, zero
import Distributions: var, entropy
using Base.Cartesian

zero_like(A::Array) = zeros(A)
zero_like(x::Number) = zero(x)

include("expfam.jl")
# data types, including VBModel, Factor, and Node
include("types.jl")

export VBModel,
    Factor, FactorInds, @factor, get_structure, project, @wrapvars, value,
    EntropyFactor, LogNormalFactor, LogGammaFactor, LogMvNormalCanonFactor,
    @deffactor,
    Node, RandomNode, ConstantNode, @~,
    register, check_conjugate, update!,
    E, Elog, Eloggamma, var, value, naturals, @defnaturals,
    get_node_size, get_name_mapping

end  # module
