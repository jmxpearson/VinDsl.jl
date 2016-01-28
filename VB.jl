"""
A Variational Bayesian modeling approach. Models are defined by
bipartite graphs in which factors defining terms in the variational
objective are connected to nodes defining variables.
"""
module VB

using Distributions
using PDMats
using Optim
using ForwardDiff
import Base: convert, call, zero
import Distributions: var, entropy, cov
using Base.Cartesian

zero_like(A::Array) = zeros(A)
zero_like(x::Number) = zero(x)
lB(x::Vector) = sum(lgamma(x)) - lgamma(sum(x))
lB(x::Matrix) = sum(lgamma(x), 1) - lgamma(sum(x, 1))  # columnwise

# define a convert method for Arrays to PDMats (positive definite matrices)
# if the array is not posdef, this will throw an exception
convert{T <: Number}(::Type{PDMat}, arr::Array{T}) = PDMat(arr)

include("expfam.jl")
# data types, including VBModel, Factor, and Node
include("HMM.jl")
include("types.jl")

export VBModel,
    Factor, FactorInds, @factor, get_structure, project, project_inds, @wrapvars, value,
    EntropyFactor, LogNormalFactor, LogGammaFactor, LogMvNormalCanonFactor,
    LogMvNormalDiagCanonFactor, LogMarkovChainFactor, LogDirichletFactor,
    LogMarkovMatrixFactor,
    _expandE, get_all_syms, @exprnode,
    @deffactor,
    Node, RandomNode, ConstantNode, ExprNode, @~,
    register, check_conjugate, update!, unroll_pars, update_pars!, reroll_pars,
    get_par_sizes, flatten,
    E, Elog, Eloggamma, Elogdet, V, H, C, value, naturals, @defnaturals,
    get_node_size, get_name_mapping, HMM, MarkovChain, MarkovMatrix

end  # module
