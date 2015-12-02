"""
A Variational Bayesian modeling approach. Models are defined by 
bipartite graphs in which factors defining terms in the variational
objective are connected to nodes defining variables.
"""
module VB

using Distributions
import Base: convert, call
import Distributions: var, entropy

# data types, including VBModel, Factor, and Node
include("expfam.jl")
include("types.jl")

export VBModel, 
    Factor, FactorInds, 
    EntropyFactor, LogNormalFactor, LogGammaFactor,
    Node, RandomNode, ConstantNode,
    register, check_conjugate, update!, get_structure,
    E, Elog, Eloggamma, var, value, naturals

end  # module