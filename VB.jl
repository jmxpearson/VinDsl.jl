"""
A Variational Bayesian modeling approach. Models are defined by 
bipartite graphs in which factors defining terms in the variational
objective are connected to nodes defining variables.
"""
module VB

using Distributions
import Base.convert
import Distributions: var, entropy

# data types, including VBModel, Factor, and Node
include("types.jl")

export VBModel, 
    Factor, EntropyFactor, LogNormalFactor,
    Node, RandomNode, ConstantNode,
    register,
    E, Elog, var, value

end  # module