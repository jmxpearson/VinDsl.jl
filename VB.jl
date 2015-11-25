"""
A Variational Bayesian modeling approach. Models are defined by 
bipartite graphs in which factors defining terms in the variational
objective are connected to nodes defining variables.
"""
module VB

using Distributions
import Base.convert

# data types, including VBModel, Factor, and Node
include("types.jl")

export VBModel, Factor, Node, RandomNode, ConstantNode,
    E, Elog, var,
    LogNormalFactor, value

end  # module