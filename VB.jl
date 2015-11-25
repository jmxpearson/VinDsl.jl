module VB
@doc """
A Variational Bayesian modeling approach. Models are defined by 
bipartite graphs in which factors defining terms in the variational
objective are connected to nodes defining variables.
""" -> current_module()

using Distributions

# data types, including VBModel, Factor, and Node
include("types.jl")

export VBModel, Factor, Node, RandomNode, E

end  # module