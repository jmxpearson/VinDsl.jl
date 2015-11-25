# let's define some types

abstract VBModel  

"Defines a factor, a term in the variational objective."
abstract Factor

"Defines a node (variable) in the model."
abstract Node

"Node corresponding to a collection of random variables."
type RandomNode{D <: Distribution, N} <: Node
    data::Array{D, N}
    factors::Vector{Factor}
end
RandomNode{D <: Distribution}(d::Type{D}, pars...) = RandomNode(map(d, pars...), Factor[])

type ConstantNode{T <: Number, N} <: Node
    data::Array{T, N}
    factors::Vector{Factor}
end

convert(::Type{Node}, x::Array) = ConstantNode(x, Factor[])

# bind a factor and a node in the graph
bind(n::Node, f::Factor) = push!(n.factors, f)

# define an expectation method on Nodes
"Calculates the expected value of a Node x."
E(x::RandomNode) = map(mean, x.data)
E(x::ConstantNode) = x.data

"Calculates the variance of a Node x."
var(x::RandomNode) = map(var, x.data)
var(x::ConstantNode) = zeros(x.data)

"Calculates the expected value of the log of a Node x."
Elog(x::ConstantNode) = map(log, x.data)


# define some factors
type LogNormalFactor 
    x::Node
    μ::Node
    τ::Node
end

value(f::LogNormalFactor) = sum(-(1/2) * (E(f.τ) .* ( var(f.x) + var(f.μ) + 
    (E(f.x) - E(f.μ)).^2 ) + log(2π) + Elog(f.τ)))
