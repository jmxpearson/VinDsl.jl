# let's define some types

"Defines a factor, a term in the variational objective."
abstract Factor

# a node is either a random variable or a constant
# constant is not necessarily scalar (e.g., Multivariate Normal)
typealias Node Distribution
typealias NodeArray{N} Array{Node, N}

# define outer constructor for making node arrays
call{D <: Distribution}(::Type{NodeArray}, ::Type{D}, pars...) = map(D, pars...)

"Defines a Variational Bayes model."
type VBModel  
    # nodes maps symbols to the nodes/groups of nodes associated with them
    nodes::Dict{Symbol, Union{Array{Node}, Node}}

    # all factors in the graph
    factors::Vector{Factor}

    # dictionary linking all random variables to a list of tuples
    # each tuple gives a factor and the name of the random variable
    # in that factor
    graph::Dict{Distribution, Vector{Tuple{Factor, Symbol}}}

    VBModel(nodes, factors) = begin
        for f in factors
            register(f)
        end
        new(nodes, factors)
    end
end


# register a factor with its associated nodes in the graph
function register(f::Factor, m::VBModel) 
    for var in fieldnames(f)
        n = getfield(f, var)
        if isa(n, Distribution)
            push!(m.graph, n => (f, var))
        end
    end
end

function check_conjugate(n::Distribution, m::VBModel)
    is_conj = Bool[method_exists(naturals, Tuple{typeof(f), Type{Val{s}}, typeof(n)}) for (f, s) in m.graph[n]]
    all(is_conj)
end

# define some factors
type EntropyFactor <: Factor
    x::Node
end

type LogNormalFactor <: Factor
    x::Union{Node, Float64}
    μ::Union{Node, Float64}  # mean
    τ::Union{Node, Float64}  # precision
end

type LogGammaFactor <: Factor
    x::Union{Node, Float64}
    α::Union{Node, Float64}  # shape
    β::Union{Node, Float64}  # rate
end

# define an expectation method on Distributions
"Calculate the expected value of a Node x."
E(x::Distribution) = mean(x)

# Define functions for nonrandom nodes.
# In each case, a specialized method is already defined for distributions.
E(x) = x
var(x) = zero(x)
entropy(x) = zero(x)
Elog(x) = log(x)
Eloggamma(x) = lgamma(x)

"Calculate the contribution of a Factor f to the objective function."
value(f::LogNormalFactor) = -(1/2) * ((E(f.τ) * ( var(f.x) + var(f.μ) + 
    (E(f.x) - E(f.μ))^2 ) + log(2π) + Elog(f.τ)))

value(f::LogGammaFactor) = (E(f.α) - 1) * E(f.x) - E(f.β) * E(f.x) + 
    E(f.α) * E(f.β) - Eloggamma(f.α)

value(f::EntropyFactor) = entropy(f.x)


"Return natural parameters from a Factor f viewed as a distribution for 
a given symbol. The last parameter is a type check for conjugacy."
naturals(f::LogNormalFactor, ::Type{Val{:x}}, ::Normal) = begin
    μ, τ = E(f.μ), E(f.τ)
    (μ .* τ, -τ/2)
end
naturals(f::LogNormalFactor, ::Type{Val{:μ}}, ::Normal) = begin
    x, τ = E(f.x), E(f.τ)
    (x .* τ, -τ/2)
end
naturals(f::LogNormalFactor, ::Type{Val{:τ}}, ::Gamma) = begin
    v = var(f.x) + var(f.μ) + (E(f.x) - E(f.μ)).^2
    (1/2, v/2)
end
naturals(f::LogGammaFactor, ::Type{Val{:x}}, ::Gamma) = (E(f.α) - 1, -E(f.β))

# "Update a RandomNode n."
# function update!{D}(n::RandomNode{D}, ::Type{Val{:conjugate}})
#     # get natural parameter vectors for each factor
#     nlist = [naturals(f, Val{s}, n) for (f, s) in n.factormap]

#     # sum all natural parameter vectors
#     # zip converts a list of natural parameter vectors for each factor into 
#     # a list of factors for each element of the natural parameter vector
#     # we then map + over each of these lists
#     totals = map(x -> +(x...), zip(nlist))

#     # update each distribution in the array
#     for idx in eachindex(n.data)
#         natpars = Any[par[idx] for par in totals]
#         n.data[idx] = D(naturals_to_params(natpars, D)...)
#     end
# end

