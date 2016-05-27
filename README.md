# VinDsl.jl: Fast and furious variational inference
![](http://www.duke.edu/~jmp33/assets/vindsl.png)

[![Documentation Status](https://readthedocs.org/projects/vindsljl/badge/?version=latest)](http://vindsljl.readthedocs.org/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/jmxpearson/VinDsl.jl.svg?branch=master)](https://travis-ci.org/jmxpearson/VinDsl.jl)
[![Coverage Status](https://coveralls.io/repos/github/jmxpearson/VinDsl.jl/badge.svg?branch=master)](https://coveralls.io/github/jmxpearson/VinDsl.jl?branch=master)
## **WARNING**:
VinDsl.jl is a work in progress. Not quite alpha, but watch this space!

## **DEPENDENCY ALERT**:
While updating of [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) for automatic differentiation is in progress, this package will rely on the *current master* of that package. Use `Pkg.checkout("Distributions")` to be sure you have the latest updates.

## **Development**:
For contributors: documentation of design, internals, and todos, see [here](http://vindsljl.readthedocs.org/en/latest/)

See also [this presentation](https://github.com/jmxpearson/VinDsl.jl/blob/master/doc/dukeML_feb_18_2016.ipynb)

## A Variational Inference Domain-Specific Language

[Variational inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) is an approximate method of statistical inference based on optimization. Unlike conventional Bayesian methods based on [Markov Chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) (MCMC), it scales well to large and streaming datasets, making it a competitive technique for machine learning applications.  

However, coding variational inference models by hand traditionally involves lots of tedious algebra and careful index accounting. New techniques like [black-box variational inference](http://www.cs.columbia.edu/~blei/papers/RanganathGerrishBlei2014.pdf) and [local expectation gradients](http://papers.nips.cc/paper/5678-local-expectation-gradients-for-black-box-variational-inference) allow much of this to be avoided, and implementations for some models are already possible in [Stan](http://mc-stan.org/), but no current framework allows these rapidly developing techniques to be mixed and matched by researchers.

The goal of VinDsl.jl is to provide a set of data abstractions and macros that take the pain out of coding variational inference models. In particular, because the syntactic sugar for defining models is implemented in the same language as the underlying building blocks, the entire framework is easily extensible and hackable.

### Things we have:
- Intelligent index handling: you define the model structure, VinDsl handles the sum over indices automatically
- A set of macros for coding conjugate models and updates
- Limited support for automatic expectation-taking
- built-in support for Hidden Markov Models

### On our hit list:
- Automatic differentiation ([it's coming](https://github.com/jmxpearson/DiffableDistributions.jl))
- support for state-space models
- variational deep networks
- GPU support

# Roadmap

## Short term:
This is what we need to get the *Neuron's eye view* models running:
- E-calculus:
    - implement V, C, Elog, etc. (need a way to define these rules more easily)
    - rename expandE ⟶ expand?
- Poisson Model:
    - [ ] factors where we take a product over some index, rather than a sum?
        - could have a function that makes an outer node in a factor inner, but this seems like a hack, and breaks the DSL facade
        - could look for `:ref`-headed `Expr`s and get all symbols used as indices; make these inner
    - [ ] gamma priors with linked shape/rate
    - [ ] pieces of Poisson with binarized exponent
    - [ ] BFGS optimization piece
        - right now, hard to do automatic differentiation until [this](https://github.com/JuliaStats/Distributions.jl/pull/430) is resolved ([more background](https://github.com/JuliaStats/Distributions.jl/issues/432))
- Log-Normal Model
    - [ ] **Jacobian of λ ⟶ η**
    - [ ] ExprNode that's a linear combo of pieces
        - [ ] V calculus for this
        - [ ] naturals for this
            - if V calculus not ready, can brute-force by breaking in pieces
    - bottleneck model (where η is a node)
        - [ ] implement Poisson
        - [ ] Lambert W for mean
        - [ ] explicit optimization for noise variance
- both models
    - [ ] iteration and convergence monitoring

## Medium-term:
- maps from each distribution's canonical (and/or natural) parameters to unconstrained parameters (like Stan)
- use these to set up optimization functions for BFGS/CG optimization steps
- when Julia 0.5 has fast anonymous functions and closures, this will be much easier
- update strategies:
    - [ ] SGD
    - [ ] LEG (local expectation gradients)
    - [ ] location-scale trick (like BBVI)


## Long-term
- build a @model macro that's fed a single (multiline) expression of tilde statements
- these statements are reparsed, replacing ~ with @factor and changing, e.g., Normal -> LogNormalFactor
- factors are made, then model is built from them
- `:=` defines ExprNode
- if a node definition contains an expression instead of a symbol in one of its arguments, gensym a new ExprNode and pass this to constructor
- implement Base.show for new types
- better error reporting for mismatched indices, etc.
- profile naturals using .+ vs + (probably bad) vs a custom in-place update for speed
- make sure project and project_inds return slices, not copies
- use CartesianRange instead of generated functions to define value(f::Factor)
- turn project and project_ind into macros that take x -> x[i_1, i_2] , etc. so as to avoid recurring project function call
- implement caching of expected values, etc. in RandomNodes (node can still be immutable if cache is a dict that gets cleared whenever data is updated); functions like E, Elog, V, etc. would need to be decorated to search cache first, then calculate, then update cache
- implement caching in ExprNodes of things like E(n)
- look carefully at Lora.jl api and graph structure of model
