# VinDsl.jl: Fast and furious variational inference
![](http://www.duke.edu/~jmp33/assets/vindsl.png)

[![Documentation Status](https://readthedocs.org/projects/vindsljl/badge/?version=latest)](http://vindsljl.readthedocs.org/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/jmxpearson/VinDsl.jl.svg?branch=master)](https://travis-ci.org/jmxpearson/VinDsl.jl)
[![Coverage Status](https://coveralls.io/repos/github/jmxpearson/VinDsl.jl/badge.svg?branch=master)](https://coveralls.io/github/jmxpearson/VinDsl.jl?branch=master)
<a href="https://zenhub.com"><img src="https://raw.githubusercontent.com/ZenHubIO/support/master/zenhub-badge.png"></a>
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
- preliminary support for [ADVI](http://arxiv.org/abs/1603.00788)

### On our hit list:
- Automatic differentiation (it's coming)
- support for state-space models
- variational deep networks/autoencoders
- GPU support
