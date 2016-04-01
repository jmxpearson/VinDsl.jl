.. VinDsl.jl documentation master file, created by
   sphinx-quickstart on Sat Mar 19 14:59:53 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VinDsl.jl: Fast and Furious Variational Inference
=================================================
`VinDsl.jl <https://github.com/jmxpearson/VinDsl.jl>`_ is a Julia package that aims to provide a fast, flexible, thoroughly hackable domain-specific language (DSL) for variational Bayesian inference.

In particular VinDsl features:

* Intelligent index handling: you define the model structure, VinDsl handles the sum over indices automatically
* A set of macros for coding conjugate models and updates
* Limited support for automatic expectation-taking
* built-in support for Hidden Markov Models

Contents:

.. toctree::
   :maxdepth: 2

   internals.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
