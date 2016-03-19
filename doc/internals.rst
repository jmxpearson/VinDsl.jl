.. _internals

Internals
=========

Overview
--------
At its core, VinDsl, like `Stan <http://mc-stan.org>`_, is nothing more than a set of tools for constructing objective functions to be optimized. But whereas Stan prioritizes stability and a broad non-statistical user base, VinDsl targets machine learning researchers attempting to prototype new algorithms. Its inference engine and domain-specific language are all in Julia, making it easily extensible. If it doesn't exist, you should be able to hack it yourself in Julia.

Mathematical framework
-----------------------
While nothing in the Variational Inference framework require that the underlying model be a graphical model, many models are. So VinDsl aims to make it easy to construct such models by organizing many of its underlying data structures around `factor graphs <https://en.wikipedia.org/wiki/Factor_graph>`_, as suggested by `this talk <http://people.csail.mit.edu/dhlin/jubayes/julia_bayes_inference.pdf>`_.

A factor graph can be thought of as a bipartite graph in which random variables form nodes and nodes connect to factors. In the case of variational inference, factors then represent terms in the optimization objective, also known as the evidence lower bound (ELBO).
