Autotuner
=========

The genetic-algorithm-based autotuner tries to optimize a TC by tuning the available mapping options.

Each autotuning session starts with a set (:code:`population`) of candidate options
which can be initialized randomly and/or from known starting points. Each
candidate is benchmarked and the best ones have a higher chance of surviving
and breeding to produce the next generation of candidates. This procedure is
repeated for a pre-defined number of generations. In the end, the best candidate
is returned.

At the end of each generation new candidates must be selected. Each candidate
is either a combination of parent candidates (:code:`crossover`) or one that survives
from the previous generation. Both types are potentially randomly changed
(mutation). The top candidates (:code:`elites`) survive intact (without mutations)
between generations.

.. _autotuner_parameters:

Parameters for Autotuning
-------------------------

The parameters that control the autotuner's behavior are the following:

* :code:`Number of generations`: The number of tuning generation to be run.
* :code:`Population size`: The number of candidates in each generation.
* :code:`Number of elites`: The number of best candidates that are preserved intact between generations (without any mutations).
* :code:`Crossover rate`: The rate at which new candidates are bred instead of just surviving across generations.
* :code:`Mutation rate`: The rate at which candidate options are randomly changed (mutated).
* :code:`Number of threads`: The number of threads that are used to compile different candidates in parallel.
* :code:`GPUs`: A comma separated list of GPUs (ids) to use for evaluating candidates (e.g., "0,1,2,3").
* :code:`RNG state`: The state used to seed the tuner's RNG.
* :code:`min_launch_total_threads`: Prune out kernels mapped to fewer than this many threads and block. Set this to :code:`1` to avoid pruning.

Caching
-------

After each autotuning session the best candidates' profiling information and compilation results are stored in a cache. They can be subsequently retrieved to seed a new autotuning session.
