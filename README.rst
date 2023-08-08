
----------------------------------------------------------------------------------------------------

This is a modified version of Scikit-Optimize. You can now add "space_constraint" to skopt. 

Additionally, you can use the following options with constrained optimization using the present modification of skopt:
initial_point_generator="grid_modified",
initial_point_generator="lhs_modified",
initial_point_generator="lhs_pydoe",
initial_point_generator="sobol_scipy",

-----------------------------------------------------------------------------------------------------
About Scikit-Optimize:

Scikit-Optimize
===============

Scikit-Optimize, or ``skopt``, is a simple and efficient library to
minimize (very) expensive and noisy black-box functions. It implements
several methods for sequential model-based optimization. ``skopt`` aims
to be accessible and easy to use in many contexts.

The library is built on top of NumPy, SciPy and Scikit-Learn.

We do not perform gradient-based optimization. For gradient-based
optimization algorithms look at
``scipy.optimize``
`here <http://docs.scipy.org/doc/scipy/reference/optimize.html>`_.

.. figure:: https://github.com/scikit-optimize/scikit-optimize/blob/master/media/bo-objective.png
   :alt: Approximated objective

Approximated objective function after 50 iterations of ``gp_minimize``.
Plot made using ``skopt.plots.plot_objective``.

Important links
---------------

-  Static documentation - `Static
   documentation <https://scikit-optimize.github.io/>`__
-  Example notebooks - can be found in examples_.
-  Issue tracker -
   https://github.com/scikit-optimize/scikit-optimize/issues
-  Releases - https://pypi.python.org/pypi/scikit-optimize

Install
-------

scikit-optimize requires

* Python >= 3.6
* NumPy (>= 1.13.3)
* SciPy (>= 0.19.1)
* joblib (>= 0.11)
* scikit-learn >= 0.20
* matplotlib >= 2.0.0

You can install the latest release with:
::

    pip install scikit-optimize

This installs an essential version of scikit-optimize. To install scikit-optimize
with plotting functionality, you can instead do:
::

    pip install 'scikit-optimize[plots]'

This will install matplotlib along with scikit-optimize.

In addition there is a `conda-forge <https://conda-forge.org/>`_ package
of scikit-optimize:
::

    conda install -c conda-forge scikit-optimize

Using conda-forge is probably the easiest way to install scikit-optimize on
Windows.