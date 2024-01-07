
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

Important links
---------------

-  Static documentation - `Static
   documentation <https://scikit-optimize.github.io/>`__
-  Example notebooks - can be found in examples_.
-  Issue tracker -
   https://github.com/scikit-optimize/scikit-optimize/issues
-  Releases - https://test.pypi.org/project/skopt-modcn/

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


    pip install -i https://test.pypi.org/simple/ skopt-modcn==0.0.1

This installs an essential version of scikit-optimize-constrained.

Use
-------

Then use the following:


    import skopt_modcn
