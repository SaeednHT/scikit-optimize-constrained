

Constrained Scikit-Optimize
===========================
![Static Badge](https://img.shields.io/badge/skopt_modcn-yellow)
![Static Badge](https://img.shields.io/badge/test.pypi-0.0.1-green)
![GitHub Repo stars](https://img.shields.io/github/stars/SaeednHT/scikit-optimize-constrained)



Constrained Scikit-Optimize  ``skopt_modcn`` is a modified version of Scikit-Optimize ``skopt``. You can now add ``space_constraint`` to skopt.

Additionally, you can use the following options to use costrained initial points with your constrained optimization ``skopt_modcn``:

initial_point_generator="grid_modified",

initial_point_generator="lhs_modified",

initial_point_generator="lhs_pydoe",

initial_point_generator="sobol_scipy",


-----------------------------------------------------------------------------------------------------
## Releases in pypi
* [PYPI Releases](https://test.pypi.org/project/skopt-modcn/)

## Prerequisites

Constrained scikit-optimize requires

* Python >= 3.6
* NumPy (>= 1.13.3)
* SciPy (>= 0.19.1)
* joblib (>= 0.11)
* scikit-learn >= 0.20
* matplotlib >= 2.0.0

How to install?
---------------

You can install the latest release with the following command:


    pip install -i https://test.pypi.org/simple/ skopt-modcn==0.0.1

This installs an essential version of scikit-optimize-constrained.

How to use?
-----------

Then use the following in python:


    import skopt_modcn


Sample optimization without constrains:
---------------------------------------
```
# Import skopt and numpy
import skopt
import numpy as np

# Define the objective function with two inputs
def f(x):
    # You can modify this function according to your problem
    return ( x[0] ) ** 2 + ( x[1] ) ** 2

# Define the domain of the inputs as two real intervals
space = [(-2.0, 2.0), (-2.0, 2.0)]

# Use skopt.gp_minimize function to find the minimum of the function
res = skopt.gp_minimize(f, space, n_calls=20, n_initial_points=10, initial_point_generator = "lhs",\
                        random_state=1, verbose=True)

# Get the best point and the best value observed so far
x_best = res.x
y_best = res.fun

print('The input of the minimum objective function is: '+ str(x_best))
print('The minimum of the objective function is: '+ str(y_best))
```
Sample optimization with constrains:
---------------------------------------
```
# Import skopt_modcn and numpy
import skopt_modcn
import numpy as np

# Define the objective function with two inputs
def f(x):
    # You can modify this function according to your problem
    return ( x[0] ) ** 2 + ( x[1] ) ** 2

def constraint(x):
    return (x[0] / x[1] <= 1) and (x[0] / x[1] >= 0)

# Define the domain of the inputs as two real intervals
space = [(-2.0, 2.0), (-2.0, 2.0)]

# Use skopt_modcn.gp_minimize function to find the minimum of the function
res = skopt_modcn.gp_minimize(f, space, n_calls=20, n_initial_points=10, initial_point_generator = "lhs_pyDOE",\
                              random_state=1, space_constraint=constraint, verbose=True)

# Get the best point and the best value observed so far
x_best = res.x
y_best = res.fun

print('The input of the minimum objective function is: '+ str(x_best))
print('The minimum of the objective function is: '+ str(y_best))
```


## Contributors
### Developer of ``skopt_modcn``
* [**SaeednHT**](https://github.com/SaeednHT/)
See also the list of [contributors](https://github.com/SaeednHT/scikit-optimize-constrained/graphs/contributors)
-----------------------------------------------------------------------------------------------------
About the original Scikit-Optimize:

Scikit-Optimize
===============

Scikit-Optimize, or ``skopt``, is a simple and efficient library to
minimize (very) expensive and noisy black-box functions. It implements
several methods for sequential model-based optimization. ``skopt`` aims
to be accessible and easy to use in many contexts.

The library is built on top of NumPy, SciPy and Scikit-Learn.

We do not perform gradient-based optimization. For gradient-based
optimization algorithms look at 
* [scipy.optimize](http://docs.scipy.org/doc/scipy/reference/optimize.html)

Important links
---------------

-  Skopt documentation - https://scikit-optimize.github.io/
-  Issue tracker -
   https://github.com/scikit-optimize/scikit-optimize/issues