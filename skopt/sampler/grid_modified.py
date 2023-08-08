"""
Inspired by https://github.com/jonathf/chaospy/blob/master/chaospy/
distributions/sampler/sequences/grid.py
"""
import numpy as np
import matplotlib.pyplot as plt
from .base import InitialPointGenerator
from ..space import Space
from sklearn.utils import check_random_state


def _quadrature_combine(args):
    args = [np.asarray(arg).reshape(len(arg), -1) for arg in args]
    shapes = [arg.shape for arg in args]

    size = np.prod(shapes, 0)[0] * np.sum(shapes, 0)[1]
    if size > 10 ** 9:
        raise MemoryError("Too large sets")

    out = args[0]
    for arg in args[1:]:
        out = np.hstack([
            np.tile(out, len(arg)).reshape(-1, out.shape[1]),
            np.tile(arg.T, len(out)).reshape(arg.shape[1], -1).T,
        ])
    return out


def _create_uniform_grid_exclude_border(n_dim, order):
    assert order > 0
    assert n_dim > 0
    x_data = np.arange(1, order + 1) / (order + 1.)
    x_data = _quadrature_combine([x_data] * n_dim)
    return x_data


def _create_uniform_grid_include_border(n_dim, order):
    assert order > 1
    assert n_dim > 0
    x_data = np.arange(0, order) / (order - 1.)
    x_data = _quadrature_combine([x_data] * n_dim)
    return x_data


def _create_uniform_grid_only_border(n_dim, order):
    assert n_dim > 0
    assert order > 1
    x = [[0., 1.]] * (n_dim - 1)
    x.append(list(np.arange(0, order) / (order - 1.)))
    x_data = _quadrature_combine(x)
    return x_data

def reduce_points(points2, num_b=20):
    """
    Reduce the number of points in a 4D space grid while maintaining a grid-like structure.

    Parameters:
        points2 (numpy array): The dense grid samples in the 4D space. It should be a numpy array of shape (n, 4),
                               where n is the number of samples, and each row represents a 4D point.
        num_b (int): The number of points to be selected (default is 20).

    Returns:
        reduced_points2 (numpy array): The selected points forming a grid in the 4D space.
    """
    num_points_total = len(points2)

    if num_points_total <= num_b:
        return points2.copy()  # If there are fewer points than required, return all points.

    # Number of points selected along each dimension
    num_points_per_dim = int(np.ceil(num_b ** (1 / 4)))

    # Generate the grid in 4D space
    grid_points = []
    for dim_idx in range(4):
        dim_min = np.min(points2[:, dim_idx])
        dim_max = np.max(points2[:, dim_idx])
        grid_points_dim = np.linspace(dim_min, dim_max, num_points_per_dim)
        grid_points.append(grid_points_dim)

    # Create a grid in the 4D space using numpy.meshgrid
    meshgrid_points = np.array(np.meshgrid(*grid_points)).T.reshape(-1, 4)

    # Find the nearest point from the dense grid for each point in the grid
    dists = np.linalg.norm(meshgrid_points[:, None] - points2, axis=-1)
    nearest_point_indices = np.argmin(dists, axis=-1)

    # Randomly select points from the nearest point indices to get approximately 20 points
    np.random.shuffle(nearest_point_indices)
    reduced_points2 = points2[nearest_point_indices[:num_b]]

    # Use numpy.unique to remove duplicates from reduced_points2
    reduced_points2 = np.unique(reduced_points2, axis=0)

    list_points = []
    for point in reduced_points2:
        list_points.append(np.ndarray.tolist(point))

    return list_points

class Grid_modified(InitialPointGenerator):
    """Generate samples from a regular grid.

    Parameters
    ----------
    border : str, default='exclude'
        defines how the samples are generated:
        - 'include' : Includes the border into the grid layout
        - 'exclude' : Excludes the border from the grid layout
        - 'only' : Selects only points at the border of the dimension
    use_full_layout : boolean, default=True
        When True, a  full factorial design is generated and
        missing points are taken from the next larger full factorial
        design, depending on `append_border`
        When False, the next larger  full factorial design is
        generated and points are randomly selected from it.
    append_border : str, default="only"
        When use_full_layout is True, this parameter defines how the missing
        points will be generated from the next larger grid layout:
        - 'include' : Includes the border into the grid layout
        - 'exclude' : Excludes the border from the grid layout
        - 'only' : Selects only points at the border of the dimension
    """

    def __init__(self, border="exclude", use_full_layout=True,
                 append_border="only"):
        self.border = border
        self.use_full_layout = use_full_layout
        self.append_border = append_border

    def generate(self, dimensions, space_constraint, n_samples, random_state=None):
        np.random.seed(random_state)
        """Creates samples from a regular grid.

        Parameters
        ----------
        dimensions : list, shape (n_dims,)
            List of search space dimensions.
            Each search dimension can be defined either as

            - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
              dimensions),
            - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
              dimensions),
            - as a list of categories (for `Categorical` dimensions), or
            - an instance of a `Dimension` object (`Real`, `Integer` or
              `Categorical`).

        n_samples : int
            The order of the Halton sequence. Defines the number of samples.
        random_state : int, RandomState instance, or None (default)
            Set random state to something other than None for reproducible
            results.

        Returns
        -------
        np.array, shape=(n_dim, n_samples)
            grid set
        """
        if space_constraint is not None:
            space = Space(dimensions, constraint=space_constraint)
            dim = space.n_dims
            bounds = space.bounds
            """
            bounds: list of tuples containing the lower and upper bounds for each dimension
            """
            # Create a meshgrid of points for each dimension
            grids = [np.linspace(bound[0], bound[1], num=40) for bound in bounds]
            
            # Generate all possible combinations of points from the meshgrid
            points = np.array(np.meshgrid(*grids)).T.reshape(-1, len(bounds))

            #  accept points that satisfy constraint
            points2 = []
            for point in points:
                if space_constraint(point):
                    points2.append(point)
            # print('points2 = ', str(points2))
            points2 = np.array(points2)
            # # If you want to ignore the constraints
            # points2 = points
            
            # n_samples = 30 # np.size(points2,0) // 5

            # final_points = points/2
            points = reduce_points(points2, n_samples)

            

            points_p = np.array(points)
            # dim = np.array(points_p).shape[1]
            # print('dim='+str(dim))
            fig, axs = plt.subplots(dim, dim, figsize=(dim*4, dim*4))
            for i in range(dim):
                # print('i='+str(i))
                for j in range(dim):
                    # print('j='+str(j))
                    if i == j:
                        axs[i, j].hist(points_p[:, i])
                    else:
                        axs[i, j].scatter(points_p[:, i], points_p[:, j])
                    if i == 0:
                        axs[i, j].set_title(f'Dimension {j+1}')
                    if j == 0:
                        axs[i, j].set_ylabel(f'Dimension {i+1}')
            # plt.show()
            plt.savefig('Initial_guess_distribution_Grid_modified.pdf')
            print('points :\n'+str(points))
            print('The number of initial points :\n'+str(len(points)))
            with open('initial_points.txt','w') as ipfile:
                ipfile.write('The number of initial points: '+str(len(points)))
                ipfile.write('\npoints :\n')
                for point in points:
                    ipfile.write(str(point))

            return points
        else:
            rng = check_random_state(random_state)
            space = Space(dimensions)
            n_dim = space.n_dims
            transformer = space.get_transformer()
            space.set_transformer("normalize")

            if self.border == "include":
                if self.use_full_layout:
                    order = int(np.floor(np.sqrt(n_samples)))
                else:
                    order = int(np.ceil(np.sqrt(n_samples)))
                if order < 2:
                    order = 2
                h = _create_uniform_grid_include_border(n_dim, order)
            elif self.border == "exclude":
                if self.use_full_layout:
                    order = int(np.floor(np.sqrt(n_samples)))
                else:
                    order = int(np.ceil(np.sqrt(n_samples)))
                if order < 1:
                    order = 1
                h = _create_uniform_grid_exclude_border(n_dim, order)
            elif self.border == "only":
                if self.use_full_layout:
                    order = int(np.floor(n_samples / 2.))
                else:
                    order = int(np.ceil(n_samples / 2.))
                if order < 2:
                    order = 2
                h = _create_uniform_grid_exclude_border(n_dim, order)
            else:
                raise ValueError("Wrong value for border")
            if np.size(h, 0) > n_samples:
                rng.shuffle(h)
                h = h[:n_samples, :]
            elif np.size(h, 0) < n_samples:
                if self.append_border == "only":
                    order = int(np.ceil((n_samples - np.size(h, 0)) / 2.))
                    if order < 2:
                        order = 2
                    h2 = _create_uniform_grid_only_border(n_dim, order)
                elif self.append_border == "include":
                    order = int(np.ceil(np.sqrt(n_samples - np.size(h, 0))))
                    if order < 2:
                        order = 2
                    h2 = _create_uniform_grid_include_border(n_dim, order)
                elif self.append_border == "exclude":
                    order = int(np.ceil(np.sqrt(n_samples - np.size(h, 0))))
                    if order < 1:
                        order = 1
                    h2 = _create_uniform_grid_exclude_border(n_dim, order)
                else:
                    raise ValueError("Wrong value for append_border")
                h = np.vstack((h, h2[:(n_samples - np.size(h, 0))]))
                rng.shuffle(h)
            else:
                rng.shuffle(h)
            h = space.inverse_transform(h)
            space.set_transformer(transformer)
            return h
