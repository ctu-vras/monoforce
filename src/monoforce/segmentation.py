"""Segmentation of points into geometric primitives (planes, etc.)."""
from .geometry import affine
from .utils import timing
from .vis import map_colors, show_cloud
from matplotlib import cm
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import open3d as o3d
from scipy.spatial import cKDTree

default_rng = np.random.default_rng(135)


def cluster_open3d(x, eps, min_points=10):
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    assert eps >= 0.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    # NB: High min_points value causes finding no points, even if clusters
    # with enough support are found when using lower min_points value.
    # clustering = pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True)
    clustering = pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    # Invalid labels < 0.
    clustering = np.asarray(clustering)
    return clustering


def keep_mask(n, indices):
    mask = np.zeros(n, dtype=bool)
    mask[indices] = 1
    return mask


def remove_mask(n, indices):
    mask = np.ones(n, dtype=bool)
    mask[indices] = 0
    return mask


def largest_cluster(x, eps, min_points=10):
    clustering = cluster_open3d(x, eps, min_points=min_points)
    clusters, counts = np.unique(clustering[clustering >= 0], return_counts=True)
    if len(counts) == 0:
        return []
    largest = clusters[counts.argmax()]
    indices = np.flatnonzero(clustering == largest)
    assert len(indices) >= min_points
    return indices


def fit_models_iteratively(x, fit_model, min_support=3, max_models=10, cluster_eps=None, cluster_k=10,
                           verbose=0, visualize=False):
    """Fit multiple models iteratively.

    @param x: Input point cloud.
    @param fit_model: Function that fits a model to a point cloud and returns the model and the inlier indices.
    @param min_support: Minimum number of inliers required for a model to be considered valid.
    @param max_models: Maximum number of models to fit.
    @param cluster_eps: If not None, cluster the inliers of each model and keep only the largest cluster.
    @param verbose: Verbosity level.
    @return: List of models and inlier indices.
    """
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    remaining = x
    indices = np.arange(len(remaining))  # Input point indices of remaining point cloud.
    models = []
    labels = np.full(len(remaining), -1, dtype=int)
    label = 0
    while True:
        model, support_tmp = fit_model(remaining)

        support_tmp = np.asarray(support_tmp)
        if verbose >= 2:
            print('Found model %i (%s) supported by %i / %i (%i) points.'
                  % (label, model, len(support_tmp). len(remaining), len(x)))

        if len(support_tmp) < min_support:
            if verbose >= 0:
                print('Halt due to insufficient model support.')
            break

        # Extract the largest contiguous cluster and keep the rest for next iteration.
        if cluster_eps:
            # largest_indices = largest_cluster(remaining[support_tmp], eps=cluster_eps, min_points=min_support)
            largest_indices = largest_cluster(remaining[support_tmp], eps=cluster_eps, min_points=cluster_k)
            if len(largest_indices) == 0 or len(largest_indices) < min_support:
                # Remove all points if there is no cluster with sufficient support.
                mask = remove_mask(len(remaining), support_tmp)
                remaining = remaining[mask]
                indices = indices[mask]
                if verbose >= 2:
                    print('No cluster from model %i has support >= %i.' % (label, min_support))
                if len(remaining) < min_support:
                    if verbose >= 1:
                        print('Not enough points to continue.')
                    break
                continue
            support_tmp = support_tmp[largest_indices]

            # support_tmp = support_tmp[clustering == largest]
            if verbose >= 1:
                print('Kept largest cluster from model %i %s supported by %i / %i (%i) points.'
                      % (label, model, len(support_tmp), len(remaining), len(x)))

        support = indices[support_tmp]
        models.append((model, support))
        labels[support] = label

        if len(models) == max_models:
            if verbose >= 1:
                print('Target number of models found.')
            break

        mask = remove_mask(len(remaining), support_tmp)
        remaining = remaining[mask]
        indices = indices[mask]
        if len(remaining) < min_support:
            if verbose >= 1:
                print('Not enough points to continue.')
            break
        label += 1

    print('%i models (highest label %i) with minimum support of %i points were found.'
          % (len(models), labels.max(), min_support))

    if visualize:
        num_primitives = len(models)
        num_points = len(x)
        labels = np.full(num_points, -1, dtype=int)
        for i in range(num_primitives):
            labels[models[i][1]] = i
        max_label = num_primitives - 1
        colors = np.zeros((num_points, 3), dtype=np.float32)
        segmented = labels >= 0
        colors[segmented] = map_colors(labels[segmented], colormap=cm.jet, min_value=0.0, max_value=max_label)
        show_cloud(x, colors)

    return models


def fit_cylinder_pcl(x, distance_threshold, radius_limits, max_iterations=1000):
    import pcl
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    assert distance_threshold >= 0.0
    assert max_iterations > 0
    cld = pcl.PointCloud(x.astype(np.float32))
    seg = cld.make_segmenter()
    # seg = cld.make_segmenter_normals(ksearch=9, searchRadius=-1.0)
    # seg = cld.make_segmenter_normals(9, -1.0)
    # seg = cld.make_segmenter_normals(int_ksearch=9)
    # seg = cld.make_segmenter_normals(double_searchRadius=0.5)
    # seg = cld.make_segmenter_normals(int_ksearch=9, double_searchRadius=-1.0)
    # seg = cld.make_segmenter_normals()
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_STICK)
    # seg.set_model_type(pcl.SACMODEL_CYLINDER)
    # seg.set_eps_angle(0.25)
    # seg.set_radius_limits(radius_limits[0], radius_limits[1])
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(distance_threshold)
    seg.set_MaxIterations(max_iterations)
    indices, model = seg.segment()
    return model, indices


def fit_cylinder_rsc(x, distance_threshold, max_iterations=1000):
    import pyransac3d as rsc
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    assert distance_threshold >= 0.0
    assert max_iterations > 0
    m = rsc.Cylinder().fit(x, thresh=distance_threshold, maxIteration=max_iterations)
    model, indices = m[:-1], m[-1]
    return model, indices


def fit_cylinder_ls(x):
    from cylinder_fitting import fit
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    w, c, r, err = fit(x, guess_angles=[(0, np.pi / 2)])
    # show_fit(w, c, r, x)
    model = w, c, r
    return model


def fit_cylinder(x, distance_threshold, radius_limits=None, max_iterations=1000):
    from .ransac import ransac
    from scipy.spatial import cKDTree
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    assert distance_threshold >= 0.0
    assert max_iterations > 0

    # min_sample = 3

    # Speed up by constructing the model only from a local neighborhood.
    # To interface with ransac, we will use minimal sample size 1 and find the
    # other two points in the local neighborhood if necessary.
    x_all = x
    tree = cKDTree(x, leafsize=64, compact_nodes=True, balanced_tree=False)
    min_sample = 1

    def get_model(x):
        if len(x) == 1:
            # Find the two other points in the local neighborhood.
            i = tree.query_ball_point(x, 5 * distance_threshold)[0]
            # Return sample from all points if no model can be constructed from
            # the local neighborhood.
            if len(i) < 5:
                sample = np.random.choice(len(x_all), 5, replace=False)
                x = x_all[sample]
            else:
                i = np.random.choice(i, size=5, replace=False)
                x = x_all[i]
        model = fit_cylinder_ls(x)
        # Limit radius
        if radius_limits:
            r = model[2]
            if r < radius_limits[0] or r > radius_limits[1]:
                print('Radius limits exceeded:', r, radius_limits)
                return None
        # Limit direction
        w = model[0]
        if abs(w[2]) < 0.8:
            print('Direction limits exceeded:', w)
            return None
        return model

    def get_inliers(model, x):
        dist = point_to_cylinder_dist(x, model)
        dist = np.abs(dist)
        inliers = np.flatnonzero(dist <= distance_threshold)
        return inliers

    # Avoid local optimization with inliers due to slow cylinder fit.
    model, inliers = ransac(x, min_sample, get_model, get_inliers,
                            fail_prob=0.01, max_iters=max_iterations, lo_iters=0, verbosity=1)
    return model, inliers


def fit_cylinders(x, distance_threshold, radius_limits=None, max_iterations=1000, **kwargs):
    """Segment points into cylinders."""
    assert isinstance(x, np.ndarray)
    assert isinstance(distance_threshold, float)
    assert distance_threshold >= 0.0
    # assert isinstance(radius_limits, tuple)
    # assert len(radius_limits) == 2
    # assert radius_limits[0] >= 0.0
    # assert radius_limits[0] <= radius_limits[1]
    if x.dtype.names:
        x = structured_to_unstructured(x[['x', 'y', 'z']])
    # models = fit_models_iteratively(x, lambda x: fit_cylinder_pcl(x, distance_threshold, radius_limits), **kwargs)
    # models = fit_models_iteratively(x, lambda x: fit_cylinder_rsc(x, distance_threshold, max_iterations=max_iterations),
    #                                 **kwargs)
    models = fit_models_iteratively(x, lambda x: fit_cylinder(x, distance_threshold, radius_limits=radius_limits,
                                                              max_iterations=max_iterations),
                                    **kwargs)
    return models


def fit_plane_pcl(x, distance_threshold, max_iterations=1000):
    import pcl
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    assert distance_threshold >= 0.0
    assert max_iterations > 0
    cld = pcl.PointCloud(x.astype(np.float32))
    seg = cld.make_segmenter()
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(distance_threshold)
    seg.set_MaxIterations(max_iterations)
    indices, model = seg.segment()
    return model, indices


def fit_plane_ls(x):
    assert len(x) >= 3
    # TODO: Speed up for minimal sample size.
    # if len(x) == 3:
    #     pass
    mean = np.mean(x, axis=0)
    cov = np.cov(x.T)
    # Eigenvalues in ascending order with their eigenvectors in columns.
    w, v = np.linalg.eigh(cov)
    n = v[:, 0]
    d = -np.dot(n, mean)
    return list(n) + [d]


def point_to_plane_dist(x, model):
    n = model[:3]
    d = model[3]
    dist = np.dot(x, n) + d
    return dist


def point_to_cylinder_dist(x, model):
    w, c, r = model
    w = np.asarray(w).reshape((1, 3))
    c = np.asarray(c).reshape((1, 3))
    w = w / np.linalg.norm(w)
    u = x - c
    dist = np.linalg.norm(u - np.dot(u, w.T) * w, axis=1) - r
    return dist


def fit_plane(x, distance_threshold, normal_z_limits=(0.0, 1.0), max_iterations=1000):
    from .ransac import ransac
    from scipy.spatial import cKDTree
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    assert distance_threshold >= 0.0
    assert max_iterations > 0

    # min_sample = 3

    # Speed up by constructing the model only from a local neighborhood.
    # To interface with ransac, we will use minimal sample size 1 and find the
    # other two points in the local neighborhood if necessary.
    x_all = x
    # tree = cKDTree(x, leafsize=64, compact_nodes=True, balanced_tree=False)
    tree = cKDTree(x, compact_nodes=True, balanced_tree=False)
    min_sample = 1

    def get_model(x):
        if len(x) == 1:
            # Find the two other points in the local neighborhood.
            i = tree.query_ball_point(x, 5 * distance_threshold)[0]
            # Return sample from all points if no model can be constructed from
            # the local neighborhood.
            if len(i) < 3:
                # print('Not enough points in local neighborhood.')
                return None
                sample = np.random.choice(len(x_all), 3, replace=False)
                x = x_all[sample]
            else:
                # print('Model constructed from local neighborhood.')
                i = np.random.choice(i, size=3, replace=False)
                x = x_all[i]
        model = fit_plane_ls(x)
        if abs(model[2]) < normal_z_limits[0] or abs(model[2]) > normal_z_limits[1]:
            return None
        # TODO: Check consistency with local neighborhood.
        return model

    def get_inliers(model, x):
        dist = point_to_plane_dist(x, model)
        dist = np.abs(dist)
        inliers = np.flatnonzero(dist <= distance_threshold)
        return inliers

    model, inliers = ransac(x, min_sample, get_model, get_inliers,
                            fail_prob=1e-6, max_iters=max_iterations, lo_iters=3, verbosity=0)
    return model, inliers


def fit_planes(x, distance_threshold, normal_z_limits=(0.0, 1.0), max_iterations=1000, **kwargs):
    """Segment points into planes."""
    from scipy.spatial import cKDTree
    assert isinstance(x, np.ndarray)
    assert isinstance(distance_threshold, float)
    assert distance_threshold >= 0.0
    if x.dtype.names:
        x = structured_to_unstructured(x[['x', 'y', 'z']])

    # Fit models to filtered cloud.
    x_filtered = filter_range(x, 0.5, 10.0)
    grid = 0.1
    x_filtered = filter_grid(x_filtered, grid)
    def fit_model(x):
        return fit_plane(x, distance_threshold, normal_z_limits=normal_z_limits, max_iterations=max_iterations)
    models = fit_models_iteratively(x_filtered, fit_model, **kwargs)

    # Construct inlier set from original points: consistent with model and close to filtered inliers.
    for i in range(len(models)):
        model, inliers = models[i]
        tree = cKDTree(x_filtered[inliers])
        dist = point_to_plane_dist(x, model)
        dist = np.abs(dist)
        orig_inliers = np.flatnonzero(dist <= distance_threshold)
        d, _ = tree.query(x[orig_inliers])
        orig_inliers = orig_inliers[d <= grid]
        models[i] = model, orig_inliers

    return models


def fit_stick_pcl(x, distance_threshold, max_iterations=1000):
    import pcl
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 3
    assert distance_threshold >= 0.0
    assert max_iterations > 0
    cld = pcl.PointCloud(x.astype(np.float32))
    seg = cld.make_segmenter()
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_STICK)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(distance_threshold)
    seg.set_MaxIterations(max_iterations)
    indices, model = seg.segment()
    return model, indices


def fit_sticks(x, distance_threshold, max_iterations=1000, **kwargs):
    """Segment points into planes."""
    assert isinstance(x, np.ndarray)
    assert isinstance(distance_threshold, float)
    assert distance_threshold >= 0.0
    if x.dtype.names:
        x = structured_to_unstructured(x[['x', 'y', 'z']])
    models = fit_models_iteratively(x, lambda x: fit_stick_pcl(x, distance_threshold, max_iterations=max_iterations),
                                    **kwargs)
    return models


def position(cloud):
    """Cloud to point positions (xyz)."""
    if cloud.dtype.names:
        x = structured_to_unstructured(cloud[['x', 'y', 'z']])
    else:
        x = cloud
    return x

def color(cloud):
    """Color to rgb."""
    if cloud.dtype.names:
        rgb = structured_to_unstructured(cloud[['r', 'g', 'b']])
    else:
        rgb = cloud
    return rgb


def filter_range(cloud, min, max, log=False, return_mask=False):
    """Keep points within range interval."""
    assert isinstance(cloud, np.ndarray), type(cloud)
    assert isinstance(min, (float, int)), min
    assert isinstance(max, (float, int)), max
    assert min <= max, (min, max)
    min = float(min)
    max = float(max)
    if min <= 0.0 and max == np.inf:
        return cloud
    if cloud.dtype.names:
        cloud = cloud.ravel()
    x = position(cloud)
    r = np.linalg.norm(x, axis=1)
    mask = (min <= r) & (r <= max)

    if log:
        print('%.3f = %i / %i points kept (range min %s, max %s).'
              % (mask.sum() / len(cloud), mask.sum(), len(cloud), min, max))

    filtered = cloud[mask]
    if return_mask:
        return filtered, mask

    return filtered


def filter_grid(cloud, grid_res, keep='first', log=False, rng=default_rng, return_mask=False):
    """Keep single point within each cell. Order is not preserved."""
    assert isinstance(cloud, np.ndarray)
    # assert cloud.dtype.names
    assert isinstance(grid_res, (float, int)) and grid_res > 0.0
    assert keep in ('first', 'random', 'last')

    if cloud.dtype.names:
        cloud = cloud.ravel()
    if keep == 'first':
        pass
    elif keep == 'random':
        rng.shuffle(cloud)
    elif keep == 'last':
        cloud = cloud[::-1]

    x = position(cloud)
    keys = np.floor(x / grid_res).astype(int)
    assert keys.size > 0
    _, ind = np.unique(keys, return_index=True, axis=0)

    if log:
        print('%.3f = %i / %i points kept (grid res. %.3f m).'
              % (len(ind) / len(keys), len(ind), len(keys), grid_res))

    filtered = cloud[ind]
    if return_mask:
        return filtered, ind

    return filtered


def valid_point_mask(arr, discard_tf=None, discard_model=None):
    assert isinstance(arr, np.ndarray)
    assert arr.dtype.names
    # Identify valid points, i.e., points with valid depth which are not part
    # of the robot (contained by the discard model).
    # x = position(arr)
    # x = x.reshape((-1, 3)).T
    x = position(arr.ravel()).T
    valid = np.isfinite(x).all(axis=0)
    valid = np.logical_and(valid, (x != 0.0).any(axis=0))
    if discard_tf is not None and discard_model is not None:
        y = affine(discard_tf, x)
        valid = np.logical_and(valid, ~discard_model.contains_point(y))
    return valid.reshape(arr.shape)


def valid_point_indices(*args, **kwargs):
    return np.flatnonzero(valid_point_mask(*args, **kwargs))


@timing
def compute_rigid_support(arr, transform=None, range=None, grid=None, scale=1.0, radius=0.1, min_support=30):
    xyz = position(arr)
    xyz = xyz.reshape((-1, 3))

    if transform is not None:
        # xyz = affine(transform, xyz.T).T
        # Only rotate so that range is applied in sensor frame.
        xyz = np.matmul(xyz, transform[:3, :3].T)

    filtered = xyz.copy()
    if range is not None:
        filtered = filter_range(xyz, *range)
    if grid is not None:
        filtered = filter_grid(xyz, grid)

    xyz = scale * xyz
    filtered = scale * filtered

    tree = cKDTree(filtered, compact_nodes=False, balanced_tree=False)
    ind = tree.query_ball_point(xyz, radius, workers=-1)

    support = np.array([len(i) for i in ind]).astype(np.uint32)
    support = support.reshape(arr.shape)

    rigid = support >= min_support

    return support, rigid
