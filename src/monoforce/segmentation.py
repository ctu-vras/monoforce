"""Primitive shapes for collision checking.

Points are 2-dim column vectors or 1-dim arrays.
Transforms are (D+1)-type-(D+1) matrices.
Distances are signed,
- positive if the point is outside the shape,
- negative if inside,
- zero on the surface of the shape.
The shape contains the point if its distance is non-negative.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from tf.transformations import euler_matrix, quaternion_matrix


def affine(tf, x):
    """Apply an affine transform to points."""
    tf = np.asarray(tf)
    x = np.asarray(x)
    assert tf.ndim == 2
    assert x.ndim == 2
    assert tf.shape[1] == x.shape[0] + 1
    y = np.matmul(tf[:-1, :-1], x) + tf[:-1, -1:]
    return y


def inverse(tf):
    """Compute the inverse of an affine transform."""
    tf = np.asarray(tf)
    assert tf.ndim == 2
    assert tf.shape[0] == tf.shape[1]
    tf_inv = np.eye(tf.shape[0])
    tf_inv[:-1, :-1] = tf[:-1, :-1].T
    tf_inv[:-1, -1:] = -np.matmul(tf_inv[:-1, :-1], tf[:-1, -1:])
    return tf_inv


class Body(object):
    def __init__(self, pose=None, origin=None, orientation=None):
        if pose is None:
            pose = np.eye(4)
        else:
            pose = np.asarray(pose)

        if origin is not None:
            origin = np.asarray(origin)
            assert origin.size == 3
            pose[:3, 3:] = np.asarray(origin).reshape((3, -1))

        if orientation is not None:
            orientation = np.asarray(orientation)
            if orientation.size == 3:
                orientation = euler_matrix(*orientation)[:3, :3]
            elif orientation.size == 4:
                orientation = quaternion_matrix(*orientation)[:3, :3]
            pose[:3, :3] = orientation

        self.pose = pose
        self.pose_inv = inverse(pose)

    def distance_to_point_local(self, x):
        raise NotImplementedError()

    def distance_to_point(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape((3, -1))
        return self.distance_to_point_local(affine(self.pose_inv, x))

    def contains_point_local(self, x):
        raise NotImplementedError()

    def contains_point(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape((3, -1))
        return self.contains_point_local(affine(self.pose_inv, x))

    def __repr__(self):
        return str(self)


class Bodies(Body):
    """Body from multiple primitives.

    Distance is the minimum distance to any of the primitives.

    >>> Bodies([Sphere(0.5), Sphere(0.5, origin=[1, 0, 0])])
    Bodies([Sphere(0.5), Sphere(0.5)])
    >>> Bodies([Sphere(0.5), Sphere(0.5, origin=[1, 0, 0])]).distance_to_point([[0], [0], [0]]).item()
    -0.5
    >>> Bodies([Sphere(0.5), Sphere(1.0, origin=[1, 0, 0])]).distance_to_point([[1], [0], [0]]).item()
    -1.0
    >>> Bodies([Sphere(0.5), Box(origin=[1, 0, 0])]).distance_to_point([[0.5], [0.5], [0.5]]).item()
    0.0
    """
    def __init__(self, bodies, **kwargs):
        super(Bodies, self).__init__(**kwargs)
        self.bodies = bodies

    def distance_to_point_local(self, x):
        d = np.full((x.shape[1],), np.inf)
        for body in self.bodies:
            d = np.minimum(d, body.distance_to_point(x))
        return d

    def contains_point_local(self, x):
        c = np.zeros((x.shape[1],), dtype=bool)
        for body in self.bodies:
            c = np.logical_or(c, body.contains_point(x))
        return c

    def __str__(self):
        return 'Bodies([%s])' % ', '.join(str(b) for b in self.bodies)


class Box(Body):
    """Bounding box, axis-aligned, unit volume, and centered at origin by
    default.

    >>> Box().contains_point([0, 0, 0]).item()
    True
    >>> Box().contains_point([0, 0, -0.5]).item()
    True
    >>> Box().contains_point([0, 0, -1.0]).item()
    False
    >>> Box().distance_to_point([0, 0, 0]).item()
    -0.5
    >>> Box().distance_to_point([3.5, 4.5, 0]).item()
    5.0
    """
    def __init__(self, extents=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)), scale=None, **kwargs):
        extents = np.asarray(extents)
        if scale is not None:
            extents = np.asarray(extents) * scale
        assert extents.shape == (3, 2)
        super(Box, self).__init__(**kwargs)
        self.extents = extents

    def distance_to_point_local(self, x):
        # Signed distance to planes.
        d0 = self.extents[:, :1] - x
        d1 = x - self.extents[:, 1:]
        d = np.maximum(d0, d1)
        # Positive distance (outside).
        dp = np.linalg.norm(np.maximum(d, 0), axis=0)
        # Negative distance (inside).
        dn = d.max(axis=0)
        d = np.where((d >= 0).any(axis=0), dp, dn)
        return d

    def contains_point_local(self, x):
        c = ((self.extents[:, :1] <= x) & (x <= self.extents[:, 1:])).all(axis=0)
        return c

    def __str__(self):
        return ('Box(extents=((%.3f, %.3f), (%.3f, %.3f), (%.3f, %.3f)))'
                % tuple(self.extents.ravel()))


class Sphere(Body):
    """Sphere of specified radius, centered at the origin by default.

    >>> Sphere().contains_point([0, 0, 0]).item()
    True
    >>> Sphere().contains_point([0, 0, -1]).item()
    True
    >>> Sphere().contains_point([0, 0, -2]).item()
    False
    >>> Sphere().distance_to_point([0, 0, 0]).item()
    -1.0
    >>> Sphere().distance_to_point([0, 1, 0]).item()
    0.0
    >>> Sphere().distance_to_point([0, 2, 0]).item()
    1.0
    """
    def __init__(self, radius=1.0, **kwargs):
        assert radius >= 0.0
        super(Sphere, self).__init__(**kwargs)
        self.radius = radius

    def distance_to_point_local(self, x):
        d = np.linalg.norm(x, axis=0) - self.radius
        return d

    def contains_point_local(self, x):
        c = self.distance_to_point_local(x) <= 0
        return c

    def __str__(self):
        return 'Sphere(%.3g)' % self.radius


class Cylinder(Body):
    """Cylinder of specified radius and height, centered at origin by default.

    >>> Cylinder(height=2.0).contains_point([0, 0, 1.0]).item()
    True
    >>> Cylinder(height=2.0).contains_point([0, 0, -2.0]).item()
    False
    >>> Cylinder().distance_to_point([0, 0, 0]).item()
    -0.5
    >>> Cylinder().distance_to_point([0, 0, -1.0]).item()
    0.5
    >>> Cylinder().distance_to_point([0, 2, 0]).item()
    1.0
    >>> Cylinder().distance_to_point([4, 0, 4.5]).item()
    5.0
    >>> np.round(Cylinder(height=2.0).distance_to_point([1.0, 1.0, 0.0]).item(), 3)
    0.414
    >>> np.round(Cylinder(height=2.0, orientation=(0.0, np.pi / 2, 0.0)).distance_to_point([1.0, 1.0, 0.0]).item(), 3)
    0.0
    """
    def __init__(self, radius=1.0, height=1.0, **kwargs):
        assert radius >= 0
        assert height >= 0
        super(Cylinder, self).__init__(**kwargs)
        self.radius = radius
        self.height = height

    def distance_to_point_local(self, x):
        dxy = np.linalg.norm(x[:2], axis=0) - self.radius
        dz = np.abs(x[2]) - self.height / 2
        # Positive distance (outside).
        dp = np.hypot(np.maximum(dxy, 0), np.maximum(dz, 0))
        # Negative distance (inside).
        dn = np.maximum(dxy, dz)
        d = np.where((dn >= 0).any(axis=0), dp, dn)
        return d

    def contains_point_local(self, x):
        c = self.distance_to_point_local(x) <= 0
        return c

    def __str__(self):
        return ('Cylinder(radius=%.3f, height=%.3f)'
                % (self.radius, self.height))


def test():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    test()
