import numpy as np


def Ra2b(a, b):
    """Gets rotation matrix for R3 vectors that rotates a -> b. This is the
    linear algebra version of rodriquez formula. See Notes for the long
    explanation. Theory:
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    Parameters
    ----------
    a: np.ndarray
        vector to rotate from
    b: np.ndarray
        vector to rotate to
    Returns
    -------
    np.ndarray 3x3 rotation matrix
    Notes
    -----
    This is an inline equation embedded :math:`a^2 + b^2 = c^2` in text.
    Let :math:`\hat{\mathbf{a}}` and :math:`\hat{\mathbf{a}}` be unit vectors,
    and :math:`\\theta` the angle between the two vectors. Also, let
    :math:`\mathbf{k} = \hat{\mathbf{a}} \\times \hat{\mathbf{b}}`, :math:`s =
    \sin \\theta = ||\mathbf{k}||`, and :math:`c = \cos \\theta =
    \hat{\mathbf{a}} \cdot \hat{\mathbf{b}}`. Then, the rotation matrix to
    rotate :math:`\hat{\mathbf{a}}` to :math:`\hat{\mathbf{b}}` is given by
    Equation :eq:`rotationmatrix`
    .. math:: :label: rotationmatrix
        \mathbf{R} = \mathbf{I}
            + \mathbf{K}
            + \\frac{1}{1+c} \mathbf{K} \cdot \mathbf{K},
    where  :math:`\mathbf{K}` is the skew-symmetric cross-product matrix of
    :math:`\mathbf{k}`, which is essentially a cross-product operator. Such that
    :math:`\mathbf{k} \\times \mathbf{v} = \mathbf{K} \cdot \mathbf{v}`.
    .. math:: :label: crossproductmatrix
        \mathbf{K}=\left[
            \\begin{array}{ccc}
                0 & -k_{z} & k_{y} \\\\
            k_{z} & 0 & -k_{x} \\\\
            -k_{y} & k_{x} & 0
            \end{array}
        \\right].
    From https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula it may seem
    that some form of angle is needed but fret not. Since we
    are defining :math:`\mathbf{k}` as the cross-product between the two vectors
    that define the rotation the :math:`\sin \\theta` cancels and the
    :math:`1-\cos \\theta` can be simplified using the quantities :math:`s` and
    :math:`c`. This is due to the fact that for the rotation matrix equation on
    Wiki :math:`\mathbf{k}` has to be a unit vector, meaning it has to be
    normalized.
    .. math:: :label: normalizek
        \mathbf{k}=\\frac{\mathbf{a} \\times \mathbf{b}}
            {|\mathbf{a} \\times \mathbf{b}|}
                =\\frac{\mathbf{a} \\times \mathbf{b}}
            {|\mathbf{a}||\mathbf{b}| \sin \\theta}.
    Since we don't normalize :math:`\mathbf{k}`, we need to apply
    :math:`\\frac{1}{\sin \\theta} = \\frac{1}{s}` to each cross product
    operation. And Wiki's
    .. math:: :label: rotationmatrixwiki
        \mathbf{R} = \mathbf{I}
            + \sin \\theta \mathbf{K}
            + (1- \cos \\theta) \mathbf{K} \cdot \mathbf{K},
    becomes
    .. math:: :label: rotationmatrixnonesimple
        \mathbf{R} = \mathbf{I}
            + \\frac{s}{s}\mathbf{K}
            + \\frac{1 - c}{s^2} \mathbf{K} \cdot \mathbf{K}.
    Finally we use
    .. math:: :label: simplification
        \\frac{1 - c}{s^2}
            = \\frac{1 - c}{1-c^2}
            = \\frac{1 - c}{(1-c)(1+c)}
            = \\frac{1}{1+c}
    to simplify and end up with
    .. math:: :label: simplified
        \mathbf{R} = \mathbf{I}
            + \mathbf{K}
            + \\frac{1}{1+c} \mathbf{K} \cdot \mathbf{K},
    which is the Rotation matrix built in the function.
    :Author: Lucas Sawade (lsawade@princeton.edu)
    :Last Modified: 2021.10.07 18.00
    """

    # compute normalized vectors
    an = a / np.linalg.norm(a)
    bn = b / np.linalg.norm(b)

    # Compute cross and dot product
    v = np.cross(an, bn)
    c = np.dot(an, bn)

    # Compute skew-symmetric cross-product matrix of n
    def S(n):
        Sn = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        return Sn

    Sv = S(v)

    # Compute the rotation matrix
    # The factor of 1/(1+c) comes as follows
    # s = norm of (v)
    # c = a dot b (cosine of angles)

    R = np.eye(3) + Sv + np.dot(Sv, Sv) * 1/(1+c)

    return R
