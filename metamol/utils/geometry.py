from typing import Iterable
import numpy as np
from metamol.exceptions import MetaError

from metamol.utils.constants import PI

__all__ = ["Tanslate", "Rotate", "uv_degree", "RotateX", "RotateY",
           "RotateZ", "normalize", "q_mult", "q_conj", "q_rot",
           "axisangle_to_q", "q_to_axisangle"]

def Translate(obj, translation_vector):
    """Translate a Molecule/System based on the translation vector."""
    if not isinstance(translation_vector, Iterable) or len(translation_vector)!=3:
        raise MetaError("Translation vector must be an array of shape (3, )")
    for atom in obj.atoms_iter():
        atom.xyz = atom.xyz + translation_vector

def Rotate(about, theta, xyz, rad=True):
    """Rotate xyz about some axis with angle theta."""
    if isinstance(about, str):
        if about.lower() == 'x':
            about = (1, 0, 0)
        elif about.lower() == 'y':
            about = (0, 1, 0)
        elif about.lower() == 'z':
            about = (0, 0, 1)
        else:
            raise MetaError("Unrecognized axis type: {0}".format(about))

    elif not isinstance(about, Iterable) or len(about) != 3:
            raise TypeError(
                "The rotation axis must be either a string or an Iterable object."
                )

    # if not rad:
    #     theta = theta / 180.0 * PI
    if sum([a*a for a in about]) < 0.0000001:
        raise MetaError("Cannot rotate around a zero vector.")
    q = axisangle_to_q(about, theta, rad)
    return q_rot(q, xyz)

def uv_degree(u, v):
    """Compute the angle between vectors u and v."""
    c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
    return np.arccos(np.clip(c, -1, 1))

# def RotateX(theta, xyz, rad=True):
#     """Rotate along X axis."""
#     if not rad:
#         theta = theta / 180.0 * PI

#     about = (1, 0, 0)

#     return Rotate(about, theta, xyz)

# def RotateY(theta, xyz, rad=True):
#     """Rotate along Y axis."""
#     if not rad:
#         theta = theta / 180.0 * PI

#     about = (0, 1, 0)

#     return Rotate(about, theta, xyz)

# def RotateZ(theta, xyz, rad=True):
#     """Rotate along Z axis."""
#     if not rad:
#         theta = theta / 180.0 * PI

#     about = (0, 0, 1)

#     return Rotate(about, theta, xyz)

###############Define quaternion functions#######################
def normalize(v, tol=0.00001):
    """Normalize the input vector."""
    mag2 = sum(x * x for x in v)
    if abs(mag2 - 1.0) > tol:
        mag = np.sqrt(mag2)
        v = tuple(x / mag for x in v)
    return v

def q_mult(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2    
    return (w, x, y, z)

def q_conj(q):
    """Calculate the conjugate quaternion."""
    w, x, y, z = q
    return (w, -x, -y, -z)

def q_rot(q, xyz):
    """use quaternion to roatate a Molecule/System."""
    q_inv = q_conj(q)
    xyz = np.asarray(xyz)
    if xyz.ndim == 1:
        xyz = np.expand_dims(xyz, axis=0)
    xyz_out = []
    for v in xyz:
        q_v = (0.0,) + tuple(v)
        xyz_out.append(q_mult(q_mult(q, q_v), q_inv)[1:])
    return xyz_out

def axisangle_to_q(about, theta, rad=True):
    """Convert axisangle to quaternion."""
    if not rad:
        theta = theta / 180.0 * PI
    about = normalize(about)
    x, y, z = about
    theta /= 2.0
    w = np.cos(theta)
    x = x * np.sin(theta)
    y = y * np.sin(theta)
    z = z * np.sin(theta)
    return (w, x, y, z)

def q_to_axisangle(q, rad=True):
    """Convert quaternion to axisangle."""
    w, v = q[0], q[1:]
    theta = np.acos(w) * 2.0
    if not rad:
        theta = theta / PI * 180.0
    return normalize(v), theta
