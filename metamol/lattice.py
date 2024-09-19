from typing import Iterable
import numpy as np
from copy import deepcopy

import metamol as meta
from metamol.exceptions import MetaError
#from metamol import System
from metamol.utils.geometry import Translate
from metamol.utils.convert_formats import *

BASIS_DICT = {
    'sc': [[0., 0., 0.]], 'hcc': [[0., 0., 0.], [0.5, 0.5, 0.5]],
    'bcc': [[0., 0., 0.], [0.5, 0.5, 0.], [0., 0.5, 0.5], [0.5, 0., 0.5]],
    'diamond': [[0., 0., 0.], [0., 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0.],
             [0.75, 0.75, 0.75], [0.75, 0.25, 0.25], [0.25, 0.75, 0.25], [0.25, 0.25, 0.75]]
    }

class Lattice(meta.System):
    """This class builds lattice structures.
        Inputs
        ---------
        spacings: array of shape (3, ), required
            Lattice spacings, (a, b, c).
        langles: array of shape (3, ), optional
            Lattice angles, (alpha, beta, gamma).
        vectors: array of ashape (3, 3), optional, one of angles or vectors must be defined
            Lattice unit vectors, (v1, v2, v3).
        occupy_points: dict, optional
            Dictionary of molecule-location pairs.
        
        Attributes
        -----------
        spacings: Lattice spacings.
        langles: Lattice angles.
        vectors: Lattice vectors.
        numCells: Number of unit cells.
        _compute_params(): Check the sanity of inputs and 
                        calculate lattice parameters from the inputs.
        _compute_coords(point): Compute the coordinates of lattice point.
        _check_consistency(): Check the consistency of lattice parameters.
        clear(): Clear all occupied locations in the lattice.
        occupy(occupy_points): Occupy locations in the lattice.
        replicate(x, y, z): replicate the lattice several times in 3 dimensions.        
    """
    def __init__(
        self, 
        spacings=None, 
        langles=None, 
        vectors=None, 
        occupy_points=dict(),
        dimensions=3,
        style=None,
        constant=1.0,
        basis=[], 
    ):
        super(Lattice, self).__init__()
        self.dimensions = dimensions
        self.basis = basis
        if style:
            self.create_lattice(style=style, constant=constant)
        else:
            self.spacings = self._sanitize_input(vals=spacings, type="spacings")
            self.langles = self._sanitize_input(vals=langles, type="langles")
            self.vectors = self._sanitize_input(vals=vectors, type="vectors")
            self._compute_params()
        self.occupied = set()
        self.occupy(occupy_points)
        self.numCells = 1

    def create_lattice(self, style, constant):
        self.spacings = np.asarray([constant] * self.dimensions, dtype=np.float64)
        self.langles = np.asarray([90.0, 90.0, 90.0], dtype=np.float64)
        self.vectors = np.asarray([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float64)
        if style in BASIS_DICT:
            self.basis = BASIS_DICT[style]
        else:
            raise MetaError("Lattice style {0} not supported".format(style))

    def occupy(self, occupy_points=dict(), particle=None):
        """Occupy the lattice points by molecules."""
        # for all (molecule, points) in occupy_points dict
        #   for every lattice point to occupy,
        #       make a copy of molecule
        #       calculate the coordinates of the point
        #       translate the molecule_copy to that point
        #       add the molecule_copy to Lattice instance.
        # update atom idx

        if not occupy_points and not particle: return
        elif not occupy_points:
            occupy_points[particle] = self.basis

        self.basis = []
        for mol in occupy_points.keys():
            lattice_points = occupy_points[mol]
            self.basis += list(lattice_points)
            for point in lattice_points:
                if isinstance(mol, meta.Atom):
                    mol = meta.Molecule(input=mol)
                mol_copy = deepcopy(mol)
                coords = self._compute_coords(point)
                if tuple(coords) not in self.occupied:
                    self.occupied.add(tuple(coords))
                    translation_vector = coords - mol_copy.center
                    Translate(mol_copy, translation_vector)
                    self.add(mol_copy)

    def clear(self):
        """Clear all occupied locations in the lattice."""
        # Clear molList
        super(Lattice, self).__init__()
        self.occupy_points = dict()
        self.occupied = set()

    def replicate(self, x=1, y=1, z=1):
        """Extend the lattice in 3 dimensions."""
        # Get original molecules in the lattice
        mols = self.molecules.copy()
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    if i == j == k == 0:
                        continue
                    for mol in mols:
                        mol_copy = deepcopy(mol)
                        translation_vector = i * self.spacings[0] * self.vectors[0] + \
                        j * self.spacings[1] * self.vectors[1] + k * self.spacings[2] * self.vectors[2]
                        new_coords = mol_copy.center + translation_vector
                        if tuple(new_coords) not in self.occupied:
                            Translate(mol_copy, translation_vector)
                            self.add(mol_copy)
                            self.occupied.add(tuple(new_coords))
        self.numCells = x * y * z
        if not self.valid_box():
            self.create_box()

    def _compute_params(self):
        """Check the sanity of inputs and calculate lattice parameters from the inputs."""
        # Check input validity.
        if self.spacings is None or not isinstance(self.spacings, Iterable) or len(self.spacings)!=3:
            raise MetaError("Lattice spacings array is required and must have a shape of (3, )")
        if self.langles is None and self.vectors is None:
            raise MetaError("At least one of lattice angles and vectors input is required")
        if self.langles is not None and self.vectors is not None:
            if not self._check_consistency():
                raise MetaError("Lattice length, angles and vectors are not consistent")
        elif self.langles is not None:
            #self.vectors = norm_lattice_vectors(normalize_vectors(box_to_vectors(self.spacings, self.langles)))
            self.vectors = box_to_vectors(self.spacings, self.langles, norm=True)
        else:
            self.langles = vectors_to_box(self.vectors)[1]

    def _sanitize_input(self, vals, type):
        """Sanitize inputs"""
        if type == "spacings":
            if not isinstance(vals, Iterable) or len(vals) != self.dimensions:
                raise MetaError("Lattice spacings must be an array of size lattice-dimension")
            try: 
                vals = np.asarray(vals, dtype=np.float64)
            except ValueError:
                raise MetaError("Lattice spacings have some non-numeric values")
            if np.any(np.isnan(vals)):
                raise MetaError("nan values present in lattice spacings input")
            if np.any(vals < 0.0):
                raise MetaError("Negative values present in lattice spacings input")
        
        elif type == "langles":
            if vals is None:
                return
            try: 
                vals = np.asarray(vals, dtype=np.float64)
            except ValueError:
                raise MetaError("Lattice angles have some non-numeric values")
            if vals.shape != (self.dimensions,):
                raise MetaError("Lattice angles must be an array of size lattice-dimension")
            if np.any(np.isnan(vals)):
                raise MetaError("nan values present in lattice angles input")
            if np.any(np.isclose(vals, 0.0)) or np.any(np.isclose(vals, 180.0)) or np.any(np.isclose(vals, -180.0)):
                raise MetaError("0 or 180 degrees present in lattie angles input")
            if abs(sum(vals)) > 360.0:
                raise MetaError("Sum of lattice angles larger than 360 degrees")
        
        elif type == "vectors":
            if vals is None:
                return
            try:
                vals = np.asarray(vals, dtype=np.float64)
            except ValueError:
                raise MetaError("Lattice vectors have some non-numeric values")
            if vals.shape != (self.dimensions, self.dimensions):
                raise MetaError("Lattice vectors must be an array of size lattice-dimension")
            if np.any(np.isnan(vals)):
                raise MetaError("nan values present in lattice vectors input")

            det = np.linalg.det(vals)
            if abs(det) == 0.0:
                raise MetaError("Lattice vectors are co-linear")

            if det < 0.0:
                raise MetaError("Negative Determinant: the determinant of lattice vectors is negative, "
                    "indicating a left-handed system."
                )

            vals = norm_lattice_vectors(vals)      

        else:
            raise MetaError("Unsupported input type {0}".format(type))

        return vals

    def _check_consistency(self):
        """Check the consistency of lattice parameters."""
        vec_compute = box_to_vectors(self.spacings, self.langles, norm=True)
        #vec_compute = norm_lattice_vectors(normalize_vectors(vec_compute))
        return np.allclose(self.vectors, vec_compute)

    def _compute_coords(self, point):
        """Compute the coordinates of lattice point."""
        trans_matrix = self.vectors.transpose()
        coords = []
        for i in range(3):
            coord = np.dot(trans_matrix[i], point) * self.spacings[i]
            coords.append(coord)
        return np.asarray(coords)
    
    def __repr__(self):
        """Representation of the Lattice."""

        desc = ["Lattice id: {}".format(id(self))]
        desc.append("Lattice Parameters: ")
        desc.append("Lattice spacings: {0}A * {1}A * {2}A"\
            .format(self.spacings[0], self.spacings[1], self.spacings[2]))
        desc.append("Lattice angles: {0}d, {1}d, {2}d"\
            .format(self.langles[0], self.langles[1], self.langles[2]))
        desc.append("Number of unit cells: {0}".format(self.numCells))
        desc.append("Number of Molecules: {}".format(self.numMols))
        if self.numWater:
            desc.append("Number of water molecules: {}".format(self.numWater))
        desc.append("Number of Atoms: {}".format(self.numAtoms))
        desc.append("Number of Bonds: {}".format(self.numBonds))
        if self.box is not None:
            # if len(self.box) == 3:
            #     len_x, len_y, len_z = self.box[0], self.box[1], self.box[2]
            # elif len(self.box) == 6:
            #     len_x, len_y, len_z = self.box[3]-self.box[0], self.box[4]-self.box[1], self.box[5]-self.box[2]
            len_x, len_y, len_z = self.box.lengths
            desc.append("Box: {0}A * {1}A * {2}A".format(len_x, len_y, len_z))
            desc.append("Box Angle: {0}d, {1}d, {2}d"\
                .format(self.box.angle[0], self.box.angle[1], self.box.angle[2]))
            periodicity = ["fixed" if p==0 else "periodic" for p in self.box.per]
            desc.append("Periodicity: {0} in x, {1} in y, {2} in z".format(periodicity[0], periodicity[1], periodicity[2]))

        if self.parametrized:
            desc.append("Parametrized: Yes")

        return "\n".join(desc)

def norm_lattice_vectors(vectors):
    for i in range(vectors.shape[0]):
        length = np.linalg.norm(vectors[i])
        vectors[i] /= length
    return vectors