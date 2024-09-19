# Contain some help functions
import numpy as np
from copy import deepcopy
import contextlib
import tempfile
import shutil
import os
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem

from metamol.exceptions import MetaError

__all__ = ["tempdir", "cd", "save_remove", "approximate_bl", "find_best_orientation", "center_of_mass", "wrap_coords",
            "random_rotation_matrix", "distance", "fix_bond_order", "optimize_config", "fix_bond_order", "parmed_load"]

@contextlib.contextmanager
def tempdir():
    dirpath = tempfile.mkdtemp()
    try:
        yield dirpath
    finally:
        shutil.rmtree(dirpath)

@contextlib.contextmanager
def cd(newpath):
    prev_path = os.getcwd()
    os.chdir(os.path.abspath(newpath))
    try:
        yield
    finally:
        os.chdir(prev_path)

def save_remove(file):
    """Remove a file if it exists."""
    if os.path.exists(file):
        os.remove(file)

def approximate_bl(atom1, atom2):
    """Approximate for the bond length between two atoms."""
    pt = Chem.GetPeriodicTable()
    return 1.1 * (pt.GetRcovalent(atom1.atomic) + pt.GetRcovalent(atom2.atomic))

def find_best_orientation(atom, bond_length, anchor, neighs, **kwargs):
    """Find the best orientation of the added bond."""
    per = kwargs.get("per", (0, 0, 0))
    box = kwargs.get("per", (20.0, 20.0, 20.0))
    maxmin_dist = 0.0
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                if i==0 and j==0 and k==0: continue
                ori = [i, j, k]
                dist_curr = bond_length * np.asarray(ori) / np.linalg.norm(ori)
                a_temp = deepcopy(atom)
                a_temp.xyz = np.asarray(anchor.xyz) + dist_curr
                min_dist = float('inf')
                for neigh in neighs:
                    min_dist = min(min_dist, distance(a_temp.xyz, neigh.xyz, per, box))
                if min_dist > maxmin_dist:
                    ori_out = ori
                    dist_out = dist_curr
                    maxmin_dist = min_dist
    return ori_out, dist_out

def center_of_mass(positions, topology, indices=None, per=(True, True, True), box=[0.0, 0.0, 0.0]):
    from openmm.unit import Quantity
    if indices is None:
        indices = list(range(len(positions)))
    unit = positions.unit
    resPositions = np.asarray([positions[idx]._value for idx in indices])
    resPositions = wrap_coords(resPositions, per=per, box=box.value_in_unit(unit))
    atoms = list(topology.atoms())
    masses = np.asarray([atoms[idx].element._mass._value for idx in indices])
    COM = (np.transpose(resPositions).dot(masses)) / sum(masses)
    COM = Quantity(value=COM, unit=unit)
    newPositions = deepcopy(positions)
    atomidx = 0
    for idx in indices:
        newPositions[idx] = Quantity(value=resPositions[atomidx], unit=unit)
        atomidx += 1
    return newPositions, COM

def wrap_coords(coords, per=(True, True, True), box=[0.0, 0.0, 0.0]):
    if len(coords) == 0: return coords
    coords = np.asarray(coords)
    x_coords = coords[:, 0]
    if per[0]:
        x0 = x_coords[0]
        for idx in range(1, len(x_coords)):
            x = x_coords[idx]
            if abs(x-x0) > box[0] / 2:
                x = x + box[0] if x < x0 else x - box[0]
                x_coords[idx] = x

    y_coords = coords[:, 1]
    if per[1]:
        y0 = y_coords[0]
        for idx in range(1, len(y_coords)):
            y = y_coords[idx]
            if abs(y-y0) > box[1] / 2:
                y = y + box[1] if y < y0 else y - box[1]
                y_coords[idx] = y

    z_coords = coords[:, 2]
    if per[2]:
        z0 = z_coords[0]
        for idx in range(1, len(z_coords)):
            z = z_coords[idx]
            if abs(z-z0) > box[2] / 2:
                z = z + box[2] if z < z0 else z - box[2]
                z_coords[idx] = z
    return np.stack((x_coords, y_coords, z_coords), axis=1)

def random_rotation_matrix(drot=1.0):
    """
    Generate a random axis and angle for rotation of coordinates.
    reference: grand source code: https://github.com/essex-lab/grand

    Returns
    -------
    rot_matrix : numpy.ndarray
        Rotation matrix generated
    """
    # First generate a random axis about which the rotation will occur
    rand1 = rand2 = 2.0

    while (rand1**2 + rand2**2) >= 1.0:
        rand1 = np.random.rand()
        rand2 = np.random.rand()
    rand_h = 2 * np.sqrt(1.0 - (rand1**2 + rand2**2))
    axis = np.array([rand1 * rand_h, rand2 * rand_h, 1 - 2*(rand1**2 + rand2**2)])
    axis /= np.linalg.norm(axis)

    # Get a random angle
    theta = np.pi * (2*np.random.rand() - 1.0) * drot

    # Simplify products & generate matrix
    x, y, z = axis[0], axis[1], axis[2]
    x2, y2, z2 = axis[0]*axis[0], axis[1]*axis[1], axis[2]*axis[2]
    xy, xz, yz = axis[0]*axis[1], axis[0]*axis[2], axis[1]*axis[2]
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[cos_theta + x2*(1-cos_theta),   xy*(1-cos_theta) - z*sin_theta, xz*(1-cos_theta) + y*sin_theta],
                           [xy*(1-cos_theta) + z*sin_theta, cos_theta + y2*(1-cos_theta),   yz*(1-cos_theta) - x*sin_theta],
                           [xz*(1-cos_theta) - y*sin_theta, yz*(1-cos_theta) + x*sin_theta, cos_theta + z2*(1-cos_theta)  ]])

    return rot_matrix

def distance(coords1, coords2, per=(False, False, False), box=[0.0, 0.0, 0.0]):
    """Calculate the distance between coords1 and coords2"""
    if not per[0]:
        dx = abs(coords1[0]-coords2[0])
    else:
        dx = min(abs(coords1[0]-coords2[0]), abs(abs(coords1[0]-coords2[0])-box[0]))

    if not per[1]:
        dy = abs(coords1[1]-coords2[1])
    else:
        dy = min(abs(coords1[1]-coords2[1]), abs(abs(coords1[1]-coords2[1]-box[1])))

    if not per[2]:
        dz = abs(coords1[2]-coords2[2])
    else:
        dz = min(abs(coords1[2]-coords2[2]), abs(abs(coords1[2]-coords2[2]-box[2])))

    return np.sqrt(dx*dx + dy*dy + dz*dz)

def optimize_config(obj, perturb_range=(-0.25, 0.25)):

    from metamol.rw import convert_to_rd
    clone = deepcopy(obj)
    left, right = perturb_range
    for atom in clone.atoms_iter():
        atom.x += np.random.uniform(left, right)
        atom.y += np.random.uniform(left, right)
        atom.z += np.random.uniform(left, right)
    rd_system = convert_to_rd(clone)
    Chem.SanitizeMol(rd_system)
    status = AllChem.UFFOptimizeMolecule(rd_system)
    if status:
        warnings.warn("Configuration optimization not successful")
        return obj
    pos = rd_system.GetConformers()[0].GetPositions()
    for idx, atom in enumerate(clone.atoms):
        atom.xyz = pos[idx]
    return clone

############# Function that corrects bond order in rdkit mol ################# 
from typing import List

def fix_bond_order(mol: Chem.Mol) -> Chem.Mol:
    """On a Mol where hydrogens are present it guesses bond order.
       Developed by Matteo Ferla. For more detail, go to:
       https://blog.matteoferla.com/2020/02/guess-bond-order-in-rdkit-by-number-of.html."""
      
    def is_sp2(atom: Chem.Atom) -> bool:
        N_neigh = len(atom.GetBonds())
        symbol = atom.GetSymbol()
        if symbol == 'H':
            return False
        elif symbol == 'N' and N_neigh < 3:
            return True
        elif symbol == 'C' and N_neigh < 4:
            return True
        elif symbol == 'O' and N_neigh < 2:
            return True
        else:
            return False

    def get_other(bond: Chem.Bond, atom: Chem.Atom) -> Chem.Atom:
        """Given an bond and an atom return the other."""
        if bond.GetEndAtomIdx() == atom.GetIdx(): # atom == itself gives false.
            return bond.GetBeginAtom()
        else:
            return bond.GetEndAtom()
    
    def find_sp2_bonders(atom: Chem.Atom) -> List[Chem.Atom]:
        return [neigh for neigh in find_bonders(atom) if is_sp2(neigh)]

    def find_bonders(atom: Chem.Atom) -> List[Chem.Atom]:
        return atom.GetNeighbors()

    def descr(atom: Chem.Atom) -> str:
        return f'{atom.GetSymbol()}{atom.GetIdx()}'

    ## main body of function
    for atom in mol.GetAtoms():
        #print(atom.GetSymbol(), is_sp2(atom), find_sp2_bonders(atom))
        if is_sp2(atom):
            doubles = find_sp2_bonders(atom)
            if len(doubles) == 1:
                #tobedoubled.append([atom.GetIdx(), doubles[0].GetIdx()])
                b = mol.GetBondBetweenAtoms( atom.GetIdx(), doubles[0].GetIdx())
                if b:
                    b.SetBondType(Chem.rdchem.BondType.DOUBLE)
                else:
                    raise ValueError('Issue with:', descr(atom), descr(doubles[0]))
            elif len(doubles) > 1:
                for d in doubles:
                    b = mol.GetBondBetweenAtoms( atom.GetIdx(), d.GetIdx())
                if b:
                    b.SetBondType(Chem.rdchem.BondType.AROMATIC)
                    b.SetIsAromatic(True)
                else:
                    raise ValueError('Issue with:', descr(atom), descr(d))
            elif len(doubles) == 0:
                print(descr(atom),' is underbonded!')
        else:
            pass
            #print(descr(atom),' is single', find_bonders(atom))
    return mol

def parmed_load(filename, structure=True, GMX='', **kwargs):
    # Modified parmed load function that fixes some element errors.

    import parmed as pmd
    if GMX:
        from parmed import gromacs
        shared_path = os.path.join(os.path.split(GMX)[0], '..', 'share', 'gromacs', 'top')
        gromacs.GROMACS_TOPDIR = shared_path

    pmd_extra_elements = {'CL': (17, 35.4532), 
                    'NA': (11, 22.989769282),
                    'MG': (12, 24.30506),
                    'BE': (4, 9.0121823),
                    'LI': (3, 6.9412),
                    'ZN': (30, 65.4094),
                }

    struct = pmd.load_file(filename, structure=structure, **kwargs)
    # Clean up misinformation of some elements
    for atom in struct.atoms:
        name = atom.name.upper()
        while len(name)>1 and name[0].isdigit():
            name = name[1:]
        name = name[:2]
        if name.startswith('K'):
            atom.atomic_number = 19
            atom.mass = 39.09831
        elif len(atom.residue) == 1 and name == 'CA':
            atom.atomic_number = 20
            atom.mass = 40.0784
        elif name in pmd_extra_elements:
            atom.atomic_number = pmd_extra_elements[name][0]
            atom.mass = pmd_extra_elements[name][1]    

    return struct