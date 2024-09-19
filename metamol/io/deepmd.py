import numpy as np
import metamol as meta
from metamol.system import Box
from metamol.utils.convert_formats import *

__all__ = ["read_dp_json", "write_dp_json"]

def read_dp_jason(filename):
    """Read a VASP POSCAR file into a Meta System."""
    sys_out = meta.System()
    lines = []
    with open(filename, "r") as f:
        for line in f.readlines():
            lines.append(line)
    lines = lines[::-1]
    sys_out.name = lines.pop()

    scale = float(lines.pop().strip())

    a = np.fromiter(lines.pop().split(), dtype="float64")
    b = np.fromiter(lines.pop().split(), dtype="float64")
    c = np.fromiter(lines.pop().split(), dtype="float64")

    lattice_vectors = np.stack((a, b, c))
    box_lengths, box_angle = vectors_to_box(lattice_vectors)
    box = Box(bounds=np.asarray([0.0, 0.0, 0.0]+list(box_lengths)), angle=box_angle)
    sys_out.box = box

    # POSCAR files do not require atom types to be specified
    # this block handles unspecified types
    line = lines.pop().split()
    if line and line[0].isdigit():
        n_types = np.fromiter(line, dtype="int")
        types = ["_" + chr(i + 64) for i in range(1, len(n_types) + 1)]
        # if no types exist, assign placeholder types "_A", "_B", "_C", etc
    else:
        types = line
        line = lines.pop().split()
        n_types = np.fromiter(line, dtype="int")

    all_types = []
    for itype, n in zip(types, n_types):
        all_types += [itype]*n

    all_types.sort()
    total_atoms = len(all_types)

    # handle optional argument "Selective dynamics"
    # and required arguments "Cartesian" or "Direct"
    switch = lines.pop()[0].upper()

    # selective_dynamics = False
    if switch == "S":
        # selective_dynamics = True
        switch = lines.pop()[0].upper()

    if switch == "C":
        cartesian = True
    else:
        cartesian = False

    # Slice is necessary to handle files using selective dynamics
    coords = np.stack(
        [
            np.fromiter(line.split()[:3], dtype="float64")
            for line in lines[total_atoms::-1]
        ]
    )

    if cartesian:
        coords = coords * scale
    else:
        coords = coords.dot(lattice_vectors) * scale

    mol = meta.Molecule()
    for idx, type in enumerate(all_types):
        mol.atomList.append(meta.Atom(idx=idx+1, symbol=type, x=coords[idx][0], y=coords[idx][1], z=coords[idx][2]))
        mol.numAtoms += 1
    sys_out.add(mol)

    if host_obj is None:
        return sys_out
    else:
        host_obj.copy(sys_out)
        return host_obj

def write_poscar(sys, filename, **kwargs):
    """Write to a VASP POSCAR file."""
    if not isinstance(sys, meta.System):
        raise TypeError("The object to write must be either a Metamol System")
    
    sys.flatten()
    atoms = [a.symbol for m in sys.molecules_iter() for a in m.atoms_iter()]
    
    # Sort element names alphabetically
    unique_atoms = np.unique(atoms)

    count_list = [str(atoms.count(i)) for i in unique_atoms]

    # Sort the coordinates so they are in the same order as the elements
    atoms = np.array([(sym, idx) for idx, sym in enumerate(atoms)], 
                     dtype=[('typ', '<U5'), ('idx', int)])
    sorted_xyz = sys.xyz[np.argsort(atoms, order=('typ', 'idx'))]

    if sys.box is None:
        box_lengths = sys.get_boundingbox()
    else:
        box_lengths = sys.box.lengths

    # elif len(sys.box) == 6:
    #     box = [sys.box[3]-sys.box[0], sys.box[4]-sys.box[1], sys.box[5]-sys.box[2]]
    # else:
    #     box = sys.box

    # if not sys.box_angle:
    #     sys.box_angle = [90.0, 90.0, 90.0]
    
    box_angle = sys.box.angle

    lattice = box_to_vectors(np.asarray(box_lengths), np.asarray(box_angle))

    coord_style = kwargs.get("coord_style", "cartesian")
    lattice_constant = kwargs.get("lattice_constant", 1.0)

    if coord_style == "cartesian":
        sorted_xyz /= lattice_constant
    elif coord_style == "direct":
        sorted_xyz = sorted_xyz.dot(np.linalg.inv(lattice)) / lattice_constant
    else:
        raise ValueError("coord_style must be either 'cartesian' or 'direct'")

    title = sys.name if sys.name else filename.split('.')[0]
    with open(filename, "w") as f:
        f.write("{0} Created by MetaMol (version={1})\n".format(title, meta.__version__))
        f.write(f"\t{lattice_constant:.15f}\n")

        f.write("\t{0:.15f} {1:.15f} {2:.15f}\n".format(*lattice[0]))
        f.write("\t{0:.15f} {1:.15f} {2:.15f}\n".format(*lattice[1]))
        f.write("\t{0:.15f} {1:.15f} {2:.15f}\n".format(*lattice[2]))
        f.write("{}\n".format("\t".join(unique_atoms)))
        f.write("{}\n".format("\t".join(count_list)))
        f.write(f"{coord_style}\n")
        for row in sorted_xyz:
            f.write(f"{row[0]:.15f} {row[1]:.15f} {row[2]:.15f}\n")

            