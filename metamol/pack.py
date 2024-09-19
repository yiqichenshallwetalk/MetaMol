"""A wrapper for PACKMOL program.

http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml
"""
import os
import sys
import tempfile
import warnings
from distutils.spawn import find_executable
from subprocess import PIPE, Popen
import numpy as np

from metamol.exceptions import MetaError
from metamol.utils.constants import AMU

PACKMOL = find_executable("packmol")
PACKMOL_HEADER = """
tolerance {0:.16f}
filetype xyz
output {1}
seed {2}
sidemax {3}
"""
PACKMOL_SOLUTE = """
structure {0}
    number 1
    center
    fixed {1:.3f} {2:.3f} {3:.3f} 0. 0. 0.
end structure
"""
PACKMOL_BOX = """
structure {0}
    number {1:d}
    inside box {2:.3f} {3:.3f} {4:.3f} {5:.3f} {6:.3f} {7:.3f}
    {8}
end structure
"""
PACKMOL_SPHERE = """
structure {0}
    number {1:d}
    inside sphere {2:.3f} {3:.3f} {4:.3f} {5:.3f}
    {6}
end structure
"""

PACKMOL_CONSTRAIN = """
constrain_rotation x 0. 0.
constrain_rotation y 0. 0.
constrain_rotation z 0. 0.
"""

def initconfig_box(
    system,
    region=None,
    density=None,
    aspect_ratio=None,
    overlap=2.0,
    seed=12345,
    sidemax=1000.0,
    edge=2.0,
    fix_orientation=False,
    temp_file=None,
):
    """Fill a box with an `System` object using PACKMOL.

    `initconfig_box` takes a `System` as input and
    returns a new `System` that has been filled to specification by PACKMOL,
    which can then be used as the initial configuration in molecular simulations.

    For the cases in which `density` is not None, the box size is
    caomputed internally. The aspect ratio of the box is based on System.box.

   Arguments
    ----------
    system : meta.System
        The model system to fill in box.
    region : list of floats, default=None
        The region where particles packed in.
    density : float, units :math:`kg/m^3` , default=None
        Target density for the system in macroscale units. 
        The box size will be determined based on the specified density.
        It will override the box argument
    aspect_ratio : list of int, default=None
        The aspect ratio of box created based on the given density.
        Only valid when density is not None.
    overlap : float, units A, default=2.0
        Minimum separation between atoms of different molecules.
    seed : int, default=12345
        Random seed to be passed to PACKMOL.
    sidemax : float, optional, default=1000.0
        Needed to build an initial approximation of the molecule distribution in
        PACKMOL. All system coordinates must fit with in +/- sidemax, so
        increase sidemax accordingly to your final box size.
    edge : float, units A, default=2.0
        Buffer at the edge of the box to not place molecules. This is necessary
        in some systems because PACKMOL does not account for periodic boundary
        conditions in its optimization.
    fix_orientation : bool or list of bools
        Specify that molecules should not be rotated when filling the box,
        default=False.
    temp_file : str, default=None
        File name to write PACKMOL raw output to.

    Returns
    -------
    filled : meta.System
    """
    # check that the user has the PACKMOL binary on their PATH
    _check_packmol(PACKMOL)

    if region is not None:
        if len(region)!=3 and len(region)!=6:
            raise ValueError("The simulation box must be 3 dimensional")
    else:
        region = system.box.bounds

    if not isinstance(fix_orientation, (list, set)):
        fix_orientation = [fix_orientation] * system.numMols

    if system:
        if system.numMols != len(fix_orientation):
            raise ValueError(
                "`number of molecules` and `fix_orientation` must be of "
                "equal length.")

    if density is not None:
        total_mass = _calculate_mass(system)
        if total_mass == 0:
            raise MetaError("The total mass of the System is 0. "
            "Cannot fill simulation box based on the density specified."
            )
        
        # Conversion from (amu/(kg/m^3))**(1/3) to A
        L = (total_mass * AMU / density)**(1/3) * 1.0e10
        if aspect_ratio is None:
            new_box_length = np.asarray([L, L, L])
        else:
            L *= np.prod(np.asarray(aspect_ratio)) ** (-1 / 3)
            new_box_length = L * np.asarray(aspect_ratio)
        
        #print(new_box_length)
        if len(region) == 6:
            region[3] = region[0] + new_box_length[0]
            region[4] = region[1] + new_box_length[1]
            region[5] = region[2] + new_box_length[2]
        elif len(region) == 3:
            region = new_box_length

    if len(region) == 6:
        box_mins = np.asarray(region[:3]) + edge
        box_maxs = np.asarray(region[3:]) - edge
    elif len(region) == 3:
        box_mins = np.asarray([edge] * 3)
        box_maxs = np.asarray(region) - edge
    else:
        raise MetaError("The system's box is not 3 dimentional")

    # Build the input file for each molecule and call packmol.
    filled_xyz = _new_xyz_file()

    # create a list to contain the file handles for the compound temp files
    mol_dict = dict()
    mol_xyz_list = list()
    try:
        input_text = PACKMOL_HEADER.format(
            overlap, filled_xyz.name, seed, sidemax
        )
        for mol, m_mols, rotate in zip(
            system.molecules, system.dup, fix_orientation
        ):
            m_mols = int(m_mols)
            if mol.name not in mol_dict:
                mol_dict[mol.name] = [mol, m_mols]
            else:
                mol_dict[mol.name][1] += 1

        for molname, vals in mol_dict.items():
            mol_xyz = _new_xyz_file()
            mol_xyz_list.append(mol_xyz)

            mol, molnum = vals[0], vals[1]
            mol.save(mol_xyz.name)
            input_text += PACKMOL_BOX.format(
                mol_xyz.name,
                molnum,
                box_mins[0],
                box_mins[1],
                box_mins[2],
                box_maxs[0],
                box_maxs[1],
                box_maxs[2],
                PACKMOL_CONSTRAIN if rotate else "",
            )
        _run_packmol(input_text, filled_xyz, temp_file)

        # Update the coordinates for the system.
        #filled = Compound()
        #filled = _create_topology(filled, compound, n_compounds)
        system.flatten()
        system.update_coords(filled_xyz.name)

    # ensure that the temporary files are removed from the machine after filling
    finally:
        for file_handle in mol_xyz_list:
            file_handle.close()
            os.unlink(file_handle.name)
        filled_xyz.close()
        os.unlink(filled_xyz.name)
    
    return system

def _calculate_mass(system):
    """Calculate the total mass of the System"""
    total_mass = 0
    find_zero_mass = False

    for idx, mol in enumerate(system.molecules):
        mol_mass = 0
        for atom in mol.atoms_iter():
            if atom.mass == 0 and not find_zero_mass:
                find_zero_mass = True
            mol_mass += atom.mass
        total_mass += mol_mass * system.dup[idx]

    if find_zero_mass:
        warnings.warn(
            "Some of the atoms in the Ststem have a mass of zero. "
            "This may have an effect on density calculations."
        )

    return total_mass


def _new_xyz_file():
    """Generate xyz file using tempfile.NamedTemporaryFile.

    Return
    ------
    _ : file-object
        Temporary xyz file.
    """
    return tempfile.NamedTemporaryFile(suffix=".xyz", delete=False)

def _packmol_error(out, err):
    """Log packmol output to files."""
    with open("log.txt", "w") as log_file:
        log_file.write(out)
    raise RuntimeError("PACKMOL failed. See 'log.txt'")

def _run_packmol(input_text, filled_xyz, temp_file):
    """Call PACKMOL to pack system based on the input text.

    Parameters
    ----------
    input_text : str, required
        String formatted in the input file syntax for PACKMOL.
    filled_xyz : `tempfile` object, required
        Tempfile that will store the results of PACKMOL packing.
    temp_file : `tempfile` object, required
        Where to copy the filled tempfile.
    """
    # Create input file
    packmol_inp = tempfile.NamedTemporaryFile(
        mode="w", delete=False, prefix="packmol-", suffix=".inp"
    )
    packmol_inp.write(input_text)
    packmol_inp.close()

    proc = Popen(
        "{} < {}".format(PACKMOL, packmol_inp.name),
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        universal_newlines=True,
        shell=True,
    )
    out, err = proc.communicate()

    if "WITHOUT PERFECT PACKING" in out:
        warnings.warn(
            "Packmol finished with imperfect packing. Using the .xyz_FORCED "
            "file instead. This may not be a sufficient packing result."
        )
        os.system("cp {0}_FORCED {0}".format(filled_xyz.name))

    if "ERROR" in out or proc.returncode != 0:
        _packmol_error(out, err)
    else:
        # Delete input file if success
        os.remove(packmol_inp.name)

    if temp_file is not None:
        os.system("cp {0} {1}".format(filled_xyz.name, temp_file))

def _check_packmol(PACKMOL): 
    if not PACKMOL:
        msg = "Packmol not found."
        if sys.platform.startswith("win"):
            msg = (
                msg + " If packmol is already installed, make sure that the "
                "packmol.exe is on the path."
            )
        raise IOError(msg)