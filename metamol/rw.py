import os
import metamol as meta
from metamol.exceptions import MetaError
from metamol.io.lammps import read_lammps_dumps
from metamol.io.xyz import *
from metamol.io.pdb import *
from metamol.io.mol2 import *
from metamol.io.lammps import *
from metamol.io.gromacs import *
from metamol.io.vasp import *
from metamol.utils.convert_formats import *

"""Read Functions"""
def readfile(filename: str, host_obj=None, asSystem=True, smiles=False, backend=None, file_type=None, **kwargs):
    """Read a file of existing topology into a System or a Molecule object
        If asSystem is True, then return a System object. 
        Otherwise return a Molecule object from the first molecule read from the input file.
    """

    if not isinstance(filename, str):
        raise TypeError("File name must be a string")
    
    if smiles or filename.endswith('txt'):
        return read_smi(filename=filename, host_obj=host_obj, asSystem=asSystem, backend=backend)
    # read from xyz file
    elif filename.endswith('xyz'):
        return read_xyz(filename=filename, host_obj=host_obj, asSystem=asSystem, backend=backend, **kwargs)
    # read from pdb file
    elif filename.endswith('pdb'):
        return read_pdb(filename=filename, host_obj=host_obj, asSystem=asSystem, backend=backend)
    # read from mol2 file
    elif filename.endswith('mol2'):
        return read_mol2(filename=filename, host_obj=host_obj, asSystem=asSystem, backend=backend)
    # read from lammps data file
    elif file_type=='lammps' or filename.endswith('data') or filename.split('/')[-1].startswith('data.') \
        or filename.endswith('lmp') or filename.split('/')[-1].startswith('lmp.') \
        or filename.endswith('lammps') or filename.split('/')[-1].startswith('lammps.'):
        return read_lammps(filename=filename, host_obj=host_obj, **kwargs)
    # read from lammps dump file
    elif filename.endswith('dump'):
        return read_lammps_dumps(filename=filename, host_obj=host_obj)
    # read from gromacs file
    elif file_type=='gromacs' or filename.endswith('gro') or filename.endswith('top'):
        return read_gromacs(filename=filename, host_obj=host_obj, **kwargs)
    # read from vasp poscar file
    elif file_type=='vasp' or filename.endswith('poscar'):
        return read_poscar(filename=filename, host_obj=host_obj, **kwargs)
    else:
        raise MetaError("File format not supported: {}".format(filename))

def read_smi(filename, host_obj, asSystem, backend):
    """Read a smiles string or file to construct a Molecule/System object."""
    file_exists = os.path.exists(filename)

    if backend is None:
        backend = 'rdkit'
    
    if backend.lower()=='rdkit' or backend.lower()=='rd':
        from rdkit import Chem
        from rdkit.Chem import AllChem

        if file_exists:
            rdmols = Chem.SmilesMolSupplier(filename)
            if not rdmols:
                raise ValueError(
                    "The SMILES file provided is invalid."
                )
            mols = [mol for mol in rdmols]
            smi_list = [Chem.MolToSmiles(mol) for mol in rdmols]
        else:
            smi_list = filename.split(',')
            mols = [Chem.MolFromSmiles(smi) for smi in smi_list]
        
        if not asSystem:
            if isinstance(host_obj, meta.System):
                raise TypeError("Host object must be Molecule when asSystem is set to False")

            return convert_from_rd(rdmol=mols[0], host_obj=host_obj, smi=smi_list[0])

        if isinstance(host_obj, meta.Molecule):
                raise TypeError("Host object must be System when asSystem is set to True")
        
        if host_obj:
            sys_out = host_obj
        else:
            sys_out = meta.System()

        for idx, mol in enumerate(mols):
            sys_out.add(convert_from_rd(rdmol=mol, host_obj=None, smi=smi_list[idx]))
        return sys_out

    else:
        raise MetaError("Right now only support rdkit backend for reading smiles")

def read_mol(filename, host_obj, asSystem, backend):
    """Read a mol file to construct a Molecule/System object."""
    file_exists = os.path.exists(filename)
    if not file_exists:
        raise FileNotFoundError("The provided mol file {} is not found on disk".format(filename))

    if backend is None:
        backend = 'rdkit'
    
    if backend.lower()=='rdkit' or backend.lower()=='rd':
        from rdkit import Chem
        from rdkit.Chem import AllChem

        rdmols = Chem.MolFromMolFile(filename, sanitize=False, removeHs=False)
        if not rdmols:
            raise ValueError(
                "The Mol file provided is invalid."
            )
        mols = [mol for mol in Chem.GetMolFrags(rdmols, asMols=True)]
        smi_list = [Chem.MolToSmiles(mol) for mol in mols]
            
        if not asSystem:
            if isinstance(host_obj, meta.System):
                raise TypeError("Host object must be Molecule when asSystem is set to False")

            return convert_from_rd(rdmol=mols[0], host_obj=host_obj, smi=smi_list[0])

        if isinstance(host_obj, meta.Molecule):
                raise TypeError("Host object must be System when asSystem is set to True")
        
        if host_obj:
            sys_out = host_obj
        else:
            sys_out = meta.System()

        for idx, mol in enumerate(mols):
            sys_out.add(convert_from_rd(rdmol=mol, host_obj=None, smi=smi_list[idx]))
        return sys_out
    
    else:
        raise MetaError("Right now only support rdkit for reading mol files")

def savefile(obj, filename: str, frameId=1, timestep=None, **kwargs):
    """Save Molecule or System to a file."""
    if frameId == 1 and timestep is None:
        obj_save = obj
    else:
        from copy import deepcopy
        obj_save = deepcopy(obj)
        frame = obj.get_frame(frameId, timestep)
        obj_save.xyz = frame.coords

    lammps = kwargs.get('lammps', False)
    gromacs = kwargs.get('gromacs', False)

    if filename.endswith('xyz'):
        write_xyz(obj_save, filename)

    elif filename.endswith('pdb'):
        backend = kwargs.get('backend', None)
        write_pdb(obj_save, filename, backend)

    elif filename.endswith('mol2'):
        backend = kwargs.get('backend', None)
        bonds = kwargs.get('bonds', True)
        write_mol2(obj_save, filename, backend, bonds=bonds)

    elif lammps or filename.endswith('lmp') or filename.startswith('lmp.') \
        or filename.endswith('lammps') or filename.startswith('lammps.') \
        or filename.endswith('data') or filename.startswith('data.'):
        atom_style = kwargs.get('atom_style', 'full')
        unit_style = kwargs.get('unit_style', 'real')
        pair_coeff_label = kwargs.get('pair_coeff_label', 'lj/long/coul/long')
        zero_dihedral_weight = kwargs.get('zero_dihedral_weight', True)
        combining_rule = kwargs.get('combining_rule', 'arithmetic')
        hybrid = kwargs.get('hybrid', dict())
        tip4p = kwargs.get('tip4p', '')

        write_lammps(sys=obj_save, filename=filename, 
                    atom_style=atom_style, unit_style=unit_style,
                    pair_coeff_label=pair_coeff_label, 
                    zero_dihedral_weight=zero_dihedral_weight,
                    combining_rule=combining_rule, hybrid=hybrid, tip4p=tip4p)

    elif gromacs or filename.endswith('gro') or filename.endswith('top'):
        write_gromacs(sys=obj_save, filename=filename, **kwargs)
    
    elif filename.endswith('poscar'):
        write_poscar(sys=obj_save, filename=filename, **kwargs)

    else:
        ext = filename.split('.')[-1]
        raise ValueError("Extension {} is currently not supported".format(ext))
