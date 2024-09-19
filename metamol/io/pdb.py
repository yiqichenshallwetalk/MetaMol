import os

import metamol as meta
from metamol.exceptions import MetaError
from metamol.utils.convert_formats import *

__all__ = ["read_pdb", "write_pdb"]

def read_pdb(filename, host_obj, asSystem, backend):
    """Read a pdb file to construct a Molecule/System object."""
    file_exists = os.path.exists(filename)
    if not file_exists:
        raise FileNotFoundError("The provided pdb file {} is not found on disk".format(filename))

    if backend is None:
        backend = 'parmed'

    if backend.lower()=='parmed' or backend.lower()=='pmd':
        from metamol.utils.help_functions import parmed_load

        struct = parmed_load(filename)

        if not asSystem:
            if isinstance(host_obj, meta.System):
                raise TypeError("Host object must be Molecule when asSystem is set to False")

            return convert_from_pmd(struct=struct, host_obj=host_obj, asSystem=False)

        if isinstance(host_obj, meta.Molecule):
                raise TypeError("Host object must be System when asSystem is set to True")
        
        return convert_from_pmd(struct=struct, host_obj=host_obj, asSystem=True)

    elif backend.lower() == 'openmm':
        import openmm as mm
        pdb = mm.app.PDBFile(filename)

        if not asSystem:
            if isinstance(host_obj, meta.System):
                raise TypeError("Host object must be Molecule when asSystem is set to False")

            return convert_from_openmm(topology=pdb.topology, positions=pdb.positions, host_obj=host_obj, asSystem=False)

        if isinstance(host_obj, meta.Molecule):
                raise TypeError("Host object must be System when asSystem is set to True")
        
        return convert_from_openmm(topology=pdb.topology, positions=pdb.positions, host_obj=host_obj, asSystem=True)

    elif backend.lower()=='rdkit' or backend.lower()=='rd':
        from rdkit import Chem
        #from rdkit.Chem import AllChem

        rdmols = Chem.MolFromPDBFile(filename, sanitize=False, removeHs=False)
        if not rdmols:
            raise ValueError(
                "The PDB file provided is invalid."
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
        raise MetaError("Right now only support parmed/rdkit backend for reading pdb files")

def write_pdb(myobj, filename, backend=None):
    """Write to a pdb file via parmed or rdkit backend."""
    if (not isinstance(myobj, meta.Molecule)) and (not isinstance(myobj, meta.System)):
        raise TypeError("The input object must be either a Molecule or a System")

    if backend is None:
        backend = 'parmed'
    
    if backend.lower()=='parmed' or backend.lower()=='pmd':
        struct = convert_to_pmd(myobj)
        struct.save(filename, overwrite=True)
    
    elif backend.lower()=='rdkit' or backend.lower()=='rd':
        from rdkit import Chem
        rdmol = convert_to_rd(myobj)
        Chem.MolToPDBFile(rdmol, filename)
    else:
        raise NotImplementedError("Right now only support parmed/rdkit backend for writing pdb files")