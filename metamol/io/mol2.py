import os

import metamol as meta
from metamol.exceptions import MetaError
from metamol.utils.convert_formats import *

__all__ = ["read_mol2", "write_mol2"]

def read_mol2(filename, host_obj, asSystem, backend):
    """Read a mol2 file into a Molecule/System object."""
    file_exists = os.path.exists(filename)
    if not file_exists:
        raise FileNotFoundError("File {} is not found on disk".format(filename))

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
        
        return convert_from_pmd(struct=struct, host_obj=host_obj)
    
    else:
        raise MetaError("Right now only support parmed backend for reading mol2 files.")

def write_mol2(myobj, filename, backend=None, bonds=True):
    """Write to a mol2 file via parmed backend."""
    if (not isinstance(myobj, meta.Molecule)) and (not isinstance(myobj, meta.System)):
        raise TypeError("The input object must be either a Molecule or a System")

    if backend is None:
        backend = 'parmed'
    
    if backend.lower()=='parmed' or backend.lower()=='pmd':
        struct = convert_to_pmd(myobj, bonds=bonds)
        struct.save(filename, overwrite=True)

    else:
        raise NotImplementedError("Right now only support parmed backend for writing mol2 files")