import os

import metamol as meta
from metamol.system import Box
from metamol.exceptions import MetaError
from metamol.utils.help_functions import distance
from metamol.bond_graph import BondGraph
from metamol.utils.convert_formats import *

__all__ = ["read_xyz", "write_xyz"]

def read_xyz(filename, host_obj=None, asSystem=False, backend=None, **kwargs):
    """Read an xyz file to construct a Molecule/System object."""
    file_exists = os.path.exists(filename)
    if not file_exists:
        raise FileNotFoundError("The provided xyz file {} is not found on disk".format(filename))

    if backend is None:
        backend = 'internal'
    
    if backend.lower() == 'internal':
        from rdkit import Chem
        pt = Chem.GetPeriodicTable()
        title = ""
        atomList = []
        with open(filename, 'r') as f:
            for line_idx, line in enumerate(f):
                if line_idx == 0:
                    continue
                elif line_idx == 1:
                    if not line.startswith("Created by MetaMol"):
                        line_spt = line.split()
                        if line_spt:
                            title = line_spt[0]
                else:
                    atomic_symbol, x, y, z = line.split()
                    atomList.append(meta.Atom(idx=line_idx-1, symbol=atomic_symbol.upper(),
                                        x=float(x), y=float(y), z=float(z)))
        
        per = kwargs.get("per", [0, 0, 0])
        box = kwargs.get("box", [0.0, 0.0, 0.0])
        cov_factor = kwargs.get("cov_factor", 1.5)

        bg = BondGraph()
        for i in range(len(atomList)-1):
            for j in range(i+1, len(atomList)):
                atom1, atom2 = atomList[i], atomList[j]
                bg.add_node(atom1)
                bg.add_node(atom2)
                Rcov_i, Rcov_j = pt.GetRcovalent(atom1.atomic)*cov_factor, pt.GetRcovalent(atom2.atomic)*cov_factor
                if distance(atom1.xyz, atom2.xyz, per, box) <= Rcov_i + Rcov_j:
                    bg.add_edge(atom1, atom2)

        connected_parts = bg.connected_nodes()
        numMols = len(connected_parts)
        sys_out = meta.System(name=title)
        sys_out.box = Box(bounds=box, per=per)
        #sys_out.per = per
        for i in range(numMols):
            mol = meta.Molecule(name=title)
            for atom in sorted(connected_parts[i], key=lambda a: a.idx):
                mol.atomList.append(atom)
                mol.numAtoms += 1
                for other in bg._adj[atom]:
                    mol.add_bond((atom, other))
            mol.numBonds //= 2
            sys_out.add(mol)

        sys_out.update_atom_idx()
        
        if not asSystem:
            if isinstance(host_obj, meta.System):
                raise TypeError("Host object must be Molecule when asSystem is set to False")
            if host_obj:
                host_obj.copy(sys_out.molecules[0])
            return sys_out.molecules[0]
        
        else:
            if host_obj:
                host_obj.copy(sys_out)
            return sys_out            

    elif backend.lower()=='rdkit' or backend.lower()=='rd':
        from rdkit import Chem
        from metamol.utils.load_xyz import xyz2mol
        rdmols = xyz2mol(filename)

        if not asSystem:
            if isinstance(host_obj, meta.System):
                raise TypeError("Host object must be Molecule when asSystem is set to False")

            return convert_from_rd(rdmol=rdmols[0], host_obj=host_obj)

        if isinstance(host_obj, meta.Molecule):
                raise TypeError("Host object must be System when asSystem is set to True")

        if host_obj:
            sys_out = host_obj
        else:
            sys_out = meta.System()

        for rdmol in rdmols:
            mols = [mol for mol in Chem.GetMolFrags(rdmol, asMols=True)]
            smi_list = [Chem.MolToSmiles(mol) for mol in mols]
            for idx, mol in enumerate(mols):
                sys_out.add(convert_from_rd(rdmol=mol, host_obj=None, smi=smi_list[idx]))
        return sys_out
    
    else:
        raise MetaError("Right now only support internal/rdkit backends for reading xyz files")

def write_xyz(myobj, filename):
    """Write to a xyz file."""
    if isinstance(myobj, meta.Molecule):
        flag = 1
    elif isinstance(myobj, meta.System):
        flag = 2
    else:
        raise TypeError("The object to write must be either a Molecule or a System")
    
    savefile = open(filename, "w")
    savefile.write(str(myobj.numAtoms) + '\n')
    savefile.write("{0} Created by MetaMol (version={1})".format(myobj.name, meta.__version__).strip())
    savefile.write("\n")

    if flag==1:
        for atom in myobj.atoms:
            savefile.write(atom.symbol + '\t' + str(atom.x) + '\t' 
                + str(atom.y) + '\t' + str(atom.z) + '\n')
    
    elif flag==2:
        for mol in myobj.molecules_iter():
            for atom in mol.atoms_iter():
                savefile.write(atom.symbol + '\t' + str(atom.x) + '\t' 
                    + str(atom.y) + '\t' + str(atom.z) + '\n')                
    
    savefile.close()