from typing import Iterable
import numpy as np
import itertools
import os
from copy import deepcopy

import metamol as meta
from metamol.molecule import Molecule
from metamol.bond_graph import BondGraph
from metamol.exceptions import MetaError
from metamol import rw

class Chain(Molecule):
    def __init__(self, index=1, name=""):
        super(Chain, self).__init__()

        self.name = name
        self.index = index
        self.molList, self.dup = [], []
        self.numMols = 0
        self.numWater = 0

    @property
    def atoms(self):
        """Return all atoms in the Chain.

        Returns
        ------
        List of all atoms in the Chain
        """
        all_atoms = []
        for mol in self.molecules_iter():
            all_atoms += mol.atoms        

        return all_atoms

    def atoms_iter(self):
        """Iterate through all atoms in the Chain.

        Yields
        ------
        Atom object
        """
        for atom in self.atoms:
            yield atom

    @property
    def molecules(self):
        """Return all molecules in the Chain."""
        return self.molList
    
    def molecules_iter(self):
        """Iterate through all molecules in the Chain."""
        for mol in self.molList:
            yield mol


    def add(self, mols=None, dup=None):
        """Add Molecules to the Chain."""
        if mols is None: return

        if isinstance(mols, meta.Molecule):
            mols = [mols]
            if isinstance(dup, int):
                dup = [dup]
        
        if not isinstance(mols, Iterable):
            raise TypeError("Only Molecule objects can be added. ")

        if dup is None:
            dup = [1]*len(mols)
        else:
            if len(dup) != len(mols):
                raise ValueError("The length of 'moList' and 'dup' must be equal")

        for idx, mol in enumerate(mols):
            if not isinstance(mol, meta.Molecule):
                raise TypeError("Only Molecule objects can be added. ")
            if dup[idx] > 1:
                self.flattened = False
            if isinstance(mol, (meta.Water3Site, meta.Water4Site)) or 'water' in mol.name:
                self.numWater += dup[idx]
            self.numMols += dup[idx]
            self.numAtoms += mol.numAtoms*dup[idx]
            self.numBonds += mol.numBonds*dup[idx]
        
        self.molList += mols
        self.dup += dup

    # @property
    # def bonds(self):
    #     """Return all bonds in the Chain.

    #     Returns
    #     ------
    #     List of all bonds in the Chain

    #     See Also
    #     --------
    #     bond_graph.edges : Return all edges in the bond graph
    #     """
    #     if self.bond_graph:
    #         return self.bond_graph.edges
    #     else:
    #         return []

    # def bonds_iter(self):
    #     """Iterate through all bonds in the Molecule.

    #     Yields
    #     ------
    #     tuple of Atom objects
    #         The next bond in the Molecule

    #     See Also
    #     --------
    #     bond_graph.edges_iter : Iterates over all edges in a BondGraph
    #     """
    #     if self.bond_graph:
    #         return self.bond_graph.edges_iter()
    #     else:
    #         return iter(())

        
    @property
    def xyz(self):
        """Return all atom coordinates in the molecule."""
        arr = np.fromiter(
                itertools.chain.from_iterable((a.x, a.y, a.z) for a in self.atoms),
                dtype=float,
        )
        return arr.reshape((-1, 3))

    def __getitem__(self, idx):
        """Get atom from Molecule."""
        if not isinstance(idx, int):
            raise TypeError("Atom index must be a integer")
        
        if idx < 0 or idx >= self.numAtoms:
            raise MetaError("Atom index out of range")

        return self.atoms[idx]

    def __repr__(self):
        """Representation of the Molecule."""

        desc = ["Molecule id: {}".format(id(self))]
        if self.name:
            desc.append("Name: {}".format(self.name))
        if self.smi:
            desc.append("SMILES: {}".format(self.smi))
        
        desc.append("Number of Atoms: {}".format(self.numAtoms))
        desc.append("Number of Bonds: {}".format(self.numBonds))

        return "\n".join(desc)

    def save(self, filename, fmt='xyz', **kwargs):
        
        ext = os.path.splitext(filename)[-1]
        if '.' not in ext:
            filename += '.' + fmt
        
        rw.savefile(self, filename, **kwargs)
        
    def copy(self, target=None):
        """Copy all info from another Molecule."""
        if target:
            if target.name:
                self.name = target.name
            if target.smi:
                self.smi = target.smi
            self.numAtoms = target.numAtoms
            self.numBonds = target.numBonds
            self.atomList = target.atomList
            self.bond_graph = target.bond_graph

    def draw(self, atomIdx=True, removeHs=False):
        """Show chemical drawing of the molecule
        Inputs
        ---------
        atomIdx: bool, default=True
            Whether to show atom index in the drawing
        removeHs: bool, default=False
            Whether to remove all hydrogen atoms in the drawing"""
        
        rdmol = rw.convert_to_rd(self)
        AllChem.Compute2DCoords(rdmol)
        #AllChem.UFFOptimizeMolecule(rdmol)

        if removeHs:
            rdmol = Chem.RemoveHs(rdmol)
        if atomIdx:
            for atom in rdmol.GetAtoms():
                atom.SetProp('atomLabel', str(atom.GetIdx()+1))
        return rdmol
        