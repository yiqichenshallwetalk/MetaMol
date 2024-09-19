from typing import Iterable
import numpy as np
import itertools
import os
from copy import deepcopy

import metamol as meta
from metamol.bond_graph import BondGraph
from metamol.exceptions import MetaError
from metamol import rw
from metamol.utils.visual import visual
from metamol.utils.help_functions import optimize_config

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from metamol.utils.geometry import Translate, Rotate, uv_degree
RDLogger.DisableLog('rdApp.*')

class Molecule(object):

    def __init__(self, input=None, smiles=False, name="", backend=None, chain=1, **kwargs):
        super(Molecule, self).__init__()

        self.name = name
        self.smi = None
        self.numAtoms, self.numBonds = 0, 0
        self.atomList = []
        self.bond_graph = None
        self.chain = chain

        #FF attributes, used for gromacs read function.
        self.angles, self.dihedrals = [], []
        self.rb_torsions, self.impropers = [], []

        if input is None:
            return

        if isinstance(input, str):
            rw.readfile(input, host_obj=self, asSystem=False, smiles=smiles, backend=backend, **kwargs)

        elif isinstance(input, Chem.Mol):
            rw.convert_from_rd(input, host_mol=self)
        
        elif isinstance(input, meta.Atom):
            #input.idx = 1
            self.atomList.append(input)
            self.numAtoms += 1
        
        elif isinstance(input, Iterable):
            for i in range(len(input)):
                a = input[i]
                assert isinstance(a, meta.Atom)
                a.idx = i+1
                self.atomList.append(a)
                self.numAtoms += 1

        else:
            raise MetaError("Cannot construct Molecule from the given input")


    def embed(self):
        rd_mol = self.to_rd()
        AllChem.EmbedMolecule(rd_mol, useRandomCoords=True)
        AllChem.UFFOptimizeMolecule(rd_mol)
        self.from_rd(rd_mol)

    @property
    def mass(self):
        """The total mass of the molecule."""
        return sum([a.mass for a in self.atoms_iter()])

    @property
    def center(self):
        """The cartesian center of the molecule."""
        if np.all(np.isfinite(self.xyz)):
            coords = self.xyz
            if len(coords)==1:
                np.expand_dims(coords, axis=0)
            return np.mean(coords, axis=0)

    @property
    def mass_center(self):
        """The mass center of the molecule."""
        if self.mass == 0.0:
            raise MetaError("Cannot calculate mass center for a molecule with zero mass")
        if np.all(np.isfinite(self.xyz)):
            mass_coords = np.asarray([np.asarray(a.xyz)*a.mass for a in self.atoms_iter()]) / self.mass
            if len(mass_coords)==1:
                np.expand_dims(mass_coords, axis=0)            
            return np.mean(mass_coords, axis=0)

    @property
    def atoms(self):
        """Return all atoms in the Molecule.

        Returns
        ------
        List of all atoms in the Molecule
        """

        return self.atomList

    def atoms_iter(self):
        """Iterate through all atoms in the Molecule.

        Yields
        ------
        Atom object
        """
        for atom in self.atoms:
            yield atom

    def assign_residues(self, starting_resid=1):
        self.residues = set()
        current_resname = ''
        for atom in self.atoms_iter():
            if atom.resname == '': atom.resname = 'RES'
            # if atom.resid != -1: 
            #     self.residues.add(atom.resname+str(atom.resid))
            #     continue
            if atom.resname != current_resname:
                current_resname = atom.resname
                numRes = len(self.residues) + starting_resid
                self.residues.add(atom.resname+str(numRes))
            atom.resid = numRes
        self._numResidues = len(self.residues)

    @property
    def numResidues(self):
        """Return number of residues in the Molecule."""
        if hasattr(self, '_numResidues'):
            return self._numResidues
        self.assign_residues()
        return self._numResidues

    @numResidues.setter
    def numResidues(self, num):
        """Set number of residues in the Molecule."""
        if not isinstance(num, int):
            raise TypeError("Residue number must be an integer")
        self._numResidues = num

    @property
    def bonds(self):
        """Return all bonds in the Molecule.

        Returns
        ------
        List of all bonds in the Molecule

        See Also
        --------
        bond_graph.edges : Return all edges in the bond graph
        """
        if self.bond_graph:
            return self.bond_graph.edges
        else:
            return []

    def bonds_iter(self):
        """Iterate through all bonds in the Molecule.

        Yields
        ------
        tuple of Atom objects
            The next bond in the Molecule

        See Also
        --------
        bond_graph.edges_iter : Iterates over all edges in a BondGraph
        """
        if self.bond_graph:
            return self.bond_graph.edges_iter()
        else:
            return iter(())

    @property
    def neighbors(self):
        """Dictionary that contains neighbor info for all atoms in the molecule."""
        return self.bond_graph._adj

    def add_atom(self, atom, connect_to, **kwargs):
        if not isinstance(atom, meta.Atom):
            raise TypeError("The atom to add must be an Atom instance")        
        
        if isinstance(connect_to, int):
            loc1 = connect_to
        elif isinstance(connect_to, meta.Atom):
            loc1 = connect_to.idx
        else:
            raise TypeError(
                "The atom to connect in the original molecule must be" 
                "referenced by either the atom index or Atom instance")

        self.connect(loc1=loc1, 
                    other=Molecule(atom), 
                    loc2=1, keep_coords=True, **kwargs)
            
    def remove_atom(self, atom):
        """Remove an atom from the molecule. 
           Input can be either metamol.atom.Atom or int.
           All hydrogen atoms connected to this atom will also be removed.
           Should only call this function to remove end groups."""
        if isinstance(atom, int):
            atom = self.atoms[atom-1]
            remove_list = [atom.idx-1]
        elif isinstance(atom, meta.Atom):
            remove_list = [atom.idx-1]
        else:
            raise TypeError(
                "The atom to remove must be represented"
                "by either the atom index or Atom instance")
        
        #Remove bonded H atoms
        for neigh in self.bond_graph._adj[atom]:
            if neigh.atomic == 1:
                remove_list.append(neigh.idx-1)
        for rm_idx in sorted(remove_list, reverse=True):
            a_temp = self.atomList.pop(rm_idx)
            self.bond_graph.remove_node(a_temp)
            self.numAtoms -= 1

        #Update atom index
        for idx, atom in enumerate(self.atoms):
            atom.idx = idx + 1
        self.numBonds = sum(1 for _ in self.bonds)
        if self.smi:
            self.update_smi()

    def connect(self, loc1, other, loc2, rm_atoms1=None, rm_atoms2=None, keep_coords=False, **kwargs):
        """Attach another Molecule to the current Molecule."""
        if isinstance(other, Molecule):
            mol_to_connect = other
        elif isinstance(other, str):
            mol_to_connect = Molecule(other, smiles=True)
        else:
            raise TypeError(
                "The molecule to connect must be either "
                "a Molecule object or a SMILES string")

        if loc1 <= 0 or loc1 > self.numAtoms:
            raise MetaError("The index of anchor atom in the original molecule is out of range")
        if loc2 <= 0 or loc2 > mol_to_connect.numAtoms:
            raise MetaError("The index of anchor atom in the molecule to add is out of range")

        clone1 = deepcopy(self)
        if rm_atoms1:
            a_connect = clone1[loc1-1]
            if isinstance(rm_atoms1, int):
                rm_atoms1 = [rm_atoms1]
            for rm_atom in sorted(rm_atoms1, reverse=True):
                clone1.remove_atom(rm_atom)
            loc1 = a_connect.idx

        clone2 = deepcopy(other)
        if rm_atoms2:
            a_connect = clone2[loc2-1]
            if isinstance(rm_atoms2, int):
                rm_atoms2 = [rm_atoms2]
            for rm_atom in sorted(rm_atoms2, reverse=True):
                clone2.remove_atom(rm_atom)
            loc2 = a_connect.idx

        if not keep_coords:
            #If not reuired to keep original coordinates, we can use rdkit to construct 
            #and optimize the final structure.
            bond_type = kwargs.get('bond_type', 'S')
            if bond_type.upper() == 'S':
                bond_order = Chem.BondType.SINGLE
            elif bond_type.upper() == 'D':
                bond_order = Chem.BondType.DOUBLE
            elif bond_type.upper() == 'T':
                bond_order = Chem.BontType.TRIPLE
            else:
                raise ValueError("Unidentified bond type {}".format(bond_type))

            rdmol_ori = clone1.to_rd()
            rdmol_to_connect = clone2.to_rd()

            combo = Chem.CombineMols(rdmol_ori, rdmol_to_connect)
            edcombo = Chem.EditableMol(combo)
            edcombo.AddBond(loc1-1, loc2-1+rdmol_ori.GetNumAtoms(), order=bond_order)
            
            # Adjust Hs
            new_rdmol = Chem.AddHs(Chem.RemoveHs(edcombo.GetMol()))

            Chem.SanitizeMol(new_rdmol)
            AllChem.EmbedMolecule(new_rdmol)
            AllChem.MMFFOptimizeMolecule(new_rdmol)

            rw.mol_convert_from_rd(new_rdmol, host_obj=clone1)
        else:
            clone1.atomList += clone2.atomList
            clone1.numAtoms += clone2.numAtoms
            #clone1.numBonds += clone2.numBonds
            atom1, atom2 = clone1[loc1-1], clone2[loc2-1]

            bl = kwargs.get('bondlength', None)
            ori = kwargs.get('orientation', None)
            if not bl:
                from metamol.utils.help_functions import approximate_bl
                bl = approximate_bl(atom1=atom1, atom2=atom2)
            if not ori:
                from metamol.utils.help_functions import find_best_orientation
                per = kwargs.get('per', (0, 0, 0))
                box_lengths = kwargs.get('box', [0.0, 0.0, 0.0])
                ori, dist_from_anchor = \
                    find_best_orientation(atom=atom2, bond_length=bl, 
                                        anchor=atom1, 
                                        neighs=clone1.neighbors[atom1],
                                        per=per, box=box_lengths)
            else:
                dist_from_anchor = bl * np.asarray(ori) / np.linalg.norm(ori)
            translation_vector = atom1.xyz + dist_from_anchor - atom2.xyz
            Translate(clone2, translation_vector)
            # #translate = new_coords - np.asarray(atom2.xyz)
            # for atom in clone2.atoms_iter():
            #     atom.xyz += translate
            
            clone1.add_bond((atom1, atom2))
            
            for bond in clone2.bonds_iter():
                clone1.add_bond(bond)

            #Update atom index
            update_aidx = kwargs.get('update_aidx', True)
            if update_aidx:
                for idx, atom in enumerate(clone1.atoms):
                    atom.idx = idx + 1

            if clone1.smi:
                clone1.update_smi()

        self.copy(clone1)

    def update_smi(self):
        """Update the smiles string when Molecule is modified."""
        try:
            smi = Chem.MolToSmiles(Chem.RemoveHs(self.to_rd()))
        except:
            smi = None
        self.smi = smi

    def add_bond(self, atom_pair):
        """Add a bond between two atoms.

        Parameters
        ----------
        atom_pair : indexable object, length=2, dtype=int/Atom
            The pair of atoms to connect
        """  
        if self.bond_graph is None:
            self.bond_graph = BondGraph()

        if not isinstance(atom_pair, Iterable) or len(atom_pair)!=2:
            raise MetaError("Atom pair must be an indexalbe oject with length of 2")

        if isinstance(atom_pair[0], int) and isinstance(atom_pair[1], int):
            atom1, atom2 = self[atom_pair[0]], self[atom_pair[1]]

        elif isinstance(atom_pair[0], meta.Atom) and isinstance(atom_pair[1], meta.Atom):
            atom1, atom2 = atom_pair[0], atom_pair[1]
            
        else:
            raise MetaError("Atom pair must be either two ints or tow Atoms")

        self.bond_graph.add_edge(atom1, atom2)
        self.numBonds += 1

    def remove_bond(self, atom_pair):
        """Remove the bond between two atoms.

        Parameters
        ----------
        atom_pair : indexable object, length=2, dtype=int/Atom
            The pair of atoms which form the bond to be deleted
        """  
        if self.bond_graph is None:
            return

        if not isinstance(atom_pair, Iterable) or len(atom_pair)!=2:
            raise MetaError("Atom pair must be an indexalbe oject with length of 2")

        if isinstance(atom_pair[0], int) and isinstance(atom_pair[1], int):
            atom1, atom2 = self[atom_pair[0]], self[atom_pair[1]]

        elif isinstance(atom_pair[0], meta.Atom) and isinstance(atom_pair[1], meta.Atom):
            atom1, atom2 = atom_pair[0], atom_pair[1]
            
        else:
            raise MetaError("Atom pair must be either two ints or tow Atoms")

        self.bond_graph.remove_edge(atom1, atom2)
        self.numBonds -= 1

    def get_element_idx(self, ele):
        return [a.idx for a in self.atoms if a.symbol==ele.upper()]

    def view(self, backend='py3Dmol', params={}, inter=False):
        """Visualize the molecule. Default backend: py3Dmol."""
        return visual(self, backend=backend, params=params, inter=inter)

    def optimize(self, perturb_range=(-0.25, 0.25)):
        new_mol = optimize_config(obj=self, perturb_range=perturb_range)
        self.xyz = new_mol.xyz
        del new_mol

    def from_pmd(self, struct):
        """Convert a parmed struct to Molecule object. """
        rw.convert_from_pmd(struct, self, asSystem=False)

    def from_rd(self, rdmol):
        """COnvert a rdkit Mol to Molecule object."""
        rw.mol_convert_from_rd(rdmol, self, smi=self.smi)

    def to_pmd(self, box=None, title="", residues=None):
        """Convert Molecule to pmd struct."""
        return rw.convert_to_pmd(
            self, 
            box=box, 
            title=title, 
            residues=residues
            )

    def to_rd(self):
        """Convert Molecule to rdkit Mol."""
        return rw.convert_to_rd(self)
        
    @property
    def xyz(self):
        """Return all atom coordinates in the molecule."""
        arr = np.fromiter(
                itertools.chain.from_iterable((a.x, a.y, a.z) for a in self.atoms),
                dtype=float,
        )
        return arr.reshape((-1, 3))

    @xyz.setter
    def xyz(self, coords):
        if not isinstance(coords, Iterable):
            raise TypeError("The new coordinates must be an Iterable")
        rows, cols = np.asarray(coords).shape
        if cols != 3:
            raise MetaError("The new coordinates must have 3 columns")
        if rows != self.numAtoms:
            raise MetaError("The size of the new coordinates must equal "
                            "the number of Atoms in the Molecule")        

        for idx, row in enumerate(coords):
            self[idx].xyz = row

    def rotate(self, about, angle, rad=True):
        """Rotate the Molecule."""
        from metamol.utils.geometry import Rotate
        newxyz = Rotate(about=about, theta=angle, xyz=self.xyz, rad=rad)
        self.xyz = newxyz

    def rotate_around_center(self, about, angle, rad=True):
        #Rotate object around its center
        from metamol.utils.geometry import Rotate
        oldxyz, center_pt = self.xyz, self.center
        #Trnaslate from center to to the origin.
        oldxyz -= center_pt
        #Rotate around the given axis
        newxyz = Rotate(about=about, theta=angle, xyz=oldxyz, rad=rad)
        #Translate back to center position
        newxyz += center_pt
        self.xyz = newxyz

    def align_to(self, ori):
        """Align Molecule to an orientation."""
        from metamol.utils.geometry import Rotate
        if not hasattr(self, 'align'):
            return
        align_dict = {'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}
        if self.align in ['x', 'y', 'z']:
            self.align = align_dict[self.align]
        if isinstance(ori, str):
            ori = align_dict[ori]
        elif not isinstance(ori, Iterable):
            raise TypeError(
                "The rotation axis must be either a string or an Iterable object."
                )
        if self.align == ori:
            return
        angle = uv_degree(self.align, ori)
        about = np.cross(self.align, ori)
        if sum([a*a for a in about]) < 0.0000001:
            self.align = ori
            return
        newxyz = Rotate(about, angle, self.xyz, rad=True)
        self.xyz = newxyz
        self.align = ori

    def __getitem__(self, idx):
        """Get atom from Molecule."""
        if not isinstance(idx, int):
            raise TypeError("Atom index must be a integer")
        
        if idx < 0 or idx >= self.numAtoms:
            raise MetaError("Atom index out of range")

        return self.atoms[idx]

    def __repr__(self):
        """Representation of the Molecule."""

        desc = ["Molecule id: {0}".format(id(self))]
        if self.name:
            desc.append("Name: {0}".format(self.name))
        if self.smi:
            desc.append("SMILES: {0}".format(self.smi))
        
        desc.append("Number of Atoms: {0}".format(self.numAtoms))
        if hasattr(self, '_numResidues'):
            desc.append("Number of Residues: {0}".format(self.numResidues))
        desc.append("Number of Bonds: {0}".format(self.numBonds))

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
            if hasattr(target, '_numResidues'):
                self.residues = target.residues
                self.numResidues = target.numResidues

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
        