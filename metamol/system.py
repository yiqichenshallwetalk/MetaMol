from collections import OrderedDict
from typing import Iterable
import os
import numpy as np
import itertools
from copy import deepcopy

import metamol as meta
from metamol.exceptions import MetaError
from metamol import pack
from metamol import rw
from metamol.utils.visual import visual
from metamol.utils.help_functions import optimize_config

__all__ = ["System", "Box", "Frame"]

class System(object):

    def __init__(self, input=None, dup=None, box=None, box_angle=None, smiles=False, name="", file_type=None, **kwargs):
        super(System, self).__init__()
        
        self.molList = []
        self.numMols = 0
        self.numAtoms = 0
        self.numBonds = 0
        per = kwargs.get('per', (False, False, False))
        self._box = None
        self.dup = []
        self.name = name
        self.numWater = 0
        self.parametrized = False
        self.flattened = True
        self.numFrames = 0
        self.frames = []

        # FF Attributes
        self.angles = []
        self.dihedrals = []
        self.rb_torsions = []
        self.impropers = []
        self.numAngles = 0
        self.numDihedrals = 0
        self.numRBs = 0
        self.numImpropers = 0
        self.use_ub = False
        self.ff_name = ""
        self.params = {
            'atom_type': OrderedDict(),
            'bond_type': OrderedDict(),
            'angle_type': OrderedDict(),
            'dihedral_type': OrderedDict(),
            'rb_torsion_type': OrderedDict(),
            'improper_type': OrderedDict(),
            'NBFIX': OrderedDict()}

        if input is None:
            return

        if isinstance(input, str):
            self.readfile(input, smiles=smiles, file_type=file_type, **kwargs)

        elif isinstance(input, meta.Molecule):
            self.add(input, dup)
        
        elif isinstance(input, Iterable):
            if isinstance(input[0], str):
                # Read Gromacs files (.gro, .top) into meta.System
                for file in input:
                    if file.endswith('top'): topfile = file
                    elif file.endswith('gro'): grofile = file
                    else: raise MetaError('File {0} is not a gromacs file format'.format(file))
                self.readfile(topfile, smiles=False, xyz=grofile, **kwargs)
            elif isinstance(input[0], meta.Molecule):
                self.add(input, dup)
            else:
                raise MetaError("Cannot construct System from the given input")

        else:
            raise MetaError("Cannot construct System from the given input")
        
        if box is not None:
            if not isinstance(box, Iterable) or (len(box) != 3 and len(box) != 6):
                raise MetaError("Simulation Box bounds not valid")
            if len(box) == 3:
                box = [0.0, 0.0, 0.0] + list(box)
            self._box = Box(bounds=box, angle=box_angle, per=per)
            for frame in self.frames:
                frame.box = self._box

    def readfile(self, filename, smiles=False, file_type=None, timestep=None, **kwargs):
        rw.readfile(filename, host_obj=self, smiles=smiles, file_type=file_type, **kwargs)
        if not filename.endswith('top') and not filename.endswith('dump'):
            timestep = 0 if timestep is None else timestep
            self.numFrames = 1
            self.frames.append(Frame(coords=self.xyz, timestep=timestep, box=self.box))

    def readframe(self, filename, timestep=None, box=None, **kwargs):
        temp_sys = rw.readfile(filename, **kwargs)
        self.numFrames += 1
        timestep = 0 if timestep is None else timestep
        box = self.box if box is None else box
        self.frames.append(Frame(coords=temp_sys.xyz, timestep=timestep, box=box))

    @property
    def box(self):
        if self._box is not None:
            return self._box
        return None

    @box.setter
    def box(self, new_box, angle=None):
        if isinstance(new_box, Box):
            self._box = new_box
        elif isinstance(new_box, Iterable):
            if (len(new_box)!=3 and len(new_box)!=6):
                raise MetaError("Simulation box must be a list or array of length 3 or 6")
            if len(new_box) == 3:
                self._box = Box(bounds=np.array([0.0, 0.0, 0.0]+list(new_box)))
            elif len(new_box) == 6:
                self._box = Box(bounds=np.asarray(new_box))
        if angle is not None:
            self._box.angle = angle
        
        for frame in self.frames:
            frame.box = self._box

    @property
    def mass(self):
        return sum([mol.mass for mol in self.molecules_iter()])

    @property
    def net_charge(self):
        return sum([a.charge for a in self.atoms_iter()])

    @property
    def center(self):
        """The cartesian center of the system."""
        if np.all(np.isfinite(self.xyz)):
            coords = self.xyz
            if len(coords)==1:
                np.expand_dims(coords, axis=0)
            return np.mean(coords, axis=0)

    def rotate(self, about, angle, rad=True):
        """Rotate the System."""
        from metamol.utils.geometry import Rotate
        newxyz = Rotate(about=about, theta=angle, xyz=self.xyz, rad=rad)
        self.xyz = newxyz

    def rotate_around_center(self, about, angle, rad=True):
        from metamol.utils.geometry import Rotate
        oldxyz, center_pt = self.xyz, self.center
        #Trnaslate from center to to the origin.
        oldxyz -= center_pt
        #Rotate around the given axis
        newxyz = Rotate(about=about, theta=angle, xyz=oldxyz, rad=rad)
        #Translate back to center position
        newxyz += center_pt
        self.xyz = newxyz
   
    def add(self, mols=None, dup=None, update_index=True, assign_residues=True):
        """Add Molecules to the System."""
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
        if assign_residues:
            self.assign_residues()
        
        if update_index:
            self.update_atom_idx()

    def remove(self, refs):
        """Remove Molecules from the System either by name or by index."""
        if isinstance(refs, int):
            self.remove([refs])

        elif isinstance(refs, str):
            remove_idx = []
            for idx, mol_to_remove in enumerate(self.molecules):
                if mol_to_remove.name==refs or mol_to_remove.smi==refs:
                    remove_idx.append(idx+1)
  
            self.remove(remove_idx)

        elif isinstance(refs, Iterable):
            atoms_to_remove = set()
            for idx in sorted(refs, reverse=True):
                if idx < 1 or idx > self.numMols:
                    raise ValueError("Molecule index out of range")

                mol_to_rm = self.molecules[idx-1]
                if isinstance(mol_to_rm, (meta.Water3Site, meta.Water4Site)):
                    self.numWater -= self.dup[idx-1]

                self.numMols -= self.dup[idx-1]
                self.numBonds -= mol_to_rm.numBonds*self.dup[idx-1]
                self.numAtoms -= mol_to_rm.numAtoms*self.dup[idx-1]
                atoms_to_remove.update(set([a.idx for a in mol_to_rm.atoms]))
                self.molList.pop(idx-1)
                self.dup.pop(idx-1)

            if self.parametrized:
                # remove angles
                for idx in range(len(self.angles)-1, -1, -1):
                    angle = self.angles[idx]
                    if set([angle.atom1.idx, angle.atom2.idx, angle.atom3.idx]).intersection(atoms_to_remove):
                        self.angles.pop(idx)
                        self.numAngles -= 1
                for idx in range(len(self.dihedrals)-1, -1, -1):
                    dih = self.dihedrals[idx]
                    if set([dih.atom1.idx, dih.atom2.idx, dih.atom3.idx, dih.atom4.idx]).intersection(atoms_to_remove):
                        self.dihedrals.pop(idx)
                        self.numDihedrals -= 1
                for idx in range(len(self.rb_torsions)-1, -1, -1):
                    RB = self.rb_torsions[idx]
                    if set([RB.atom1.idx, RB.atom2.idx, RB.atom3.idx, RB.atom4.idx]).intersection(atoms_to_remove):
                        self.rb_torsions.pop(idx)
                        self.numRBs -= 1
                for idx in range(len(self.impropers)-1, -1, -1):
                    imp = self.impropers[idx]
                    if set([imp.atom1.idx, imp.atom2.idx, imp.atom3.idx, imp.atom4.idx]).intersection(atoms_to_remove):
                        self.impropers.pop(idx)
                        self.numImpropers -= 1
            
            self.update_atom_idx()

        else:
            raise MetaError("The molecule to remove must be index/name/smiles")

    def __add__(self, other):
        clone = deepcopy(self)
        clone.merge(deepcopy(other))
        return clone

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def merge(self, sysList=[], newBox=None):
        if not sysList: return

        if isinstance(sysList, meta.System):
            sysList = [sysList]

        for sys in sysList:
            self.molList += sys.molList
            self.numMols += sys.numMols
            self.numWater += sys.numWater
            self.dup += sys.dup
            self.numAtoms += sys.numAtoms
            self.numBonds += sys.numBonds

        if newBox:
            self.box = newBox

        self.update_atom_idx()

        if any([sys.parametrized==False for sys in [self]+sysList]):
            print("Not all systems are parametrized. Did not merge FF parameters.")
        else: 
            self._update_ff_params(sysList=sysList)

    def _update_ff_params(self, sysList=[]):
        for sys in sysList:
            # for atom in sys.atoms_iter():
            #     atom.idx += curr
            for at_key, atom_type in sys.params['atom_type'].items():
                if at_key not in self.params['atom_type']:
                    atom_type.idx = len(self.params['atom_type']) + 1
                    self.params['atom_type'][at_key] = atom_type

            for bond_key, bond_type in sys.params['bond_type'].items():
                if bond_key not in self.params['bond_type']:
                    bond_type.idx = len(self.params['bond_type']) + 1
                    self.params['bond_type'][bond_key] = bond_type
            
            for angle_key, angle_type in sys.params['angle_type'].items():
                if angle_key not in self.params['angle_type']:
                    angle_type.idx = len(self.params['angle_type']) + 1
                    self.params['angle_type'][angle_key] = angle_type
            
            for angle in sys.angles:
                angle.typeID = self.params['angle_type'][(angle.atom1.type, angle.atom2.type, angle.atom3.type)].idx
                self.angles.append(angle)
            self.numAngles += sys.numAngles

            for dihedral_key, dihedral_type in sys.params['dihedral_type'].items():
                if dihedral_key not in self.params['dihedral_type']:
                    dihedral_type.idx = len(self.params['dihedral_type']) + 1
                    self.params['dihedral_type'][dihedral_key] = dihedral_type
            
            for dih in sys.dihedrals:
                dih.typeID = self.params['dihedral_type'][(dih.atom1.type, dih.atom2.type, dih.atom3.type, dih.atom4.type)].idx
                self.dihedrals.append(dih)
            self.numDihedrals += sys.numDihedrals

            for rb_key, rb_type in sys.params['rb_torsion_type'].items():
                if rb_key not in self.params['rb_torsion_type']:
                    rb_type.idx = len(self.params['rb_torsion_type']) + 1
                    self.params['rb_torsion_type'][rb_key] = rb_type

            for RB in sys.rb_torsions:
                RB.typeID = self.params['rb_torsion_type'][(RB.atom1.type, RB.atom2.type, RB.atom3.type, RB.atom4.type)].idx
                self.rb_torsions.append(RB)
            self.numRBs += sys.numRBs

            for im_key, im_type in sys.params['improper_type'].items():
                if im_key not in self.params['improper_type']:
                    im_type.idx = len(self.params['improper_type']) + 1
                    self.params['improper_type'][im_key] = im_type

            for improper in sys.impropers:
                improper.typeID = self.params['improper_type'][(improper.atom1.type, improper.atom2.type, improper.atom3.type, improper.atom4.type)].idx
                self.impropers.append(improper)
            self.numImpropers += sys.numImpropers

            for nb_key, nb_params in sys.params['NBFIX'].items():
                if nb_key not in self.params['NBFIX']:
                    self.params['NBFIX'][nb_key] = nb_params

        for atom in self.atoms_iter():
            atom.atidx = self.params['atom_type'][atom.type].idx
            
    def append_surface(self, surface, location='bottom', cushion=0.0, edge=1.0):
        """Append surface to the system."""
        if not isinstance(surface, meta.System):
            raise MetaError("Surface must be a System instance")
        if self.box is None:
            raise MetaError("System Box must be valid")
        ori_box_bottom = self.box.bounds[2]
        ori_box_top = self.box.bounds[5]

        surf_temp = deepcopy(surface)

        if location == 'bottom':
            transition_in_z = ori_box_bottom - cushion - max(surf_temp.xyz[:, 2])
        elif location == 'top':
            #Flip the surface upside down
            surf_temp.rotate_around_center(
                about='x', angle=180.0, rad=False,
            )
            transition_in_z = ori_box_top + cushion - min(surf_temp.xyz[:, 2])
        # TODO: add more locations
        else:
            raise MetaError("Not implemented yet")

        for atom in surf_temp.atoms_iter():
            atom.z = atom.z + transition_in_z

        self.merge(surf_temp)
        self.box = surf_temp.box
        new_top = max(ori_box_top, max(self.xyz[:, 2])+edge)
        new_bottom = min(ori_box_bottom, min(self.xyz[:, 2])-edge)
        if self.box:
            self.box.bounds[2] = new_bottom
            self.box.bounds[5] = new_top

    def assign_residues(self):
        # Assign residues (for self constructed structures)
        self.numResidues = 0
        self.residues = set()
        for mol in self.molecules:
            starting_resid = self.numResidues + 1
            mol.assign_residues(starting_resid)
            self.residues.update(mol.residues)
            self.numResidues += mol.numResidues

    def count_residues(self):
        # Count residues (for read-in Systems with built-in residue information)
        self.numResidues = 0
        self.residues = set()
        for mol in self.molecules:
            mol.residues = set()
            for atom in mol.atoms_iter():
                mol.residues.add(atom.resname+str(atom.resid))
            mol.numResidues = len(mol.residues)
            self.numResidues += mol.numResidues
            self.residues.update(mol.residues)

    @property
    def atoms(self):
        """Return all atoms in the System."""
        all_atoms = []
        for mol in self.molecules_iter():
            all_atoms += mol.atoms
        return all_atoms
    
    def atoms_iter(self):
        """Iterate through all atoms in the System."""
        for mol in self.molecules_iter():
            for atom in mol.atoms:
                yield atom

    @property
    def molecules(self):
        """Return all molecules in the System."""
        return self.molList
    
    def molecules_iter(self):
        """Iterate through all molecules in the System."""
        for mol in self.molList:
            yield mol

    @property
    def bonds(self):
        """Return all bonds (Atom pair tuple) in the System."""
        all_bonds = []
        for mol in self.molecules_iter():
            for bond in mol.bonds_iter():
                all_bonds.append(bond)
        return all_bonds

    def bonds_iter(self):
        """Iterate though all bonds in the System."""
        for mol in self.molecules_iter():
            for bond in mol.bonds_iter():
                yield bond

    def update_atom_idx(self):
        idx = 1
        for atom in self.atoms_iter():
            atom.update_idx(idx)
            idx += 1

    def update_coords(self, file):
        if not os.path.exists(file):
            raise FileNotFoundError("File {} does not exist".format(file))
        
        new_coords = np.genfromtxt(file, skip_header=2, usecols=(1, 2, 3,))

        if len(new_coords)!=self.numAtoms:
            raise MetaError("The size of the updated system must equal the original one")
        
        idx = 0
        for mol in self.molecules_iter():
            for atom in mol.atoms_iter():
                atom.xyz = new_coords[idx]
                idx += 1
                if idx==self.numAtoms:
                    break

    def get_boundingbox(self):
        pos = self.xyz
        return [max(pos[:, 0])-min(pos[:, 0]), max(pos[:, 1])-min(pos[:, 1]), max(pos[:, 2])-min(pos[:, 2])]
    
    def valid_box(self):
        if self.box is None: return False
        if not isinstance(self.box, Box):
            return False
        if self.box.bounds is None: return False
        
        xyz = self.xyz
        min_x, max_x = min(xyz[:, 0]), max(xyz[:, 0])
        min_y, max_y = min(xyz[:, 1]), max(xyz[:, 1])
        min_z, max_z = min(xyz[:, 2]), max(xyz[:, 2])

        if not self.box.per[0] and self.box.lengths[0] < max_x - min_x:
            return False
        if not self.box.per[1] and self.box.lengths[1] < max_y - min_y:
            return False
        if not self.box.per[2] and self.box.lengths[2] < max_z - min_z:
            return False
        return True

    def create_box(self, edge=1.0):
        xyz = self.xyz
        min_x, max_x = min(xyz[:, 0]), max(xyz[:, 0])
        min_y, max_y = min(xyz[:, 1]), max(xyz[:, 1])
        min_z, max_z = min(xyz[:, 2]), max(xyz[:, 2])
        self.box = Box([min_x-edge, min_y-edge, min_z-edge, 
                        max_x+edge, max_y+edge, max_z+edge])

    def flatten(self):
        if self.flattened:
            return
        newMolList = []
        newNumMols = 0
        newNumAtoms = 0
        newNumBonds = 0
        for idx, mol in enumerate(self.molecules_iter()):
            for _ in range(self.dup[idx]):
                mol_to_add = deepcopy(mol)
                newMolList.append(mol_to_add)

            newNumMols += self.dup[idx]
            newNumAtoms += mol.numAtoms * self.dup[idx]
            newNumBonds += mol.numBonds * self.dup[idx]
        
        self.molList = newMolList
        self.numMols = newNumMols
        self.numAtoms = newNumAtoms
        self.numBonds = newNumBonds
        self.dup = self.numMols * [1]
        self.update_atom_idx()
        self.assign_residues()

        for mol in self.molecules_iter():
            self.angles += mol.angles
            self.dihedrals += mol.dihedrals
            self.rb_torsions += mol.rb_torsions
            self.impropers += mol.impropers

        self.numAngles = len(self.angles)
        self.numDihedrals = len(self.dihedrals)
        self.numRBs = len(self.rb_torsions)
        self.numImpropers = len(self.impropers)
        
    def initial_config(
        self, 
        region=None,
        density=None,
        aspect_ratio=None,
        overlap=2.0,
        seed=12345,
        sidemax=1000.0,
        edge=2.0,
        fix_orientation=False,
        temp_file=None):

        if (self.box is None) and (region is None):
            raise MetaError("Need to specify simulation box when use packmol to create initial configuration.") 
        
        pack.initconfig_box(
            system=self, 
            region=region, 
            density=density, 
            aspect_ratio=aspect_ratio, 
            overlap=overlap, 
            seed=seed, 
            sidemax=sidemax, 
            edge=edge, 
            fix_orientation=fix_orientation, 
            temp_file=temp_file
            )

    def optimize(self, perturb_range=(-0.25, 0.25)):
        new_sys = optimize_config(obj=self, perturb_range=perturb_range)
        self.xyz = new_sys.xyz
        del new_sys

    def from_pmd(self, struct):
        from metamol.utils.convert_formats import convert_from_pmd
        convert_from_pmd(struct, self, asSystem=True)

    def from_rd(self, rdmol):
        from metamol.utils.convert_formats import convert_from_rd
        convert_from_rd(rdmol, self, asSystem=True)

    def to_pmd(self, box=None, title="", residues=None, parametrize=False):
        from metamol.utils.convert_formats import convert_to_pmd
        return convert_to_pmd(
            self,
            box=box,
            title=title,
            residues=residues,
            parametrize=parametrize,
            )

    def to_openmm(self, createSystem=False, forcefield_files=None, **kwargs):
        from metamol.utils.convert_formats import convert_to_openmm
        return convert_to_openmm(
            self,
            createSystem=createSystem,
            forcefield_files=forcefield_files,
            **kwargs
        )

    def to_rd(self):
        from metamol.utils.convert_formats import convert_to_rd
        return convert_to_rd(self)

    def save(self, filename, **kwargs):
        
        # ext = os.path.splitext(filename)[-1]
        # if '.' not in ext:
        #     filename += '.' + fmt
        
        rw.savefile(self, filename, **kwargs)

    @property
    def xyz(self):
        """Return all atom coordinates in the system."""
        arr = np.fromiter(
                itertools.chain.from_iterable(mol.xyz.flatten() for mol in self.molecules_iter()),
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
                            "the number of Atoms in the System")        

        atoms = self.atoms
        for idx, row in enumerate(coords):
            atoms[idx].xyz = row

    def get_atom(self, index):
        """Get an Atom in the System."""
        if not isinstance(index, int) or index <= 0 or index > self.numAtoms:
            raise MetaError("Atom index not valid or out of range")
        return self.atoms[index-1]

    def get_molecule(self, selection):
        """Get molecule(s) in the System."""
        if isinstance(selection, int):
            if selection <= 0 or selection > self.numMols:
                raise MetaError("Molecule index out of range")
            return self.molecules[selection-1]

        if isinstance(selection, str):
            sl = []
            for mol in self.molecules_iter():
                if selection==mol.name or selection==mol.smi:
                    sl.append(mol)
            if not sl:
                raise MetaError("No molecule name or smiles match label {}".format(selection))
            return sl
        
        raise TypeError("Selection label must be int or string")

    def get_element(self, element): 
        if not isinstance(element, str):
            raise TypeError("element not valid")
        return [a for a in self.atoms_iter() if a.symbol.lower() == element.lower()]

    def get_frame(self, frameId=1, timestep=None):
        """Get atom coordinates in a frame"""
        if frameId is None and timestep is None:
            raise MetaError("At least one of frameId or timestep should be provided")
        if timestep is not None:
            if not isinstance(timestep, int) or timestep < 0:
                raise MetaError("Timestep not valid")
            for frame in self.frames:
                if frame.timestep == timestep: return frame
            raise MetaError("Frame not found!")

        else:
            if frameId == 1 and self.numFrames == 0:
                self.numFrames += 1
                self.frames.append(Frame(coords=self.xyz, box=self.box))
                return self.frames[0]

            if not isinstance(frameId, int) or frameId <=0 or frameId > self.numFrames:
                raise MetaError("Frame index not valid or out of range")
            return self.frames[frameId-1]

    def RDF(self, frameId=1, timestep=None, element_pair=None, per=None, nk=40, rmax=None):
        """Calculate the radial distribution function (RDF) of the System.
        Arguments
        ----------
        frameId : int, default=1
            The frame Id to calculate RDF for.
        timestep: int, default=None
            The frame at certain timestep to calculate RDF for. If specified, will ignore frame.
        element_pair : Iterable, length 2
            The desirable pair of elements to calculate RDF for.
        per : Iterable, length 3, defualt=None
            The periodicity of box in 3 dimensions. If None, then will use frame.per.
        nk : int, default=40
            Number of points in RDF.
        rmax: float, default=None
            The maximum distance for RDF. If not set, then will use half of the minimum box length.
        
        Returns
        ----------
        bins, rdf
        """
        from metamol.utils.help_functions import distance

        if element_pair is not None:
            if not isinstance(element_pair, Iterable) or len(element_pair) != 2:
                raise MetaError("element_pair must be an Iteralbe of length 2")
                
            atomlist = self.get_element(element_pair[0])
            if element_pair[0] != element_pair[1]:
                atomlist_other = self.get_element(element_pair[1])
        else:
            atomlist = self.atoms

        frame = self.get_frame(frameId=frameId, timestep=timestep)
        per = frame.box.per if per is None else per
        if np.any(np.asarray(per)==True) and not np.allclose(frame.box.angle, 90.0):
            raise MetaError("Distance calculation for periodic non-orthorgonal box")

        rmax = frame.box.lengths.min()/2 if rmax is None else rmax
        distances = []
        N = len(atomlist)
        vol = frame.box.volume
        bins = np.linspace(0.0, rmax, nk+1)
        bin = bins[1]
        rdf = np.zeros(nk+1)
        if 'atomlist_other' not in locals():
            rho_all = N*N / vol
            for i in range(N-1):
                for j in range(i+1, N):
                    coords1, coords2 = frame.coords[atomlist[i].idx-1], frame.coords[atomlist[j].idx-1]
                    d = distance(coords1, coords2, per=frame.box.per, box=frame.box.lengths)
                    distances.append(d)
                    if d < rmax:
                        rdf[int(d/bin)+1] += 2

        else:
            M = len(atomlist_other)
            rho_all = N*M / vol
            for i in range(N):
                for j in range(M):
                    coords1, coords2 = frame.coords[atomlist[i].idx-1], frame.coords[atomlist_other[j].idx-1]
                    d = distance(coords1, coords2, per=frame.box.per, box=frame.box.lengths)
                    distances.append(d)
                    if d < rmax:
                        rdf[int(d/bin)+1] += 1

        for idx in range(1, nk+1):
            rdf[idx] /= rho_all * 4/3 * np.pi * (bins[idx]**3 - bins[idx-1]**3)

        return bins, rdf

    def RMSD(self, timestep, ref=None, molid=None, per=None):
        """Calculate the root mean square deviation (RMSD) of a certain frame 
        to the referece frame.

        Arguments
        ----------
        timestep : int
            The timestep to calculate RMSD for.
        ref: int, default=0
            The reference frame. default is the initial frame (timestep=0).
        molid: list of int, default=None
            The index list of mols to calculate RMSD for.
        per : Iterable, length 3, defualt=None
            The periodicity of box in 3 dimensions. If None, then will use frame.per.

    
        Returns
        ----------
        rmsd value
        """       
        from metamol.utils.help_functions import distance

        if isinstance(ref, System):
            ref_frame = ref.get_frame()
        elif isinstance(ref, Frame):
            ref_frame = ref
        else:
            ref_frame = self.get_frame(timestep=ref)

        calc_frame = self.get_frame(timestep=timestep)

        per = calc_frame.box.per if per is None else per
        if np.any(np.asarray(per)==True) and not np.allclose(ref_frame.box.angle, 90.0):
            raise MetaError("Distance calculation for periodic non-orthorgonal box")
        box_lengths = calc_frame.box.lengths

        if molid is None:
            atom_mass = [a.mass for a in self.atoms_iter()]
            coords_ref = ref_frame.coords
            coords_calc = calc_frame.coords
        else:
            if isinstance(molid, int):
                molid = [molid]
            elif not isinstance(molid, Iterable):
                raise MetaError("Molid must be an Iterable object")
            atom_list = []
            for mid in molid:
                mol = self.molecules[mid-1]
                atom_list += mol.atoms
            atom_mass, coords_ref, coords_calc = [], [], []
            for atom in atom_list:
                atom_mass.append(atom.mass)
                coords_ref.append(ref_frame.coords[atom.idx-1])
                coords_calc.append(calc_frame.coords[atom.idx-1])

        total_mass = sum(atom_mass)
        if total_mass == 0:
            raise MetaError("Cannot calculate RMSD for systems with 0 mass")
        
        rmsd = 0.0
        for idx in range(len(atom_mass)):
            rmsd += atom_mass[idx]*distance(coords_ref[idx], coords_calc[idx], per, box_lengths)**2
        return np.sqrt(rmsd / total_mass)

    def RMSF(self, atomid, timesteps=[], per=None):
        """Calculate the root mean square fluctuation (RMSF) of atom positions.

        Arguments
        ----------
        atomid : int
            The index of atom to compute RMSF
        timesteps : list of int
            The timesteps to calculate RMSF for.
        per : Iterable, length 3, defualt=None
            The periodicity of box in 3 dimensions. If None, then will use frame.per.

    
        Returns
        ----------
        rmsf value for atomid
        """       

        from metamol.utils.help_functions import wrap_coords
        atom_pos = []
        if not timesteps:
            for frame in self.frames:
                atom_pos.append(frame.coords[atomid-1])
        else:
            for timestep in timesteps:
                frame = self.get_frame(timestep=timestep)
                atom_pos.append(frame.coords[atomid-1])

        nt = len(atom_pos)
        if nt == 0: return 0.0
        per = self.box.per if per is None else per
        atom_pos = wrap_coords(atom_pos, per, self.box.lengths)
        avg_pos = np.average(atom_pos, axis=0)
        rdsf = 0.0
        for coord in atom_pos:
            rdsf += sum(np.square(coord-avg_pos))

        return np.sqrt(rdsf / nt)
        
    def __getitem__(self, idx):
        """Return a molecule in the system by index."""

        if not isinstance(idx, int):
            raise TypeError("Molecule index must be an int")

        return self.get_molecule(idx+1)

    def __repr__(self):
        """Representation of the System."""

        desc = ["System id: {0}".format(id(self))]

        desc.append("Number of Molecules: {0}".format(self.numMols))    
        if self.numWater:
            desc.append("Number of water molecules: {0}".format(self.numWater))    
        desc.append("Number of Atoms: {0}".format(self.numAtoms))
        if hasattr(self, "numResidues"):
            desc.append("Number of Residues: {0}".format(self.numResidues))
        desc.append("Number of Bonds: {0}".format(self.numBonds))
        if self.numFrames > 0:
            desc.append("Number of frames: {0}".format(self.numFrames))

        if self.box is not None:
            desc.append("Box: {0}".format(self.box.bounds))
            desc.append("Box Angle: {0}".format(self.box.angle))
            periodicity = ["periodic" if p else "fixed" for p in self.box.per]
            desc.append("Periodicity: {0} in x, {1} in y, {2} in z".format(periodicity[0], periodicity[1], periodicity[2]))
        if self.parametrized:
            desc.append("Parametrized: Yes")
            desc.append("Number of Angles: {0}".format(self.numAngles))
            desc.append("Number of Proper Dihedrals: {0}".format(self.numDihedrals))
            desc.append("Number of RB Torsions: {0}".format(self.numRBs))
            desc.append("Number of Improper Dihedrals: {0}".format(self.numImpropers))
        else:
            desc.append("Parametrized: No")

        return "\n".join(desc)
    
    def view(self, backend='py3Dmol', params={}, inter=False, bonds=True, frameId=1, timestep=None):
        """Visualize the system. Default backend: py3Dmol."""
        #frame = self.get_frame(frameId=frameId, timestep=timestep)
        return visual(self, backend=backend, params=params, inter=inter, bonds=bonds, frameId=frameId, timestep=timestep)

    def copy(self, target=None, coords=True):
        if target:
            self.name = target.name
            if not coords:
                ori_coords = self.xyz
            self.molList = target.molList
            if not coords:
                self.xyz = ori_coords
            self.numMols, self.numAtoms, self.numBonds = target.numMols, target.numAtoms, target.numBonds
            self.numWater = target.numWater
            self.dup = target.dup
            if hasattr(target, 'numResidues'):
                self.numResidues = target.numResidues
                self.residues = target.residues
            if target.box is not None:
                self.box = target.box
            # if target.box_angle is not None:
            #     self.box_angle = target.box_angle
            if isinstance(target, meta.Lattice):
                self.lengths = target.lengths
                self.langles = target.langles
                self.vectors = target.vectors
            if target.parametrized:
                self.parametrized = True
                self.params = target.params
                self.angles = target.angles
                self.dihedrals = target.dihedrals
                self.rb_torsions = target.rb_torsions
                self.impropers = target.impropers
                self.numAngles = target.numAngles
                self.numDihedrals = target.numDihedrals
                self.numRBs = target.numRBs
                self.numImpropers = target.numImpropers
                self.use_ub = target.use_ub
                self.ff_name = target.ff_name

    def copy_coords(self, target=None, box=True):
        """Only copy coordinates from target object."""
        if target:
            if self.numAtoms == 0:
                self.molList = target.molList
                self.numAtoms = target.numAtoms
                self.numMols = target.numMols

            self.xyz = target.xyz

            if box:
                self.box = target.box

    # Parametrize the System by `Foyer`
    def parametrize(self, forcefield_files=None, forcefield_name='opls', struct=None, **kwargs):
        from metamol.parametrization import _parametrize
        _parametrize(
            sys=self, 
            forcefield_files=forcefield_files, 
            forcefield_name=forcefield_name, 
            struct=struct, 
            **kwargs)

class Box(object):
    def __init__(self, bounds=None, angle=None, per=(False, False, False)):
        self.bounds = bounds
        if angle is None:
            self.angle = [90.0, 90.0, 90.0]
        else:
            self.angle = angle
        self.per = per

    @property
    def lengths(self):
        if self.bounds is not None:
            return np.asarray([self.bounds[3]-self.bounds[0], self.bounds[4]-self.bounds[1], self.bounds[5]-self.bounds[2]])
        else:
            return np.array([0.0, 0.0, 0.0])

    @property
    def volume(self):
        if self.bounds is None: return 0.0
        if self.angle is None or np.allclose(self.angle, 90.0):
            return np.prod(self.lengths)
        else:
            cos_a, cos_b, cos_c = np.cos(np.radians(self.angle))
            return np.prod(self.lengths) * \
                np.sqrt(1.0 + 2 * cos_a * cos_b * cos_c \
                - cos_a * cos_a - cos_b * cos_b - cos_c * cos_c)

    def __repr__(self):
        desc = ["Box id: {}".format(id(self))]

        desc.append("Box Bounds: {0}A to {1}A in x,  {2}A to {3}A in y, {4}A to {5}A in z"\
            .format(self.bounds[0], self.bounds[3], self.bounds[1], self.bounds[4], self.bounds[2], self.bounds[5]))
        desc.append("Box Length:{0}A * {1}A * {2}A".format(self.lengths[0], self.lengths[1], self.lengths[2]))
        periodicity = ["periodic" if p else "fixed" for p in self.per]
        desc.append("Periodicity: {0} in x, {1} in y, {2} in z".format(periodicity[0], periodicity[1], periodicity[2]))
        
        return "\n".join(desc)

class Frame(object):
    def __init__(self, timestep=0, coords=None, box=None):
        self.timestep = timestep
        self.coords = coords
        self.box = box
    
    def __repr__(self):
        desc = ["Frame id: {}".format(id(self))]

        desc.append("Timestep: {0}".format(self.timestep))
        if self.box is not None:
            desc.append("Box Length:{0}A * {1}A * {2}A".format(self.box.lengths[0], self.box.lengths[1], self.box.lengths[2]))
            periodicity = ["periodic" if p else "fixed" for p in self.box.per]
            desc.append("Periodicity: {0} in x, {1} in y, {2} in z".format(periodicity[0], periodicity[1], periodicity[2]))
        
        return "\n".join(desc)