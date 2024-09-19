from typing import Iterable
from copy import deepcopy

import metamol as meta
from metamol.exceptions import MetaError

class Polymer(meta.Molecule):
    """a generic Polymer class"""
    def __init__(self, monomers, seq='A', name="polymer", head=None, tail=None):
        super(Polymer, self).__init__(name=name)
        if isinstance(monomers, meta.Molecule):
            self.monomers = [monomers]
        else:
            self.monomers = monomers
            
        for monomer in self.monomers:
            if monomer.align != 'x':
                monomer.align_to('x')
        self.seq = seq

        self.head = deepcopy(head)
        if self.head and self.head.align != 'x':
            self.head.align_to('x')
        
        self.tail = deepcopy(tail)
        if self.tail:
            if self.tail.align != 'x':
                self.tail.align_to('x')
            self.tail.rotate(about='y', angle=180.0, rad=False)
            # self.tail.head, self.tail.tail = self.tail.tail, self.tail.head
            # assert isinstance(self.tail.head, int)

    def build(self, N, align='x'):
        if N < 1:
            raise MetaError("Number of repeating units must be at least one")
        
        seq_dict = {self.seq[idx]: self.monomers[idx] for idx in range(len(self.seq))}

        frags = []
        for _ in range(N):
            for s in self.seq:
                frags.append(deepcopy(seq_dict[s]))

        assert len(frags) == len(self.seq) * N
        
        temp_mol = deepcopy(frags[0])
        if self.head:
            temp_mol.connect(loc1=temp_mol.head, 
                             other=self.head, 
                             loc2=self.head.tail,
                             keep_coords=True,
                             orientation=[-1, 0, 0],
                             update_aidx=False)
        else:
            temp_mol.add_atom(meta.Atom(atomic=1), temp_mol.head, orientation=[-1, 0, 0])
        
        tail, addon = temp_mol.tail, temp_mol.numAtoms
        for idx, frag in enumerate(frags[1:]):
            # if (idx+1) % 10 == 0:
            #     print("Building No {0:d}th block".format(idx+1))
            temp_mol.connect(loc1=tail, 
                            other=frag, 
                            loc2=frag.head,
                            keep_coords=True,
                            orientation=[1, 0, 0],
                            update_aidx=False)
            tail = frag.tail + addon
            addon = temp_mol.numAtoms
        
        if self.tail:
            temp_mol.connect(loc1=tail, 
                             other=self.tail, 
                             loc2=self.tail.tail,
                             keep_coords=True,
                             orientation=[1, 0, 0],
                             update_aidx=True)
        else:
            temp_mol.add_atom(meta.Atom(atomic=1), tail, orientation=[1, 0, 0])
        
        temp_mol.name = self.name
        self.copy(temp_mol)

        if align == 'y':
            self.rotate(about='z', angle=90.0, rad=False)
        elif align == 'z':
            self.rotate(about='y', angle=-90.0, rad=False)

        elif isinstance(align, Iterable):
            self.align_to(align)

        else:
            raise TypeError(
                "align argument must be either a sting or an Iterable."
                )       

        self.align = align

        for atom in self.atoms_iter():
            atom.resname = 'POL'