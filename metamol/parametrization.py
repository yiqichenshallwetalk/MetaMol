import os
from collections import OrderedDict

import metamol as meta
from metamol.exceptions import MetaError
from metamol.utils.ffobjects import (
    Angle, Dihedral, RB_Torsion, Improper, AtomType, BondType, 
    AngleType, DihedralType, RBTorsionType, ImproperType
)

__all__ = ["_parametrize", "_parameterize_custom", "_parametrize_water", "_parametrize_openmm_to_pmd"]

def _parametrize(sys, forcefield_files=None, forcefield_name='opls', struct=None, backend='foyer', **kwargs):

    # Custom parametrization
    custom = kwargs.get("custom", False)
    if custom:
        parameters = kwargs.get("parameters", dict())
        if not parameters:
            raise MetaError("Must provide parameters for custom forcefields")
        _parametrize_custom(sys, parameters)
        sys.parametrized = True
        return

    if struct is not None and len(struct.bonds) > 0 and struct.bonds[0].type is not None:
        pass

    elif backend == 'openmm':
        struct = _parametrize_openmm_to_pmd(sys, forcefield_files=forcefield_files, **kwargs)

    else:
        # Parametrize water molecules
        if sys.numWater>0:
            water_system = meta.System()
            water_mols = []
            for idx, mol in enumerate(sys.molecules):
                if isinstance(mol, (meta.Water3Site, meta.Water4Site)):
                    water_system.add(mol, sys.dup[idx])
                    water_mols.append(idx+1)
            sys.remove(water_mols)
            _parametrize_water(water_system)

        if sys.numMols == 0:
            if 'water_system' in locals():
                sys.copy(water_system)
            return

        sys.update_atom_idx()
    
        from foyer import Forcefield
        from pkg_resources import resource_filename

        if struct is None:
            struct = sys.to_pmd()

        if forcefield_files:
            forcefield_name = os.path.basename(forcefield_files).split('.')[0]
        elif forcefield_name:
            ff_file = resource_filename("metamol", os.path.join("ff_files", forcefield_name.lower()+".xml"))

            if forcefield_files is None:
                forcefield_files = ff_file
        
        else:
            raise MetaError("Neither the force field file location nor force field name is specified.")

        if isinstance(forcefield_files, str) and not os.path.exists(forcefield_files):
            raise FileNotFoundError("Forcefield files not found on disk")
        
        ff = Forcefield(forcefield_files=forcefield_files)
        struct = ff.apply(struct)

    if not sys.parametrized:
        sys.flatten()
    else:
        sys.angles, sys.dihedrals, sys.rb_torsions, sys.impropers = [], [], [], []
        sys.numAngles, sys.numDihedrals, sys.numRBs, sys.numImpropers = 0, 0, 0, 0
        sys.use_ub = False
    
    sys.params = {
        'atom_type': OrderedDict(),
        'bond_type': OrderedDict(),
        'angle_type': OrderedDict(),
        'dihedral_type': OrderedDict(),
        'rb_torsion_type': OrderedDict(),
        'improper_type': OrderedDict(),
        'NBFIX': OrderedDict()}

    atom_map = dict()
    for idx, atom in enumerate(sys.atoms):
        pmdatom = struct[idx]
        atom.type = pmdatom.type
        atom.mass = pmdatom.mass
        atom.charge = pmdatom.charge
        #atom.xyz = (pmdatom.xx, pmdatom.xy, pmdatom.xz)
        if atom.type not in sys.params['atom_type']:
            atidx = len(sys.params['atom_type'])+1
            sys.params['atom_type'][atom.type] = AtomType(
                                                    idx=atidx, 
                                                    atomic=atom.atomic, 
                                                    symbol=atom.symbol, 
                                                    name=atom.type, 
                                                    sigma=pmdatom.sigma, 
                                                    epsilon=pmdatom.epsilon,  
                                                    mass=pmdatom.mass, 
                                                    charge=pmdatom.charge,
                                                    )
        else:
            atidx = sys.params['atom_type'][atom.type].idx
        atom.atidx = atidx
        atom_map[pmdatom] = atom

    if forcefield_files is not None and 'gaff' in forcefield_files:
        #For GAFF forcefield, calculate charges by ambertools
        from metamol.antechamber import ante_charges
        charge_dict = dict()
        for mol in sys.molecules_iter():
            if mol.name in charge_dict or mol.smi in charge_dict:
                sub_charges = charge_dict[mol.name] if mol.name else charge_dict[mol.smi]
                for idx, atom in enumerate(mol.atoms):
                    #atom.charge = sub_charge_dict[atom.type]
                    atom.charge = sub_charges[idx]
                continue
            
            charge_methd = kwargs.get('charge_method', 'bcc')
            net_charge = kwargs.get('net_charge', 0.0)
            multiplicity = kwargs.get('multiplicity', 1)
            charge_tol = kwargs.get('charge_tol', 0.005)
            scfconv = kwargs.get('scfconv', 1.0e-10)
            ndiis_attempts = kwargs.get('ndiis_attempts', 0)
            mol, sub_charges = ante_charges(mol, 
                                                charge_methd, 
                                                net_charge, 
                                                multiplicity,
                                                charge_tol,
                                                scfconv,
                                                ndiis_attempts,)
            
            if mol.name:
                charge_dict[mol.name] = sub_charges
            elif mol.smi is not None:
                charge_dict[mol.smi] = sub_charges

    for bond in struct.bonds:
        bond_key = tuple(sorted((bond.atom1.type, bond.atom2.type)))
        btype = bond.type[0] if isinstance(bond.type, list) else bond.type
        if bond_key not in sys.params['bond_type']:
            bondidx = len(sys.params['bond_type']) + 1
            # Deal with constraint bonds
            if btype is None:
                k = 1.0
                from metamol.utils.help_functions import distance
                req = distance(atom_map[bond.atom1].xyz, atom_map[bond.atom2].xyz)
            else:
                k, req = btype.k, btype.req
            sys.params['bond_type'][bond_key] = BondType(idx=bondidx, k=k, req=req)
        
    for angle in struct.angles:
        angle_key = (angle.atom1.type, angle.atom2.type, angle.atom3.type)
        agtype = angle.type[0] if isinstance(angle.type, list) else angle.type
        if angle_key not in sys.params['angle_type']:
            angleidx = len(sys.params['angle_type']) + 1
            sys.params['angle_type'][angle_key] = AngleType(idx=angleidx, k=agtype.k, theteq=agtype.theteq)
        else:
            angleidx = sys.params['angle_type'][angle_key].idx

        sys.angles.append(Angle(atom_map[angle.atom1], atom_map[angle.atom2], atom_map[angle.atom3], angleidx))
        sys.numAngles += 1

    for ub in struct.urey_bradleys:
        ub_key = tuple(sorted((ub.atom1.type, ub.atom2.type)))
        ubtype = ub.type[0] if isinstance(ub.type, list) else ub.type
        sys.use_ub = True
        for angle_key in sys.params['angle_type'].keys():
            if ub_key == (angle_key[0], angle_key[2]):
                sys.params['angle_type'][angle_key].ubk = ubtype.ubk
                sys.params['angle_type'][angle_key].ubreq = ubtype.ubreq

    for dihedral in struct.dihedrals:
        dihedral_key = (dihedral.atom1.type, dihedral.atom2.type, dihedral.atom3.type, dihedral.atom4.type)
        dtype = dihedral.type[0] if isinstance(dihedral.type, list) else dihedral.type
        if dihedral_key not in sys.params['dihedral_type']:
            dtidx = len(sys.params['dihedral_type']) + 1
            sys.params['dihedral_type'][dihedral_key] = DihedralType(idx=dtidx, phi_k=dtype.phi_k, per=dtype.per, phase=dtype.phase)
        else:
            dtidx = sys.params['dihedral_type'][dihedral_key].idx

        sys.dihedrals.append(Dihedral(atom_map[dihedral.atom1], atom_map[dihedral.atom2], atom_map[dihedral.atom3], atom_map[dihedral.atom4], dtidx))
        sys.numDihedrals += 1

    for rb_torsion in struct.rb_torsions:
        rb_key = (rb_torsion.atom1.type, rb_torsion.atom2.type, rb_torsion.atom3.type, rb_torsion.atom4.type)
        rbtype = rb_torsion.type[0] if isinstance(rb_torsion.type, list) else rb_torsion.type
        if rb_key not in sys.params['rb_torsion_type']:
            rbidx = len(sys.params['rb_torsion_type']) + 1
            sys.params['rb_torsion_type'][rb_key] = RBTorsionType(idx=rbidx, c0=rbtype.c0, c1=rbtype.c1, 
                c2=rbtype.c2, c3=rbtype.c3, c4=rbtype.c4, c5=rbtype.c5)
        else:
            rbidx = sys.params['rb_torsion_type'][rb_key].idx

        sys.rb_torsions.append(RB_Torsion(atom_map[rb_torsion.atom1], atom_map[rb_torsion.atom2], atom_map[rb_torsion.atom3], atom_map[rb_torsion.atom4], rbidx))
        sys.numRBs += 1

    for improper in struct.impropers:
        im_key = (improper.atom1.type, improper.atom2.type, improper.atom3.type, improper.atom4.type)
        imtype = improper.type[0] if isinstance(improper.type, list) else improper.type
        if im_key not in sys.params['improper_type']:
            itidx = len(sys.params['improper_type']) + 1
            sys.params['improper_type'][im_key] = ImproperType(idx=itidx, psi_k=imtype.psi_k, psi_eq=imtype.psi_eq)
        else:
            itidx = sys.params['improper_type'][im_key].idx
        sys.impropers.append(Improper(atom_map[improper.atom1], atom_map[improper.atom2], atom_map[improper.atom3], atom_map[improper.atom4], itidx))
        sys.numImpropers += 1

    if struct.has_NBFIX():
        for atom in struct.atoms:
            if atom.atom_type.nbfix:
                other_atoms = list(atom.atom_type.nbfix.keys())
                for other_atom in other_atoms:
                    NB_key = tuple(sorted((other_atom, atom.type)))
                    if NB_key in sys.params['NBFIX']:
                        continue
                    (rmin, epsilon, rmin14, epsilon14) = atom.atom_type.nbfix[other_atom]
                    sys.params[NB_key] = (epsilon, rmin)

    sys.parametrized = True
    sys.ff_name = forcefield_name

    try:
        sys.merge(water_system)
    except:
        return

def _parametrize_custom(sys, parameters):
    element_dict = dict()
    for atom_type in parameters['atom_type']:
        if atom_type.name not in sys.params["atom_type"]:
            atidx = len(sys.params["atom_type"]) + 1
            atom_type.idx = atidx
            sys.params["atom_type"][atom_type.name] = atom_type
            element_dict[atom_type.symbol] = atom_type
        else:
            element_dict[atom_type.symbol] = sys.params["atom_type"][atom_type.name]
    
    # Assign atidx for all atoms
    for atom in sys.atoms_iter():
        atom.type = element_dict[atom.symbol].name
        atom.atidx = element_dict[atom.symbol].idx

def _parametrize_water(sys):
    """Parametrize water systems."""
    water_params = {
        'Water-TIP3P': {'atom_type': {'OW': AtomType(name='OWT3', atomic=8, symbol='O', sigma=3.188, epsilon=0.102, mass=15.9994, charge=-0.830), 
                                'HW': AtomType(name='HWT3', atomic=1, symbol='H', sigma=0.0, epsilon=0.0, mass=1.008, charge=0.415)},
                    'bond_type': BondType(name='OHT3', k=450, req=0.9572), 'angle_type': AngleType(name='HOHT3', k=55, theteq=104.52)},
        'Water-SPCE': {'atom_type': {'OW': AtomType(name='OWSPCE', atomic=8, symbol='O', sigma=3.166, epsilon=0.1553, mass=15.9994, charge=-0.8476), 
                                'HW': AtomType(name='HWSPCE', atomic=1, symbol='H', sigma=0.0, epsilon=0.0, mass=1.008, charge=0.4238)},
                    'bond_type': BondType(name='OHSPCE', k=0.0, req=1.0), 'angle_type': AngleType('HOHSPCE', k=0.0, theteq=109.47)},
        'Water-TIP4P': {'atom_type': {'OW': AtomType(name='OWT4', atomic=8, symbol='O', sigma=3.16435, epsilon=0.16275, mass=15.9994, charge=0.0), 
                                'HW': AtomType(name='HWT4', atomic=1, symbol='H', sigma=0.0, epsilon=0.0, mass=1.008, charge=0.5242),
                                'MW': AtomType(name='MWT4', atomic=0, symbol='D', sigma=0.0, epsilon=0.0, mass=0.0, charge=-1.0484)},
                    'bond_type': BondType(name='OHT4', k=0.0, req=0.9572), 'angle_type': AngleType(name='HOHT4', k=0.0, theteq=104.52)},
        'Water-TIP4P/2005': {'atom_type': {'OW': AtomType(name='OWT405', atomic=8, symbol='O', sigma=3.1589, epsilon=0.1852, mass=15.9994, charge=0.0),
                                        'HW': AtomType(name='HWT405', atomic=1, symbol='H', sigma=0.0, epsilon=0.0, mass=1.008, charge=0.5564),
                                        'MW': AtomType(name='MWT405', atomic=0, symbol='D', sigma=0.0, epsilon=0.0, mass=0.0, charge=-1.1128)},
                    'bond_type': BondType(name='OHT405', k=0.0, req=0.9572), 'angle_type': AngleType(name='HOHT4', k=0.0, theteq=104.52)},
    }

    sys.params = {
        'atom_type': OrderedDict(),
        'bond_type': OrderedDict(),
        'angle_type': OrderedDict(),
        'dihedral_type': OrderedDict(),
        'rb_torsion_type': OrderedDict(),
        'improper_type': OrderedDict(),
        'NBFIX': OrderedDict()}
    sys.flatten()
    
    for mol in sys.molecules_iter():
        for atom in mol.atoms_iter():
            at = water_params[mol.name]['atom_type'][atom.name[:2]]
            atom.mass = at.mass
            atom.charge = at.charge
            atom.type = at.name
            if atom.type not in sys.params['atom_type']:
                at.idx = len(sys.params['atom_type']) + 1
                sys.params['atom_type'][atom.type] = at
            atom.atidx = at.idx
    
        for bond in mol.bonds_iter():
            bond_key = tuple(sorted((bond[0].type, bond[1].type)))
            if  bond_key not in sys.params['bond_type']:
                bt = water_params[mol.name]['bond_type']
                bt.idx = len(sys.params['bond_type']) + 1
                sys.params['bond_type'][bond_key] = bt
        
        angle_key = (mol.atoms[1].type, mol.atoms[0].type, mol.atoms[2].type)
        if angle_key not in sys.params['angle_type']:
            angleidx = len(sys.params['angle_type']) + 1
            at = water_params[mol.name]['angle_type']
            at.idx = angleidx
            sys.params['angle_type'][angle_key] = at
        else:
            angleidx = sys.params['angle_type'][angle_key].idx
        sys.angles.append(Angle(atom1=mol.atoms[1], atom2=mol.atoms[0], atom3=mol.atoms[2], angleidx=angleidx))
        sys.numAngles += 1
    sys.parametrized = True


def _parametrize_openmm_to_pmd(sys, forcefield_files, **kwargs):
    import parmed as pmd
    topo, pos, system = sys.to_openmm(createSystem=True, forcefield_files=forcefield_files, **kwargs)
    struct = pmd.openmm.load_topology(topo, system=system, xyz=pos)
    return struct

