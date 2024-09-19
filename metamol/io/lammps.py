#TODO: Add Lammps<->Gromacs input file conversion 

import numpy as np
import pandas as pd
import warnings
from collections import OrderedDict
from copy import deepcopy

import metamol as meta
from metamol.system import Box
from metamol.exceptions import LammpsError, MetaError
from metamol.utils.ffobjects import Angle, AngleType, AtomType, BondType, Dihedral, DihedralType, RB_Torsion, Improper, RBTorsionType, ImproperType
from metamol.utils.constants import EPS0, EC, NAV, AMU, KCAL_TO_J
from metamol.utils.convert_formats import *

__all__ = ["read_lammps", "read_lammps_dumps", "write_lammps"]

def read_lammps(filename, host_obj=None, coords_only=False, **kwargs):
    """Read System from lammps data file."""
    sys_out = meta.System()
    nmass, ndiheds = 0, 0
    nbt, nat, ndt, nit = 0, 0, 0, 0
    with open(filename, 'r') as f:
        for num, line in enumerate(f, 1):
            if 'atom types' in line:
                nmass = int(line.split()[0])
            if 'bond types' in line:
                nbt = int(line.split()[0])
            if 'angle types' in line:
                nat = int(line.split()[0])
            if 'dihedral types' in line:
                ndt = int(line.split()[0])
            if 'improper types' in line:
                nit = int(line.split()[0])
            if 'atoms' in line:
                sys_out.numAtoms = int(line.split()[0])
            if 'bonds' in line:
                sys_out.numBonds = int(line.split()[0])
            if 'angles' in line:
                sys_out.numAngles = int(line.split()[0])
            if 'dihedrals' in line:
                ndiheds = int(line.split()[0])
            if 'impropers' in line:
                sys_out.numImpropers = int(line.split()[0])
            if 'xlo' in line:
                xlo, xhi = float(line.split()[0]), float(line.split()[1])
            if 'ylo' in line:
                ylo, yhi = float(line.split()[0]), float(line.split()[1])
            if 'zlo' in line:
                zlo, zhi = float(line.split()[0]), float(line.split()[1])
            if 'Masses' in line:
                startm = num+2
            
            if 'Pair Coeffs' in line:
                startpc = num+2
            if 'PairIJ Coeffs' in line:
                startpcij = num+2
            if 'Bond Coeffs' in line:
                startbc = num+2
            if 'Angle Coeffs' in line:
                startac = num+2
            if 'Improper Coeffs' in line:
                startic = num+2                
            if 'Dihedral Coeffs' in line:
                startdc = num+2
                dtype = line.split()[-1]
                if dtype=='opls':
                    sys_out.numRBs = ndiheds
                else:
                    sys_out.numDihedrals = ndiheds

            if 'Atoms' in line:
                startl = num+2
            if 'Bonds' in line:
                startb = num+2
            if 'Angles' in line:
                starta = num+2
            if 'Dihedrals' in line:
                startd = num+2
            if 'Impropers' in line:
                starti = num+2

    if nmass==0:
        print("No atoms in the data file. Return with an empty System.")
        if host_obj is None:
            return sys_out
        else:
            host_obj.copy(sys_out)
            return host_obj
    unit_style = kwargs.get('unit_style', 'real')
    # Get sigma, mass, epsilon, charge conversions
    if unit_style=='lj':
        # if units=='lj', need to specify conversion factors in key args
        sigma_conversion_factor = kwargs['scf']
        epsilon_conversion_factor = kwargs['ecf']
        mass_conversion_factor = kwargs['mcf']
        assert (sigma_conversion_factor!=0 and epsilon_conversion_factor!=0 and mass_conversion_factor!=0)
        charge_conversion_factor = np.sqrt(
            4 
            * np.pi
            * (sigma_conversion_factor * 1e-10)
            * (epsilon_conversion_factor * KCAL_TO_J)
            * EPS0) / EC

    elif unit_style=='si':
        sigma_conversion_factor = 1.0e10
        epsilon_conversion_factor = 1.0 / KCAL_TO_J * NAV
        mass_conversion_factor = 1.0 / AMU
        charge_conversion_factor = 1.0 / EC
    else:
        sigma_conversion_factor = 1
        epsilon_conversion_factor = 1
        mass_conversion_factor = 1
        charge_conversion_factor = 1

    sys_out.box = Box(bounds=np.asarray([xlo, ylo, zlo, xhi, yhi, zhi])*sigma_conversion_factor)
    
    if 'per' in kwargs:
        sys_out.box.per = list(kwargs['per'])
    
    df_mass = pd.read_csv(filename, sep=r'\s+', names=["id", "mass"], usecols=(0, 1,), skiprows = startm-1, nrows = nmass)
    masses = df_mass['mass'].values * mass_conversion_factor
    del df_mass

    ats = []
    if 'startpc' in locals():
        #df_pair = pd.read_csv(filename, sep=r'\s+', names=["id", "epsilon", "sigma", "type"], usecols=(0, 1, 2, 3,), skiprows = startpc-1, nrows = nmass)
        df_pair = pd.read_csv(filename, sep=r'\s+', names=["id", "epsilon", "sigma", "type"], skiprows = startpc-1, nrows = nmass)
        for idx, row in df_pair.iterrows():
            epsilon, sigma = row['epsilon'] * epsilon_conversion_factor, row['sigma'] * sigma_conversion_factor
            at = 'at'+str(idx+1) if pd.isna(row['type']) or not isinstance(row['type'], str) else row['type'][1:]
            ats.append(at)
            sys_out.params['atom_type'][at] = AtomType(idx=idx+1, mass=masses[idx], epsilon=epsilon, sigma=sigma, name=at)
        del df_pair
    elif 'startpcij' in locals():
        #df_pairIJ = pd.read_csv(filename, sep=r'\s+', names=["at1", "at2", "epsilon", "sigma", "type1"], usecols=(0, 1, 3, 4, 5,), skiprows = startpcij-1, nrows = nmass*(nmass+1)//2)
        df_pairIJ = pd.read_csv(filename, sep=r'\s+', names=["at1", "at2", "epsilon", "sigma", "type1"], skiprows = startpcij-1, nrows = nmass*(nmass+1)//2)
        for idx, row in df_pairIJ.iterrows():
            at1, at2, epsilon, sigma = int(row['at1']), int(row['at2']), row['epsilon'] * epsilon_conversion_factor, row['sigma'] * sigma_conversion_factor
            if at1==at2:
                at = 'at'+str(at1) if pd.isna(row['type1']) else row['type1'][1:]
                ats.append(at)
                sys_out.params['atom_type'][at] = AtomType(idx=at1, mass=masses[at1-1], epsilon=epsilon, sigma=sigma, name=at)
        del df_pairIJ
    else:
        ats = ['at'+str(idx) for idx in range(1, nmass+1)]

    assert len(ats) == nmass

    if nbt>0:
        df_bt = pd.read_csv(filename, sep=r'\s+', names=["type", "k", "req"], usecols=(0, 1, 2,), skiprows = startbc-1, nrows = nbt)
    if nat>0:
        df_at = pd.read_csv(filename, sep=r'\s+', names=["type", "k", "theteq"], usecols=(0, 1, 2,), skiprows = startac-1, nrows = nat)
    if ndt>0:
        if dtype=='opls':
            df_dt = pd.read_csv(filename, sep=r'\s+', names=["type", "f1", "f2", "f3", "f4"], usecols=(0, 1, 2, 3, 4,), skiprows = startdc-1, nrows = ndt)
        else:
            df_dt = pd.read_csv(filename, sep=r'\s+', names=["type", "phi_k", "per", "phase", "weight"], usecols=(0, 1, 2, 3, 4,), skiprows = startdc-1, nrows = ndt)
    if nit>0:
        df_it = pd.read_csv(filename, sep=r'\s+', names=["type", "psi_k", "psi_eq"], usecols=(0, 1, 2, ), skiprows = startic-1, nrows = nit)

    df_atoms = pd.read_csv(filename, sep=r'\s+', names=["aid","molid", "atomty", "charge", "x", "y", "z", "nx", "ny", "nz"], skiprows=startl-1, nrows=sys_out.numAtoms)
    df_atoms.sort_values(by=['aid'], inplace=True)
    df_atoms.reset_index(inplace=True)

    if sys_out.numBonds > 0:
        df_bonds = pd.read_csv(filename, sep=r'\s+', names=["bid", "type","at1","at2"], skiprows=startb-1, nrows=sys_out.numBonds)
    if sys_out.numAngles > 0:
        df_angles = pd.read_csv(filename, sep=r'\s+', names = ["angid", "type","at1","at2","at3"], skiprows=starta-1, nrows=sys_out.numAngles)
    if ndiheds > 0:
        df_diheds = pd.read_csv(filename, sep=r'\s+', names = ["did", "type","at1","at2","at3", "at4"], skiprows=startd-1, nrows=ndiheds)
    if sys_out.numImpropers > 0:
        df_imps = pd.read_csv(filename, sep=r'\s+', names = ["impid", "type","at1","at2","at3", "at4"], skiprows=starti-1, nrows=sys_out.numImpropers)    

    sys_out.numMols = int(df_atoms['molid'].max())
    sys_out.molList = [meta.Molecule() for _ in range(sys_out.numMols)]
    at_updated = set()
    for idx, row in df_atoms.iterrows():
        aid, molid, massid = int(row['aid']), int(row['molid']), int(row['atomty'])
        atom_tmp = meta.Atom(
                    idx=aid,
                    mass=masses[massid-1], 
                    charge=row['charge'] * charge_conversion_factor,
                    x=row['x'] * sigma_conversion_factor,
                    y=row['y'] * sigma_conversion_factor,
                    z=row['z'] * sigma_conversion_factor,
                    atomtype=ats[massid-1],
                    atidx=massid,
                )
        if atom_tmp.atidx not in at_updated:
            atom_type = sys_out.params['atom_type'][atom_tmp.type]
            atom_type.atomic = atom_tmp.atomic
            atom_type.symbol = atom_tmp.symbol
            atom_type.charge = atom_tmp.charge
            at_updated.add(atom_tmp.atidx)

        sys_out.molecules[molid-1].atomList.append(atom_tmp)
        sys_out.molecules[molid-1].numAtoms += 1
    
    if not coords_only:

        all_atoms = sys_out.atoms

        if sys_out.numBonds > 0:
            for idx, row in df_bonds.iterrows():
                bid, aid1, aid2 = int(row['type']), int(row['at1']), int(row['at2'])
                molid = int(df_atoms.iloc[aid1-1].at['molid'])
                atom1, atom2 = all_atoms[aid1-1], all_atoms[aid2-1]
                sys_out.molecules[molid-1].add_bond((atom1, atom2))
                bond_key = tuple(sorted((atom1.type, atom2.type)))
                if bond_key not in sys_out.params['bond_type']:
                    k = df_bt.iloc[bid-1].at['k'] * epsilon_conversion_factor / sigma_conversion_factor / sigma_conversion_factor
                    req = df_bt.iloc[bid-1].at['req'] * sigma_conversion_factor
                    sys_out.params['bond_type'][bond_key] = BondType(idx=bid, k=k, req=req)
            del df_bonds, df_bt
            sys_out.params['bond_type'] = OrderedDict(sorted(sys_out.params['bond_type'].items(), key=lambda x: x[1].idx))

        if sys_out.numAngles > 0:
            for idx, row in df_angles.iterrows():
                angid = int(row['type'])
                aid1, aid2, aid3 = int(row['at1']), int(row['at2']), int(row['at3'])
                atom1, atom2, atom3 = all_atoms[aid1-1], all_atoms[aid2-1], all_atoms[aid3-1]
                ang_key = (atom1.type, atom2.type, atom3.type)
                sys_out.angles.append(Angle(
                                        atom1=all_atoms[aid1-1], 
                                        atom2=all_atoms[aid2-1],
                                        atom3=all_atoms[aid3-1],
                                        angleidx=angid,
                                        )
                                    )
                if ang_key not in sys_out.params['angle_type']:
                    k = df_at.iloc[angid-1].at['k'] * epsilon_conversion_factor
                    theteq = df_at.iloc[angid-1].at['theteq']
                    sys_out.params['angle_type'][ang_key] = AngleType(idx=angid, k=k, theteq=theteq)
            del df_angles, df_at
            sys_out.params['angle_type'] = OrderedDict(sorted(sys_out.params['angle_type'].items(), key=lambda x: x[1].idx))

        if sys_out.numDihedrals > 0:
            for idx, row in df_diheds.iterrows():
                did = int(row['type'])
                aid1, aid2, aid3, aid4 = int(row['at1']), int(row['at2']), int(row['at3']), int(row['at4'])
                atom1, atom2, atom3, atom4 = all_atoms[aid1-1], all_atoms[aid2-1], all_atoms[aid3-1], all_atoms[aid4-1]
                dih_key = (atom1.type, atom2.type, atom3.type, atom4.type)
                sys_out.dihedrals.append(Dihedral(
                                        atom1=all_atoms[aid1-1], 
                                        atom2=all_atoms[aid2-1],
                                        atom3=all_atoms[aid3-1],
                                        atom4=all_atoms[aid4-1],
                                        dtidx=did,
                                        )
                                    )
                if dih_key not in sys_out.params['dihedral_type']:
                    phi_k = df_dt.iloc[did-1].at['phi_k'] * epsilon_conversion_factor
                    per = df_dt.iloc[did-1].at['per']
                    phase = df_dt.iloc[did-1].at['phase']
                    sys_out.params['dihedral_type'][dih_key] = DihedralType(idx=did, phi_k=phi_k, per=per, phase=phase)             
            del df_diheds, df_dt
            sys_out.params['dihedral_type'] = OrderedDict(sorted(sys_out.params['dihedral_type'].items(), key=lambda x: x[1].idx))
    
        if sys_out.numRBs > 0:
            for idx, row in df_diheds.iterrows():
                rid = int(row['type'])
                aid1, aid2, aid3, aid4 = int(row['at1']), int(row['at2']), int(row['at3']), int(row['at4'])
                atom1, atom2, atom3, atom4 = all_atoms[aid1-1], all_atoms[aid2-1], all_atoms[aid3-1], all_atoms[aid4-1]
                rb_key = (atom1.type, atom2.type, atom3.type, atom4.type)
                #if idx%100==0: print(idx)
                sys_out.rb_torsions.append(RB_Torsion(
                                        atom1=all_atoms[aid1-1], 
                                        atom2=all_atoms[aid2-1],
                                        atom3=all_atoms[aid3-1],
                                        atom4=all_atoms[aid4-1],
                                        rbidx=rid,
                                        )
                                    )
                if rb_key not in sys_out.params['rb_torsion_type']:
                    c1, c2 = df_dt.iloc[rid-1].at['f1'] * epsilon_conversion_factor, df_dt.iloc[rid-1].at['f2'] * epsilon_conversion_factor
                    c3, c4 = df_dt.iloc[rid-1].at['f3'] * epsilon_conversion_factor, df_dt.iloc[rid-1].at['f4'] * epsilon_conversion_factor
                    sys_out.params['rb_torsion_type'][rb_key] = RBTorsionType(idx=rid, c1=c1, c2=c2, c3=c3, c4=c4, opls=True)
            del df_diheds, df_dt
            sys_out.params['rb_torsion_type'] = OrderedDict(sorted(sys_out.params['rb_torsion_type'].items(), key=lambda x: x[1].idx))

        if sys_out.numImpropers > 0:
            for idx, row in df_imps.iterrows():
                iid = int(row['type'])
                aid1, aid2, aid3, aid4 = int(row['at1']), int(row['at2']), int(row['at3']), int(row['at4'])
                atom1, atom2, atom3, atom4 = all_atoms[aid1-1], all_atoms[aid2-1], all_atoms[aid3-1], all_atoms[aid4-1]
                im_key = (atom1.type, atom2.type, atom3.type, atom4.type)
                sys_out.dihedrals.append(Improper(
                                        atom1=all_atoms[aid1-1], 
                                        atom2=all_atoms[aid2-1],
                                        atom3=all_atoms[aid3-1],
                                        atom4=all_atoms[aid4-1],
                                        itidx=iid,
                                        )
                                    )
                if im_key not in sys_out.params['improper_type']:
                    psi_k = df_it.iloc[iid-1].at['psi_k'] * epsilon_conversion_factor
                    psi_eq = df_it.iloc[iid-1].at['psi_eq']
                    sys_out.params['improper_type'][im_key] = ImproperType(idx=iid, psi_k=psi_k, psi_eq=psi_eq)         
            del df_imps, df_it
            sys_out.params['improper_type'] = OrderedDict(sorted(sys_out.params['improper_type'].items(), key=lambda x: x[1].idx))
        sys_out.parametrized = True

    sys_out.flattened = True
    sys_out.dup = [1] * sys_out.numMols
    
    if host_obj is None:
        return sys_out
    else:
        host_obj.copy(sys_out)
        return host_obj

def read_lammps_dumps(filename, host_obj=None):
    """Read dump files from disk"""
    from metamol.system import Frame
    frames = []
    #read_step, read_box, read_atoms = False, False, False
    with open(filename, 'r') as f:
        current_section = None
        for line in f.readlines():
            line = line.strip()
            if not line: continue
            if line.split()[0] == 'ITEM:':
                current_section = line.split()[1]
                if current_section == 'BOX':
                    px, py, pz = line.split()[3:6]
                    per = (px=='pp', py=='pp', pz=='pp')
                    box_bounds = []
                elif current_section == 'ATOMS':
                    coords = []

            elif current_section == 'TIMESTEP':
                timestep = int(line)
                if "coords" in locals():
                    coords.sort(key=lambda x: x[0])
                    coords = np.asarray([x[1] for x in coords])
                    frame.coords = coords
                    frames.append(frame)
                frame = Frame(timestep=timestep)
            
            elif current_section == 'BOX':
                line = line.split()
                box_bounds += [np.float64(line[0]), np.float64(line[1])] 
            
            elif current_section == 'ATOMS':
                frame.box = Box(
                    bounds=np.asarray([box_bounds[0], box_bounds[2], box_bounds[4], box_bounds[1], box_bounds[3], box_bounds[5]]), 
                    per=per)

                line = line.split()
                xyz = [np.float64(line[2])*frame.box.lengths[0]+frame.box.bounds[0], 
                       np.float64(line[3])*frame.box.lengths[1]+frame.box.bounds[1],
                       np.float64(line[4])*frame.box.lengths[2]+frame.box.bounds[2]]
                coords.append((int(line[0]), xyz))

        coords.sort(key=lambda x: x[0])
        coords = np.asarray([x[1] for x in coords])
        frame.coords = coords
        frames.append(frame)

    if host_obj is None: return frames
    if not isinstance(host_obj, meta.System):
        raise MetaError("The host object must be a System instance")
    host_obj.frames += frames
    host_obj.numFrames += len(frames)
    return host_obj

def write_lammps(
    sys,
    filename,
    atom_style='full',
    unit_style='real',
    pair_coeff_label = 'lj/long/coul/long',
    combining_rule='arithmetic',
    zero_dihedral_weight=True,
    hybrid=dict(),
    tip4p='',
):
    """Write a meta.System to a lammps data file.

    Parameters
    ----------
    sys : metamol.System
        The model system to write.
    filename : str
        Name of the file to write to.
    atom_style : str, default=full
        The the style of atoms to be saved in the LAMMPS data file. The
        supported styles are `full`, `atomic`, `charge` and `molecular`.
    unit_style : str, default=real
        The the style of units in the LAMMPS data file. The supported styles are 
        `lj`, `real`, `si` and `metal`.
    pair_coeff_label : str, default=lj/long/coul/long
        Label to the pair_coeffs section of the LAMMPS data file.
    zero_dihedral_weighting_factor : bool, default=True
        If True, will set weighting parameter to zero in CHARMM-style dihedrals.
        This should be True if the CHARMM dihedral style is used in non-CHARMM forcefields.
    combining_rule : str, default=arithmetic
        The rule used to combine pair coefficients. The supported styles are 
        `arithmetic` (`lorentz`) and `geometric`.
    hybrid : dict: key->AtomType, value->(pair style, priority), default: empty
        If hybrid pair style is used. This dict maps atom types to pair styles assigned with a priority value.
    tip4p: str, default=''
        If System contains TIP4P water molecules, this value can be either `cut` or `long`.
        `cut` indicates lj/cut/tip4p/* pair style while `long` indicates lj/long/tip4p/*.
        *=`long` when TIP4P parameters are used and *=`cut` when TIP4P/2005 parameters are used. 

    Outputs
    -------
    filename : LAMMPS data file    
    """
    # Make sure the input argment sys is a System instance
    if not isinstance(sys, meta.System):
        raise TypeError("Input sys must be a System instance")
    
    if sys.numAtoms == 0:
        print("No atoms in the system. No data file is written.")
        return

    atom_style = atom_style.lower()
    unit_style = unit_style.lower()

    # Check the validity of atom style
    if atom_style not in ['atomic', 'full', 'charge', 'molecular']:
        raise ValueError("Atom style {} is not supported".format(atom_style))
    
    # Check the validity of unit style
    if unit_style not in ['real', 'lj', 'si', 'metal']:
        raise ValueError("Unit style {} is not supported".format(unit_style))

    if atom_style == 'atomic' and unit_style == 'si':
        warnings.warn("It is not recommedned to simulate atomic systems using SI units.")


    clone = deepcopy(sys)
    
    # Check for TIP4P
    if tip4p:
        tip4p_long = []
        tip4p_2005 = []
        for mol in clone.molecules_iter():
            if isinstance(mol, meta.Water4Site):
                # Assign charge to oxygen atom
                mol.atoms[0].charge = mol.atoms[-1].charge
                if isinstance(mol, meta.TIP4P):
                    if not tip4p_long:
                        tip4p_long = [mol.atoms[0].type, mol.atoms[1].type]                    
                elif isinstance(mol, meta.TIP4P2005):
                    if not tip4p_2005:
                        tip4p_2005 = [mol.atoms[0].type, mol.atoms[1].type]
                mw = mol.atoms.pop()
                clone.numAtoms -= 1
                if mw.type in clone.params['atom_type']:
                    del clone.params['atom_type'][mw.type]

        atidx = 1
        for at in clone.params['atom_type']:
            clone.params['atom_type'][at].idx = atidx
            atidx += 1
        for idx, atom in enumerate(clone.atoms):
            atom.idx = idx + 1
            atom.atidx = clone.params['atom_type'][atom.type].idx
        
        if not hybrid:
            for at in tip4p_2005:
                hybrid[at] = ('lj/'+tip4p+'/tip4p/cut', 1) # hybrid value = (pair_style, priority)
            for at in tip4p_long:
                hybrid[at] = ('lj/'+tip4p+'/tip4p/long', 1)

            others = [at for at in clone.params['atom_type'] if at not in tip4p_long+tip4p_2005]
            for at in others:              
                hybrid[at] = (pair_coeff_label, len(hybrid) + 1)

    if hybrid:
        use_hybrid = True
    
    else:
        use_hybrid = False

    if not clone.parametrized:
        clone.params = {'atom_type': OrderedDict()}
        for atom in clone.atoms_iter():
            if atom.name not in clone.params['atom_type']:
                clone.params['atom_type'][atom.name] = AtomType(idx=len(clone.params['atom_type'])+1, mass=atom.mass)
            atom.atidx = clone.params['atom_type'][atom.name].idx

    # Get sigma, mass, epsilon, charge conversions
    if unit_style=='lj':
        sigma_conversion_factor = np.max([at.sigma for at in clone.params['atom_type']])
        epsilon_conversion_factor = np.max([at.epsilon for at in clone.params['atom_type']])
        mass_conversion_factor = np.max([at.mass for at in clone.params['atom_type']])
        assert (sigma_conversion_factor!=0 and epsilon_conversion_factor!=0 and mass_conversion_factor!=0)
        charge_conversion_factor = np.sqrt(
            4 
            * np.pi
            * (sigma_conversion_factor * 1e-10)
            * (epsilon_conversion_factor * KCAL_TO_J)
            * EPS0) / EC

    elif unit_style=='si':
        sigma_conversion_factor = 1.0e10
        epsilon_conversion_factor = 1.0 / KCAL_TO_J * NAV
        mass_conversion_factor = 1.0 / AMU
        charge_conversion_factor = 1.0 / EC
    
    elif unit_style=='metal':
        sigma_conversion_factor = 1
        epsilon_conversion_factor = 1.0 / KCAL_TO_J * NAV * EC
        mass_conversion_factor = 1
        charge_conversion_factor = 1

    else:
        sigma_conversion_factor = 1
        epsilon_conversion_factor = 1
        mass_conversion_factor = 1
        charge_conversion_factor = 1
    
    if clone.numDihedrals > 0 and clone.numRBs > 0:
        raise LammpsError("Cannot support multiple dihedral types")

    with open(filename, "w") as data:
        data.write("LAMMPS data file created by MetaMol(version: {0}); units: {1} \n\n".format(meta.__version__, unit_style.lower()))
        
        #Write number of atoms, bonds, angles, dihedrals and impropers
        data.write("{:d} atoms \n".format(clone.numAtoms))
        if atom_style in ['full', 'molecular']:
            if clone.numBonds > 0:
                data.write("{:d} bonds \n".format(clone.numBonds))
            if clone.numAngles > 0:
                data.write("{:d} angles \n".format(clone.numAngles))
            if clone.numDihedrals+clone.numRBs > 0:
                data.write("{:d} dihedrals \n".format(clone.numDihedrals + clone.numRBs))
            if clone.numImpropers > 0:
                data.write("{:d} impropers \n".format(clone.numImpropers))
        data.write("\n")

        #Write number of atom types, bond types, angle types,
        #dihedral types and improper types.
        data.write("{:d} atom types\n".format(len(clone.params['atom_type'])))
        if atom_style in ['full', 'molecular']:
            numBondTypes = len(clone.params['bond_type'])
            if numBondTypes > 0:
                data.write("{:d} bond types\n".format(numBondTypes))
            numAngleTypes = len(clone.params['angle_type'])
            if numAngleTypes > 0:
                data.write("{:d} angle types\n".format(numAngleTypes))
            numDihTypes = len(clone.params['dihedral_type']) + len(clone.params['rb_torsion_type'])
            if numDihTypes > 0:
                data.write("{:d} dihedral types\n".format(numDihTypes))
            numImproperTypes = len(clone.params['improper_type'])
            if numImproperTypes > 0:
                data.write("{:d} improper types\n".format(numImproperTypes))
        
        data.write("\n")

        #Write Box info
        if clone.box is None:
            raise LammpsError("Simulation Box not set")
        xlo, ylo, zlo, xhi, yhi, zhi = np.asarray(clone.box.bounds) * (1.0 / sigma_conversion_factor)
        
        # elif len(clone.box) == 3:
        #     xlo, ylo, zlo = 0.0, 0.0, 0.0
        #     xhi, yhi, zhi = np.asarray(clone.box) * (1.0 / sigma_conversion_factor)
        # else:
        #     raise LammpsError("Simulation Box must be 3 dimensional")

        if unit_style == 'si':
            data.write("{0:.7e} {1:.7e} xlo xhi \n".format(xlo, xhi))
            data.write("{0:.7e} {1:.7e} ylo yhi \n".format(ylo, yhi))
            data.write("{0:.7e} {1:.7e} zlo zhi \n".format(zlo, zhi))
        else:
            data.write("{0:.6f} {1:.6f} xlo xhi \n".format(xlo, xhi))
            data.write("{0:.6f} {1:.6f} ylo yhi \n".format(ylo, yhi))
            data.write("{0:.6f} {1:.6f} zlo zhi \n".format(zlo, zhi))

        #Write mass section
        data.write("\nMasses")
        if unit_style == 'lj' or unit_style == 'metal':
            data.write("\nType\tmass\n")
        elif unit_style == 'real':
            data.write("\nType\tmass (amu)\n")
        elif unit_style == 'si':
            data.write("\nType\tmass (kilogram)\n")
        mass_line = "{0:d}\t{1:.7e}\t\t#{2}\n" if unit_style=='si' else "{0:d}\t{1:.6f}\t\t#{2}\n"
        for atname, at in clone.params['atom_type'].items():
            data.write(mass_line.format(
                at.idx, 
                at.mass * (1.0 / mass_conversion_factor),
                atname,
                )
            )
        
        #Write pair coeffs section
        if not clone.parametrized:
            pass
        
        else:
            #data.write("\nPair Coeffs # {0}\n".format(pair_coeff_label))
            
            if len(clone.params['NBFIX']) > 0:
                data.write("\nPairIJ Coeffs # {0}\n".format(pair_coeff_label))
                #Write modified cross-interaction
                if unit_style == 'lj':
                    data.write("Type1\tType2\treduced_epsilon\t\treduced_sigma\n")
                elif unit_style == 'real':
                    data.write("Type1\tType2\tepsilon (kcal/mol)\t\tsigma (A)\n")
                elif unit_style == 'si':
                    data.write("Type1\tType2\tepsilon (J)\t\tsigma (m)\n")
                
                pair_style_line = "{0:d}\t{1:d}\t\t{2:.7e}\t\t{3:.7e}\t\t#{4}\t{5}\n" if unit_style=='si' \
                    else "{0:d}\t{1:d}\t\t{2:.6f}\t\t{3:.6f}\t\t#{4}\t{5}\n"
                atnames = list(clone.params['atom_type'].keys())
                for id1 in range(len(atnames)):
                    for id2 in range(id1, len(atnames)):
                        at_key = tuple(sorted((atnames[id1], atnames[id2])))
                        at1, at2 = clone.params['atom_type'][at_key[0]], clone.params['atom_type'][at_key[1]]
                        if at1.idx > at2.idx:
                            at1, at2 = at2, at1
                        if at_key in clone.params['NBFIX']:
                            data.write(pair_style_line.format(
                                at1.idx, 
                                at2.idx,
                                clone.params['NBFIX'][at_key][1] * (1.0 / epsilon_conversion_factor), 
                                clone.params['NBFIX'][at_key][0] / 2**(1 / 6) * (1.0 / sigma_conversion_factor),
                                at1.name,
                                at2.name,
                                )
                            )
                        else:
                            if combining_rule.lower() == 'geometric':
                                sigma = (at1.sigma * at2.sigma)**0.5
                            elif combining_rule.lower() == 'arithmetic' or combining_rule.lower() == 'lorentz':
                                sigma = (at1.sigma + at2.sigma) * 0.5
                            else:
                                raise LammpsError("Right now only support lorentz(arithmetic) and geometric combining rules")
                            epsilon = (at1.epsilon * at2.epsilon)**0.5

                            data.write(pair_style_line.format(
                                at1.idx, 
                                at2.idx,
                                epsilon * (1.0 / epsilon_conversion_factor), 
                                sigma * (1.0 / sigma_conversion_factor),
                                at1.name,
                                at2.name,
                                )
                            )
            elif use_hybrid:
                data.write("\nPairIJ Coeffs # Hybrid\n")
                #Write modified cross-interaction
                if unit_style == 'lj':
                    data.write("Type1\tType2\tpair style\treduced_epsilon\t\treduced_sigma\n")
                elif unit_style == 'real':
                    data.write("Type1\tType2\tpair style\tepsilon (kcal/mol)\t\tsigma (A)\n")
                elif unit_style == 'si':
                    data.write("Type1\tType2\tpair style\tepsilon (J)\t\tsigma (m)\n")
                
                pair_style_line = "{0:d}\t{1:d}\t{2}\t\t{3:.7e}\t\t{4:.7e}\t\t#{5}\t{6}\n" if unit_style=='si' \
                    else "{0:d}\t{1:d}\t{2}\t\t{3:.6f}\t\t{4:.6f}\t\t#{5}\t{6}\n" 

                atnames = list(clone.params['atom_type'].keys())
                for id1 in range(len(atnames)):
                    for id2 in range(id1, len(atnames)):
                        at_key = tuple(sorted((atnames[id1], atnames[id2])))
                        at1, at2 = clone.params['atom_type'][at_key[0]], clone.params['atom_type'][at_key[1]]
                        if at1.idx > at2.idx:
                            at1, at2 = at2, at1

                        if combining_rule.lower() == 'geometric':
                            sigma = (at1.sigma * at2.sigma)**0.5
                        elif combining_rule.lower() == 'arithmetic' or combining_rule.lower() == 'lorentz':
                            sigma = (at1.sigma + at2.sigma) * 0.5
                        else:
                            raise LammpsError("Right now only support lorentz(arithmetic) and geometric combining rules ")
                        epsilon = (at1.epsilon * at2.epsilon)**0.5   

                        pair_style = hybrid[at1.name][0] if hybrid[at1.name][1] <= hybrid[at2.name][1] else hybrid[at2.name][0]      

                        data.write(pair_style_line.format(
                            at1.idx, 
                            at2.idx,
                            pair_style,
                            epsilon * (1.0 / epsilon_conversion_factor), 
                            sigma * (1.0 / sigma_conversion_factor),
                            at1.name,
                            at2.name,
                            )
                        )

            else:
                data.write("\nPair Coeffs # {0}\n".format(pair_coeff_label))
                if unit_style == 'lj':
                    data.write("Type\treduced_epsilon\t\treduced_sigma\n")
                elif unit_style == 'real':
                    data.write("Type\tepsilon (kcal/mol)\t\tsigma (A)\n")
                elif unit_style == 'si':
                    data.write("Type\tepsilon (J)\t\tsigma (m)\n")
                atom_style_line = "{0:d}\t{1:.7e}\t\t{2:.7e}\t\t#{3}\n" if unit_style=='si' \
                    else "{0:d}\t{1:.6f}\t\t{2:.6f}\t\t#{3}\n"

                for atname, at in clone.params['atom_type'].items():       
                    data.write(atom_style_line.format(
                        at.idx, 
                        at.epsilon * (1.0 / epsilon_conversion_factor), 
                        at.sigma * (1.0 / sigma_conversion_factor),
                        atname,
                        )
                    )

            #Write bonds coeffs section
            if clone.numBonds > 0:
                data.write("\nBond Coeffs # harmonic\n")
                if unit_style == 'lj':
                    data.write("Type\treduced_K\t\treduced_req\n")
                elif unit_style == 'real':
                    data.write("Type\tK (kcal/mol/A^2)\t\treq (A)\n")
                elif unit_style == 'si':
                    data.write("Type\tK (J/m^2)\t\treq (m)\n")
                
                bond_style_line = "{0:d}\t{1:.7e}\t\t{2:.7e}\t\t#{3}\t{4}\n" if unit_style=='si' \
                    else "{0:d}\t{1:.6f}\t\t{2:.6f}\t\t#{3}\t{4}\n"
                for bond_key, bt in clone.params['bond_type'].items():
                    data.write(bond_style_line.format(
                        bt.idx,
                        bt.k * (
                            sigma_conversion_factor ** 2
                            / epsilon_conversion_factor
                        ),
                        bt.req * (1.0 / sigma_conversion_factor),
                        bond_key[0],
                        bond_key[1],
                        )
                    )

            #Write Angle coeffs section
            if clone.numAngles > 0:
                if clone.use_ub:
                    data.write("\nAngle Coeffs # charmm \n")
                    if unit_style == 'real':
                        data.write(
                            "Type\tk(kcal/mole/deg^2)\t\ttheteq(deg)\t\tk_ub(kcal/mol/A^2)\t\treq_ub(A)\n"
                        )
                    elif unit_style == 'si':
                        data.write(
                            "Type\tk(J/deg^2)\t\ttheteq(deg)\t\tk_ub(J/A^2)\t\treq_ub(A)\n"
                        )
                    elif unit_style == 'lj':
                        data.write(
                            "Type\treduced_k\t\ttheteq(deg)\t\treduced_k_ub\t\treduced_req_ub\n"
                        )

                    angle_style_line = "{0:d}\t{1:.7e}\t\t{2:.3f}\t\t{3:.7e}\t\t{4:.7e}\t\t#{5}\t{6}\t{7}\n" if unit_style=='si' \
                        else "{0:d}\t{1:.6f}\t\t{2:.3f}\t\t{3:.6f}\t\t{4:.6f}\t\t#{5}\t{6}\t{7}\n"
                    for angle_key, angle_type in clone.params['angle_type'].items():
                        data.write(
                            angle_style_line.format(
                                angle_type.idx,
                                angle_type.k * (1.0 / epsilon_conversion_factor),
                                angle_type.theteq,
                                angle_type.ubk * (
                                    sigma_conversion_factor**2
                                    / epsilon_conversion_factor),
                                angle_type.ubreq * (1.0 / sigma_conversion_factor),
                                angle_key[0],
                                angle_key[1],
                                angle_key[2],
                                )
                            )
            
                else:
                    data.write("\nAngle Coeffs # harmonic \n")
                    if unit_style == 'real':
                        data.write(
                            "Type\tK(kcal/mole/deg^2)\t\ttheteq(deg)\n"
                        )
                    elif unit_style == 'si':
                        data.write(
                            "Type\tK(J/deg^2)\t\ttheteq(deg)\n"
                        )
                    elif unit_style == 'lj':
                        data.write(
                            "Type\treduced_K\t\ttheteq(deg)\n"
                        )

                    angle_style_line = "{0:d}\t{1:.7e}\t\t{2:.3f}\t\t#{3}\t{4}\t{5}\n" if unit_style=='si' \
                        else "{0:d}\t{1:.6f}\t\t{2:.3f}\t\t#{3}\t{4}\t{5}\n"
                    for angle_key, angle_type in clone.params['angle_type'].items():
                        data.write(
                            angle_style_line.format(
                                angle_type.idx,
                                angle_type.k * (1.0 / epsilon_conversion_factor),
                                angle_type.theteq,
                                angle_key[0],
                                angle_key[1],
                                angle_key[2],
                                )
                            )

            #Write Dihedral coeffs section
            if clone.numRBs > 0:
                data.write("\nDihedral Coeffs # opls\n")
                if unit_style == 'real':
                    data.write("Type\tf1(kcal/mol)\t\tf2(kcal/mol)\t\tf3(kcal/mol)\t\tf4(kcal/mol)\n")
                elif unit_style == 'si':
                    data.write("Type\tf1(J)\t\tf2(J)\t\tf3(J)\t\tf4(J)\n")
                elif unit_style == 'lj':
                    data.write("Type\treduced_f1\t\treduced_f2\t\treduced_f3\t\treduced_f4\n")

                RB_style_line = "{0:d}\t{1:.7e}\t\t{2:.7e}\t\t{3:.7e}\t\t{4:.7e}\t\t#{5}\t{6}\t{7}\t{8}\n" if unit_style=='si' \
                    else "{0:d}\t{1:.6f}\t\t{2:.6f}\t\t{3:.6f}\t\t{4:.6f}\t\t#{5}\t{6}\t{7}\t{8}\n"
                for RB_key, RB_type in clone.params['rb_torsion_type'].items():
                    if RB_type.opls:
                        opls_params = [RB_type.c0 * (1.0 / epsilon_conversion_factor),
                                       RB_type.c1 * (1.0 / epsilon_conversion_factor),
                                       RB_type.c2 * (1.0 / epsilon_conversion_factor),
                                       RB_type.c3 * (1.0 / epsilon_conversion_factor),
                                       RB_type.c4 * (1.0 / epsilon_conversion_factor),
                                       RB_type.c5 * (1.0 / epsilon_conversion_factor)]

                    else:
                        opls_params = RB_to_OPLS(
                            RB_type.c0 * (1.0 / epsilon_conversion_factor),
                            RB_type.c1 * (1.0 / epsilon_conversion_factor),
                            RB_type.c2 * (1.0 / epsilon_conversion_factor),
                            RB_type.c3 * (1.0 / epsilon_conversion_factor),
                            RB_type.c4 * (1.0 / epsilon_conversion_factor),
                            RB_type.c5 * (1.0 / epsilon_conversion_factor),
                            error_if_outside_tolerance=False,
                        )
                    data.write(RB_style_line.format(
                        RB_type.idx,
                        opls_params[1],
                        opls_params[2],
                        opls_params[3],
                        opls_params[4],
                        RB_key[0],
                        RB_key[1],
                        RB_key[2],
                        RB_key[3],
                        )
                    )
            elif clone.numDihedrals > 0:
                data.write("\nDihedral Coeffs # charmm\n")
                if unit_style == 'real':
                    data.write("Type\tK(kcal/mol)\t\tn\t\td(deg)\t\tweight\n")
                elif unit_style == 'si':
                    data.write("Type\tK(J)\t\tn\t\td(deg)\t\tweight\n")
                elif unit_style == 'lj':
                    data.write("Type\treduced_K\t\tn\t\td(deg)\t\tweight\n")

                if zero_dihedral_weight:
                    weight = 0.0
                else:
                    weight = 1.0 / len(clone.params['dihedral_type'])

                dih_type_line = "{0:d}\t{1:.7e}\t\t{2:d}\t\t{3:d}\t\t{4:.3f}\t\t#{5}\t{6}\t{7}\t{8}\n" if unit_style=='si' \
                    else "{0:d}\t{1:.6f}\t\t{2:d}\t\t{3:d}\t\t{4:.3f}\t\t#{5}\t{6}\t{7}\t{8}\n"
                for dih_key, dih_type in clone.params['dihedral_type'].items():
                    data.write(dih_type_line.format(
                        dih_type.idx,
                        dih_type.phi_k * (1.0 / epsilon_conversion_factor),
                        dih_type.per,
                        int(dih_type.phase),
                        weight,
                        dih_key[0],
                        dih_key[1],
                        dih_key[2],
                        dih_key[3],
                        )
                    )
            
            #Write Improper Coeffs section
            if clone.numImpropers > 0:
                data.write("\nImproper Coeffs # harmonic\n")
                if unit_style == 'real':
                    data.write("Type\tK(kcal/mol)\t\tpsi(deg)\n")
                elif unit_style == 'si':
                    data.write("Type\tK(J)\t\tpsi(deg)\n")
                elif unit_style == 'lj':
                    data.write("Type\treduced_K\t\treduced_psi\n")

                imp_type_line = "{0:d}\t{1:.7e}\t\t{2:.3f}\t\t#{3}\t{4}\t{5}\t{6}\n" if unit_style=='si' \
                    else "{0:d}\t{1:.6f}\t\t{2:.3f}\t\t#{3}\t{4}\t{5}\t{6}\n"
                for imp_key, imp_type in clone.params['improper_type'].items():
                    data.write(imp_type_line.format(
                        imp_type.idx,
                        imp_type.psi_k * (1.0 / epsilon_conversion_factor),
                        imp_type.psi_eq,
                        imp_key[0],
                        imp_key[1],
                        imp_key[2],
                        )
                    )

        #Write Atom section
        data.write("\nAtoms # {0}\n\n".format(atom_style))
        if atom_style == 'atomic':
            atom_line = "{idx:d}\t{type_id:d}\t{x:.7e}\t{y:.7e}\t{z:.7e}\n" if unit_style=='si' \
                else "{idx:d}\t{type_id:d}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n"
        elif atom_style == 'charge':
            if unit_style == 'real' or unit_style == 'metal':
                atom_line = "{idx:d}\t{type_id:d}\t{charge:.6f}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n"
            elif unit_style == 'lj':
                atom_line = "{idx:d}\t{type_id:d}\t{charge:.4ef}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n"
            elif unit_style == 'si':
                atom_line = "{idx:d}\t{type_id:d}\t{charge:.4e}\t{x:.7e}\t{y:.7e}\t{z:.7e}\n"
        elif atom_style == 'molecular':
            atom_line = "{idx:d}\t{mol_id:d}\t{type_id:d}\t{x:.7e}\t{y:.7e}\t{z:.7e}\n" if unit_style=='si' \
                else "{idx:d}\t{mol_id:d}\t{type_id:d}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n"
        elif atom_style == 'full':
            if unit_style == 'real' or unit_style == 'metal':
                atom_line = "{idx:d}\t{mol_id:d}\t{type_id:d}\t{charge:.6f}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n"
            elif unit_style == 'lj':
                atom_line = "{idx:d}\t{mol_id:d}\t{type_id:d}\t{charge:.4e}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n"
            elif unit_style == 'si':
                atom_line = "{idx:d}\t{mol_id:d}\t{type_id:d}\t{charge:.4e}\t{x:.7e}\t{y:.7e}\t{z:.7e}\n"

        for molid, mol in enumerate(clone.molecules):
            for atomid, atom in enumerate(mol.atoms):
                data.write(
                    atom_line.format(
                        idx=atom.idx,
                        mol_id=molid + 1,
                        type_id=atom.atidx,
                        charge=atom.charge *(1.0 / charge_conversion_factor),
                        x = atom.x * (1.0 / sigma_conversion_factor),
                        y = atom.y * (1.0 / sigma_conversion_factor),
                        z = atom.z * (1.0 / sigma_conversion_factor),
                    )
                )

        if atom_style in ['full', 'molecular']:
            #Write Bond section
            if clone.numBonds > 0:
                data.write("\nBonds\n\n")
                bond_idx = 0
                for bond in clone.bonds_iter():
                    bond_idx += 1
                    bond_key = sorted((bond[0].type, bond[1].type))
                    bond_type_idx = clone.params['bond_type'][tuple(bond_key)].idx
                    data.write(
                        "{0:d}\t{1:d}\t{2:d}\t{3:d}\n".format(
                            bond_idx,
                            bond_type_idx,
                            bond[0].idx,
                            bond[1].idx,
                        )
                    )

            #Write Angle section
            if clone.numAngles > 0:
                data.write("\nAngles\n\n")
                for idx, angle in enumerate(clone.angles):
                    data.write(
                        "{0:d}\t{1:d}\t{2:d}\t{3:d}\t{4:d}\n".format(
                            idx+1, 
                            angle.typeID, 
                            angle.atom1.idx, 
                            angle.atom2.idx, 
                            angle.atom3.idx,
                        )
                    )

            #Write Dihedral section
            if clone.numDihedrals > 0:
                data.write("\nDihedrals\n\n")
                for idx, dih in enumerate(clone.dihedrals):
                    data.write(
                        "{0:d}\t{1:d}\t{2:d}\t{3:d}\t{4:d}\t{5:d}\n".format(
                            idx+1,
                            dih.typeID,
                            dih.atom1.idx,
                            dih.atom2.idx,
                            dih.atom3.idx,
                            dih.atom4.idx,
                        )
                    )
            
            elif clone.numRBs > 0:
                data.write("\nDihedrals\n\n")
                for idx, RB in enumerate(clone.rb_torsions):
                    data.write(
                        "{0:d}\t{1:d}\t{2:d}\t{3:d}\t{4:d}\t{5:d}\n".format(
                            idx+1,
                            RB.typeID,
                            RB.atom1.idx,
                            RB.atom2.idx,
                            RB.atom3.idx,
                            RB.atom4.idx,
                        )
                    )
            
            #Write Improper section
            if clone.numImpropers > 0:
                data.write("\nImpropers\n\n")
                for idx, improper in enumerate(clone.impropers):
                    data.write(
                        "{0:d}\t{1:d}\t{2:d}\t{3:d}\t{4:d}\t{5:d}\n".format(
                            idx+1,
                            improper.typeID,
                            improper.atom1.idx,
                            improper.atom2.idx,
                            improper.atom3.idx,
                            improper.atom4.idx,
                        )
                    )