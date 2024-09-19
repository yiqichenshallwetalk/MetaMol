import pytest
import numpy as np

from metamol.lib.molecules.alkane import *
from metamol.lib.molecules.polymer import *
from metamol.lib.fragments.monomers import *

@pytest.mark.parametrize('N', [1, 2, 3, 4, 5, 10, 15, 20, 100, 200])
def test_alkane(N):
    alkane = Alkane(N, align='x')

    assert alkane.numAtoms == 3*N + 2
    assert alkane.numBonds == 3*N + 1

    assert len([a for a in alkane.atoms_iter() if a.symbol=='C']) == N
    assert len([a for a in alkane.atoms_iter() if a.symbol=='H']) == 2*N + 2

    alkane_y = Alkane(N, align='y')

    alkane.rotate(about='z', angle=90.0, rad=False)

    assert np.allclose(alkane.xyz, alkane_y.xyz)

    return

def __get_param__(monomers_seq=[], head=[], tail=[], N=[]):
    for m_seq in monomers_seq:
        for h in head:
            for t in tail:
                for num in N:
                    yield (m_seq[0], m_seq[1], h, t, num)
                    
@pytest.mark.parametrize('monomers,seq,head,tail,N', __get_param__(
    monomers_seq = [([CH2()], 'A'), ([PEGMonomer()], 'A'), ([CH2(), PEGMonomer()], 'AB')],
    head = [None, CH3(), OH()], tail = [None, CH3(), OH()],
    N = [1, 5, 10, 20, 40],
    )
)
def test_polymer_template(monomers, seq, head, tail, N):
    pol = Polymer(monomers, seq=seq, head=head, tail=tail)
    pol.build(N)

    headAtoms = 1 if head is None else head.numAtoms
    tailAtoms = 1 if tail is None else tail.numAtoms
    assert pol.numAtoms == sum([m.numAtoms for m in monomers]) * N + headAtoms + tailAtoms

    headBonds = 0 if head is None else head.numBonds
    tailBonds = 0 if tail is None else tail.numBonds
    assert pol.numBonds == sum([m.numBonds+1 for m in monomers]) * N + headBonds + tailBonds + 1

    all_ele_frags = ['H'] if head is None else [a.symbol for a in head.atoms_iter()]
    all_ele_frags += ['H'] if tail is None else [a.symbol for a in tail.atoms_iter()]
    for monomer in monomers:
        all_ele_frags += [a.symbol for a in monomer.atoms_iter()] * N
    
    all_ele_pol = [a.symbol for a in pol.atoms_iter()]
    
    assert len(all_ele_frags) == len(all_ele_pol)

    from collections import Counter
    assert Counter(all_ele_frags) == Counter(all_ele_pol)

    pol_y = deepcopy(pol)
    pol.build(N, align='y')

    pol_y.rotate(about='z', angle=90.0, rad=False)
    assert np.allclose(pol.xyz, pol_y.xyz)

    return