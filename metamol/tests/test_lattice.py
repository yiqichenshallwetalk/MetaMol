import pytest
import os
import numpy as np
from pkg_resources import resource_filename

import metamol as meta
from metamol.exceptions import MetaError

@pytest.mark.parametrize('x', [2, 3, 4])
@pytest.mark.parametrize('y', [2, 3, 4])
@pytest.mark.parametrize('z', [5, 10, 20])
def test_cu(x, y, z):
    cu_lattice = meta.Lattice(
        spacings=[3.6149, 3.6149, 3.6149],
        langles=[90.0, 90.0, 90.0],
        )
    vecs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert np.allclose(vecs, cu_lattice.vectors)

    cu = meta.Molecule(meta.Atom(symbol='Cu'))
    cu_locations = [[0., 0., 0.], [.5, .5, 0.], [.5, 0., .5], [0., .5, .5]]
    occupy_points = {cu: cu_locations}
    numAtoms = sum([len(occupy_points[key]) * key.numAtoms \
                for key in occupy_points.keys()])

    cu_lattice.occupy(occupy_points)
    assert cu_lattice.numAtoms == numAtoms

    cu_lattice.replicate(x=x, y=y, z=z)
    assert cu_lattice.numAtoms == numAtoms * x * y * z

    return

@pytest.mark.parametrize('x', [2, 3, 4])
@pytest.mark.parametrize('y', [2, 3, 4])
@pytest.mark.parametrize('z', [5, 10, 20])
def test_cscl(x, y, z):
    cscl_lattice = meta.Lattice(
        spacings=[4.123, 4.123, 4.123],
        langles=[90.0, 90.0, 90.0],
        )
    vecs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert np.allclose(vecs, cscl_lattice.vectors)

    cs = meta.Molecule(meta.Atom(symbol='Cs'))
    cl = meta.Molecule(meta.Atom(symbol='Cl'))

    occupy_points = {cs: [[.5, .5, .5]], cl: [[0., 0., 0.]]}
    numAtoms = sum([len(occupy_points[key]) * key.numAtoms \
                for key in occupy_points.keys()])

    cscl_lattice.occupy(occupy_points)
    assert cscl_lattice.numAtoms == numAtoms

    cscl_lattice.replicate(x=x, y=y, z=z)
    assert cscl_lattice.numAtoms == numAtoms * x * y * z

    return

@pytest.mark.parametrize('x', [3, 5, 10])
@pytest.mark.parametrize('y', [3, 5, 10])
@pytest.mark.parametrize('z', [1])
def test_graphene(x, y ,z):
    gc_lattice = meta.Lattice(
        spacings=[2.456, 2.456, 0.], 
        vectors=[[1.0, 0., 0.], [-0.5, np.sqrt(3/4), 0.], [0., 0., 1.]],
        )
    langles = [90.0, 90.0, 120.0]
    assert np.allclose(langles, gc_lattice.langles)

    C = meta.Molecule(meta.Atom(symbol='C'))
    occupy_points = {C: [[0., 0., 0.], [2/3, 1/3, 0.]]}
    numAtoms = sum([len(occupy_points[key]) * key.numAtoms \
                for key in occupy_points.keys()])

    gc_lattice.occupy(occupy_points)
    assert gc_lattice.numAtoms == numAtoms

    gc_lattice.replicate(x=x, y=y, z=z)
    assert gc_lattice.numAtoms == numAtoms * x * y * z

    return

@pytest.mark.parametrize('x', [2, 3, 4])
@pytest.mark.parametrize('y', [2, 3, 4])
@pytest.mark.parametrize('z', [5, 10, 20])
def test_cholesterol(x, y ,z):
    cholesterol_lattice = meta.Lattice(
        spacings=[14.172, 34.209, 10.481],
        langles=[94.64, 90.67, 96.32], 
        )
    vectors = [[ 1.        ,  0.        ,  0.        ],
       [-0.11008126,  0.99392259,  0.        ],
       [-0.01169344, -0.08268452,  0.99650717]]
    assert np.allclose(vectors, cholesterol_lattice.vectors)

    filename = resource_filename("metamol", os.path.join("tests", "files", "cholesterol.pdb"))
    cholesterol = meta.Molecule(filename)

    occupy_points = {cholesterol: [[0., 0., 0.]]}
    numAtoms = sum([len(occupy_points[key]) * key.numAtoms \
                for key in occupy_points.keys()])
                    
    cholesterol_lattice.occupy(occupy_points)
    assert cholesterol_lattice.numAtoms == numAtoms

    cholesterol_lattice.replicate(x=x, y=y, z=z)
    assert cholesterol_lattice.numAtoms == numAtoms * x * y * z

    return

@pytest.mark.parametrize('spacings', ["xyz", None, [1., 1.], [1., 1.5, -0.5]])
def test_invalid_lengths(spacings):
    with pytest.raises(MetaError):
        lattice_test = meta.Lattice(spacings=spacings, langles=[90.0, 90.0, 90.0])

@pytest.mark.parametrize('langles', [[90., 90.], [90., 90., 0.], [90., 190., 140.]])
def test_invalid_langles(langles):
    with pytest.raises(MetaError):
        lattice_test = meta.Lattice(spacings=[10.0, 10.0, 10.0], langles=langles)

@pytest.mark.parametrize('vectors', [[[1.0, 0.0], [0.0, 1.0]], 
[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]
)
def test_invalid_vectors(vectors):
    with pytest.raises(MetaError):
        lattice_test = meta.Lattice(spacings=[10.0, 10.0, 10.0], vectors=vectors)

@pytest.mark.parametrize('inputs', 
[{"spacings": [10., 10., 10.], "langles": [90., 90., 90.],
"vectors": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]},
{"spacings": [2.456, 2.456, 0.], "langles": [90.0, 90.0, 120.0],
"vectors": [[1.0, 0., 0.], [-0.5, np.sqrt(3/4), 0.], [0., 0., 1.]]}]
)
def test_consistent_inputs(inputs):
    spacings, langles, vectors = inputs["spacings"], inputs["langles"], inputs["vectors"]
    lattice_test = meta.Lattice(spacings=spacings, langles=langles, vectors=vectors)

@pytest.mark.parametrize('inputs', 
[{"spacings": [10., 10., 10.], "langles": [90., 90., 120.],
"vectors": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]},
{"spacings": [2.456, 2.456, 0.], "langles": [90., 90., 30.],
"vectors": [[1.0, 0., 0.], [-0.5, -np.sqrt(3/4), 0.], [0., 0., 1.]]}]
)
def test_inconsistent_inputs(inputs):
    spacings, langles, vectors = inputs["spacings"], inputs["langles"], inputs["vectors"]
    with pytest.raises(MetaError):
        lattice_test = meta.Lattice(spacings=spacings, langles=langles, vectors=vectors)
