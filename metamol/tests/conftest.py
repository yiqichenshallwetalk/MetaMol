def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", default=False, help="Test MD runs on GPU")
    parser.addoption("--gpu_backend", action="store", default="kokkos", help="Lammps GPU backend, possible choices: kokkos, gpu")

def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    gpu_option = metafunc.config.getoption("--gpu")
    gpu_backend = metafunc.config.getoption("--gpu_backend")
    if "gpu" in metafunc.fixturenames:
        metafunc.parametrize("gpu", [gpu_option])
        metafunc.parametrize("gpu_backend", [gpu_backend])
