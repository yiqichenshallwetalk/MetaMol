import pytest
import os
import json
from pkg_resources import resource_filename
import shutil

#import metamol as meta

try:
    from metamol.workflows.gromacs_fep import Solvate, PrepRunFEP, FEP_workflow
    from metamol.workflows.utils import download_local
    from dflow import Task, Workflow, upload_artifact, config
    from dflow.python import OP, OPIO, Artifact, OPIOSign, PythonOPTemplate, Parameter, BigParameter
    # Debug mode , indepedent of k8s
    config["mode"] = "debug"
    has_wf = True
except (ModuleNotFoundError or ImportError):
    has_wf = False

class TestFEPWorkflow:
    @pytest.mark.skipif(not has_wf, reason="workflow module not installed")
    def test_FEP_workflow(self):
        inp_file_prefix = resource_filename("metamol", os.path.join("tests", "files", "workflow", "FEP"))
        
        wf, step_list = FEP_workflow(config_file="input_FEP.json", 
                          inp_file_prefix = inp_file_prefix)
        wf.submit()

        shutil.rmtree("upload", ignore_errors=True)
        assert wf.query_status() == "Succeeded"
        
        local_dir = "download"
        download_local(wf, step_list[-1], local_path=local_dir)

        shutil.rmtree(wf.id, ignore_errors=True)
        shutil.rmtree(local_dir, ignore_errors=True)