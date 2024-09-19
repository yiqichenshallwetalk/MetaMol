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
#    @pytest.mark.skipif(not has_wf, reason="workflow module not installed")
#    def test_prep_run_enmin(self):
#        inp_file_prefix = resource_filename("metamol", os.path.join("tests", "files", "workflow", "FEP"))
#        with open(os.path.join(inp_file_prefix, "input_FEP.json")) as f:
#            FEP_configs = json.load(f)   
#        solvation_configs = FEP_configs["solvation"]
#        prep_run_configs = FEP_configs["prep-run-block1"]
#        art_gro = upload_artifact(os.path.join(inp_file_prefix, solvation_configs["input_data"]["inp_gro"]))
#        art_top = upload_artifact(os.path.join(inp_file_prefix, solvation_configs["input_data"]["inp_top"]))
#        art_mdp = upload_artifact(os.path.join(inp_file_prefix, prep_run_configs["input_data"]["inp_mdp"]))
#
#        solvation = Task(
#            name="solvate",
#            template=PythonOPTemplate(Solvate),
#            parameters={"GMX": FEP_configs["GMX"],
#                    "box": solvation_configs["gmx_cmds"]["box"],
#                    "screen": solvation_configs["gmx_cmds"]["screen"],
#                    },
#            artifacts={"inp_gro": art_gro, "inp_top": art_top},
#        )
#
#        prep_run = Task(
#            name="enmin-prep-run",
#            template=PythonOPTemplate(PrepRunFEP),
#            parameters={"GMX": FEP_configs["GMX"],
#                        "screen": prep_run_configs["gmx_cmds"]["screen"],
#                        "run_type": prep_run_configs["run_type"], 
#                        "num_states": prep_run_configs["num_states"],
#                        "num_threads": prep_run_configs["gmx_cmds"]["num_threads"],
#                        "extra_args": prep_run_configs["gmx_cmds"]["extra_args"]},
#            artifacts={"inp_gro": solvation.outputs.artifacts["out_gro"],
#                    "inp_top": solvation.outputs.artifacts["out_top"], 
#                    "inp_mdp": art_mdp, }
#        )
#
#        wf = Workflow(name="test-enmin-run")
#        wf.add(prep_run)
#        wf.add(solvation)
#        wf.submit()
#
#        assert wf.query_status() == "Succeeded"
#
#        local_dir = "download"
#        download_local(wf, prep_run, local_path=local_dir)
#
#        shutil.rmtree(wf.id, ignore_errors=True)
#        shutil.rmtree("upload", ignore_errors=True)
#        shutil.rmtree(local_dir, ignore_errors=True)

    @pytest.mark.skipif(not has_wf, reason="workflow module not installed")
    def test_FEP_workflow(self):
        inp_file_prefix = resource_filename("metamol", os.path.join("tests", "files", "workflow", "FEP"))
        
        wf, step_list = FEP_workflow(config_file="input_FEP.json", 
                          inp_file_prefix = inp_file_prefix)
        wf.submit()

        assert wf.query_status() == "Succeeded"
        
        local_dir = "download"
        download_local(wf, step_list[-1], local_path=local_dir)

        shutil.rmtree(wf.id, ignore_errors=True)
        shutil.rmtree("upload", ignore_errors=True)
        shutil.rmtree(local_dir, ignore_errors=True)