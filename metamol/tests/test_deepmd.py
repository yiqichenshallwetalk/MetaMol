import pytest
import os
from pkg_resources import resource_filename
from distutils.spawn import find_executable

import metamol as meta
from metamol.utils.help_functions import save_remove

try:
    from metamol.engines.metaDP import metaDP
    has_deepmd = True
except (ModuleNotFoundError or ImportError):
    has_deepmd = False

LMP = find_executable("lmp_serial") or find_executable("lmp_mpi") or find_executable("lmp")

class Test_DeePMD:

    @pytest.mark.skipif(not has_deepmd, reason="DeepMD module not installed")
    def test_train(self):
        # Train the metaDP class with descriptors se_e2_a/se_e2_r
        work_dir = resource_filename("metamol", os.path.join("tests", "files", "deepMD", "water"))
        mdp = metaDP(INPUT=work_dir+'/input_torch.json')
        new_params ={"loss": {"type": "ener", "start_pref_e": 0.06, "limit_pref_e": 1},
              "training": {"training_data": 
              {"systems": ["data/data_0/", "data/data_1/", "data/data_2/"]},
                           "validation_data": {"systems": ["data/data_3/"]},
                           "mixed_precision": {
                                "output_prec": "float32",
                                "compute_prec": "float16"},
                           "disp_freq": 10, "save_freq": 100, "numb_steps": 400},
             }

        mdp.update(new_params)
        #mdp.delete("training:stat_file")

        output = "test_out.json"
        fout = "frozen_model"

        # Train the model from scratch
        mdp.train(work_dir=work_dir, output=output, log_level=3, skip_neighbor_stat=True,
                keep_output=True, freeze_model=False, fout=fout)
        
        # Re-initialize training from checkpoint
        mdp.train(work_dir=work_dir, output=output, log_level=3, init_model="model.ckpt.pt", 
                keep_output=True, freeze_model=True, fout=fout)
        
        # Continue training from frozen model
        mdp.train(work_dir=work_dir, output=output, log_level=3, init_frz_model=fout, 
                keep_output=True, freeze_model=True, fout=fout)

        # Change mdp params and start training from scratch
        mdp.update({"training": {"numb_steps": 800}})
        mdp.delete("training:mixed_precision")
        mdp.train(work_dir=work_dir, output=output, log_level=3, 
                keep_output=False, freeze_model=True, fout=fout)

        #INPUT='input_torch.json' 
        save_remove(os.path.join(work_dir, fout+".pth"))
        
        return

    @pytest.mark.skipif(not has_deepmd, reason="DeepMD module not installed")
    def test_infer(self):
        # Train the metaDP class with descriptors se_e2_a
        work_dir = resource_filename("metamol", os.path.join("tests", "files", "deepMD", "water"))

        mdp = metaDP(INPUT=work_dir+'/input_torch.json')
        new_params ={"loss": {"type": "ener", "start_pref_e": 0.06, "limit_pref_e": 1},
              "training": {"training_data": 
              {"systems": ["data/data_0/", "data/data_1/", "data/data_2/"]},
                           "validation_data": {"systems": ["data/data_3/"]},
                           "disp_freq": 10, "save_freq": 100, "numb_steps": 400},
             }

        mdp.update(new_params)

        output = "test_out.json"
        fout = "frozen_model"
        # Train the model from scratch
        mdp.train(work_dir=work_dir, output=output, log_level=3, 
                keep_output=False, freeze_model=True, fout=fout)

        # create water system
        water = meta.SPCE()
        sys_water = meta.System(water, dup=5, box=[10.0, 10.0, 10.0], box_angle=[90.0, 90.0, 90.0])
        sys_water.initial_config(seed=54321)

        # compute energy, force and virial of the system by inference of the trained model.
        e, f, v = mdp.infer(sys_water, model=os.path.join(work_dir, fout))

        sys_water.initial_config(seed=12345)
        e_new, f_new, v_new = mdp.infer(sys_water, model=os.path.join(work_dir, fout))

        assert e != e_new
        assert (f != f_new).all()
        assert (v != v_new).all()

        save_remove(os.path.join(work_dir, fout+".pth"))
        save_remove(os.path.join(work_dir, "se_e2_a"))

        return

    @pytest.mark.skipif(not has_deepmd, reason="DeepMD module not installed")
    @pytest.mark.skipif(not LMP, reason="Lammps package not installed")
    def test_lmp(self, gpu, gpu_backend):
        # Train the metaDP class with descriptors se_e2_a
        work_dir = resource_filename("metamol", os.path.join("tests", "files", "deepMD", "water"))

        mdp = metaDP(INPUT=work_dir+'/input_torch.json')
        new_params ={"loss": {"type": "ener", "start_pref_e": 0.06, "limit_pref_e": 1},
              "training": {"training_data": 
              {"systems": ["data/data_0/", "data/data_1/", "data/data_2/"]},
                           "validation_data": {"systems": ["data/data_3/"]},
                           "disp_freq": 10, "save_freq": 100, "numb_steps": 400},
             }

        mdp.update(new_params)
        #mdp.delete("training:stat_file")

        output = "test_out.json"
        fout = "frozen_model"
        # Train the model from scratch
        mdp.train(work_dir=work_dir, output=output, log_level=3, skip_neighbor_stat=True,
                keep_output=False, freeze_model=True, fout=fout)
        # create water system
        water = meta.SPCE()
        water_sys = meta.System(water, dup=20, box=[20.0, 20.0, 20.0], box_angle=[90.0, 90.0, 90.0])
        water_sys.initial_config(seed=54321)
        water_sys.save(os.path.join(work_dir, "water.lmp"), atom_style='atomic', unit_style='metal')

        from metamol.engines.metaLammps import metaLammps
        mlmp = metaLammps()
        mlmp.file(os.path.join(work_dir, "in.plugin.lammps"))
        mlmp.delete("mass")

        mlmp.launch(work_dir=work_dir, gpu=gpu, gpu_backend=gpu_backend)

        save_remove(os.path.join(work_dir, "water.lmp"))
        save_remove(os.path.join(work_dir, 'lammps_log.txt'))
        save_remove(os.path.join(work_dir, "log.lammps"))
        save_remove(os.path.join(work_dir, fout+".pth"))

        return
