# metaDP wrapper that manage deepmd-kit.
from typing import (Iterable, Dict, Type)
import numpy as np
from collections import defaultdict
import os
import glob
import warnings
import json
import yaml

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from deepmd.backend.backend import (
        Backend,
    )
    BACKENDS: Dict[str, Type[Backend]] = Backend.get_backends_by_feature(
        Backend.Feature.ENTRY_POINT
    )
    #TODO: Add backend check to ensure at least one backend (tf or pt) is available. Default: pt.
    from deepmd.utils.argcheck import *
from metamol.utils.help_functions import cd 

VARIANTS = {"descriptor": ("loc_frame", "se_e2_a", "se_e2_r", "se_e3", "se_a_tpe", "hybrid"),
            "fitting_net": ("ener", "dipole", "polar"),
            "modifier": ("dipole_charge"),
            "model compression": ("se_e2_a"),
            "learning_rate": ("exp"),
            "loss": ("ener", "tensor"),
        }

ALIASES = {"se_a": "se_e2_a", "n_axis_neuron": "axis_neuron", "se_at": "se_e3",
           "se_a_3be": "se_e3", "se_t": "se_e3", "se_a_ebd": "se_a_tpe",
           "se_r": "se_e2_r", "n_neuron": "neuron", "pol_type": "sel_type",
           "dipole_type": "sel_type", "auto_prob_style": "auto_prob", 
           "sys_weights": "sys_probs", "numb_batch": "numb_btch", 
           "stop_batch": "numb_steps",
        }

class metaDP(object):
    def __init__(self, INPUT=None, **kwargs):
        self._params = defaultdict(dict)
        if INPUT:
            with open(INPUT) as f:
                self.params = json.load(f)
            self.params = normalize(self._params)
        else:
            self.params = gen_default_params(**kwargs)
        self._data = np.asarray([])

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, p):
        if not isinstance(p, dict):
            raise TypeError("DeepMD Params must be a dictionary")
        self._params = p

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        if not isinstance(d, np.ndarray):
            raise TypeError("DeepMD training data must be a numpy array")
        self._data = d

    def close(self):
        del self

    def clear(self):
        self.params = defaultdict(dict)

    def update(self, new_params):
        """Update the current params dict with new params. (BFS)
        Rule for changing variants: 
        If change to a new variant, need to adopt all params of the new variant.
        """
        new_param_dicts = [new_params]
        ori_param_dicts = [self.params]
        while new_param_dicts:
            temp_new, temp_ori = [], []
            for idx, new_param in enumerate(new_param_dicts):
                ori_param = ori_param_dicts[idx]
                for key in new_param.keys():
                    if key in ALIASES:
                        alias_key = ALIASES[key]
                        new_param[alias_key] = new_param[key]
                        key = alias_key
                    #key = ALIASES.get(key, key)
                    if key not in ori_param or ori_param[key] is None \
                    or not isinstance(new_param[key], dict):
                        ori_param[key] = new_param[key]
                    elif key in VARIANTS and "type" in new_param[key]:
                        new_var_type = new_param[key]["type"]
                        new_var_type = ALIASES.get(new_var_type, new_var_type)
                        ori_var_type = ori_param[key]["type"]
                        if new_var_type != ori_var_type:
                            if new_var_type not in VARIANTS[key]:
                                raise KeyError("{0} is not a valid variant for {1} type".format(new_param[key]["type"], key))
                            #print(key, new_param[key])
                            ori_param[key] = new_param[key]
                    else:
                        temp_new.append(new_param[key])
                        temp_ori.append(ori_param[key])
            new_param_dicts, ori_param_dicts = temp_new, temp_ori
        self.params = normalize(self.params)
        
    def delete(self, params):
        """Delete parameter fields.
        params: List of parameters to delete.
                format: majorfiled:subfield1:subfield2:subfield3
        """
        if isinstance(params, str):
            params = [params]
        if not isinstance(params, Iterable):
            raise TypeError("Parameters to delete must be a collection of strings")

        for param in params:
            d = self.params
            key_not_found = False
            fields = param.split(":")
            for field in fields[:-1]:
                if field not in d: 
                    key_not_found = True
                    break
                d = d[field]
            if fields[-1] not in d: key_not_found = True
            if key_not_found:
                warnings.warn("Parameter {0} not found in original parameter dict.".format(param))
                continue
            del d[fields[-1]]
                
    def read_input(self, filename: str):
        if filename.endswith('json'):
            with open(filename) as fp:
                self.params = json.load(fp)
        elif filename.endswith('yaml'):
            with open(filename) as fp:
                self.params = yaml.safe_load(fp)
        else:
            raise TypeError("Input file must be either json or yaml format")

    def write_input(self, filename: str):
        if filename.endswith('json'):
            with open(filename, 'w') as f:
                json.dump(self.params, f, indent=4)
        elif filename.endswith('yaml'):
            with open(filename, 'w') as f:
                yaml.dump(self.params, f, default_flow_style=False,
                  sort_keys=False)
        else:
            raise TypeError("Input file to write must be either json or yaml format")

    def train(self, 
              work_dir=None,
              backend='pt',
              INPUT=None, 
              init_model=None,
              restart=None,
              output="out.json",
              init_frz_model=None,
              mpi_log="master",
              log_level=2,
              log_path=None,
              fine_tune=None,
              model_branch=None,
              force_load=False,
              skip_neighbor_stat=False,
              #is_compress=False,
              keep_output=True,
              freeze_model=False,
              **kwargs,
    ):
        deepmd_main = BACKENDS[backend]().entry_point_hook
       # Go to the working directory if specified else stay in the current dir.
        if work_dir is None:
            work_dir = os.getcwd()

        with cd(work_dir):
            keep_input = True
            if not INPUT: 
                keep_input = False
                self.write_input('in.json')
                INPUT = 'in.json'
            ARGS = ['train']
            # Input arg
            ARGS.append(INPUT)

            # verbose arg
            if 0<=log_level<=3: 
                ARGS.append('-v')
                ARGS.append(str(log_level))
            else:
                raise ValueError("dp trainning log level must in the range of [0, 3].")

            # log path arg
            if log_path:
                ARGS.append('-l')
                ARGS.append(log_path)

            # mpi-log arg
            ARGS.append('-m')
            ARGS.append(mpi_log)

            # init-model arg
            if init_model:
                ARGS.append('-i')
                ARGS.append(init_model)

            # restart arg
            if restart:
                ARGS.append('-r')
                ARGS.append(restart)            
            
            # init-frz-model arg
            if init_frz_model:
                if backend == 'pt' or backend == 'pytorch':
                    init_frz_model += '.pth'
                elif backend == 'tf' or backend == 'tensorflow':
                    init_frz_model += '.pb'
                else:
                    raise ValueError("Unknown backend.")
                
                ARGS.append('-f')
                ARGS.append(init_frz_model)

            # fine-tune arg
            if fine_tune:
                ARGS.append('-t')
                ARGS.append(fine_tune) 
            
            # output arg
            ARGS.append('-o')
            ARGS.append(output)

            # model-branch arg
            if model_branch:
                ARGS.append('--model-branch')
                ARGS.append(fine_tune) 

            if force_load:
                ARGS.append('--force-load')
            if skip_neighbor_stat:
                ARGS.append('--skip-neighbor-stat')
            
            # start training process
            deepmd_main(ARGS)

            if freeze_model: 
                fout = kwargs.get("fout", "frozen_model")
                #node_names = kwargs.get("node_names", None)
                self.freeze(backend=backend, 
                            log_level=log_level, 
                            log_path=log_path, 
                            output=fout, )
                print("Frozen model is save as {0}".format(os.path.join(work_dir, fout)))
            
            # Read training data
            if init_model is None and init_frz_model is None:
                self.data = np.genfromtxt("lcurve.out", names=True)
            else:
                data_curr = np.genfromtxt("lcurve.out", names=True)
                data_curr["step"] += max(self.data["step"])
                data_all = np.append(self.data, data_curr[1:])
                self.data = data_all

            # Clean up files
            if not keep_input:
                os.remove(INPUT)
            if not keep_output:
                out_files = list()
                out_files += glob.glob("model.ckpt*")
                out_files += glob.glob(output)
                out_files += glob.glob("lcurve.out")
                out_files += glob.glob("checkpoint")
                for out_file in out_files:
                    os.remove(out_file)

    def freeze(self,
              backend='pt',
              log_level=2,
              log_path=None,
              checkpoint_folder=".", 
              output='frozen_model',
              #node_names=None,
              head=None,
    ):
        deepmd_main = BACKENDS[backend]().entry_point_hook
        if backend == 'pt' or backend == 'pytorch':
            output += '.pth'
        elif backend == 'tf' or backend == 'tensorflow':
            output += '.pb'
        else:
            raise ValueError("Unknown backend.")
        
        ARGS = ['freeze']

        # verbose arg
        if 0<=log_level<=3: 
            ARGS.append('-v')
            ARGS.append(str(log_level))
        else:
            raise ValueError("dp freeze log level must in the range of [0, 3].")

        # log path arg
        if log_path:
            ARGS.append('-l')
            ARGS.append(log_path)
        
        # checkpoint-folder arg
        ARGS.append('-c')
        ARGS.append(checkpoint_folder)

        # output arg
        ARGS.append('-o')
        ARGS.append(output)

        # # node-names arg
        # if node_names:
        #     ARGS.append('-n')
        #     ARGS.append(node_names)

        # head arg
        if head:
            ARGS.append('--head')
            ARGS.append(head)

        deepmd_main(ARGS)

    # def test(self,
    #         work_dir=None,
    #         model="frozen_model.pb",
    #         system=".",
    #         set_prefix="set",
    #         numb_test=100,
    #         rand_seed=None,
    #         shuffle_test=False,
    #         detail_file=None,
    #         atomic=False,
    # ):
    #     tf.compat.v1.reset_default_graph()
    #     if work_dir is None:
    #         work_dir = os.getcwd()
            
    #     with cd(work_dir):
    #         if not os.path.exists(model):
    #             raise FileNotFoundError("Model file {0} not found on disk".format(model))
    #         test(model=model, system=system, set_prefix=set_prefix,
    #             numb_test=numb_test, rand_seed=rand_seed, shuffle_test=shuffle_test,
    #             detail_file=detail_file, atomic=atomic,)
    
    def infer(self, system, backend='pt', model="frozen_model"):
        from deepmd.infer import DeepPot
        from metamol.utils.convert_formats import box_to_vectors

        if backend == 'pt' or backend == 'pytorch':
            model += '.pth'
        elif backend == 'tf' or backend == 'tensorflow':
            model += 'pb'
        else:
            raise ValueError("Unknown backend.")
        
        dp_model = DeepPot(model)
        # Change after adding frames in meta.System class
        coord = system.xyz.reshape([1, -1])
        if system.box is None:
            box_lengths = system.get_boundingbox()
        else:
            box_lengths = system.box.lengths
            
        box_angle = system.box.angle

        cell = box_to_vectors(np.asarray(box_lengths), np.asarray(box_angle)).reshape([1, -1])
        at_dict = dict()
        for atom in system.atoms:
            if atom.symbol in at_dict:
                continue
            at_dict[atom.symbol] = len(at_dict)
        atype = [at_dict[atom.symbol] for atom in system.atoms]

        e, f, v = dp_model.eval(coord, cell, atype)
        return e, f, v

    def plot_lr(self, scale="log"):
        import matplotlib.pyplot as plt
        plt.plot(self.data["step"], self.data["lr"])
        plt.xlabel('Step', fontsize=15)
        plt.ylabel('Learning Rate', fontsize=15) 
        if scale == "log":
            plt.xscale('symlog')
            plt.yscale('log')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.show()
        
    def plot_loss(self, cols="all", scale="log"):
        import matplotlib.pyplot as plt
        if cols=="all":
            cols = self.data.dtype.names[1:-1]
        elif isinstance(cols, str):
            cols = [cols]
        elif not isinstance(cols, Iterable):
            raise TypeError("The loss colmuns must be an iterable object")

        for col in cols:
            if col not in self.data.dtype.names[1:-1]:
                raise ValueError("Loss column {0} not found in training data",format(col))
            plt.plot(self.data["step"], self.data[col], label=col)
        plt.legend(loc='best', fontsize=10)
        plt.xlabel('Step', fontsize=15)
        plt.ylabel('Loss', fontsize=15) 
        if scale == "log":
            plt.xscale('symlog')
            plt.yscale('log')
        plt.xticks(fontsize=15)  
        plt.yticks(fontsize=15) 
        plt.show()         

def gen_default_params(**kwargs):
    descriptor = kwargs.get("descriptor", "se_e2_a")
    systems = kwargs.get("systems", ["."])
    numb_steps = kwargs.get("numb_steps", 1000000)

    params = {"model": {"type_map": ["O","H"], "descriptor": {"type": descriptor}, 
              "fitting_net": {}},
              "learning_rate": {}, "loss": {},
              "training": {"training_data":{"systems": systems}, 
              "numb_steps": numb_steps}, 
        }

    params = normalize(params)
    return params
    