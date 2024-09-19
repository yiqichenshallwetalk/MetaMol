MetaMol:
A molecular simulation workflow management tool developed by Yiqi Chen, researcher at MetaX.

# Installation of Metamol via conda
## 1.Create anaconda virtual environment
```bash
cd ${metamol_dir}
conda env create -f environment.yml
conda env list
``` 
After the installation of all packages, make sure that the metamol env is in the list.

## 2.Install MetaMol
```bash
cd ${metamol_dir}
conda activate metamol
pip3 install .
```
To verify the success of installation, please run:
```bash
python3 -c "import metamol"
```
## 3.Run Unit Tests
```bash
cd ${metamol_dir}/metamol
pytest (--gpu) -vv
```
If installed correctly, 100% of tests should be passed. 
