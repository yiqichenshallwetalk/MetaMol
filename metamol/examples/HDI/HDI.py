import metamol as meta
from metamol.lib.fragments.monomers import Monomer
from metamol.lib.molecules.polymer import Polymer

#构建PFPE(全氟聚醚)润滑剂分子
#头基团
headgroup = meta.Molecule('OCC(O)COCC(F)(F)C(F)(F)O', smiles=True)
#将第14号O原子标记为连接点(tail), 并移除24号H原子(remove_atoms)
headgroup = Monomer(headgroup, tail=14, remove_atoms=[24])

#使用同样的方法构建尾基团
tailgroup = meta.Molecule('OCC(O)COCC(F)(F)C(F)(F)', smiles=True)
tailgroup = Monomer(tailgroup, tail=11, remove_atoms=[23])

#构建重复单元
repunit = meta.Molecule('C(F)(F)C(F)(F)C(F)(F)O', smiles=True)
repunit = Monomer(repunit, head=1, tail=10, remove_atoms=[11, 12])

#组合成全氟聚醚分子
PFPE = Polymer(monomers=[repunit], name='PFPE', head=headgroup, tail=tailgroup)
#此例中，将重复单元N设置成3
PFPE.build(N=3)
PFPE.embed()

#构建HDI体系，首先添加10个PFPE分子并分配分子力场参数
HDI_sys = meta.System(PFPE, 10, box=[40.0, 40.0, 40.0])
HDI_sys.parametrize(forcefield_name='gaff')

#生成初始构象
HDI_sys.initial_config(region=(40.0, 40.0, 15.0))

#构建磁盘表面模型
COC_surf = meta.Lattice(
        spacings=[3.567, 3.567, 3.567],
        langles=[90.0, 90.0, 90.0])
locations = [[0., 0., 0.], [0., 0.5, 0.5], 
            [0.5, 0, 0.5], [0.5, 0.5, 0.],
            [0.75, 0.75, 0.75], [0.75, 0.25, 0.25], 
            [0.25, 0.75, 0.25], [0.25, 0.25, 0.75]]

C = meta.Atom(symbol='C')
occupy_points = {C: locations}
COC_surf.clear()
COC_surf.occupy(occupy_points)
COC_surf.replicate(x=12, y=12, z=2)

#为磁盘表面模型分配力场参数         
from metamol.utils.ffobjects import AtomType
COC_atom = AtomType(name='coc_custom', atomic=6, symbol='C', sigma=3.39967, epsilon=0.086, mass=12.01)
COC_params = {'atom_type': [COC_atom]}
COC_surf.parametrize(custom=True, parameters=COC_params)

#搭建HDI全模型, 上下边缘皆放置一层COC
HDI_sys.append_surface(COC_surf, location='bottom')
HDI_sys.append_surface(COC_surf, location='top')

#将HDI体系储存为lammps数据文件
HDI_sys.save('HDI.data')
#创建metaLammps对象来管理和运行lammps
from metamol.metaLammps import metaLammps
mlmp = metaLammps()
#读入lammps 指令文件并运行lammps
mlmp.file('HDI.in')
#当前案例中使用4个MPI processes在CPU上进行模拟。如需使用GPU模拟，可设定gpu=True，gpu库默认为kokkos，可通过gpu_backend='gpu'修改为gpu库。
mlmp.launch(out_file='out.dat', mpi=True, nprocs=4, gpu=False)

import matplotlib.pyplot as plt
#模拟数据读取和处理
#温度随步数的变化趋势
temp_out = mlmp.get_thermo('c_T')
temp_out.plot(x='Step', y='c_T', kind='scatter')
plt.xlabel('Step', fontsize=15)
plt.ylabel('Temperature(K)', fontsize=15)
plt.savefig('results/temperature.jpg', dpi=300)
temp_out.to_csv('results/temperature.csv')

#体系势能随步数的变化趋势
pe_out = mlmp.get_thermo('PotEng')
pe_out.plot(x='Step', y='PotEng', kind='scatter')
plt.xlabel('Step', fontsize=15)
plt.ylabel('Potential Energy(Kcal/mole)', fontsize=15)
plt.savefig('results/pe.jpg', dpi=300)
pe_out.to_csv('results/potential_energy.csv')

#清理模拟输出文件
mlmp.close()
import os
onlyfiles = [f for f in os.listdir() if os.path.isfile(f)]
for f in onlyfiles:
    if f not in ['HDI.py', 'HDI.data', 'HDI.in', 'HDI.ipynb']:
        os.remove(f)
