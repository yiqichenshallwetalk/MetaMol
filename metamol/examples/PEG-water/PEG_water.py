import metamol as meta
from metamol.lib.fragments.monomers import CH3, PEGMonomer 
from metamol.lib.molecules.polymer import Polymer 
from metamol.lib.molecules import water

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 构建聚乙二醇单体 
pegmonomer = PEGMonomer()

#构建聚乙二醇高分子
ch3 = CH3()
PEG = Polymer(name='PEG', monomers=pegmonomer, seq='A', head=ch3, tail=ch3)
PEG.build(N=5)
PEG.embed()

#构建聚乙二醇-水体系, 包含10个PEG和100个水分子
water_spce = water.SPCE()
PEG_water_sys = meta.System([PEG, water_spce], dup=[5, 50], box=[20, 20, 20])

#分配力场参数并生成初始构象
PEG_water_sys.parametrize(forcefield_name='opls')
PEG_water_sys.initial_config()

#保存构象为gromacs输入格式
PEG_water_sys.save('PEG_water.top')
PEG_water_sys.save('PEG_water.gro')

#通过metaGromacs进行Gromacs模拟
from metamol.metaGromacs import metaGromacs
mgro = metaGromacs()
# 设置Gromacs模拟参数
mgro.command('title = dppc')
mgro.command('cpp = /lib/cpp')
mgro.command('integrator = md')
mgro.command('nsteps = 2000')
mgro.command('nstlist = 10')
mgro.commands_list(['nstfout = 0', 'nstxout = 0', 'nstvout = 0', 'nstxtcout = 0', 'nstlog = 0'])
mgro.command('dt = 0.001')
mgro.command('constraints = hbonds')
mgro.command('nstenergy = 50')
mgro.command('ns_type = grid')
mgro.command('coulombtype = PME')
mgro.commands_list(['rlist = 0.8', 'rvdw = 0.8', 'rcoulomb = 0.8', 'tcoupl = v-rescale', 'tc_grps = system'])
mgro.command('tau_t = 0.1')
mgro.command('ref_t = 300')
mgro.command('fourier_spacing = 0.125')
mgro.command('nstcalcenergy = 50')
mgro.command('cutoff-scheme = verlet')
#开始Gromacs模拟
mgro.grompp(gro_file='PEG_water.gro', top_file='PEG_water.top')
mgro.mdrun()

#通过metaLammps进行Lammps模拟
#保存构象为lammps输入格式(data file)
PEG_water_sys.save('PEG_water.data')

#创建metaLammps对象来管理和运行lammps
from metamol.metaLammps import metaLammps
mlmp = metaLammps()

#读入lammps 指令文件并运行lammps
mlmp.file('PEG_water.in')
mlmp.launch(output='out.dat', mpi=True, nprocs=4, gpu=False)

#模拟数据读取和处理
#温度随步数的变化趋势
temp_out = mlmp.get_thermo('Temp')
temp_out.plot(x='Step', y='Temp', kind='scatter')
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
    if f not in ['PEG_water.py', 'PEG_water.in', 'PEG_water.ipynb']:
        os.remove(f)
