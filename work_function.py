import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams 
from matplotlib import axes

flatten = lambda t: [item for sublist in t for item in sublist]

# USAGE: python [cube_file] [e_fermi] [a] [b]

# Conversion factors
au2ev = 27.211386245989
bohr2ang = 0.529177

# Read in command line arguments
filename = sys.argv[1]
e_fermi = float(sys.argv[2])*au2ev
a = float(sys.argv[3])
b = float(sys.argv[4])

# Read in data from cube file containing potential
file = open(filename)
data = file.readlines()
file.close()

# Get grid info
n_atoms, x_origin, y_origin, z_origin = data[2].split()
n_atoms = int(n_atoms)
origin = [float(x_origin), float(y_origin), float(z_origin)]

n_vox_x, x1, x2, x3 = data[3].split()
n_vox_x = int(n_vox_x)
x_axis = [float(x1), float(x2), float(x3)]

n_vox_y, y1, y2, y3 = data[4].split()
n_vox_y = int(n_vox_y)
y_axis = [float(y1), float(y2), float(y3)]

n_vox_z, z1, z2, z3 = data[5].split()
n_vox_z = int(n_vox_z)
z_axis = [float(z1), float(z2), float(z3)]

x_grid = bohr2ang*np.arange(0, n_vox_x * x_axis[0], x_axis[0])
y_grid = bohr2ang*np.arange(0, n_vox_y * y_axis[1], y_axis[1])
z_grid = bohr2ang*np.arange(0, n_vox_z * z_axis[2], z_axis[2])

# Read potential into array - TO DO: use pandas?
hart_start = n_atoms + 6
hartree = []
for d in data[hart_start:len(data)]:
    dsplit = d.split()
    hartree.append( [ float(i) for i in dsplit ] )
hartree = flatten(hartree)
hartree = np.reshape(hartree, (n_vox_x, n_vox_y, n_vox_z))

# Average over z-direction and save to file
hartree_z = au2ev*np.mean(hartree, (0,1))
pd.DataFrame(np.column_stack( (z_grid, hartree_z) ) ).to_csv('vh_z.dat',header = False, index = False)

# Plot potential
params = { 
    'font.family': 'sans-serif',
    'font.serif': 'Arial Narrow',
    'axes.labelsize': 18,
    'axes.xmargin': 0,
    'legend.fontsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    'figure.figsize': [6.0, 6.0]
    }
rcParams.update(params)

fig,ax=plt.subplots()
ax.plot(z_grid, hartree_z)
ax.set_xlabel(r'$z$ $(\mathrm{\AA})$')
ax.set_ylabel(r'$V_H$ (eV)')
fig.tight_layout()
fig.savefig('vh_z',dpi=600)

# Compute work function phi
v_h_filename = 'hartree_z.dat'
v_h_df = pd.read_csv(v_h_filename, header=None)
z = v_h_df.iloc[:,0].to_list()
v_h = v_h_df.iloc[:,1].to_list()
v_h_avg = np.mean( [ v for j,v in enumerate(v_h) if z[j] >= a and z[j] <= b ] )
phi = v_h_avg - e_fermi
print(phi)
