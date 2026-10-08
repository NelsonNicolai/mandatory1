import numpy as np
import sympy as sp
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
import sympy as sp
from scipy import sparse

from Wave2D import Wave2D
from Wave2D import Wave2D_Neumann


#class initialization
solver = Wave2D_Neumann()

#constants defined
N  = 50
Nt = 1000
cfl= 1 / (2**0.5)
c  = 1
mx = 2
my = 2
store_data = 1

#mesh initialization
xij,yij = solver.create_mesh(N)

#solving
sol_neumann = solver(N,Nt,cfl,c,mx,my,store_data)

#animation
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
frames = []
for n, val in sol_neumann.items():
    frame = ax.plot_wireframe(xij, yij, val, rstride=2, cstride=2);
    #frame = ax.plot_surface(xij, yij, val, vmin=-0.5*data[0].max(),
    #                        vmax=data[0].max(), cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    frames.append([frame])

ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True,
                                repeat_delay=1000)

ani.save("NeumannWaveMovie.gif")

print("all done")