import numpy as np 
from cellFlow.forward.solver import naivesolver_with_extra_hydrostatic, chorin_projection_with_extra_hydrostatic
import matplotlib.pyplot as plt
from cellFlow.forward.plot_utils import velocity_animation


def concentration(x,y,t, scale_x = 0.1, scale_y = 0.4, scale_t = 0.1, t0 = -0.01):
    local_t = max( t - t0, 0)
    gau = 1 - ( min((local_t)/scale_t,1)) * np.exp(-.5 * ((x / scale_x)**2+np.abs(y / scale_y)**2))
    gau *= (gau >= 0)
    return gau


dt = 0.001
nt = 500
dx = 0.05
dy = 0.05
x = np.arange(-3, 3, dx)
y = np.arange(-3, 3, dy)
X, Y = np.meshgrid(x, y)
mask = 1 * ((X/1.) ** 2 + Y ** 2 < 2.4 ** 2)
#mask = 1 * (np.logical_and(np.logical_and(X > -2, X < 2), np.logical_and(Y > -2, Y < 2)))

u = np.zeros_like(X)
v = np.zeros_like(X)
p = np.zeros_like(X)
b = np.zeros_like(X)


u, v, p, stress_inner, \
u_save, v_save, pressure_save,\
stress_inner_save = \
            naivesolver_with_extra_hydrostatic(nt = nt, u = u, v = v, 
               dt = dt, dx = dx, 
               dy = dy, mask = mask,
                X = X, Y = Y, hydro_stress = concentration,
                upwind = True,
               rho = 1, nu = 0.6, save_every = 2, save_from = 150,
               kwargs = {"tol": 1e-10, "maxit": 1000})
#naivesolver_with_extra_hydrostatic(nt = nt, u = u, v = v, 
#chorin_projection_with_extra_hydrostatic(nt = nt, u = u, v = v,
# plot size
stress_inner[np.logical_not( mask)] = np.nan
plt.figure(figsize=(7.2, 6))
plt.contourf(X, Y, p-stress_inner, alpha=0.5, cmap='viridis')  
cbar = plt.colorbar()
cbar.set_label('Hydrostatic stress', rotation=270, labelpad=15)
# plotting velocity field

plt.quiver(X[::5, ::5], 
           Y[::5, ::5], 
           u[::5, ::5], 
           v[::5, ::5], scale = v.std() * 60, color = "blue", width = 0.005) 
'''
plt.streamplot(X, Y, u, v, color='blue')
'''

plt.xlabel('X')
plt.ylabel('Y')

plt.show()
plt.savefig("hydrostatic.png", dpi=500)
plt.close()


velocity_animation(X,Y,u_save,v_save,pressure_save,stress_inner_save,mask)