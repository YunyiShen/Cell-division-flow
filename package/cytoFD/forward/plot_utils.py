import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

def velocity_animation(X,Y,u_save,v_save, 
                       p_save,
                       stress_inner_save, 
                       mask, save_loc = "quiver_animation.mp4"):
    fig, ax = plt.subplots()
    #ax.contourf(X, Y, stress_inner_save[0], alpha=0.5, cmap='viridis')
    min_inner = np.nanmin(p_save[0] - stress_inner_save[0])
    max_inner = np.nanmax(p_save[0] - stress_inner_save[0])
    stress_inner_save[0][np.logical_not(mask)] = np.nan
    global contour 
    contour = ax.contourf(X, Y, p_save[0] - stress_inner_save[0], 
                          alpha=0.5, cmap='viridis')
    quiver = ax.quiver(X[::5, ::5], Y[::5, ::5], 
              u_save[0][::5, ::5], 
              v_save[0][::5, ::5], 
              scale = v_save[-1].std() * 80, color = "blue", width = 0.005)
    def update(frame):
        global contour
        for coll in contour.collections:
            coll.remove()
        stress_inner_save[frame][np.logical_not(mask)] = np.nan
        contour = ax.contourf(X, Y, p_save[frame] - stress_inner_save[frame], alpha=0.5, cmap='viridis')
        quiver.set_UVC(u_save[frame][::5, ::5], v_save[frame][::5, ::5])
        return quiver,*contour.collections
    anim = FuncAnimation(fig, update, frames=len(u_save), blit=True)
    anim.save(save_loc, writer='ffmpeg', fps=20)
    plt.show()
    plt.close()
    return anim
