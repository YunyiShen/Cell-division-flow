import numpy as np 
import gpytorch
from cytoFD.inverse.solenoidal import spSolenoidal2DGPModel, Solenoidal2DVelocityGPModel
from cytoFD.inverse.denoising import denoisingGP
from cytoFD.inverse.datautil import cellPTVdata
from cytoFD.inverse.naivevel import spvelocityGPModel
import torch

ptvdata = cellPTVdata("../../newdata/xyuv-cen_files/20241206_ctrl003_maxIP_xyuv_dt=10s.mat", 
                  sampling_rate = .1, pixel_size = 65/1000, read = True)


model, likelihood, res, \
X_mesh, t_mesh = denoisingGP(ptvdata, kernel = Solenoidal2DVelocityGPModel,
                             subsampling_rate = 0.08, predict = True, iter = 300)


torch.save(model.state_dict(), f'./res/20241206_inducing_model_state_Sol.pth')
torch.save(likelihood.state_dict(), f'./res/20241206_inducing_likelihood_state_Sol.pth')
np.savez(f'./res/20241206_inducing_res_Sol.npz', 
         denoised = res, X_mesh = X_mesh, t_mesh = t_mesh)

