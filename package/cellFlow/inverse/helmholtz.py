
import torch
import gpytorch
from curlFree import ConstantMeanGradonly, RBFKernelGradonly
from solenoidal import RBFKernelsolspatialGradonly
from gpytorch.kernels import ScaleKernel, InducingPointKernel
from inducing_points import InducingPointKernel2


class Helm2DGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, out_dims=[1, 2]):
        '''
        Idea is to have a vector field that is [df/dy, -df/dx] for a scalar field f
        '''
        assert len(out_dims) == 2, 'out_dims must be a list of length 2 for two dimensional locations'
        super(Helm2DGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMeanGradonly()
        self.covar_module = ScaleKernel(RBFKernelsolspatialGradonly(spatial_dim=out_dims)) + gpytorch.kernels.ScaleKernel(RBFKernelGradonly())
        self.out_dims = out_dims
        self.inverted_dims = [out_dims[1], out_dims[0]]
        self.curlfree_mean_module = ConstantMeanGradonly()
        #self.curlfree_cov_module = gpytorch.kernels.ScaleKernel(RBFKernelGradonly())

    def forward(self, x):
        n1= x.shape[-2]
        #breakpoint()
        # take out the right part of cov
        dimm = x.shape[-1]
        pi1 = (torch.arange(n1).unsqueeze(1)*(dimm) + (torch.tensor(self.out_dims).unsqueeze(0) )).reshape(len(self.out_dims)*n1)  
        #breakpoint()
        mean_x = self.mean_module(x)[...,self.inverted_dims]
        mean_x[..., -1] *= -1
        mean_x += self.curlfree_mean_module(x)[..., self.out_dims]
        covar_x = self.covar_module(x)[..., pi1, :][..., :, pi1] #+ self.curlfree_cov_module(x)[..., pi1, :][..., :, pi1]
        
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    

class spHelm2DGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, out_dims=[1, 2], n_inducing = 1000):
        '''
        Idea is to have a vector field that is [df/dy, -df/dx] for a scalar field f
        '''
        assert len(out_dims) == 2, 'out_dims must be a list of length 2 for two dimensional locations'
        assert n_inducing <= train_x.shape[0], 'n_inducing must be less than the number of training points'
        super(spHelm2DGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMeanGradonly()
        self.covar_module_base = ScaleKernel(RBFKernelsolspatialGradonly(spatial_dim=out_dims)) + gpytorch.kernels.ScaleKernel(RBFKernelGradonly())
        self.covar_module = InducingPointKernel2(self.covar_module_base, inducing_points=train_x[:n_inducing, :].clone(), likelihood=likelihood, out_dims=out_dims)
        self.out_dims = out_dims
        self.inverted_dims = [out_dims[1], out_dims[0]]
        self.curlfree_mean_module = ConstantMeanGradonly()
        #self.curlfree_cov_module = gpytorch.kernels.ScaleKernel(RBFKernelGradonly())

    def forward(self, x):
        #n1= x.shape[-2]
        #breakpoint()
        # take out the right part of cov
        #dimm = x.shape[-1]
        #pi1 = (torch.arange(n1).unsqueeze(1)*(dimm) + (torch.tensor(self.out_dims).unsqueeze(0) )).reshape(len(self.out_dims)*n1)  
        #breakpoint()
        mean_x = self.mean_module(x)[...,self.inverted_dims]
        mean_x[..., -1] *= -1
        mean_x += self.curlfree_mean_module(x)[..., self.out_dims]
        #breakpoint()
        gpytorch.settings.debug._set_state(False) # this is terrible but works, since we only take part of the covariance matrix, some dimension check cannot pass
        covar_x = self.covar_module(x)[:,:]
        gpytorch.settings.debug._set_state(True)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    

