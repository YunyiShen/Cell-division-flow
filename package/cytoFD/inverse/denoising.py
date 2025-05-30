import gpytorch
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from .datautil import cellPTVdata, form_mesh
from tqdm import tqdm
from .naivevel import velocityGPModel
import numpy as np


def denoisingGP(ptvdata, subsampling_rate = 0.5,
              kernel = velocityGPModel, 
              likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2), 
              n_inducing = 1000, optimizer = torch.optim.Adam,
              lr = 0.1, iter = 300, grid_size_pred = None, predict = False):
    '''
    Denoising function for PTV data using Gaussian Process regression.
    
    Args:
        ptvdata (PTVData): PTVData object containing the data to be denoised.
        subsampling_rate (float): Subsampling rate for the data.
        kernel (gpytorch.models.ExactGP): GP model to be used for denoising.
        likelihood (gpytorch.likelihoods.Likelihood): Likelihood to be used for denoising.
        n_inducing (int): Number of inducing points to be used in the GP model.
        optimizer (torch.optim.Optimizer): Optimizer to be used for training the GP model.
        lr (float): Learning rate for the optimizer.
        iter (int): Number of iterations for the optimizer.
        grid_size_pred (list): Size of the grid for prediction in time, x, y e.g., [5, 70, 70] will generate 5 frames, 70 by 70 each.
        predict (bool): Whether to predict the velocity field or not.
    Returns:
        gp_model: The trained GP model.
    '''
    assert isinstance(ptvdata, cellPTVdata), 'ptvdata must be an instance of PTVData'
    assert subsampling_rate > 0 and subsampling_rate < 1, 'subsampling_rate must be between 0 and 1'

    if grid_size_pred is None:
        grid_size_pred = [ptvdata.nframes, 70, 70]

    X_train, Y_train, X_test, Y_test = ptvdata.get_XY_for_fit(True, subsampling_rate)
    model = kernel(X_train, Y_train, likelihood, out_dims=[1, 2], n_inducing=n_inducing)
    model = model.to(device)
    likelihood = likelihood.to(device)
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    model.train()
    likelihood.train()
    optimizer = optimizer(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    print('Training GP model...')
    for i in tqdm(range(iter)):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, Y_train)
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            print(f'Iter {i + 1}/{iter} - Loss: {loss.item():.3f}')
    
    model.eval()
    likelihood.eval()
    if not predict:
        return model, likelihood

    t_grid = torch.linspace(ptvdata.t_range[0], ptvdata.t_range[1], grid_size_pred[0]).to(device)
    grid_size_pred[0] = 1
    X_mesh = form_mesh(ptvdata.x_range, ptvdata.y_range, ptvdata.t_range, grid_size_pred)  
    X_mesh = X_mesh.to(device)

    res = []
    print('Predicting velocity field on grid...')
    for t in tqdm(t_grid):
        X_mesh[:, 0] = t
        with torch.no_grad():
            pred = likelihood(model(X_mesh))
            mean = pred.mean.cpu().numpy()
            res.append(mean)
    res = np.array(res)
    return model, likelihood, res, X_mesh[:, 1:].cpu().numpy(), t_grid.cpu().numpy()