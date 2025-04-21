import scipy.io
import torch
import numpy as np


class cellPTVdata():
    def __init__(self, mat, sampling_rate = .1, pixel_size = 1,#65/1000, 
                 x_range = None, y_range = None, t_range = None, read = True):
        '''
        Args:
            mat (str): path to .mat file
            sampling_rate (float): sampling rate of the data, in 1/s
            pixel_size (float): pixel size of the data in mm
            x_range (list): range of x values to be used
            y_range (list): range of y values to be used
        '''
        super().__init__()
        self.mat = mat
        self.sampling_rate = sampling_rate
        self.pixel_size = pixel_size

        self.data = scipy.io.loadmat(self.mat)
        keys = list(self.data.keys())
        self.data = self.data[keys[-1]] # take the last key
        self.tensor_data = None # will be filled in by read_in_data, [n_particle, 5] 
        self.x_range = x_range
        self.y_range = y_range
        self.ts = None
        self.t_range = t_range

        if read:
            self.read_in_data()


    def read_in_data(self):
        self.nframes = self.data.shape[0]
        n_frames = self.data.shape[0]
        ts = 1. * np.linspace(0, n_frames - 1, n_frames) #/ self.sampling_rate
        self.ts = ts

        frames = self.data[0][0]
        frames[:,0] = ts[0] # set the first time point to 0
        frames[:, 1:] *= self.pixel_size # convert to um
        frames[:, 3:] *= self.sampling_rate # convert to um/s

        for i in range(1, n_frames - 1):
            tmp = self.data[i][0]
            tmp[:,0] = ts[i]
            tmp[:, 1:] *= self.pixel_size
            tmp[:, 3:] *= self.sampling_rate
            frames = np.vstack((frames, tmp))
        
        self.tensor_data = torch.tensor(frames, dtype=torch.float32)

        if self.x_range is not None:
            self.tensor_data = self.tensor_data[(self.tensor_data[:,1] >= self.x_range[0]) & (self.tensor_data[:,1] <= self.x_range[1])]
        if self.y_range is not None:
            self.tensor_data = self.tensor_data[(self.tensor_data[:,2] >= self.y_range[0]) & (self.tensor_data[:,2] <= self.y_range[1])]
        
        if self.t_range is not None:
            self.tensor_data = self.tensor_data[(self.tensor_data[:,0] >= self.t_range[0]) & (self.tensor_data[:,0] <= self.t_range[1])]


        if self.x_range is None:
            self.x_range = [self.tensor_data[:,1].min().item(), self.tensor_data[:,1].max().item()]
        if self.y_range is None:
            self.y_range = [self.tensor_data[:,2].min().item(), self.tensor_data[:,2].max().item()]
        #if self.t_range is None:
        self.t_range = [self.tensor_data[:,0].min().item(), self.tensor_data[:,0].max().item()]

    def get_XY_for_fit(self, permute = True, train_split = .8, standardize_time = False):
        '''
        Args:
            permute (bool): whether to shuffle the data
            train_split (float): proportion of data to be used for training
        Returns:
            X (torch.tensor): [n_particle, 3], t, x and y coordinates
            Y (torch.tensor): [n_particle, 2], x and y velocities
        '''
        assert train_split > 0 and train_split < 1, 'train_split must be between 0 and 1'
        if self.tensor_data is None:
            self.read_in_data()
        
        if permute:
            perm_tensor_data = self.tensor_data[torch.randperm(self.tensor_data.shape[0])]
        else:
            perm_tensor_data = self.tensor_data
        n_train = int(perm_tensor_data.shape[0] * train_split)
        X_train = perm_tensor_data[:n_train, :3]
        Y_train = perm_tensor_data[:n_train, 3:]
        X_test = perm_tensor_data[n_train:, :3]
        Y_test = perm_tensor_data[n_train:, 3:]
        return X_train, Y_train, X_test, Y_test


def form_mesh(x_range, y_range, t_range, out_size = [100,100,100]):
    x = torch.linspace(x_range[0], x_range[1], out_size[1])
    y = torch.linspace(y_range[0], y_range[1], out_size[2])
    t = torch.linspace(t_range[0], t_range[1], out_size[0])
    X, Y, T = torch.meshgrid(x, y, t)
    return torch.stack([T, X, Y], dim=-1).view(-1, 3)



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from celluloid import Camera


# Function to extract the particle positions and velocities at a given time step
def extract_data(tensor, time_step, istensor = False):
    if istensor:
        tensor = tensor.detach().numpy()
    tensor = tensor[tensor[:,0] == time_step, :]
    x = tensor[:, 1]
    y = tensor[:, 2]
    dx = tensor[:, 3]
    dy = tensor[:, 4]
    return x, y, dx, dy

def make_animation(data, t_range, x_range, y_range, 
                   save_path = 'particle_animation.mp4'):
    # Create a figure and axis
    fig, ax1 = plt.subplots(1, 1, figsize=(7, 6))
    camera = Camera(fig)

    # Function to initialize the plot
   
    ax1.set_xlim(x_range[0], x_range[1])  
    ax1.set_ylim(y_range[0], y_range[1])
    for frame in t_range:  
        x, y, dx, dy = extract_data(data, frame)
        #breakpoint()
        ax1.quiver(x, y, dx, dy)
        camera.snap()
    ani = camera.animate()

    # Save the animation
    ani.save(save_path)