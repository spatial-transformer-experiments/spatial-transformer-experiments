import torch
import torch.nn as nn
import torch.nn.functional as F
from coordconv import CoordConv2d
import kornia

######################################################################
# Depicting spatial transformer networks
# --------------------------------------
#
# Spatial transformer networks boils down to three main components :
#
# -  The localization network is a regular CNN which regresses the
#    transformation parameters. The transformation is never learned
#    explicitly from this dataset, instead the network learns automatically
#    the spatial transformations that enhances the global accuracy.
# -  The grid generator generates a grid of coordinates in the input
#    image corresponding to each pixel from the output image.
# -  The sampler uses the parameters of the transformation and applies
#    it to the input image.
#
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

######################################################################
#Implement CoordConv and first experiment ideas
# allows to use CoordConv on the localisation network or the classifier network or both
# further the localisation can be by passed to determine the localisation of the 


class Net_CoordConv(nn.Module):
    def __init__(self,use_coordconf_localisation=False,use_coordconf_classifier=False,bypass_localisation=False):
        super(Net_CoordConv,self).__init__()
        self.use_coordconf_localisation=use_coordconf_localisation
        self.use_coordconf_classifier=use_coordconf_classifier
        self.bypass_localisation=bypass_localisation
        if use_coordconf_classifier:
            self.conv1 = CoordConv2d(1, 10, kernel_size=5,use_cuda=False)
            self.conv2 = CoordConv2d(10, 20, kernel_size=5,use_cuda=False)
        else:    
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        if bypass_localisation:
            self.localization = None
        elif use_coordconf_localisation:         
            self.localization = nn.Sequential(
                CoordConv2d(1, 8, kernel_size=7,use_cuda=False),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                CoordConv2d(8, 10, kernel_size=5,use_cuda=False),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
            )
        else:
            self.localization = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
            )
        


        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        if self.bypass_localisation:
            return x
        else:
            xs = self.localization(x)
            xs = xs.view(-1, 10 * 3 * 3)
            theta = self.fc_loc(xs)
            theta = theta.view(-1, 2, 3)

            grid = F.affine_grid(theta, x.size())
            x = F.grid_sample(x, grid)

            return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net_CoordConv_Homography(Net_CoordConv):
    def __init__(self,use_coordconf_localisation=False,use_coordconf_classifier=False,bypass_localisation=False):
        super(Net_CoordConv_Homography, self).__init__(use_coordconf_localisation=use_coordconf_localisation,use_coordconf_classifier=use_coordconf_classifier,bypass_localisation=bypass_localisation)
        
        
        # Overrite attributes for homography

        # Regressor for the 3 * 3 homography matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 3)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float))

    def stn(self, x):
        if self.bypass_localisation:
            return x
        else:
            xs = self.localization(x)
            xs = xs.view(-1, 10 * 3 * 3)
            theta = self.fc_loc(xs)
            theta = theta.view(-1, 3, 3)
            theta = torch.nn.functional.normalize(theta,dim=0)
            x = torch.nn.functional.normalize(x,dim=0)
            x = kornia.geometry.transform.warp_perspective(x, theta, dsize=(x.shape[-2],x.shape[-1]),)

            return x