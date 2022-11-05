
import numpy as np

import torch
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.nn.functional as F           # layers, activations and more


config = [
    (7, 64, 2),
    "m",
    (3, 192),
    "m",
    (1, 128),
    (3, 256),
    (1, 256),
    (3, 512),
    "m",
    [4, [
        (1, 256),
        (3, 512)
    ]],
    (1, 512),
    (3, 1024),
    "m",
    [2, [
        (1, 512),
        (3, 1024)
    ]],
    (3, 1024),
    (3, 1024, 2),
    (3, 1024),
    (3, 1024),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(CNNBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, bias=False, **kwargs)
        self.BN = nn.BatchNorm2d(num_features=self.out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.leakyrelu(self.BN(self.conv(x)))


class YoloV1(nn.Module):

    def __init__(self, in_channels = 3, **kwargs) -> None:
        super(YoloV1, self).__init__()

        self.architecture = config
        self.in_channels = in_channels

        self.conv_layers = self.build_conv_layers(self.architecture)
        self.fully_connected = self.build_fully_connected(**kwargs)


    def forward(self, x):
        x = self.conv_layers(x)
        return self.fully_connected(torch.flatten(x, start_dim = 1))

    def build_fully_connected(self, split_size = 7, num_boxes = 2, num_classes = 20):

        S = split_size
        B = num_boxes
        C = num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features= (S * S * 1024), out_features = 496), # 4096 in original paper
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features= 496, out_features= S * S * ( C + B * 5) ), # 1470
        )

    def build_conv_layers(self, config):

        layers = []

        in_channels = self.in_channels
        out_channels = None # to be changed

        for c in config:
            print("config" , str(c))

            if isinstance(c, str) and c == "m":
                # print("m")
                layers.append(nn.MaxPool2d(2, stride= 2))

            elif isinstance(c, tuple):
                # print("tuple")

                if len(c) == 2:
                    # print("len 2")
                    padding = 1 if c[0] == 3 else 0
                    # layers.append(nn.Conv2d(in_channels= in_channels, out_channels=c[1], padding = padding, kernel_size=c[0]))
                    layers.append(CNNBlock(in_channels= in_channels, out_channels=c[1], padding = padding, kernel_size=c[0]))
                    out_channels = c[1]

                elif len(c) == 3: # kernel, out_channels, stride
                    # print("len 3")
                    if c[0] == 7:
                        #calculé pour que ça marche, que premier layer !
                        # layers.append(nn.Conv2d(in_channels= in_channels, out_channels=c[1], kernel_size=c[0], stride= c[2], padding=3))
                        layers.append(CNNBlock(in_channels= in_channels, out_channels=c[1], kernel_size=c[0], stride= c[2], padding=3))
                        out_channels = c[1]
                    else:
                        padding = 1 if c[2] == 2 else 0
                        print("padding", str(padding))
                        # layers.append(nn.Conv2d(in_channels= in_channels, padding = padding, out_channels=c[1], kernel_size=c[0], stride= c[2]))
                        layers.append(CNNBlock(in_channels= in_channels, padding = padding, out_channels=c[1], kernel_size=c[0], stride= c[2]))
                        out_channels = c[1]
            
            elif isinstance(c, list):
                print("list")
                nbr_loops = c[0]
                configs_loop = c[1]
                for l in range(nbr_loops):
                    for c_loop in configs_loop:
                        padding = 1 if c_loop[0] == 3 else 0
                        # layers.append(nn.Conv2d(in_channels= in_channels, out_channels=c_loop[1], padding = padding, kernel_size=c_loop[0]))
                        layers.append(CNNBlock(in_channels= in_channels, out_channels=c_loop[1], padding = padding, kernel_size=c_loop[0]))
                        out_channels = c_loop[1]
                        in_channels = out_channels

            in_channels = out_channels
            # print("in_channels", str(in_channels))
            # print("x shape", str(x.shape), "\n")
        # print(layers)
        return nn.Sequential(*layers)




def test():
    print(torch.__version__)
    print(torch.cuda.is_available())

    yolov1 = YoloV1(3)

    batch = 31
    img = np.ndarray((batch, 3, 448, 448))
    img[:,:] = 42
    x = Tensor(img)
    print("  x shape", str(x.shape), "\n")

    x = torch.Tensor(img)
    out = yolov1(x)

    print("out.shape : ", str(out.shape))

if __name__ == "__main__":
    test()
    