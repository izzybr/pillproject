def define_pytorch_model():
    class PyTorch_Model(nn.Module):
        def __init__(self):
            super().__init__()
        
            # the original image is 32x32x3
            # first convolution layer - after this, the image is 16x16
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels = 3, # number of in-channels for first convolution layer is number of dimensions
                    out_channels = 56, # number of filters to try
                    kernel_size = 11, # dimensions of the kernel
                    stride = 4, # stride of 1 indicates it should go to each pixel, stride of 2 indicates every other pixel, stride of n indicates every n pixels
                    padding = 1 # adding padding when applying filters to the border (this affects the resulting dimension)
                ),
                nn.ReLU(), # activation function
                nn.MaxPool2d(kernel_size=2) # halves the images from 32x32 to 16x16
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(224, 55, 11, 4),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.conv3 = nn.Sequential(
                
            )

            # linear layer for the ten classes
            self.fc1 = nn.Linear(5 * 16 * 16, 10) # make sure the input in matches with output from the previous layer

        # now we want pass data along in our forward function
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            # flatten the output of convolution layers
            x = x.view(x.size(0), -1)  