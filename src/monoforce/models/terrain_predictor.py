import numpy as np
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU, LeakyReLU, Tanh
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch


__all__ = [
    'TerrainPredictor',
    'Geom2Trav',
    'create_torchvision_model',
]


class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        # store the convolution and RELU layers
        self.conv1 = Conv2d(inChannels, outChannels, 2, bias=False)
        self.nonlin = LeakyReLU(0.1, inplace=True)
        # self.nonlin = ReLU()
        # self.nonlin = Tanh()
        self.conv2 = Conv2d(outChannels, outChannels, 2, bias=False)

    def forward(self, x):
        # apply CONV => NONLIN => CONV block to the inputs and return it
        return self.conv2(self.nonlin(self.conv1(x)))


class Encoder(Module):
    def __init__(self, channels):
        super().__init__()
        # store the encoder blocks and maxpooling layer
        self.encBlocks = ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)

    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        blockOutputs = []
        # loop through the encoder blocks
        for block in self.encBlocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            blockOutputs.append(x)
            # x = self.pool(x)
        # return the list containing the intermediate outputs
        return blockOutputs


class Decoder(Module):
    def __init__(self, channels):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels
        self.upconvs = ModuleList([ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])

    def forward(self, x, encFeatures):
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)
            # crop the current features from the encoder blocks,
            # concatenate them with the current upsampled features,
            # and pass the concatenated output through the current
            # decoder block
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
        # return the final decoder output
        return x

    def crop(self, encFeatures, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        H, W = x.shape[-2:]
        encFeatures = CenterCrop([H, W])(encFeatures)
        # return the cropped features
        return encFeatures


class TerrainPredictor(Module):
    def __init__(self, encChannels, decChannels, nbClasses=1, retainDim=True):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        # initialize the regression head and store the class variables
        self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim

    def forward(self, x):
        # grab the features from the encoder
        encFeatures = self.encoder(x)
        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1:])
        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        map = self.head(decFeatures)
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retainDim:
            map = F.interpolate(map, x.shape[-2:])
        # return the segmentation map
        return map


class LinearPredictor(Module):
    def __init__(self, w=None, b=None):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(w) if w is not None else torch.randn(1))
        self.b = torch.nn.Parameter(torch.tensor(b) if b is not None else torch.randn(1))

    def forward(self, x):
        y = self.w * x + self.b
        return y
    

class Geom2Trav(torch.nn.Module):
    """
    Geom2Trav is a simple model that predicts the traversability height map from a estimated (geometrical) height map.
    For example, it should correct the height map of a high vegetation area to a flat terrain.
    It consists of two convolutional layers followed by two transposed convolutional layers.
    CONV => RELU => CONV => RELU => UPCONV => RELU => UPCONV => RELU
    """
    def __init__(self):
        super(Geom2Trav, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.upconv1 = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.upconv2 = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.nonlin = torch.nn.LeakyReLU()

        # Initialize weights
        self.init_weights()

    def forward(self, x):
        x = self.conv2(self.nonlin(self.conv1(x)))
        x = self.upconv2(self.nonlin(self.upconv1(x)))
        return x

    def init_weights(self):
        # Initialize weights of the model
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


def create_torchvision_model(architecture, n_inputs, n_outputs, pretrained_backbone=True):
    assert architecture in ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101',
                            'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large']

    print('Creating model %s with %i inputs and %i outputs' % (architecture, n_inputs, n_outputs))
    Architecture = eval('torchvision.models.segmentation.%s' % architecture)
    model = Architecture(pretrained=pretrained_backbone)

    arch = architecture.split('_')[0]
    encoder = '_'.join(architecture.split('_')[1:])

    # Change input layer to accept n_inputs
    if encoder == 'mobilenet_v3_large':
        model.backbone['0'][0] = torch.nn.Conv2d(n_inputs, 16,
                                                 kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    else:
        model.backbone['conv1'] = torch.nn.Conv2d(n_inputs, 64,
                                                  kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Change final layer to output n classes
    if arch == 'lraspp':
        model.classifier.low_classifier = torch.nn.Conv2d(40, n_outputs, kernel_size=(1, 1), stride=(1, 1))
        model.classifier.high_classifier = torch.nn.Conv2d(128, n_outputs, kernel_size=(1, 1), stride=(1, 1))
    elif arch == 'fcn':
        model.classifier[-1] = torch.nn.Conv2d(512, n_outputs, kernel_size=(1, 1), stride=(1, 1))
    elif arch == 'deeplabv3':
        model.classifier[-1] = torch.nn.Conv2d(256, n_outputs, kernel_size=(1, 1), stride=(1, 1))

    return model


def learn_identity(n_iters=100, lr=1e-2, bs=512):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from tqdm import tqdm

    model = TerrainPredictor(encChannels=(1, 2, 4), decChannels=(4, 2), retainDim=False)
    # model = LinearPredictor()

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.train()

    losses = []
    for i in tqdm(range(n_iters)):
        inpt1 = torch.randn((bs // 2, 1, 10, 10))
        inpt2 = torch.as_tensor(np.random.random((bs // 2, 1, 10, 10)), dtype=torch.float32)
        inpt = torch.cat([inpt1, inpt2], dim=0)
        inpt = inpt[torch.randperm(bs)]

        label = inpt.clone()

        pred = model(inpt)
        loss = loss_fn(pred, label)

        optim.zero_grad()
        loss.backward()
        optim.step()
        print(loss.item())
        losses.append(loss.item())

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    plt.title('Prediction')
    plt.imshow(pred.squeeze().detach().cpu().numpy()[0])

    plt.subplot(1, 3, 2)
    plt.title('GT')
    plt.imshow(label.squeeze().detach().cpu().numpy()[0])

    plt.subplot(1, 3, 3)
    plt.title('Loss')
    plt.plot(losses)
    plt.grid()

    plt.show()

    # save model
    path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../../config/weights/identity.pth'))
    print('Saving model to: %s' % path)
    torch.save(model.state_dict(), path)


def demo():
    import matplotlib.pyplot as plt
    from ..vis import plot_grad_flow

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    height_true = torch.from_numpy(5 * np.random.random((1, 1, 256, 256)))
    
    # model that transforms rigid height map to traversability map
    model = Geom2Trav()

    height_init = height_true.clone() + torch.randn_like(height_true) * 0.1
    print(model(height_init).shape)

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    model = model.train()

    losses = []
    for i in range(500):
        height_pred = model(height_init)
        loss = loss_fn(height_pred, height_true)

        optim.zero_grad()
        loss.backward()
        # plot_grad_flow(model.named_parameters())
        # plt.draw()
        # plt.pause(0.01)
        optim.step()
        losses.append(loss.item())
    print(loss.item())

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.title('Prediction')
    plt.imshow(height_pred.squeeze().detach().cpu().numpy(), cmap='jet')
    plt.colorbar()

    plt.subplot(1, 4, 2)
    plt.title('GT')
    plt.imshow(height_true.squeeze().detach().cpu().numpy(), cmap='jet')
    plt.colorbar()

    plt.subplot(1, 4, 3)
    plt.title('Loss')
    plt.plot(losses)
    plt.grid()

    plt.subplot(1, 4, 4)
    plot_grad_flow(model.named_parameters())
    plt.show()


def main():
    # learn_identity(n_iters=400, lr=0.01, bs=1024)
    demo()


if __name__ == '__main__':
    main()
