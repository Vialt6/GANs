import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

#according to generator architecture
factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]

class WSConv2d(nn.Module):
    #gain for the initializ const: paper formula -> sqrt(2/kernel^2*channel)
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5
        #bias dont has to be norm
        self.bias = self.conv.bias
        self.conv.bias = None

        #Initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

#Pixel Normalization
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        #square of every pixel value
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)

#Paper architecture: block used in generator and discriminator. (Remember: The discriminator doesn't use pixelNorm )
#Implemented 2 3x3 conv layeer
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelNorm=True,):
        super().__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()
        self.use_pn = use_pixelNorm

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x




class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super().__init__()
        #Initial: the first one in the figure. Different from other
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0), # 1x1 -> 4x4
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )
        #When generating 4x4 we need a torgb -> convert the 512 to RGB
        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)
        #create the progressive block
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList([self.initial_rgb])

        # create the progressive block: remember -> 4 of 512, then 512 divided by 2,4,8,16,32 progressively
        for i in range(len(factors) -1):
            # factors[i] -> factors[i+1]
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i+1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0))


    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1-alpha) * upscaled)

    # steps indicates the current resolution we are working with
    def forward(self, x, alpha, steps): # steps=0 (4x4) steps=1 (8x8) ...
        out = self.initial(x) # 4x4

        if steps == 0:
            return self.initial_rgb(out)

        #running to the progressive block
        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        # work on the layer l-1
        final_upscaled = self.rgb_layers[steps - 1](upscaled)

        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)



class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super().__init__()
        # prog blocks from 3x3 conv ecc, rgb: we have rgb layers between two blocks
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList()
        self.leaky = nn.LeakyReLU(0.2)

        # Iterate from the back, opposite of generator. add block in opposite order The first block has the highest resolution
        for i in range (len(factors) -1, 0, -1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i-1])
            # The first we append is the highest resolution
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c, use_pixelNorm=False))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in_c, kernel_size=1, stride=1, padding=0))

        # Last block same as the first block in generator
        # Initial rgb is the mirror of Initial rgb from the generator
        # is for 4x4 resolution
        self.initial_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # progblock sometimes change number of channels
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        #block for 4x4 resolution
        self.final_block = nn.Sequential(
            # in_channels + 1 in the figure wwe have 513 (512 + 1)
            WSConv2d(in_channels+1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            # Not realy a linear layer, but similar. To take a single value in the end
            WSConv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),

        )



    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled


    def minibatch_std(self, x):
        #Compute std for every example of x
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3]) # N x C x H x W -> N
        return torch.cat([x, batch_statistics], dim=1) #Concatenate along the channels

    def forward(self, x, alpha, steps): #steps = 0 (4x4), steps=1 (8x8) ...
        # If steps = 1 we want the last one
        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](x))

        # Need because the difference from prog_blocks and rgb_layers lenhgth is of 1
        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        #Run trhow the progressive block and then do avg pool
        # doing cur step here
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)

        # cur_step + 1 because we did cur step above
        for step in range(cur_step + 1, len(self.prog_blocks)):
            # Here we made img smaller and smaller
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)



if __name__ == "__main__":
    Z_DIM = 50
    IN_CHANNELS = 256
    gen = Generator(Z_DIM, IN_CHANNELS, img_channels=3)
    critic = Discriminator(IN_CHANNELS, img_channels=3)

    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(log2(img_size / 4))
        x = torch.randn((1, Z_DIM, 1, 1))
        z = gen(x, 0.5, steps=num_steps)
        assert z.shape == (1, 3, img_size, img_size)
        out = critic(z, alpha=0.5, steps=num_steps)
        assert out.shape == (1, 1)
        print(f"Success! At img size: {img_size}")













