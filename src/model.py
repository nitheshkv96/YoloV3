import torch
import torch.nn as nn

'''
Tuple: (outchannel, kernel_size, stride)
B: Residual Block with second argument meaning repeats
S: Staging with loss computation
U: Upsampling the feature and concatenating with a previous layer
'''

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bn_act = True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, bias = not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_ch)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_res = True, num_repeats = 1):
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels,channels//2, kernel_size = 1),
                    CNNBlock(channels//2, channels, kernel_size = 3, padding = 1)  
                )
            ]
        self.use_residual = use_res
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
        return x

class ScalPrediction(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_ch, 2*in_ch, kernel_size = 3, padding = 1),
            CNNBlock(2*in_ch, (num_classes + 5)*3, bn_act = False, kernel_size = 1),
        )
        self.num_classes = num_classes

    def forward(self,x):
        return (
            self.pred(x).reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0,1,3,4,2)
        )
        # N * 2 * (13 * 13) * (5+numclasses)
    


class YoloV3(nn.Module):
    def __init__(self, in_ch = 3, num_classes = 20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_ch
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, ScalPrediction):
                outputs.append(layer(x))
                continue
            
            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim = 1)
                route_connections.pop()
        return outputs


    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_ch = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_ch, kernel_size, stride = module
                layers.append(CNNBlock(in_ch, out_ch, 
                                       kernel_size = kernel_size,
                                       stride = stride,
                                       padding = 1 if kernel_size ==3 else 0,))
                in_ch = out_ch
            elif isinstance(module, list):
                num_reps = module[1]
                layers.append(ResidualBlock(in_ch, num_repeats = num_reps))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_ch, use_res = False, num_repeats = 1),
                        CNNBlock(in_ch, in_ch//2, kernel_size = 1),
                        ScalPrediction(in_ch//2, num_classes = self.num_classes)
                    ]
                    in_ch = in_ch//2
                elif module == "U":
                     layers.append(nn.Upsample(scale_factor=2))
                     in_ch = in_ch * 3   
        return layers


## Testing the Model Architecture
if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YoloV3(num_classes=num_classes)
    print(model.eval())
    # x = torch.randn((2,3,IMAGE_SIZE, IMAGE_SIZE))
    # out = model(x)
    # assert model(x)[0].shape == (2,3, IMAGE_SIZE//32,IMAGE_SIZE//32, num_classes + 5)
    # assert model(x)[1].shape == (2,3, IMAGE_SIZE//16,IMAGE_SIZE//16, num_classes + 5)
    # assert model(x)[2].shape == (2,3, IMAGE_SIZE//8,IMAGE_SIZE//8, num_classes + 5)
    # print("Success!!!!")