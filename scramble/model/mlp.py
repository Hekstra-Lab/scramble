import torch
import math

class Linear(torch.nn.LazyLinear):
    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            mean = 0.
            fan_in,fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            fan_avg = 0.5 * (fan_in + fan_out)
            scale = 1 / 10.
            std = math.sqrt(scale / fan_avg)
            low,high = -2*std,2*std
            torch.nn.init.trunc_normal_(self.weight, mean, std, low, high)
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)


class FeedForward(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear_1 = Linear(2*width)
        self.linear_2 = Linear(width)
        self.activation = torch.nn.ReLU()

    def forward(self, data):
        out = data
        out = self.activation(out)
        out = self.linear_1(out)
        out = self.activation(out)
        out = self.linear_2(out)
        return out + data

class MLP(torch.nn.Sequential):
    def __init__(self, width, depth, input_layer=True):
        layers = []
        if input_layer:
            layers.append(Linear(width))
        for i in range(depth):
            layers.append(FeedForward(width))
        super().__init__(*layers)

