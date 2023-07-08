from torch import nn


class MeasureLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, center_point, point_1_1, point_1_2, point_2_1, point_2_2):
        pass
