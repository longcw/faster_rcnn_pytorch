from torch.nn.modules.module import Module
# from functions.roi_pool import RoIPoolFunction
from ..functions.roi_pool import RoIPoolFunction


class RoIPool(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoIPool, self).__init__()
        self.roi_pool = RoIPoolFunction(pooled_height, pooled_width, spatial_scale)

    def forward(self, features, rois):
        return self.roi_pool(features, rois)
