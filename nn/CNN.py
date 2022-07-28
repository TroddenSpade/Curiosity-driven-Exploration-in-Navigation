import torch
import torch.nn as nn
import torch.nn.functional as F


''' Learning by Prediction ICLR 2017 paper
    (their final output was 64 changed to 256 here)
    input: [None, 120, 160, 1]; output: [None, 1280] -> [None, 256];
'''
class FeatureExtractor(nn.Module):
    def __init__(self, output_size=256):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(5, 5), stride=(4, 4)),
            nn.ELU(),
            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2)),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2)),
            nn.ELU(),
            nn.Flatten(start_dim=1, end_dim=-1),

            nn.Linear(in_features=1344, out_features=output_size, bias=True),
            nn.ELU(),
        )

    def forward(self, state):
        x = F.normalize(state)
        x = self.feature_extractor(state)
        return x


if __name__ == "__main__":
    state = torch.randn(1, 3, 144, 256)
    feature_extractor = FeatureExtractor()
    x = feature_extractor(state)
    print(x.shape)