from torch import nn


class SmallCNN(nn.Module):
    """Four-stage CNN used as the project baseline."""

    def __init__(self, num_classes: int):
        super().__init__()
        blocks = []
        input_channels = 3
        for output_channels in (32, 64, 128, 256):
            blocks.extend(
                [
                    nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            input_channels = output_channels
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.30), nn.Linear(256, num_classes)
        )

    def forward(self, images):
        return self.head(self.features(images))
