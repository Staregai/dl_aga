from __future__ import annotations

from typing import Any

import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, num_classes: int, img_size: int, dropout: float = 0.45) -> None:
        super().__init__()
        input_dim = 3 * img_size * img_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.75),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcite(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + identity
        return self.act(out)


ARCH_CONFIGS = {
    "small": {
        "stem_channels": 32,
        "stage_channels": (48, 96, 192, 320),
        "stage_blocks": (2, 2, 2, 2),
        "stage_dropout": (0.03, 0.06, 0.10, 0.14),
        "head_hidden": 160,
    },
    "medium": {
        "stem_channels": 48,
        "stage_channels": (64, 128, 256, 384),
        "stage_blocks": (2, 2, 3, 2),
        "stage_dropout": (0.04, 0.08, 0.12, 0.16),
        "head_hidden": 192,
    },
    "large": {
        "stem_channels": 64,
        "stage_channels": (80, 160, 320, 512),
        "stage_blocks": (2, 3, 4, 2),
        "stage_dropout": (0.05, 0.09, 0.14, 0.18),
        "head_hidden": 256,
    },
}


class RoomResNet(nn.Module):
    """ResNet-style CNN trained from scratch for room classification."""

    def __init__(self, num_classes: int, arch: str = "small", dropout: float = 0.30) -> None:
        super().__init__()
        if arch not in ARCH_CONFIGS:
            raise ValueError(f"Unsupported CNN arch: {arch}. Choose one of: {sorted(ARCH_CONFIGS)}")
        cfg = ARCH_CONFIGS[arch]
        stem_channels = cfg["stem_channels"]
        stage_channels = cfg["stage_channels"]
        stage_blocks = cfg["stage_blocks"]
        stage_dropout = cfg["stage_dropout"]
        head_hidden = cfg["head_hidden"]

        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
        )
        self.stage1 = self._make_stage(
            stem_channels,
            stage_channels[0],
            blocks=stage_blocks[0],
            stride=2,
            dropout=stage_dropout[0],
        )
        self.stage2 = self._make_stage(
            stage_channels[0],
            stage_channels[1],
            blocks=stage_blocks[1],
            stride=2,
            dropout=stage_dropout[1],
        )
        self.stage3 = self._make_stage(
            stage_channels[1],
            stage_channels[2],
            blocks=stage_blocks[2],
            stride=2,
            dropout=stage_dropout[2],
        )
        self.stage4 = self._make_stage(
            stage_channels[2],
            stage_channels[3],
            blocks=stage_blocks[3],
            stride=2,
            dropout=stage_dropout[3],
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.BatchNorm1d(stage_channels[-1]),
            nn.Dropout(dropout),
            nn.Linear(stage_channels[-1], head_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
        dropout: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [ResidualBlock(in_channels, out_channels, stride=stride, dropout=dropout)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


class RoomResNetSmall(RoomResNet):
    """Backward-compatible alias for the original small architecture."""

    def __init__(self, num_classes: int, dropout: float = 0.30) -> None:
        super().__init__(num_classes=num_classes, arch="small", dropout=dropout)


def _conv_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int,
    batch_norm: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=not batch_norm,
        )
    ]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _lecture_stage(
    in_channels: int,
    out_channels: int,
    conv_count: int,
    batch_norm: bool,
    dropout2d: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_channels = in_channels
    for _ in range(conv_count):
        layers.append(
            _conv_relu(
                current_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                batch_norm=batch_norm,
            )
        )
        current_channels = out_channels
    layers.append(nn.MaxPool2d((2, 2)))
    if dropout2d > 0:
        layers.append(nn.Dropout2d(dropout2d))
    return nn.Sequential(*layers)


class LectureCNNClassifier(nn.Module):
    """Lecture-scope CNNs: Julia/Flux port plus VGG-style improvements."""

    def __init__(
        self,
        num_classes: int,
        img_size: int,
        topology: int = 2,
        dropout: float = 0.30,
        batch_norm: bool = False,
        variant: str = "notebook",
    ) -> None:
        super().__init__()
        if variant == "notebook" and topology == 1:
            self.features = nn.Sequential(
                _conv_relu(3, 16, kernel_size=3, padding=1, batch_norm=batch_norm),
                nn.MaxPool2d((2, 2)),
                _conv_relu(16, 32, kernel_size=3, padding=1, batch_norm=batch_norm),
                nn.MaxPool2d((2, 2)),
                _conv_relu(32, 64, kernel_size=3, padding=1, batch_norm=batch_norm),
                nn.MaxPool2d((2, 2)),
            )
            feature_channels = 64
            pool_count = 3
            hidden_units = 128
            second_hidden_units = None
        elif variant == "notebook" and topology == 2:
            self.features = nn.Sequential(
                _conv_relu(3, 16, kernel_size=3, padding=1, batch_norm=batch_norm),
                _conv_relu(16, 16, kernel_size=3, padding=1, batch_norm=batch_norm),
                nn.MaxPool2d((2, 2)),
                _conv_relu(16, 32, kernel_size=3, padding=1, batch_norm=batch_norm),
                _conv_relu(32, 32, kernel_size=3, padding=1, batch_norm=batch_norm),
                nn.MaxPool2d((2, 2)),
            )
            feature_channels = 32
            pool_count = 2
            hidden_units = 256
            second_hidden_units = None
        elif variant == "notebook" and topology == 3:
            self.features = nn.Sequential(
                _conv_relu(3, 16, kernel_size=5, padding=2, batch_norm=batch_norm),
                nn.MaxPool2d((2, 2)),
                _conv_relu(16, 32, kernel_size=3, padding=1, batch_norm=batch_norm),
                nn.MaxPool2d((2, 2)),
            )
            feature_channels = 32
            pool_count = 2
            hidden_units = 64
            second_hidden_units = None
        elif variant == "wide":
            self.features = nn.Sequential(
                _lecture_stage(3, 32, conv_count=2, batch_norm=batch_norm, dropout2d=0.05),
                _lecture_stage(32, 64, conv_count=2, batch_norm=batch_norm, dropout2d=0.10),
                _lecture_stage(64, 128, conv_count=2, batch_norm=batch_norm, dropout2d=0.15),
            )
            feature_channels = 128
            pool_count = 3
            hidden_units = 256
            second_hidden_units = 128
        elif variant == "deep":
            self.features = nn.Sequential(
                _lecture_stage(3, 32, conv_count=2, batch_norm=batch_norm, dropout2d=0.05),
                _lecture_stage(32, 64, conv_count=2, batch_norm=batch_norm, dropout2d=0.10),
                _lecture_stage(64, 128, conv_count=2, batch_norm=batch_norm, dropout2d=0.15),
                _lecture_stage(128, 256, conv_count=1, batch_norm=batch_norm, dropout2d=0.20),
            )
            feature_channels = 256
            pool_count = 4
            hidden_units = 384
            second_hidden_units = 192
        else:
            raise ValueError(
                "Unsupported lecture CNN configuration. Use variant='notebook' with topology 1/2/3, "
                "or variant in {'wide', 'deep'}."
            )

        feature_dim = self._feature_dim(img_size, feature_channels=feature_channels, pool_count=pool_count)
        classifier_layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ]
        if second_hidden_units is not None:
            classifier_layers.extend(
                [
                    nn.Linear(hidden_units, second_hidden_units),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout * 0.5),
                ]
            )
            hidden_units = second_hidden_units
        classifier_layers.append(nn.Linear(hidden_units, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        self.apply(self._init_weights)

    @staticmethod
    def _feature_dim(img_size: int, feature_channels: int, pool_count: int) -> int:
        spatial_size = img_size
        for _ in range(pool_count):
            spatial_size //= 2
        if spatial_size <= 0:
            raise ValueError(f"img_size={img_size} is too small for {pool_count} pooling stages")
        return feature_channels * spatial_size * spatial_size

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.classifier(self.features(x))


def build_mlp(num_classes: int, img_size: int, dropout: float = 0.45) -> nn.Module:
    return MLPClassifier(num_classes=num_classes, img_size=img_size, dropout=dropout)


def normalize_cnn_arch(arch: str | None) -> str:
    if arch in (None, "", "room_resnet_small"):
        return "small"
    if arch in ("room_resnet_medium", "medium"):
        return "medium"
    if arch in ("room_resnet_large", "large"):
        return "large"
    if arch == "small":
        return "small"
    raise ValueError(f"Unsupported CNN arch: {arch}")


def build_custom_cnn(num_classes: int, dropout: float = 0.30, arch: str = "small") -> nn.Module:
    return RoomResNet(num_classes=num_classes, arch=normalize_cnn_arch(arch), dropout=dropout)


def build_lecture_cnn(
    num_classes: int,
    img_size: int,
    topology: int = 2,
    dropout: float = 0.30,
    batch_norm: bool = False,
    variant: str = "notebook",
) -> nn.Module:
    return LectureCNNClassifier(
        num_classes=num_classes,
        img_size=img_size,
        topology=topology,
        dropout=dropout,
        batch_norm=batch_norm,
        variant=variant,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_from_metadata(metadata: dict[str, Any], num_classes: int) -> nn.Module:
    model_type = metadata["model_type"]
    if model_type == "mlp":
        return build_mlp(num_classes=num_classes, img_size=int(metadata["img_size"]), dropout=float(metadata["dropout"]))
    if model_type == "cnn":
        return build_custom_cnn(
            num_classes=num_classes,
            dropout=float(metadata["dropout"]),
            arch=metadata.get("arch", "small"),
        )
    if model_type == "cnn_aug":
        return build_custom_cnn(
            num_classes=num_classes,
            dropout=float(metadata["dropout"]),
            arch=metadata.get("arch", "small"),
        )
    if model_type in {"lecture_cnn", "lecture_cnn_aug"}:
        return build_lecture_cnn(
            num_classes=num_classes,
            img_size=int(metadata["img_size"]),
            topology=int(metadata.get("topology", 2)),
            dropout=float(metadata["dropout"]),
            batch_norm=bool(metadata.get("batch_norm", False)),
            variant=metadata.get("variant", "notebook"),
        )
    raise ValueError(f"Unsupported model_type in checkpoint: {model_type}")
