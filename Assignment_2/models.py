##########################################################################
#   The two segmentation models compared in A2                           #
#   U-Net + ResNet-50 encoder (CNN)  vs  SegFormer-B2 (Transformer)      #
#   Both return full-resolution logits of shape (B, NUM_CLASSES, H, W)   #
##########################################################################

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding = 1, bias = False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace = True),
            nn.Conv2d(out_ch, out_ch, 3, padding = 1, bias = False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace = True),
        )


class UNetResNet50(nn.Module):
    # Encoder: ResNet-50 feature maps at strides 2, 4, 8, 16, 32 (64, 256, 512, 1024, 2048 channels).
    # Decoder: 5 upsampling steps; each doubles the resolution, concatenates the encoder skip of the same
    # stride (if any) and applies two 3x3 conv-BN-ReLU (Ronneberger et al., 2015)

    def __init__(self, num_classes, pretrained = True, decoder_channels = (256, 128, 64, 32, 16)):
        super().__init__()
        self.encoder = timm.create_model(config.ARCHS["unet_r50"], pretrained = pretrained, features_only = True)
        enc_ch = self.encoder.feature_info.channels()                  # [64, 256, 512, 1024, 2048]
        skip_ch = enc_ch[-2::-1] + [0]                                 # Skips from stride 16 down to 2, none at stride 1
        in_ch = [enc_ch[-1]] + list(decoder_channels[:-1])
        self.decoder = nn.ModuleList(ConvBlock(i + s, o) for i, s, o in zip(in_ch, skip_ch, decoder_channels))
        self.head = nn.Conv2d(decoder_channels[-1], num_classes, 1)

    def forward(self, x):
        features = self.encoder(x)
        y, skips = features[-1], features[-2::-1]
        for i, block in enumerate(self.decoder):
            y = F.interpolate(y, scale_factor = 2, mode = "nearest")
            if i < len(skips):
                y = torch.cat([y, skips[i]], dim = 1)
            y = block(y)
        return self.head(y)


class SegFormerB2(nn.Module):
    # Hugging Face SegFormer (Xie et al., NeurIPS 2021): hierarchical MiT-B2 encoder + all-MLP decoder.
    # The decoder predicts at 1/4 resolution; logits are upsampled bilinearly to the input size

    def __init__(self, num_classes, pretrained = True):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        name = config.ARCHS["segformer_b2"]
        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(name, num_labels = num_classes)
        else:
            self.net = SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(name, num_labels = num_classes))
        self.encoder = self.net.segformer                              # Alias used for per-group learning rates

    def forward(self, x):
        logits = self.net(pixel_values = x).logits
        return F.interpolate(logits, size = x.shape[-2:], mode = "bilinear", align_corners = False)


def build_model(arch, pretrained = True):
    if arch == "unet_r50":
        return UNetResNet50(config.NUM_CLASSES, pretrained)
    if arch == "segformer_b2":
        return SegFormerB2(config.NUM_CLASSES, pretrained)
    raise ValueError(f"Unknown architecture: {arch}")


def param_groups(model):
    # Pretrained encoder and randomly initialized decoder/head get different learning rates
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    encoder = [p for p in model.parameters() if id(p) in encoder_ids]
    decoder = [p for p in model.parameters() if id(p) not in encoder_ids]
    return encoder, decoder


def count_parameters(model):
    encoder, decoder = param_groups(model)
    return {"total": sum(p.numel() for p in model.parameters()),
            "encoder": sum(p.numel() for p in encoder),
            "decoder": sum(p.numel() for p in decoder)}
