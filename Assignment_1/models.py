#########################################################################
#   Build ResNet-18 (CNN) and DeiT-Tiny (Transformer) with an           #
#   init mode (scratch / pretrained) and freezing (head/partial/none)   #
#########################################################################

import timm
import torch.nn as nn

import config

# Early part of the network frozen in partial mode (about half of the backbone)
PARTIAL_FROZEN = {
    "resnet18": ["conv1", "bn1", "layer1", "layer2"],                         # layer3, layer4, fc stay trainable
    "deit_tiny": ["patch_embed", "cls_token", "pos_embed"] + [f"blocks.{i}." for i in range(6)],  # blocks 6-11, norm, head stay trainable
}


def build_model(arch, pretrained, freeze = "none"):
    # Create the model with a new 100-class head. pretrained=True loads ImageNet-1k weights
    model = timm.create_model(config.ARCHS[arch], pretrained = pretrained, num_classes = config.NUM_CLASSES)
    apply_freeze(model, arch, freeze)
    return model


def apply_freeze(model, arch, freeze):
    # freeze = "none": train everything | "head": train only the final classifier | "partial": freeze the first half of the backbone
    if freeze == "none":
        return

    head_params = {id(p) for p in model.get_classifier().parameters()}
    for name, param in model.named_parameters():
        if freeze == "head":
            param.requires_grad = id(param) in head_params
        elif freeze == "partial":
            param.requires_grad = not any(name.startswith(prefix) for prefix in PARTIAL_FROZEN[arch])
        else:
            raise ValueError(f"Kiểu đóng băng không hợp lệ: {freeze}")


def set_train_mode(model):
    # Switch to train mode, but keep BatchNorm layers of frozen parts in eval mode
    # so the pretrained running mean/var statistics are not changed
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
            module.eval()


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
