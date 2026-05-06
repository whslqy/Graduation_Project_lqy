import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    import timm
except ImportError:
    timm = None


MODEL_ALIASES = {
    "mobilenet": "mobilenet_v2",
    "efficientnet": "efficientnet_b0",
}

SUPPORTED_MODEL_NAMES = [
    "inception_resnet_v2",
    "resnet50",
    "resnet18",
    "mobilenet_v2",
    "shufflenet_v2_x0_5",
    "vgg16",
    "densenet121",
    "efficientnet_b0",
]


def normalize_model_name(model_name):
    return MODEL_ALIASES.get(model_name, model_name)


def _replace_classifier(model, model_name, num_classes):
    if model_name in {"resnet18", "resnet50"}:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if model_name == "mobilenet_v2":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if model_name == "shufflenet_v2_x0_5":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if model_name == "vgg16":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if model_name == "densenet121":
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if model_name == "inception_resnet_v2":
        in_features = model.classif.in_features
        model.classif = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def build_model(model_name, num_classes, pretrained=True):
    model_name = normalize_model_name(model_name)

    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
    elif model_name == "shufflenet_v2_x0_5":
        model = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None)
    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    elif model_name == "inception_resnet_v2":
        if timm is None:
            raise ImportError(
                "Building inception_resnet_v2 requires the 'timm' package. "
                "Install it before enabling this teacher model."
            )
        model = timm.create_model("inception_resnet_v2", pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return _replace_classifier(model, model_name, num_classes)


def build_model_group(model_names, num_classes, pretrained):
    models_dict = {}
    for model_name in model_names:
        normalized_name = normalize_model_name(model_name)
        models_dict[normalized_name] = build_model(
            model_name=normalized_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )
    return models_dict


def build_teacher_student_models(distillation_config, num_classes):
    teacher_models = build_model_group(
        distillation_config["teacher_models"],
        num_classes=num_classes,
        pretrained=distillation_config.get("pretrained_teacher", True),
    )
    student_models = build_model_group(
        distillation_config["student_models"],
        num_classes=num_classes,
        pretrained=distillation_config.get("pretrained_student", True),
    )
    return teacher_models, student_models


def summarize_pretraining_setup(teacher_models, student_models):
    summary_lines = []
    for role, models_dict in (("teacher", teacher_models), ("student", student_models)):
        for model_name, model in models_dict.items():
            if hasattr(model, "fc"):
                head = model.fc.__class__.__name__
            elif hasattr(model, "classifier"):
                head_layer = model.classifier[-1] if isinstance(model.classifier, nn.Sequential) else model.classifier
                head = head_layer.__class__.__name__
            elif hasattr(model, "classif"):
                head = model.classif.__class__.__name__
            else:
                head = "unknown"
            summary_lines.append(f"{role}:{model_name}: classifier={head}")
    return summary_lines


def extract_features(model, model_name, x):
    model_name = normalize_model_name(model_name)

    if model_name in {"resnet18", "resnet50"}:
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        return torch.flatten(x, 1)

    if model_name == "mobilenet_v2":
        x = model.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)

    if model_name == "shufflenet_v2_x0_5":
        x = model.conv1(x)
        x = model.maxpool(x)
        x = model.stage2(x)
        x = model.stage3(x)
        x = model.stage4(x)
        x = model.conv5(x)
        x = x.mean([2, 3])
        return x

    if model_name == "vgg16":
        x = model.features(x)
        x = model.avgpool(x)
        return torch.flatten(x, 1)

    if model_name == "densenet121":
        x = model.features(x)
        x = F.relu(x, inplace=False)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)

    if model_name == "efficientnet_b0":
        x = model.features(x)
        x = model.avgpool(x)
        return torch.flatten(x, 1)

    if model_name == "inception_resnet_v2":
        x = model.forward_features(x)
        x = model.global_pool(x)
        if x.ndim > 2:
            x = torch.flatten(x, 1)
        return x

    raise ValueError(f"Unsupported model: {model_name}")


def classify_from_features(model, model_name, features):
    model_name = normalize_model_name(model_name)

    if model_name in {"resnet18", "resnet50", "shufflenet_v2_x0_5"}:
        return model.fc(features)
    if model_name in {"mobilenet_v2", "efficientnet_b0"}:
        return model.classifier(features)
    if model_name == "vgg16":
        return model.classifier(features)
    if model_name == "densenet121":
        return model.classifier(features)
    if model_name == "inception_resnet_v2":
        return model.classif(features)
    raise ValueError(f"Unsupported model: {model_name}")


def forward_with_features(model, model_name, x):
    features = extract_features(model, model_name, x)
    logits = classify_from_features(model, model_name, features)
    return logits, features


def get_feature_dim(model, model_name):
    model_name = normalize_model_name(model_name)

    if model_name in {"resnet18", "resnet50", "shufflenet_v2_x0_5"}:
        return model.fc.in_features
    if model_name == "mobilenet_v2":
        return model.classifier[-1].in_features
    if model_name == "vgg16":
        return model.classifier[-1].in_features
    if model_name == "densenet121":
        return model.classifier.in_features
    if model_name == "efficientnet_b0":
        return model.classifier[-1].in_features
    if model_name == "inception_resnet_v2":
        return model.classif.in_features
    raise ValueError(f"Unsupported model: {model_name}")
