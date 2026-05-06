from pathlib import Path


SEED = 42

DATA_ROOT = Path("e:/Dataset/CK+/data")
EXTENDED_DIR = DATA_ROOT / "extended-cohn-kanade-images" / "cohn-kanade-images"
EMOTION_DIR = DATA_ROOT / "Emotion_labels" / "Emotion"

EMOTION_ID_TO_NAME = {
    1: "anger",
    2: "contempt",
    3: "disgust",
    4: "fear",
    5: "happy",
    6: "sadness",
    7: "surprise",
}
CLASS_NAMES = [EMOTION_ID_TO_NAME[i] for i in sorted(EMOTION_ID_TO_NAME)]

IMG_SIZE = 256
CROP_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 50

VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

DISTILLATION_CONFIG = {
    "teacher_models": ["inception_resnet_v2", "resnet50"],
    "student_models": ["mobilenet_v2", "resnet18", "shufflenet_v2_x0_5"],
    "experiments": [
        {"teacher": "resnet50", "student": "resnet18"},
        {"teacher": "inception_resnet_v2", "student": "resnet18"},
        {"teacher": "resnet50", "student": "mobilenet_v2"},
        {"teacher": "inception_resnet_v2", "student": "mobilenet_v2"},
        {"teacher": "resnet50", "student": "shufflenet_v2_x0_5"},
        {"teacher": "inception_resnet_v2", "student": "shufflenet_v2_x0_5"},
    ],
    "temperature": 4.0,
    "alpha": 0.55,
    "feature_loss_weight": 0.08,
    "warmup_epochs": 5,
    "kd_ramp_epochs": 5,
    "feature_start_epoch": 5,
    "gradient_clip_norm": 1.0,
    "pretrained_teacher": True,
    "pretrained_student": True,
    "student_specific_config": {
        "mobilenet_v2": {
            "temperature": 5.0,
            "alpha": 0.25,
            "feature_loss_weight": 0.03,
            "feature_loss_type": "mse_regressor",
            "warmup_epochs": 8,
            "kd_ramp_epochs": 10,
            "feature_start_epoch": 15,
        },
        "resnet18": {
            "temperature": 4.0,
            "alpha": 0.55,
            "feature_loss_weight": 0.08,
            "warmup_epochs": 5,
            "kd_ramp_epochs": 5,
            "feature_start_epoch": 8,
        },
        "shufflenet_v2_x0_5": {
            "temperature": 5.0,
            "alpha": 0.2,
            "feature_loss_weight": 0.015,
            "warmup_epochs": 8,
            "kd_ramp_epochs": 10,
            "feature_start_epoch": 16,
        },
    },
    "student_training_config": {
        "mobilenet_v2": {
            "epochs": 50,
            "learning_rate": 2e-4,
            "weight_decay": 5e-5,
            "optimizer": "adamw",
            "label_smoothing": 0.05,
        },
        "shufflenet_v2_x0_5": {
            "epochs": 50,
            "learning_rate": 2e-4,
            "weight_decay": 5e-5,
        },
    },
}

MODEL_TRAINING_CONFIG = {
    "mobilenet_v2": {
        "epochs": 50,
        "learning_rate": 2e-4,
        "weight_decay": 5e-5,
        "optimizer": "adamw",
        "label_smoothing": 0.05,
    },
}

BASELINE_MODEL_NAMES = [
    "inception_resnet_v2",
    "resnet50",
    "mobilenet_v2",
    "resnet18",
    "shufflenet_v2_x0_5",
    "vgg16",
    "densenet121",
    "efficientnet_b0",
]

TRAINING_CONFIG = {
    "epochs": NUM_EPOCHS,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "optimizer": "adam",
    "scheduler": "cosine",
    "label_smoothing": 0.1,
}

AUG_TYPES = [
    "original",
    "horizontal_flip",
    "rotate15",
    "rotate_minus15",
    "randomaffine",
    "randomresizedcrop",
    "autocontrast",
    "equalize",
    "colorjitter",
    "hist_eq",
    "clahe",
    "gaussian_blur",
    "sharpen",
    "adjust_sharpness",
    "randomerasing",
    "gaussian_noise",
]

AUG_DISPLAY_NAMES = [
    "Original",
    "HorizontalFlip",
    "Rotate15",
    "RotateMinus15",
    "RandomAffine",
    "RandomResizedCrop",
    "AutoContrast",
    "Equalize",
    "ColorJitter",
    "HistEq",
    "CLAHE",
    "GaussianBlur",
    "Sharpen",
    "AdjustSharpness",
    "RandomErasing",
    "GaussianNoise",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
