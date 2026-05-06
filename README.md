# CK+ Facial Expression Recognition with Knowledge Distillation

This repository contains the core code for a graduation project on lightweight facial expression recognition using knowledge distillation.

The project uses the CK+ dataset, trains baseline student models, trains teacher models, runs multiple teacher-student distillation variants, and exports experiment results for later comparison and analysis.

The recommended entry point is [main.ipynb](/E:/Dataset/main.ipynb). Most reusable logic is implemented in the Python modules rather than inside the notebook.

## Project Purpose

The main goal of the project is to study whether lightweight facial expression recognition models can be improved through knowledge distillation while keeping the deployed student model compact.

The project focuses on:
- baseline training for student models,
- teacher-student distillation experiments,
- comparison of multiple distillation variants,
- focused parameter sweeps for selected teacher-student pairs,
- result export for later reporting and visualisation.

## Main Files

Core project files:
- [main.ipynb](/E:/Dataset/main.ipynb): recommended interactive entry point.
- [kd_config.py](/E:/Dataset/kd_config.py): central configuration file for data paths, training settings, augmentation, model lists, and distillation defaults.
- [kd_data.py](/E:/Dataset/kd_data.py): CK+ sample loading, preprocessing, augmentation, and dataloader construction.
- [kd_models.py](/E:/Dataset/kd_models.py): model construction and feature-extraction helpers.
- [trainer.py](/E:/Dataset/trainer.py): baseline classifier training and evaluation.
- [distill.py](/E:/Dataset/distill.py): distillation loss, wrapper, adapter, and student distillation training.
- [pipeline.py](/E:/Dataset/pipeline.py): high-level training and evaluation workflow helpers.
- [variant_experiments.py](/E:/Dataset/variant_experiments.py): full report generation, variant comparison, and sweep utilities.
- [visualize.py](/E:/Dataset/visualize.py): plotting and comparison helpers used by the workflow.

Local-only helper scripts:
- [local_only_scripts](/E:/Dataset/local_only_scripts): thesis/figure/export scripts that are not required for the main workflow.

## Files and Folders Usually Not Uploaded

These are normally local-only:
- `CK+`
- `checkpoints`
- `outputs`
- `local_only_scripts`
- `__pycache__`

## Environment and Dependencies

This project uses Python with PyTorch-based training and common data/plotting libraries.

Typical dependencies include:
- `torch`
- `torchvision`
- `timm`
- `numpy`
- `pandas`
- `matplotlib`
- `Pillow`
- `opencv-python`

If you run the notebook on a different machine, make sure these packages are installed first.

## Dataset Setup

The CK+ data path is currently configured in [kd_config.py](/E:/Dataset/kd_config.py):

```python
DATA_ROOT = Path("e:/Dataset/CK+/data")
```

If your dataset is stored elsewhere, this is the first place you should update.

Related paths derived from `DATA_ROOT`:
- `EXTENDED_DIR`
- `EMOTION_DIR`

## CK+ Dataset Access

This repository does not redistribute the CK+ dataset. You must obtain access separately and place the files in the local directory structure expected by the code.

Useful official or source-related references:
- CK database project page at Carnegie Mellon Robotics Institute: https://www.ri.cmu.edu/project/cohn-kanade-au-coded-facial-expression-database/
- CMU face database page: https://www.cs.cmu.edu/~face/database.htm
- CK / CK+ agreement form: https://www.jeffcohn.net/wp-content/uploads/2020/04/CK-AgreementForm.pdf
- CK+ paper PDF: https://www.jeffcohn.net/wp-content/uploads/2020/02/CVPR2010_CK2.pdf.pdf

Important note:
- The Jeffrey Cohn resources page stated that distribution of Cohn-Kanade and Extended Cohn-Kanade ended on May 1, 2025.
- In practical terms, this means the official source is no longer providing normal public dataset download in the way older CK / CK+ reproductions often assumed.
- Reproduction therefore depends on already having an authorised local copy, obtaining access through prior institutional arrangements, or confirming an alternative access route directly with the dataset owners.

Practical expectation for this project:
- the image data should be placed under `CK+/data`
- [kd_config.py](/E:/Dataset/kd_config.py) should then be updated if your local dataset root differs from the current path

The code currently expects the following structure under `DATA_ROOT`:

```text
CK+/data/
├── extended-cohn-kanade-images/
│   └── cohn-kanade-images/
└── Emotion_labels/
    └── Emotion/
```

## Recommended Way to Run the Project

The recommended workflow is to open [main.ipynb](/E:/Dataset/main.ipynb) and run only the section you need.

Suggested order:
1. Import modules.
2. Run quick checks if needed.
3. Run the full experiment report only when the configuration is correct.
4. Use checkpoint loading for inspection of saved models.
5. Use single-pair sweep sections only for focused follow-up experiments.

## What Each Notebook Section Does

### 1. Quick Checks

Purpose:
- verify dataset loading,
- verify train/validation/test split,
- preview augmentation,
- check device and model setup.

Typical cell:

```python
# context = prepare_context(show_augments=True)
# model_summary = summarize_model_setup()
# device = get_device()
```

What you can change:
- `show_augments=True` to preview training augmentations.

### 2. Full Experiment Report

Purpose:
- train or reuse student baselines,
- train or reuse teacher models,
- run all configured distillation variants,
- export histories, tables, and report files.

Typical cell:

```python
report = run_complete_variant_report(
    baseline_save_dir='checkpoints/baselines',
    teacher_save_dir='checkpoints/teachers',
    student_save_dir='checkpoints/students',
    output_dir='outputs/reports/full_variant_report',
)
```

What you can change in this cell:
- `baseline_save_dir`: where baseline checkpoints are saved.
- `teacher_save_dir`: where teacher checkpoints are saved.
- `student_save_dir`: where distilled student checkpoints are saved.
- `output_dir`: where report CSVs and generated outputs are stored.

What you can change globally for this workflow:
- student model list in `kd_config.py`
- experiment pairs in `kd_config.py`
- default distillation settings in `kd_config.py`
- training defaults in `kd_config.py`

### 3. Checkpoint Loading

Purpose:
- reload an existing baseline model,
- reload an existing distilled student checkpoint,
- inspect saved models without retraining.

Typical editable fields:
- `model_name`
- `teacher_name`
- `student_name`
- `checkpoint_path`
- `pretrained` / `pretrained_student`

### 4. Single Pair Parameter Sweep

Purpose:
- change one parameter for one teacher-student pair,
- observe how performance changes under a controlled setup.

Typical cell:

```python
# sweep_report = run_single_pair_parameter_sweep(
#     teacher_name='resnet50',
#     student_name='resnet18',
#     parameter_name='temperature',
#     parameter_values=[2.0, 3.0, 4.0, 5.0],
#     distill_mode='logits_cosine',
#     baseline_save_dir='checkpoints/baselines',
#     teacher_save_dir='checkpoints/teachers',
#     student_save_dir='checkpoints/students',
#     output_dir='outputs/reports/single_pair_parameter_sweep',
# )
```

What you can change directly:
- `teacher_name`
- `student_name`
- `parameter_name`
- `parameter_values`
- `distill_mode`
- output directories

Supported `distill_mode` values are defined in [variant_experiments.py](/E:/Dataset/variant_experiments.py):
- `logits_only`
- `logits_cosine`
- `logits_mse_regressor`
- `logits_mse_regressor_cosine`

### 5. Single Pair Multi-Sweep

Purpose:
- sweep several parameter types for one teacher-student pair,
- compare multiple settings more systematically.

Typical editable fields:
- `teacher_name`
- `student_name`
- `distill_mode`
- `sweep_types`
- `output_dir`

Default sweep candidates are defined in `SWEEP_TYPE_LIBRARY` inside [variant_experiments.py](/E:/Dataset/variant_experiments.py).

## Where to Modify Things

### Change dataset path

Edit [kd_config.py](/E:/Dataset/kd_config.py):
- `DATA_ROOT`

### Change image size, crop size, batch size, epoch count, split ratio

Edit [kd_config.py](/E:/Dataset/kd_config.py):
- `IMG_SIZE`
- `CROP_SIZE`
- `BATCH_SIZE`
- `NUM_EPOCHS`
- `VAL_SPLIT`
- `TEST_SPLIT`

### Change class mapping

Edit [kd_config.py](/E:/Dataset/kd_config.py):
- `EMOTION_ID_TO_NAME`
- `CLASS_NAMES`

### Change training hyperparameters

Edit [kd_config.py](/E:/Dataset/kd_config.py):
- `TRAINING_CONFIG`
- `MODEL_TRAINING_CONFIG`
- `DISTILLATION_CONFIG["student_training_config"]`

Important fields include:
- `epochs`
- `learning_rate`
- `weight_decay`
- `optimizer`
- `scheduler`
- `label_smoothing`

### Change which teacher/student models are used

Edit [kd_config.py](/E:/Dataset/kd_config.py):
- `DISTILLATION_CONFIG["teacher_models"]`
- `DISTILLATION_CONFIG["student_models"]`
- `DISTILLATION_CONFIG["experiments"]`

Example experiment pair format:

```python
{"teacher": "resnet50", "student": "resnet18"}
```

### Change default distillation behaviour

Edit [kd_config.py](/E:/Dataset/kd_config.py):
- `temperature`
- `alpha`
- `feature_loss_weight`
- `warmup_epochs`
- `kd_ramp_epochs`
- `feature_start_epoch`
- `gradient_clip_norm`
- `pretrained_teacher`
- `pretrained_student`

### Change student-specific distillation settings

Edit [kd_config.py](/E:/Dataset/kd_config.py):
- `DISTILLATION_CONFIG["student_specific_config"]`

This is the place to override settings such as:
- `temperature`
- `alpha`
- `feature_loss_weight`
- `feature_loss_type`
- `warmup_epochs`
- `kd_ramp_epochs`
- `feature_start_epoch`

### Change augmentation strategy

Edit [kd_config.py](/E:/Dataset/kd_config.py):
- `AUG_TYPES`
- `AUG_DISPLAY_NAMES`

### Change distillation variants available for suite experiments

Edit [variant_experiments.py](/E:/Dataset/variant_experiments.py):
- `DISTILLATION_VARIANT_LIBRARY`

This controls:
- variant label,
- whether feature distillation is enabled,
- feature loss type,
- feature loss weights.

### Change parameter sweep presets

Edit [variant_experiments.py](/E:/Dataset/variant_experiments.py):
- `SWEEP_TYPE_LIBRARY`

This controls default values used in sweep experiments, for example:
- `temperature`
- `alpha`
- `feature_loss_weight`
- `feature_start_epoch`
- `warmup_epochs`
- `kd_ramp_epochs`

## Outputs

Typical generated files include:
- baseline checkpoints in `checkpoints/baselines`
- teacher checkpoints in `checkpoints/teachers`
- distilled student checkpoints in `checkpoints/students`
- report CSVs under `outputs/reports`

The full report workflow usually creates:
- baseline summary tables,
- per-pair variant result tables,
- saved training histories,
- report directories for later plotting and thesis analysis.

## Notes About This Repository

This repository is structured so that the notebook remains relatively clean and the reusable logic stays inside Python modules.

For GitHub presentation, this is usually better than placing all training logic directly into notebook cells.

## Should the Notebook Contain More Guidance?

My recommendation is:
- keep detailed usage documentation in this README,
- keep the notebook guidance lighter and more task-oriented.

The notebook already has section-level explanations, which is good.

If you want to improve it slightly further, the best additions would be short notes such as:
- which fields in the current cell are meant to be edited,
- which output folder the cell writes to,
- whether the cell is quick or time-consuming.

That kind of extra guidance is helpful.

However, the full detailed explanation of configurable options belongs in this README rather than being repeated throughout the notebook.
