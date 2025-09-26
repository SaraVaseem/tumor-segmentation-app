import os
import time
from pixellib.custom_train import instance_custom_training

# ===== CONFIGURATION =====
MODEL_DIR = "mask_rcnn_models"  # Folder for model saves
DATASET_PATH = "Tumor"  # Contains train/test folders
PRETRAINED_MODEL = "mask_rcnn_coco.h5"


class TrainingConfig:
    # Model Architecture
    BACKBONE = "resnet101"  # Only "resnet101" works in v0.7.1
    NUM_CLASSES = 1  # Background + tumor

    # Training Phases
    PHASE1_EPOCHS = 20  # Head training
    PHASE2_EPOCHS = 80  # Full model fine-tuning
    BATCH_SIZE = 4  # Reduce if OOM errors occur

    # Augmentation (Limited options in v0.7.1)
    AUGMENTATION = True  # Only boolean, no dict configuration


def clean_artifacts():
    """Remove problematic files before training"""
    json_path = os.path.join(DATASET_PATH, "train.json")
    try:
        if os.path.exists(json_path):
            if os.path.isdir(json_path):
                import shutil
                shutil.rmtree(json_path, ignore_errors=True)
            else:
                os.remove(json_path)
        time.sleep(2)  # Crucial for Windows filesystem
    except Exception as e:
        print(f"Cleanup warning: {str(e)}")


def train_model():
    # 1. Pre-training cleanup
    clean_artifacts()

    # 2. Initialize trainer
    trainer = instance_custom_training()
    trainer.modelConfig(
        network_backbone=TrainingConfig.BACKBONE,
        num_classes=TrainingConfig.NUM_CLASSES,
        batch_size=TrainingConfig.BATCH_SIZE
    )

    # 3. Load pretrained weights
    if PRETRAINED_MODEL and os.path.exists(PRETRAINED_MODEL):
        trainer.load_pretrained_model(PRETRAINED_MODEL)

    # 4. Load dataset
    trainer.load_dataset(DATASET_PATH)

    # 5. Two-phase training
    print("\n=== PHASE 1: Training Heads ===")
    trainer.train_model(
        num_epochs=TrainingConfig.PHASE1_EPOCHS,
        augmentation=TrainingConfig.AUGMENTATION,
        path_trained_models=MODEL_DIR,
        layers='heads'
    )

    print("\n=== PHASE 2: Fine-tuning ===")
    trainer.train_model(
        num_epochs=TrainingConfig.PHASE2_EPOCHS,
        augmentation=TrainingConfig.AUGMENTATION,
        path_trained_models=MODEL_DIR,
        layers='all'  
    )


if __name__ == "__main__":
    # Verify paths
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset folder '{DATASET_PATH}' not found")
    else:
        os.makedirs(MODEL_DIR, exist_ok=True)
        train_model()
        print(f"\nTraining completed. Models saved to {MODEL_DIR}")