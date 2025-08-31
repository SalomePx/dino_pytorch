# DINO_Pytorch

This repository provides a structured PyTorch (2.7.0) implementation of the DINO (Self-Distillation with No Labels) framework for self-supervised learning on images. It uses Vision Transformers (ViT) and PyTorch Lightning modules for scalability and modularity. 

## Project Structure

```
DINO_Pytorch/
├── data/
│   ├── augmentation.py      # Data augmentation for DINO
│   ├── datamodule.py        # PyTorch Lightning DataModule for loading datasets
├── dataset/
│   └── imagenet-mini/       # Example dataset (ImageNet-mini) with train/val folders
├── loss/
│   └── dino_loss.py         # DINO loss implementation
├── models/
│   ├── dino.py              # Main DINO LightningModule
│   ├── models.py            # Vision Transformer and related modules
├── train_model.py           # Training script
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/salomepx/dino_pytorch.git
   cd dino_pytorch
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   - Place the [ImageNet-Mini](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000) dataset in `dataset/imagenet-mini/` with `train/` and `val/` folders.

## Usage

### Training

Run the training script with desired arguments:

```bash
python train_model.py --data_path dataset/imagenet-mini --epochs 100 --arch vit_tiny
```

**Arguments:**
- `--data_path`: Path to the dataset folder.
- `--epochs`: Number of training epochs.
- `--arch`: Model architecture (`vit_tiny`, `vit_small`, `vit_base`, etc.)

Additional arguments for DINO, VisionTransformer, and DINOLoss can be set as needed.

### Checkpoints

Model checkpoints are saved in the `./checkpoints` directory.

## Main Components

- **Data Augmentation:** Implements DINO-specific augmentations for global and local crops.
- **DataModule:** Handles dataset loading and batching using PyTorch Lightning.
- **Vision Transformer:** Custom ViT implementation for use as student and teacher networks.
- **DINO Loss:** Implements the self-distillation loss function.
- **LightningModule:** Wraps the training logic, optimizer, and EMA teacher updates.

## Customization

- Modify `models/models.py` to change the Vision Transformer architecture.
- Adjust augmentations in `data/augmentation.py`.
- Add or change loss parameters in `loss/dino_loss.py`.

## References

- [DINO: Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
- [Original DINO code (Facebook Research)](https://github.com/facebookresearch/dino)

---

**Contact:**  
For questions or issues, please open an issue on GitHub.

**To Do List:**  
- [x] Create the training script
- [ ] Create the evaluation script
