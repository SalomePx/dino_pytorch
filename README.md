# DINO_Pytorch

This repository provides a PyTorch implementation of the DINO (Self-Distillation with No Labels) framework for self-supervised learning on images, using Vision Transformers (ViT) and PyTorch Lightning.

## Project Structure

```
DINO_Pytorch/
├── data/
│   ├── augmentation.py      # Data augmentation for DINO
│   ├── datamodule.py        # PyTorch Lightning DataModule for loading datasets
├── dataset/
│   └── imagenet-mini/       # Example dataset (ImageNet-Mini) with train/val folders
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
   git clone https://github.com/yourusername/DINO_Pytorch.git
   cd DINO_Pytorch
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision pytorch-lightning
   ```

3. **Download the dataset:**
   - Place the [ImageNet-Mini](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000) dataset in `dataset/imagenet-mini/` with `train/` and `val/` folders.

## Usage

### Training

Run the training script with desired arguments:

```bash
python train_model.py --data_path dataset/imagenet-mini --epochs 100 --batch_size_per_gpu 64 --num_workers 4 --arch vit_tiny
```

**Arguments:**
- `--data_path`: Path to the dataset folder.
- `--epochs`: Number of training epochs.
- `--batch_size_per_gpu`: Batch size per GPU.
- `--num_workers`: Number of workers for data loading.
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

## License

This project is released under the MIT License.

---

**Contact:**  
For questions or issues, please open an issue on GitHub.