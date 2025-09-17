# segpaste

This package is a torch-only reimplementation of the copy-paste augmentation: https://arxiv.org/abs/2012.07177.

A PyTorch implementation of "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation" with minimal dependencies and full type safety.

## Features

- 🔥 **Pure PyTorch**: Uses only `torch` and `torchvision` - no additional dependencies
- 🎯 **Type Safe**: Full type hints with mypy validation (`strict = true`)
- ⚡ **Efficient**: Vectorized operations using native PyTorch tensors
- 🧩 **Flexible**: Multiple blending modes (alpha, gaussian, poisson)
- 🎨 **Easy Integration**: Works with torchvision transforms and datasets
- 🧪 **Well Tested**: Comprehensive test suite with pytest

## Installation

```bash
pip install segpaste
```

Or install from source:

```bash
git clone https://github.com/NoeFontana/segpaste.git
cd segpaste
pip install -e .
```

## Quick Start

```python
import torch
from segpaste import CopyPasteAugmentation, CopyPasteConfig, DetectionTarget

# Configure copy-paste augmentation
config = CopyPasteConfig(
    paste_probability=0.5,
    max_paste_objects=3,
    scale_range=(0.5, 2.0),
    blend_mode="gaussian"
)

# Initialize augmentation
copy_paste = CopyPasteAugmentation(config)

# Create detection target
target_data = DetectionTarget(
    image=torch.rand(3, 512, 512),     # [C, H, W]
    boxes=torch.tensor([[50, 50, 150, 150]]),  # [N, 4] in xyxy format
    labels=torch.tensor([1]),           # [N]
    masks=torch.rand(1, 512, 512) > 0.5  # [N, H, W]
)

# Apply augmentation
source_objects = [...]  # List of DetectionTarget objects to paste
augmented = copy_paste(target_data, source_objects)

print(f"Objects before: {target_data.boxes.shape[0]}")
print(f"Objects after: {augmented.boxes.shape[0]}")
```

## Integration with torchvision

### Using CopyPasteTransform

```python
from torchvision import transforms
from segpaste import CopyPasteTransform, CopyPasteConfig

# Create source objects for pasting
source_objects = [...]  # Your source DetectionTarget objects

# Create transform
copy_paste_transform = CopyPasteTransform(
    source_objects=source_objects,
    config=CopyPasteConfig(paste_probability=0.3)
)

# Use in transform pipeline
transform = transforms.Compose([
    # ... other transforms
    copy_paste_transform,
    # ... more transforms
])

# Apply to sample
sample = {
    'image': image,
    'boxes': boxes,
    'labels': labels,
    'masks': masks
}
augmented_sample = transform(sample)
```

### Batch-time Augmentation

```python
from torch.utils.data import DataLoader
from segpaste import CopyPasteCollator

# Use copy-paste collator for batch-time augmentation
collator = CopyPasteCollator(
    config=CopyPasteConfig(paste_probability=0.4)
)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=collator
)

for batch in dataloader:
    # batch contains augmented data with objects
    # copied between images in the same batch
    images = batch['images']  # [B, C, H, W]
    boxes = batch['boxes']    # List of [N_i, 4] tensors
    labels = batch['labels']  # List of [N_i] tensors
    masks = batch['masks']    # List of [N_i, H, W] tensors
```

## Configuration

The `CopyPasteConfig` class provides comprehensive configuration options:

```python
from segpaste import CopyPasteConfig

config = CopyPasteConfig(
    paste_probability=0.5,           # Probability of applying copy-paste
    max_paste_objects=10,           # Maximum objects to paste per image
    min_paste_objects=1,            # Minimum objects to paste per image
    scale_range=(0.1, 2.0),         # Range for random scaling of pasted objects
    blend_mode="alpha",             # Blending mode: "alpha", "gaussian", "poisson"
    occluded_area_threshold=0.3,    # Objects with >30% occlusion are removed
    box_update_threshold=10.0,      # Minimum pixel difference for box updates
    enable_rotation=False,          # Enable random rotation (not implemented)
    enable_flip=True                # Enable random horizontal flip
)
```

## Blending Modes

### Alpha Blending (default)

Simple linear blending between source and target:

```python
config = CopyPasteConfig(blend_mode="alpha")
```

### Gaussian Blending

Smooth transitions with Gaussian-weighted masks:

```python
config = CopyPasteConfig(
    blend_mode="gaussian",
    # Additional parameters can be passed during blending
)
```

### Poisson Blending

Seamless composition (simplified implementation):

```python
config = CopyPasteConfig(blend_mode="poisson")
```

## Advanced Usage

### Custom Source Object Selection

```python
class CustomCopyPaste(CopyPasteAugmentation):
    def _select_objects_to_paste(self, source_objects):
        # Custom logic for selecting which objects to paste
        # e.g., class-balanced sampling, size filtering, etc.
        return super()._select_objects_to_paste(source_objects)
```

### Integrating with Detection Frameworks

The library is designed to work seamlessly with popular detection frameworks:

**Detectron2:**

```python
# Convert from Detectron2 format
def detectron2_to_detection_target(instances, image):
    return DetectionTarget(
        image=image,
        boxes=instances.gt_boxes.tensor,
        labels=instances.gt_classes,
        masks=instances.gt_masks.tensor
    )
```

**MMDetection:**

```python
# Convert from MMDet format
def mmdet_to_detection_target(ann, image):
    return DetectionTarget(
        image=image,
        boxes=torch.from_numpy(ann['bboxes']),
        labels=torch.from_numpy(ann['labels']),
        masks=torch.from_numpy(ann['masks'])
    )
```

## Technical Details

### Key Features from the Paper

1. **Large Scale Jittering (LSJ) Compatibility**: Works with multi-scale training
2. **Proper Occlusion Handling**: Updates masks and removes heavily occluded objects
3. **Scale-aware Placement**: Intelligent object placement with collision detection
4. **Box Recomputation**: Bounding boxes are recomputed from updated masks

### Performance Optimizations

- **Vectorized Operations**: All operations use PyTorch tensors for GPU acceleration
- **Memory Efficient**: Minimizes tensor copying and uses in-place operations where possible
- **Batch Processing**: Supports efficient batch-time augmentation

### Type Safety

The entire codebase is fully typed and validated with mypy in strict mode:

```bash
mypy src tests --strict
```

## Development

### Code formatting, linting and type checking

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting and [mypy](http://mypy-lang.org/) for type checking.

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
ruff format src tests examples

# Run linter with auto-fix
ruff check --fix src tests examples

# Run type checking
mypy src tests examples
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=segpaste --cov-report=html --cov-report=term-missing

# Run specific test file
pytest tests/test_copy_paste.py -v
```

### Example Scripts

See the `examples/` directory for complete usage examples:

```bash
# Run basic example
python examples/basic_usage.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{ghiasi2021simple,
  title={Simple copy-paste is a strong data augmentation method for instance segmentation},
  author={Ghiasi, Golnaz and Cui, Yin and Srinivas, Aravind and Qian, Rui and Lin, Tsung-Yi and Cubuk, Ekin D and Le, Quoc V and Zoph, Barret},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

## License

MIT License - see LICENSE file for details.
