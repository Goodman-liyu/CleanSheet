# Hijacking-Attacks-against-Neural-Network
Hijacking Attacks against Neural Network by Analyzing Training Data Implementation

## Preparation

1. Download python 3.10

2. Install poetry according to [official document](https://python-poetry.org/docs/#installation)

3. Run following command

```bash
poetry install --without tools
```

4. Run unittests

```bash
poetry run pytest tests/
```

## Sample trigger
```Python
# load trigger and mask
a = torch.load('epoch_99.pth')
tri = a['trigger']
mask = a['mask']
```
