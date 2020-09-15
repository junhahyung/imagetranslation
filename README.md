# Imagetranslation
image translation research

## Get started
`pip install .`

## Pretraining with CelebA dataset

`python scripts/train_munit.py --config configs/celeba/celeba_munit.yaml` 

## Supervised training of linear layer & testing with MAFL dataset
`python scripts/train_regressor.py --config configs/mafl/mafl_regressor.yaml`
