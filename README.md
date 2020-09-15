# Imagetranslation
image translation research

## Get started
`$ cd imagetranslation`

`$ pip install .`

`$ mkdir data`

download following dataset and place it in `data`


## Pretraining with CelebA dataset (without overlapping identities with MAFL)

`python scripts/train_munit.py --config configs/celeba/celeba4mafl_munit.yaml` 

## Supervised training of linear layer & testing with MAFL dataset
`python scripts/train_regressor.py --config configs/mafl/mafl_regressor.yaml`

TODO
- add other datasets
- add random crop in transform
