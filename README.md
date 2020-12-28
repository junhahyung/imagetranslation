# Imagetranslation
image translation research

## Get started
`$ cd imagetranslation`

`$ pip install .`

`$ mkdir data`

download [following](https://drive.google.com/file/d/1x1Dm9hNbqda30KEYC8hLX1_s_cIE5Aoc/view?usp=sharing) dataset and place it in `data`


## Pretraining with CelebA dataset (without overlapping identities with MAFL)

`python scripts/train_munit.py --config configs/celeba/celeba4mafl_munit.yaml` 

`python scripts/train_tunit.py --dataconfig configs/celeba/celeba4mafl_tunit.yaml` 

## Supervised training of linear layer & testing with MAFL dataset
MUNIT (not finished yet, but working): `python scripts/train_regressor_deprecated.py --config configs/mafl/mafl_regressor.yaml`

TUNIT: `python scripts/train_regressor.py --load_model GAN_20201007-224815`


### TODO

