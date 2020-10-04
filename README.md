# Imagetranslation
image translation research

## Get started
`$ cd imagetranslation`

`$ pip install .`

`$ mkdir data`

download [following](https://drive.google.com/file/d/1x1Dm9hNbqda30KEYC8hLX1_s_cIE5Aoc/view?usp=sharing) dataset and place it in `data`


## Pretraining with CelebA dataset (without overlapping identities with MAFL)

`python scripts/train_munit.py --config configs/celeba/celeba4mafl_munit.yaml` 
`python scripts/train_tunit.py  --data_path '../data/300w/300w' --p_semi 0.0 --dataconfig configs/celeba/celeba4mafl_tunit.yaml` 

## Supervised training of linear layer & testing with MAFL dataset
`python scripts/train_regressor.py --config configs/mafl/mafl_regressor.yaml`


### TODO
- add other datasets
- add random crop in transform
- refactor regressor_trainer by separating its model parts to models file
