import yaml
import datasets
from utils import get_instance

with open('../configs/newcode.yaml') as stream:
    config = yaml.load(stream)

cel = get_instance(datasets, config, 'data', 'dataset', train=False)
print(cel)
print(len(cel))
