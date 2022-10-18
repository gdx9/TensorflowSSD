import sys
sys.path.append('..')# parent directory

from ssd_settings import *
from ssd_data_generator import *

if __name__ == '__main__':
    pic_generator = SyntheticDatasetGenerator()

    # avg speed: 1000 pics in 11 mins
    pic_generator.generateDataset(save_dir=data_dir, generate_number=800, save=True)
