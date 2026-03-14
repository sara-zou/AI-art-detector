import shutil
import os

source = "human_art_data"
train_dest = "dataset/train/human"
val_dest = "dataset/val/human"
test_dest = "dataset/test/human"

os.makedirs(train_dest, exist_ok=True)
os.makedirs(val_dest, exist_ok=True)
os.makedirs(test_dest, exist_ok=True)