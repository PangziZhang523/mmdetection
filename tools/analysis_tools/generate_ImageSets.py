import os
import random
from glob import glob

im_dir = '/home/data/51/'

im_list = glob(os.path.join(im_dir, '*.jpg'))

def split(full_list, shuffle=False, ratio=0.2):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total==0 or offset<1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


train, val = split(im_list, shuffle=True, ratio=0.8)


train_file_name = '/project/train/data/train.txt'
train_path = open(train_file_name, "a+")

for im_path in train:
    img_name = im_path.split('/')[-1].split('.jpg')[0]

    train_path.write(f"{img_name}\n")

train_path.flush()
train_path.close()



val_file_name = '/project/train/data/val.txt'
val_path = open(val_file_name, "a+")

for im_path in val:
    img_name = im_path.split('/')[-1].split('.jpg')[0]

    val_path.write(f"{img_name}\n")

val_path.flush()
val_path.close()

