import os
import shutil

src_path = './cirtorch/dataset/'
dst_train = './cirtorch/data/train/'
dst_valid = './cirtorch/data/valid/'


# partition the original data into train and valid set
train_size = 0.8
for root, dir, file in os.walk(src_path):
    train_len = int(train_size*len(file))

    for FILENAME in file[:train_len]:
        SRC = root + "/" + FILENAME
        DST = dst_train + FILENAME
        shutil.move(SRC,DST)

    for FILENAME in file[train_len:]:
        SRC = root + "/" +  FILENAME
        DST = dst_valid + FILENAME
        shutil.move(SRC,DST)
