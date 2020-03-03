import math
import os.path
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence

from config import text_to_labels


class ShiXinSequence(Sequence):
    def __init__(
        self,
        data_dir,
        width=160,
        height=70,
        n_len=4,
        input_length=38,
        label_length=4,
        batch_size=32,
    ):
        self.data_dir = data_dir
        self.width = width
        self.height = height
        self.n_len = n_len
        self.input_length = input_length
        self.label_length = label_length
        self.batch_size = batch_size

        self.init()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]

        input_length = np.ones(self.batch_size) * self.input_length
        label_length = np.ones(self.batch_size) * self.label_length
        return (batch_x, batch_y, input_length, label_length), np.ones(self.batch_size)

    def init(self):
        all_image_files = list(Path(self.data_dir).glob("*.jpg"))
        n = len(all_image_files)

        self.x = np.zeros((n, self.height, self.width, 1))
        self.y = np.zeros((n, self.n_len))

        for idx, filename in enumerate(all_image_files):
            text = os.path.split(filename)[-1].split("_")[0].lower()
            self.x[idx] = np.expand_dims(np.array(Image.open(filename)), -1) / 255.0
            self.y[idx] = text_to_labels(text)
