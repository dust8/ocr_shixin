import os
import string

HEIGHT, WIDTH, CHANNELS = 70, 160, 1

# 列出可用的标签, 这里不分大小写
LABEL_NAMES = string.digits + string.ascii_lowercase
# 为每个标签分配索引
LABEL_TO_INDEX = dict((name, index) for index, name in enumerate(LABEL_NAMES))
INDEX_TO_LABEL = dict((index, name) for index, name in enumerate(LABEL_NAMES))


# 定义路径
OUTPUT_PATH = os.path.join(".", "output")
TARGET_PATH = os.path.join(OUTPUT_PATH, "checkpoint_weights.hdf5")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 导出 tensorflow serving 模型
MODEL_DIR = "Shixin"
MODEL_VERSION = 1


def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(LABEL_TO_INDEX.get(char))
    return ret


def labels_to_text(labels):
    ret = []
    for label in labels:
        ret.append(INDEX_TO_LABEL.get(label, ""))
    return "".join(ret)
