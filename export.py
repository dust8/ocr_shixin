import os
import shutil

import tensorflow as tf

from config import MODEL_DIR, MODEL_VERSION, TARGET_PATH
from network.model import ShiXinModel


def main():
    sx_predict_model = ShiXinModel()
    sx_predict_model.compile()
    sx_predict_model.load_checkpoint(TARGET_PATH)

    export_path = os.path.join(MODEL_DIR, str(MODEL_VERSION))

    print(f"Export path: {export_path}")
    if os.path.isdir(export_path):
        print("Already saved a model, cleaning up")
        shutil.rmtree(export_path)

    tf.saved_model.save(
        sx_predict_model.model, export_path,
    )
    print(f"Saved mode: {export_path}")


if __name__ == "__main__":
    main()
