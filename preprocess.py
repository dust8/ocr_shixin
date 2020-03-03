import pathlib

from image_utils import adaptive_threshold, load_image, rgb2gray, save_image


def preprocessing_image(path):
    origin_root = pathlib.Path(path)
    origin_all_image_paths = list(origin_root.glob("*.jpg"))
    print(f"image count: {len(origin_all_image_paths)}")

    for img_path in origin_all_image_paths:
        img = load_image(img_path)
        img = rgb2gray(img)
        img = img.astype("uint8")
        img = adaptive_threshold(img)
        dst = "/".join(img_path.parts).replace("origin", "binary")
        save_image(img, dst)


def main():
    preprocessing_image("./dataset/origin")
    preprocessing_image("./dataset/val_origin")


if __name__ == "__main__":

    main()
