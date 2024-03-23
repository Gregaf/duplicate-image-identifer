import logging
from utils.args import parse_args
from utils.logger import config_logger
import os


def find_similar_images():
    pass


def get_image_paths(dir_path: str) -> list[str]:
    image_paths = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file))
    return image_paths


def main():
    config_logger()

    args = parse_args()
    logging.debug(f"Target directory was: '{args.dir}'")

    image_file_paths = get_image_paths(args.dir)
    logging.info(f"Found '{len(image_file_paths)}' image files in the directory")
    logging.debug(f"Image file paths: {image_file_paths}")


if __name__ == "__main__":
    main()
