import argparse
import dataclasses


@dataclasses.dataclass
class Args:
    dir: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Identify similar images in a directory."
    )
    parser.add_argument("dir", type=str, help="Directory containing images")

    args = parser.parse_args()

    return Args(dir=args.dir)
