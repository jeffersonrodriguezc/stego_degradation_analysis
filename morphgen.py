import os
import csv
import sys
import argparse
import itertools
import multiprocessing

#sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from pathlib import Path
from libs.morpher import morph

def parse_launch_cmd():
    parser = argparse.ArgumentParser(
        prog="MorphGEN", description="Landmark-based morph creation tool"
    )

    parser.add_argument(
        "--from_images",
        nargs="+",
        help="Generate morphs from a list of images passed as arguments. At least 2 images are required",
    )
    parser.add_argument(
        "--from_file",
        nargs=1,
        help="Generate morphs from a file containing a list of image pairs. The file must be a CSV or TXT file",
    )
    parser.add_argument(
        "--from_dir",
        nargs=1,
        help="Generate morphs from a directory of images. The directory must contain at least 2 images",
    )
    parser.add_argument("--output_dir", default="output", help="Output directory")
    parser.add_argument("--alpha", nargs=1, default=0.5, help="Blending Factor")
    parser.add_argument(
        "--background",
        nargs=1,
        default="seamless",
        help="Background type. Select between: 'black', 'transparent', 'seamless'",
    )

    args = parser.parse_args()

    if args.from_images:
        parse_image_list(args.from_images, args.output_dir, args.alpha, args.background)
    elif args.from_file:
        parse_image_file(args.from_file, args.output_dir, args.alpha, args.background)
    elif args.from_dir:
        parse_image_dir(args.from_dir, args.output_dir, args.alpha, args.background)
    else:
        parser.print_help()


def parse_image_list(image_list, output_dir, alpha, background):
    if len(image_list) < 2:
        raise ValueError("Must provide at least 2 images")
    else:
        image_list = [Path(x).resolve() for x in image_list]
        output_dir = Path(output_dir).resolve()

        output_dir.mkdir(parents=True, exist_ok=True)

        permutations = itertools.permutations(image_list, 2)
        args = [
            [*x, output_dir / f"{x[0].stem}_{x[1].stem}.png", alpha, background]
            for x in permutations
        ]

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        with pool as p:
            p.starmap(morph, args)
            p.close()
            p.join()


def parse_image_file(image_file, output_dir, alpha, background):
    # Reading CSV file

    image_file = Path(image_file[0]).resolve()
    if image_file.suffix == ".csv":
        with open(image_file, "r") as f:
            reader = csv.reader(f, delimiter=",")
            args = [
                [
                    Path(x[0]).resolve(),
                    Path(x[1]).resolve(),
                    (Path(output_dir) / f"{Path(x[0]).stem}_{Path(x[1]).stem}.png").resolve(),
                    alpha,
                    background,
                ]
                for x in reader
            ]
    elif image_file.suffix == ".txt":
        with open(image_file, "r") as f:
            args = [
                [
                    Path(x.split(" ")[0]).resolve(),
                    Path(x.split(" ")[1]).resolve(),
                    (Path(output_dir) / f"{Path(x.split(' ')[0]).stem}_{Path(x.split(' ')[1]).stem}.png").resolve(),
                    alpha,
                    background,
                ]
                for x in f.readlines()
            ]
    else:
        raise ValueError(f"Unsupported file type: {image_file.suffix}")

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    with pool as p:
        p.starmap(morph, args)
        p.close()
        p.join()


def parse_image_dir(image_dir, output_dir, alpha, background):
    if len(image_dir) != 1:
        raise ValueError("Must provide at least 2 images")
    else:
        files = [Path(x).resolve() for x in image_dir.iterdir if x.lower().endswith(".jpg", ".jpeg", ".png")]
        output_dir = Path(output_dir).resolve()

        output_dir.mkdir(parents=True, exist_ok=True)

        permutations = itertools.permutations(files, 2)
        args = [
            [*x, output_dir / f"{x[0].stem}_{x[1].stem}.png", alpha, background]
            for x in permutations
        ]

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        with pool as p:
            p.starmap(morph, args)
            p.close()
            p.join()


if __name__ == "__main__":
    parse_launch_cmd()
