import argparse
import json
import os

from dipy.align.reslice import reslice
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
import SimpleITK as sitk


def preprocess_img(inputfile, output_preprocessed, zooms):
    img = nib.load(inputfile)
    data = img.get_data()
    affine = img.affine
    zoom = img.header.get_zooms()[:3]
    data, affine = reslice(data, affine, zoom, zooms, 1)
    data = np.squeeze(data)
    data = np.pad(data, [(0, 256 - len_) for len_ in data.shape], "constant")

    data_sub = data - gaussian_filter(data, sigma=1)
    img = sitk.GetImageFromArray(np.copy(data_sub))
    img = sitk.AdaptiveHistogramEqualization(img)
    data_clahe = sitk.GetArrayFromImage(img)[:, :, :, None]
    data = np.concatenate((data_clahe, data[:, :, :, None]), 3)
    data = (data - np.mean(data, (0, 1, 2))) / np.std(data, (0, 1, 2))
    assert data.ndim == 4, data.ndim
    assert np.allclose(np.mean(data, (0, 1, 2)), 0.), np.mean(data, (0, 1, 2))
    assert np.allclose(np.std(data, (0, 1, 2)), 1.), np.std(data, (0, 1, 2))
    data = np.float32(data)

    img = nib.Nifti1Image(data, affine)
    nib.save(img, output_preprocessed)


def preprocess_label(inputfile,
                     output_label,
                     n_classes,
                     zooms,
                     df=None,
                     input_key=None,
                     output_key=None):
    img = nib.load(inputfile)
    data = img.get_data()
    affine = img.affine
    zoom = img.header.get_zooms()[:3]
    data, affine = reslice(data, affine, zoom, zooms, 0)
    data = np.squeeze(data)
    data = np.pad(data, [(0, 256 - len_) for len_ in data.shape], "constant")

    if df is not None:
        tmp = np.zeros_like(data)
        for target, source in zip(df[output_key], df[input_key]):
            tmp[np.where(data == source)] = target
        data = tmp
    data = np.int32(data)
    assert np.max(data) < n_classes
    img = nib.Nifti1Image(data, affine)
    nib.save(img, output_label)


def main():
    parser = argparse.ArgumentParser(description="preprocess dataset")
    parser.add_argument(
        "--input_directory", "-i", type=str,
        help="directory of original dataset"
    )
    parser.add_argument(
        "--subjects", "-s", type=str, nargs="*", action="store",
        help="subjects to be preprocessed"
    )
    parser.add_argument(
        "--weights", "-w", type=int, nargs="*", action="store",
        help="sample weight for each subject"
    )
    parser.add_argument(
        "--input_image_suffix", type=str,
        help="suffix of input images"
    )
    parser.add_argument(
        "--output_image_suffix", type=str,
        help="suffix of output images"
    )
    parser.add_argument(
        "--label_suffix", type=str,
        help="suffix of labels"
    )
    parser.add_argument(
        "--output_file", "-f", type=str, default="dataset.json",
        help="json file of preprocessed dataset, default=dataset.json"
    )
    parser.add_argument(
        "--label_file", "-l", type=str, default=None,
        help="csv file with label translation rule, default=None"
    )
    parser.add_argument(
        "--input_key", type=str, default=None,
        help="specifies column for input of label translation, default=None"
    )
    parser.add_argument(
        "--output_key", type=str, default=None,
        help="specifies column for output of label translation, default=None"
    )
    parser.add_argument(
        "--n_classes", type=int,
        help="number of classes to classify"
    )
    parser.add_argument(
        "--zooms", type=float, nargs="*", action="store", default=[1., 1., 1.],
        help="zooming resolution"
    )
    args = parser.parse_args()
    if args.weights is None:
        args.weights = [1. for _ in args.subjects]
    assert len(args.subjects) == len(args.weights)
    print(args)

    if args.label_file is None:
        df = None
    else:
        df = pd.read_csv(args.label_file)

    dataset = {"in_channels": 2, "n_classes": args.n_classes}
    dataset_list = []

    for subject, weight in zip(args.subjects, args.weights):
        if not os.path.exists(subject):
            os.makedirs(subject)
        filedict = {"subject": subject, "weight": weight}

        if args.input_image_suffix is not None:
            filedict["image"] = os.path.join(
                subject,
                subject + args.output_image_suffix
            )
            preprocess_img(
                os.path.join(
                    args.input_directory,
                    subject,
                    subject + args.input_image_suffix
                ),
                filedict["image"],
                args.zooms
            )

        if args.label_suffix is not None:
            filedict["label"] = os.path.join(
                subject,
                subject + args.label_suffix
            )
            preprocess_label(
                os.path.join(
                    args.input_directory,
                    subject,
                    subject + args.label_suffix
                ),
                filedict["label"],
                args.n_classes,
                args.zooms,
                df=df,
                input_key=args.input_key,
                output_key=args.output_key
            )

        dataset_list.append(filedict)
    dataset["data"] = dataset_list

    with open(args.output_file, "w") as f:
        json.dump(dataset, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()