import sys
import os
import argparse
import json
import numpy as np
from tqdm import tqdm


def main(args: argparse.Namespace) -> None:
    """
    Running the code for dividing the original training data into training and validation dataset,
    taking into account parameter 'video_name'.
    :param args: all parameters necessary for launch
    :return:
    """
    with open(args.train_list, 'r') as fin:
        full_train_data = json.load(fin)

    all_videos = set()
    for sample in full_train_data:
        all_videos.add(sample['video_name'])

    all_videos = list(all_videos)
    number_val_samples = int(len(all_videos) * args.val_size)
    val_videos = np.random.choice(all_videos, size=number_val_samples, replace=False)

    train_data = []
    val_data = []
    for sample in tqdm(full_train_data):
        if sample[args.bbox_key] is None:
            continue

        new_sample = {
            'frame_path': os.path.join(args.pp, sample['frame_path']),
            'label': sample['label'],
            'bbox': sample[args.bbox_key]
        }

        if sample['video_name'] in val_videos:
            val_data.append(new_sample)
        else:
            train_data.append(new_sample)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'train.json'), 'w') as fout:
        json.dump(train_data, fout)
    with open(os.path.join(args.output_dir, 'val.json'), 'w') as fout:
        json.dump(val_data, fout)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', type=str, default='./lists/train_with_bboxes.json', help='Path to json file ')
    parser.add_argument('--output_dir', type=str, default='./lists/baseline_exp/', help='Path to test list file.')
    parser.add_argument('--bbox_key', type=str, default='bbox_RetinaNetResNet50', help='Which bbox to use')
    parser.add_argument('--val_size', type=float, default=0.15, help='What part of the data to use for validation')
    parser.add_argument('--pp', type=str,
                        help='Path to the directory where images from the trainset are unpacked')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))