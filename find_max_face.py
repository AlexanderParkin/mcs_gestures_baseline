import os
import json

import pandas as pd
import sys
import argparse
import cv2
import torch
import face_detection as fd
from tqdm import tqdm

from utils import get_max_bbox

CLASS_NAME2LABEL_DICT = {
    'no_gesture': 0,
    'stop': 1,
    'victory': 2,
    'mute': 3,
    'ok': 4,
    'like': 5,
    'dislike': 6
}


def main(args):
    detector = fd.build_detector(args.detector_type,
                                 confidence_threshold=.5,
                                 nms_iou_threshold=.3,
                                 device=torch.device("cuda"),
                                 max_resolution=640)
    data_df = pd.read_csv(args.data_list)
    result_arr = []
    for idx, image_path in tqdm(enumerate(data_df.frame_path.values), total=len(data_df)):
        image_path = os.path.join(args.prefix_path, image_path)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        detections = detector.detect(img)

        all_faces = []
        for det in detections:
            x1, y1, x2, y2, s = det.tolist()
            w = x2 - x1
            h = y2 - y1
            bbox = [round(x1), round(y1), round(w), round(h)]
            all_faces.append(bbox)

        if len(all_faces) > 0:
            max_bbox = get_max_bbox(all_faces)
            item = {
                'frame_path': image_path,
                'video_name': data_df.video_name.iloc[idx],
                'frame_id': int(data_df.frame_id.iloc[idx]),
                'class': CLASS_NAME2LABEL_DICT.get(data_df.class_name.iloc[idx]),
                'bbox': max_bbox
            }
            result_arr.append(item)

    with open(args.output_json_path, 'w') as fout:
        json.dump(result_arr, fout, indent=4)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix_path', type=str, help='Path to directory with data')
    parser.add_argument('--data_list', type=str, default='./train.csv', help='Path to data list file.')
    parser.add_argument('--output_json_path', type=str, default='./bboxes.json', help='Path to output json file.')
    parser.add_argument('--detector_type', type=str, default="RetinaNetResNet50", help='detector name')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
