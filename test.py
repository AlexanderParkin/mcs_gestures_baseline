import sys
import yaml
import argparse

import numpy as np
import pandas as pd
import cv2
import torch
import face_detection as fd
from tqdm import tqdm
from data.augmentations import ValidationAugmentations
from torchvision import transforms as tfs

from utils import convert_dict_to_tuple
from utils import get_max_bbox
from utils import load_resnet


def save_results(scores, save_path):
    result_df = pd.DataFrame({
        'no_gesture': scores[:, 0],
        'stop': scores[:, 1],
        'victory': scores[:, 2],
        'mute': scores[:, 3],
        'ok': scores[:, 4],
        'like': scores[:, 5],
        'dislike': scores[:, 6]
    })
    result_df.to_csv(save_path, index=False)


def main(args):
    with open(args.cfg) as f:
        data = yaml.safe_load(f)
    config = convert_dict_to_tuple(data)

    detector = fd.build_detector(args.detector_type,
                                 confidence_threshold=.5,
                                 nms_iou_threshold=.3,
                                 device=torch.device("cuda"),
                                 max_resolution=640)

    model = load_resnet(args.model_path, config.model.model_type, config.dataset.num_of_classes)

    softmax_func = torch.nn.Softmax(dim=1)
    val_augs = ValidationAugmentations(config)
    preproc = tfs.Compose([tfs.ToTensor(), tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    test_df = pd.read_csv(args.test_list)

    scores = np.zeros((len(test_df), 7), dtype=np.float32)
    scores[:, 0] = 1

    for idx, image_path in tqdm(enumerate(test_df.frame_path.values), total=len(test_df)):
        image_path = image_path
        img = cv2.imread(image_path,
                         cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        detections = detector.detect(img)

        all_faces = []
        for det in detections:
            x1, y1, x2, y2, s = det.tolist()
            w = x2 - x1
            h = y2 - y1
            bbox = [x1, y1, w, h]
            all_faces.append(bbox)

        if len(all_faces) > 0:
            max_bbox = get_max_bbox(all_faces)
            crop, *_ = val_augs(img, max_bbox, None)
            crop = preproc(crop).unsqueeze(0)
            crop = crop.to('cuda')
            out = model(crop)
            out = softmax_func(out).squeeze().detach().cpu().numpy()
            scores[idx] = out

    save_results(scores, args.out_path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='Path to config file.')
    parser.add_argument('--test_list', type=str, default='', help='Path to test list file.')
    parser.add_argument('--model_path', type=str, default='', help='Path to model file.')
    parser.add_argument('--detector_type', type=str, default='RetinaNetMobileNetV1', help='detector name')
    parser.add_argument('--out_path', type=str, default='submit.csv', help='Path to save submit file')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
