import sys
import yaml
from collections import namedtuple

import numpy as np
import pandas as pd
import cv2
import torch
import face_detection as fd
from torchvision import models
from torchvision import transforms as tfs
from tqdm import tqdm

from utils import convert_dict_to_tuple
from data.augmentations import ValidationAugmentations


def convert_dict_to_tuple(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert_dict_to_tuple(value)
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def get_max_bbox(bboxes):
    bbox_sizes = [x[2] * x[3] for x in bboxes]
    max_bbox_index = np.argmax(bbox_sizes)
    return bboxes[max_bbox_index]


def load_resnet(path, model_type, num_classes, device='cuda'):
    if model_type == 'resnet34' or model_type == 'resnet34':
        model = models.resnet34(pretrained=False)
        model.fc = torch.nn.Linear(512, num_classes)
        model.load_state_dict(torch.load(path, map_location='cpu')["state_dict"])
    else:
        raise Exception("Unknown model type: {}".format(model_type))
    model.to(device)
    model.eval()
    return model


def save_results(scores, frame_pathes, save_path):
    result_df = pd.DataFrame({
        'no_gesture': scores[:, 0],
        'stop': scores[:, 1],
        'victory': scores[:, 2],
        'mute': scores[:, 3],
        'ok': scores[:, 4],
        'like': scores[:, 5],
        'dislike': scores[:, 6],
        'frame_path': frame_pathes
    })

    result_df.to_csv(save_path, index=False)


CONFIG_PATH = './baseline_mcs.yml'
DETECTOR_TYPE = 'RetinaNetMobileNetV1'
MODEL_PATH = './model_0023.pth'
OUT_PATH = './answers.csv'


def main(args):
    with open(CONFIG_PATH) as f:
        data = yaml.safe_load(f)
    config = convert_dict_to_tuple(data)

    detector = fd.build_detector(DETECTOR_TYPE,
                                 confidence_threshold=.5,
                                 nms_iou_threshold=.3,
                                 device=torch.device("cuda"),
                                 max_resolution=640)

    model = load_resnet(MODEL_PATH, config.model.model_type, config.dataset.num_of_classes)

    softmax_func = torch.nn.Softmax(dim=1)
    val_augs = ValidationAugmentations(config)
    preproc = tfs.Compose([tfs.ToTensor(), tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    test_df = pd.read_csv(args[0])

    scores = np.zeros((len(test_df), 7), dtype=np.float32)
    scores[:, 0] = 1

    for idx, image_path in tqdm(enumerate(test_df.frame_path.values), total=len(test_df)):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
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

        if idx % 1000 == 0 and idx > 0:
            save_results(scores, test_df.frame_path.values, OUT_PATH)

    save_results(scores, test_df.frame_path.values, OUT_PATH)

if __name__ == '__main__':
    main(sys.argv[1:])
