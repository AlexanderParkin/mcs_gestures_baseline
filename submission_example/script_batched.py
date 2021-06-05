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

def read_image(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img is None:
        raise ValueError('Failed to read {}'.format(image_file))
    return img


class TestDataset(object):
    def __init__(self, image_list, max_resolution=640):
        self.image_list = image_list
        self.max_resolution = max_resolution

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        cv2.setNumThreads(6)
        image_fpath = self.image_list[idx]
        img = read_image(image_fpath)

        orgl_h, orgl_w = img.shape[:-1]
        max_size = max(orgl_h, orgl_w)
        scale_f = self.max_resolution / max_size

        resized_img = cv2.resize(img, (round(orgl_w * scale_f), round(orgl_h * scale_f)))
        # resized_img = np.array(resized_img)

        new_h, new_w = resized_img.shape[:-1]
        border_arr = np.zeros((self.max_resolution, self.max_resolution, 3), dtype=np.uint8)
        x_offset = (self.max_resolution - new_w) // 2
        y_offset = (self.max_resolution - new_h) // 2

        border_arr[y_offset:y_offset + new_h, x_offset: x_offset + new_w] = resized_img
        change_log = {
            'x_offset': x_offset,
            'y_offset': y_offset,
            'scale_f': scale_f
        }

        return border_arr, change_log, image_fpath


def rescale_bbox(bbox, change_log):
    x, y, w, h = bbox
    x_offset, y_offset, scale_f = change_log
    x = x - x_offset
    y = y - y_offset

    x = round(x / scale_f)
    y = round(y / scale_f)
    w = round(w / scale_f)
    h = round(h / scale_f)

    return [x, y, w, h]

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
BATCH_SIZE = 32
NUM_WORKERS = 8

def main(args):
    with open(CONFIG_PATH) as f:
        data = yaml.safe_load(f)
    config = convert_dict_to_tuple(data)

    detector = fd.build_detector(DETECTOR_TYPE,
                                 confidence_threshold=.5,
                                 nms_iou_threshold=.3,
                                 device=torch.device("cuda"))

    model = load_resnet(MODEL_PATH, config.model.model_type, config.dataset.num_of_classes)

    softmax_func = torch.nn.Softmax(dim=1)
    val_augs = ValidationAugmentations(config)
    preproc = tfs.Compose([tfs.ToTensor(), tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    test_df = pd.read_csv(args[0])

    scores = np.zeros((len(test_df), 7), dtype=np.float32)
    scores[:, 0] = 1

    test_dataset = TestDataset(test_df.frame_path.values, max_resolution=640)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        pin_memory=True
    )

    for batch_idx, (batch_images, change_logs, image_pathes) in tqdm(enumerate(test_loader),
                                                                     total=len(test_loader)):
        batch_images = np.array(batch_images)
        detections = detector.batched_detect(batch_images)

        crops = []
        score_using_indexes = []
        for idx, img_detections in enumerate(detections):
            all_faces = []
            for det in img_detections:
                x1, y1, x2, y2, s = det.tolist()
                w = x2 - x1
                h = y2 - y1
                bbox = [x1, y1, w, h]
                all_faces.append(bbox)

            if len(all_faces) > 0:
                max_bbox = get_max_bbox(all_faces)
                change_log = (change_logs['x_offset'][idx].numpy(),
                              change_logs['y_offset'][idx].numpy(),
                              change_logs['scale_f'][idx].numpy())
                max_bbox = rescale_bbox(max_bbox, change_log)
                img = read_image(image_pathes[idx])

                crop, *_ = val_augs(img, max_bbox, None)
                crops.append(preproc(crop))
                score_using_indexes.append(batch_idx * BATCH_SIZE + idx)

        if len(crops) > 0:
            clf_tensor = torch.stack(crops)
            clf_tensor = clf_tensor.to('cuda')
            with torch.no_grad():
                out = model(clf_tensor)
            out = softmax_func(out).squeeze().detach().cpu().numpy()
            scores[score_using_indexes] = out

        if batch_idx % 100 == 0 and batch_idx > 0:
            save_results(scores, test_df.frame_path.values, OUT_PATH)

    save_results(scores, test_df.frame_path.values, OUT_PATH)

if __name__ == '__main__':
    main(sys.argv[1:])
