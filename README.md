This is a repository with a baseline solution for the [MCS2021. Gesture Recognition competition](https://boosters.pro/championship/machinescansee2021).
In this competition, participants need to train a model to recognize one of 6 gestures (Stop, Victory, Mute, Ok, Like, Dislike), or class No_gesture.

The idea behind baseline solution is to get the largest face in the frame using the [face detector](https://github.com/hukkelas/DSFD-Pytorch-Inference.git), expand the bbox by x2.5 times and train the classifier for 7 classes on the resulting crop (6 gestures + No gesture). If no face is found in the image, then there is no gesture class with a confidence score of 1.0.

## Repository structure
config - directory for difference yml config files;\
data - directory with supporting code for creating a dataloaders;\
models - directory with model initializing code;\
submission_example - directory with everything you need to prepare a submission zip archive to run the inference on the server. You can replace `script.py` with `script_batched.py` to use batches, which will speed up the inference, but can degrade accuracy due to box scaling.\
main.py - start model training process;\
requirements.txt - all required packages used in this repository;\
train.py - training and validation code for the classification task;\
utils.py - functions for initializing various components for train.py;\
find_max_faces.py - run detector from [face_detection](https://github.com/hukkelas/DSFD-Pytorch-Inference.git) package to get bbox on training data;\
prepare_train_data.py - code for preparing a training dataset, dividing into training data and validation data.

## Steps for working with this baseline
#### 0. Run the face detector on the training data
Launch the face detector on the entire training set to get a bbox of the largest face. You can skip this step, since we have done it with `train.csv` and also posted the results of all three detectors, combined in [train_with_bboxes.json](https://drive.google.com/file/d/1JCr6gTTPQsq1GnwibJdAUNOQ0q1vesYM/view?usp=sharing)
```python
CUDA_VISIBLE_DEVICES=0 python find_all_faces.py --prefix_path /path/to/your/data --data_list /path/to/train.csv \
                                                --output_json_path ./lists/train_RetinaResNet50_bboxes.json \
                                                --detector_type RetinaNetResNet50
```
#### 1. Prepare training and validation lists
When splitting the dataset into training and validation data, it is important to split by video_name, since otherwise very similar frames can get into training and validation.
```python
python prepare_train_data.py --train_data ./lists/train_with_bboxes.json \
                             --output_dir ./lists/baseline_exp/ \
                             --bbox_key bbox_RetinaNetResNet50 \
                             --val_size 0.15
```

#### 2. Run model training
```
python ./main.py --cfg ./config/baseline_mcs.yml
```

#### 3. Prepare a submission
To prepare a submission, you need to copy the weights of the trained model to the `submission_example` folder, change global variables `CONFIG_PATH`, `MODEL_PATH` and `DETECTOR_TYPE` in `script.py` (if necessary) and zip the directory `submission_example`.