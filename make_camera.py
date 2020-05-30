import os
import sys
import time
import json
import numpy as np

from utils.util import write_pickle_file
from utils.blender_utils import add_camera, get_calibration_matrix_K_from_blender

if __name__ == '__main__':
    data_root = 'data/Future-3D-Reconstruction'
    camera_dir = os.path.join(data_root, 'train', 'cameras')
    with open(os.path.join(data_root, 'train', 'data_info', 'train_set.json')) as f:
        train_set_info = json.load(f)
    for i in range(len(train_set_info)):
        model_ID = train_set_info[i]['model']
        trans = train_set_info[i]['pose']['translation']
        rot = train_set_info[i]['pose']['rotation']
        fov = train_set_info[i]['fov']

        cam = add_camera((0, 0, 0), fov, 'camera')
        K_blender = get_calibration_matrix_K_from_blender(cam.data)
        K = np.array(K_blender)

        write_pickle_file(os.path.join(camera_dir, model_ID+'.pkl'), {'K': K})



