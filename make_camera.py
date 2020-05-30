import os
import sys
import time
import math
import json
import bpy
import numpy as np
from PIL import Image

from utils.util import write_pickle_file
from utils.blender_utils import clear_mv, clear_scene, add_camera, get_calibration_matrix_K_from_blender

def init_blender(size):
    scene = bpy.context.scene
    scene.render.resolution_x = size[0]
    scene.render.resolution_y = size[1]
    scene.render.resolution_percentage = 100

    # Delete default cube
    for obj in bpy.data.objects:
        obj.select = True
    bpy.ops.object.delete()

if __name__ == '__main__':
    data_root = 'data/Future-3D-Reconstruction'
    camera_dir = os.path.join(data_root, 'train', 'cameras')
    with open(os.path.join(data_root, 'train', 'data_info', 'train_set.json')) as f:
        train_set_info = json.load(f)
    total = len(train_set_info)
    for i in range(total):
        model_ID = train_set_info[i]['model']
        img = Image.open(os.path.join(data_root, 'train', 'image', train_set_info[i]['image'])).convert('RGB')
        trans = np.array(train_set_info[i]['pose']['translation'])
        rot = np.array(train_set_info[i]['pose']['rotation'])
        init_blender(img.size)
        try:
            #fov = math.radians(train_set_info[i]['fov'])
            fov = train_set_info[i]['fov']
            cam = add_camera((0, 0, 0), fov, 'camera')
            K_blender = get_calibration_matrix_K_from_blender(cam.data)
            K = np.array(K_blender)
            clear_scene()
            clear_mv()
        except:
            continue
        cam_path = os.path.join(camera_dir, model_ID+'.pkl')
        write_pickle_file(cam_path, {'K': K})
        print('[ {} / {} ] Saved to {}'.format(i, total, cam_path))
    print('Finished!')



