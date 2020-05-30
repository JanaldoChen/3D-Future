import os
import sys
import time
import math
import json
import numpy as np

from utils.util import write_pickle_file
from utils.blender_utils import clear_mv, clear_scene, add_camera, get_calibration_matrix_K_from_blender

def init_blender():
    scene = bpy.context.scene
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 1200
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
        trans = np.array(train_set_info[i]['pose']['translation'])
        rot = np.array(train_set_info[i]['pose']['rotation'])
        try:
            init_blender()
            fov = math.radians(train_set_info[i]['fov'])
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



