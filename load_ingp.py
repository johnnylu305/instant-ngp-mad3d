import os
import commentjson as json
import numpy as np
import time
import sys
import datetime
import argparse
from scipy.spatial.transform import Rotation as R

sys.path.append("./build")
sys.path.append("./scripts")
sys.path.append(".")

from common import *
from scenes import *

from tqdm import tqdm

import pyngp as ngp # noqa
import cv2


#rendering = True

def parse_args():
    parser = argparse.ArgumentParser(description="Run I-NGP")
    parser.add_argument("--scene_train_json", required=True, default="", help="Path to the .json file.")
    parser.add_argument("--near_distance", default=0.2, type=float, help="Set the distance from the camera at which training rays start for nerf. <0 means use ngp default")
    parser.add_argument("--aabb", type=int, default=16)
    return parser.parse_args()

def ingp_render_online(args):
    f = open(args.scene_train_json)
    params = json.load(f)
    # python binding
    testbed = ngp.Testbed()
    parent = os.path.split(args.scene_train_json)[0]
    # setup root dir
    testbed.root_dir = parent

    # reload a snapshot
    snapshot_path = os.path.join(parent, 'train_models', 'offline.ingp')
    testbed.load_snapshot(snapshot_path)

    # initialize a dataset
    #testbed.create_empty_nerf_dataset(n_images=len(params['frames']), aabb_scale=args.aabb)
    testbed.shall_train = False

    #testbed.load_training_data(args.pose)
    # near plane
    if args.near_distance>=0.0:
        testbed.nerf.training.near_distance = args.near_distance
    # opera house
    #testbed.render_aabb.min = np.array([-0.179, -0.154, -0.204])
    #testbed.render_aabb.max = np.array([1.230, 0.905, 1.429])
    # taipei
    testbed.render_aabb.min = np.array([-0.179, -0.430, -0.204])
    testbed.render_aabb.max = np.array([1.230, 1.456, 1.429])
    # outdoor opera house
    testbed.render_aabb.min = np.array([-0.104, -1.152, -1.009])
    testbed.render_aabb.max = np.array([1.474, 0.867, 1.468])

    sw = 1920 #width 
    sh = 1080 #height 
    testbed.init_window(sw, sh)
    testbed.nerf.visualize_cameras = False

    while testbed.frame():
        pass
    

def main():
    args = parser.parse_args()
    ingp_render(args)

if __name__=="__main__":
    main()
