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


def ingp_render(args):
    f = open(args.pose)
    params = json.load(f)
    # python binding
    testbed = ngp.Testbed()
    parent = os.path.split(args.pose)[0]
    # setup root dir
    testbed.root_dir = parent

    # reload a snapshot
    snapshot_path = os.path.join(parent, 'train_models', 'offline.ingp')
    testbed.load_snapshot(snapshot_path)
    # initialize a dataset
    testbed.create_empty_nerf_dataset(n_images=len(params['frames']), aabb_scale=args.aabb)
    testbed.shall_train = False   
    #testbed.load_training_data(args.pose)
    # near plane
    if args.near_distance>=0.0:
        testbed.nerf.training.near_distance = args.near_distance
    if args.batch>0:
        testbed.training_batch_size = args.batch
    testbed.render_aabb.min = np.array([args.x_scale[0], args.y_scale[0], args.z_scale[0]])
    testbed.render_aabb.max = np.array([args.x_scale[1], args.y_scale[1], args.z_scale[1]])

    for i in range(len(params['frames'])):
        height, width = int(params['frames'][i]['h']), int(params['frames'][i]['w'])
        depth_img = np.zeros((height, width))
        img = np.zeros((height, width, 4))
        testbed.nerf.training.set_image(i, img, depth_img)
        testbed.nerf.training.set_camera_extrinsics(i, params['frames'][i]['transform_matrix'][:3], convert_to_ngp=True)
        testbed.nerf.training.set_camera_intrinsics(
            i,
            fx=params['frames'][i]["fl_x"], fy=params['frames'][i]["fl_y"],
            cx=params['frames'][i]["cx"], cy=params['frames'][i]["cy"],
        )

    #for i in range(len(params['frames'])):
    with tqdm(range(testbed.nerf.training.dataset.n_images), unit="images", desc=f"Rendering test frame") as t:
        for i in t:
            #resolution = testbed.nerf.training.dataset.metadata[i].resolution
            resolution = [int(params['frames'][i]['w']), int(params['frames'][i]['h'])]
            testbed.set_camera_to_training_view(i)
            testbed.render_ground_truth = False
            image = testbed.render(resolution[0], resolution[1], 1, True)
            save_root_path = args.dst
            if not os.path.isdir(save_root_path):
                os.makedirs(save_root_path)

            save_path = os.path.join(save_root_path, f'{i:05d}.png')
            # write_image should only accept linear color as input
            #print(np.unique(image))
            write_image(save_path, image)
 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", required=True, help="Source of json file")
    parser.add_argument("--near_distance", default=0.2, type=float, help="Set the distance from the camera at which training rays start for nerf. <0 means use ngp default")
    parser.add_argument("--dst", required=True, help="Destination for output images")
    parser.add_argument("--network", type=str, default=os.path.join(".", "configs", "nerf", "base.json"))
    parser.add_argument("--x_scale", nargs=2, default=[-0.179, 1.23], type=float)
    parser.add_argument("--y_scale", nargs=2, default=[-0.154, 0.905], type=float)
    parser.add_argument("--z_scale", nargs=2, default=[-0.204, 1.429], type=float)
    #parser.add_argument("--x_scale", nargs=2, default=[-2.5, 2.5], type=float)
    #parser.add_argument("--y_scale", nargs=2, default=[-2.5, 2.5], type=float)
    #parser.add_argument("--z_scale", nargs=2, default=[-2.5, 2.5], type=float)
    parser.add_argument("--aabb", type=int, default=4) 
    parser.add_argument("--batch", type=int, default=-1)
    args = parser.parse_args()
    #dataset_path = "/home/chlu/Data/Projects/Drones/MyWebotTools/Worlds/circular_traj_demo_small/dataset/"
    ingp_render(args)

if __name__=="__main__":
    main()
