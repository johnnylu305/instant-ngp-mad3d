import argparse
import os
import commentjson as json
import numpy as np
import time
import sys
sys.path.append("./build")
sys.path.append("./scripts")
sys.path.append(".")

from common import *
from scenes import *

from tqdm import tqdm

import pyngp as ngp # noqa
import cv2
#import torch
#torch.cuda.set_per_process_memory_fraction(0.8, 0)


#training = True

def parse_args():
    parser = argparse.ArgumentParser(description="Run I-NGP")
    parser.add_argument("--scene_train_json", required=True, default="", help="Path to the .json file.")
    parser.add_argument("--network", default="", help="Path to the network config (e.g. base.json).")
    parser.add_argument("--near_distance", default=0.2, type=float, help="Set the distance from the camera at which training rays start for nerf. <0 means use ngp default")
    parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
    parser.add_argument("--no_opt", action="store_false", help="Optimize pose and distortion.")
    parser.add_argument("--n_steps", type=int, default=25000, help="Number of steps to train for before quitting.")
    parser.add_argument("--aabb", type=int, default=16)
    parser.add_argument("--batch", type=int, default=-1)
    return parser.parse_args()

def run_ingp(args):
    # load transform_*.json file
    f = open(args.scene_train_json)
    params = json.load(f)
    f.close()
    parent = os.path.split(args.scene_train_json)[0]
    print(parent)
    testbed = ngp.Testbed()
    testbed.reload_network_from_file(args.network)

    if args.gui:
        sw = 1920 #4000 #width 
        sh = 1080 #1848 #height 
        testbed.init_window(sw, sh)
        testbed.nerf.visualize_cameras = True

    # setup root dir
    testbed.root_dir = parent
    # initialize a dataset
    testbed.create_empty_nerf_dataset(n_images=len(params['frames']), aabb_scale=args.aabb)
    imgs = []
    
    # setup dataset
    for idx, param in enumerate(params['frames']):
        img_path = os.path.join(parent, param['file_path'])
        print(img_path, idx)
        
        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        img = img.astype(np.float32)
        img /= 255.0
        # set_image should only accept linear color
        img = srgb_to_linear(img)
        imgs.append(img)
        # premultiply
        img[..., :3] *= img[..., 3:4]
        height, width = img.shape[:2]
        
        depth_img = np.zeros((img.shape[0], img.shape[1]))
        testbed.nerf.training.set_image(idx, img, depth_img)
        testbed.nerf.training.set_camera_extrinsics(idx, param['transform_matrix'][:3], convert_to_ngp=True)
        testbed.nerf.training.set_camera_intrinsics(
            idx,
            fx=param["fl_x"], fy=param["fl_y"],
            cx=param["cx"], cy=param["cy"],
        )

    if args.no_opt:
        print(f"Optimizing Camera Parameters !!")
        print("--------------------------------------------")
        testbed.nerf.training.optimize_extrinsics = True
        testbed.nerf.training.optimize_distortion = True
        testbed.nerf.training.optimize_extra_dims = True

    # Training
    testbed.nerf.training.n_images_for_training = len(params['frames'])
    total_n_steps = args.n_steps
    testbed.shall_train = True
    training_step = 0
    tqdm_last_update = 0
    min_loss = 1e10
    # near plane
    if args.near_distance>=0.0:
        testbed.nerf.training.near_distance = args.near_distance
    if args.batch>0:
        testbed.training_batch_size = args.batch
    # Start training
    if total_n_steps > 0:
        with tqdm(desc="Training", total=total_n_steps, unit="step") as t:
            while testbed.frame():
                # What will happen when training is done?
                if testbed.training_step >= total_n_steps:
                    break

                # Update progress bar
                now = time.monotonic()
                if now - tqdm_last_update > 0.1:
                    t.update(testbed.training_step - training_step)
                    t.set_postfix(loss=testbed.loss)
                    training_step = testbed.training_step
                    tqdm_last_update = now

    os.makedirs(os.path.join(parent, "train_models"), exist_ok=True)
    testbed.save_snapshot(os.path.join(parent, "train_models", "offline.ingp"), False) 

def main():
    args = parse_args()
    run_ingp(args)

if __name__=="__main__":
    main()
