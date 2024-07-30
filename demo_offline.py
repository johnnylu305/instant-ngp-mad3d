import argparse
from pynput import keyboard
import threading
import time
import subprocess

# Importing the functions from the provided scripts
from load_ingp import ingp_render_online
from train_ingp import run_ingp

# Global state flags
render_thread = None
rendering = False
training = False
training_thread = None
colmapping = False
colmap_thread = None

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
    parser.add_argument("--colmap_matcher", default="sequential", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="Select which matcher colmap should use. Sequential for videos, exhaustive for ad-hoc images.")
    parser.add_argument("--images", default="images", help="Input path to the images.")
    parser.add_argument("--colmap_aabb_scale", default=32, choices=["1", "2", "4", "8", "16", "32", "64", "128"], help="Large scene scale factor. 1=scene fits in unit cube; power of 2 up to 128")
    parser.add_argument("--out", default="transforms.json", help="Output JSON file path.")
    return parser.parse_args()

# Parse the command line arguments
args = parse_args()

def run_ingp_wrapper():
    global rendering
    run_ingp(args)

def start_ingp():
    global training, training_thread
    if not training:
        training = True
        training_thread = threading.Thread(target=run_ingp_wrapper)
        training_thread.start()

def stop_ingp():
    global training, training_thread
    if training_thread:
        training_thread.join()
    training = False

def ingp_render_online_wrapper():
    ingp_render_online(args)

def start_rendering():
    global rendering, render_thread
    if not rendering:
        rendering = True
        render_thread = threading.Thread(target=ingp_render_online_wrapper)
        render_thread.start()

def stop_rendering():
    global rendering, render_thread
    rendering = False
    if render_thread:
        render_thread.join()  # Wait for the thread to finish

def run_colmap():
    # python3 scripts/colmap2nerf.py --colmap_matcher exhaustive --run_colmap --aabb_scale 32 --images ../MAD3DDataset/single_drone_test/ --out ../MAD3DDataset/single_drone_test/transform.json 
    command = ["python3", "scripts/colmap2nerf.py", "--colmap_matcher", f"{args.colmap_matcher}", "--run_colmap", "--aabb_scale", f"{args.colmap_aabb_scale}", "--images", f"{args.images}", "--out", "{args.out}", "--overwrite"]
    subprocess.run(command)

def start_colmap():
    global colmapping, colmap_thread
    if not colmapping:
        colmapping = True
        colmap_thread = threading.Thread(target=run_colmap)
        colmap_thread.start()

def stop_colmap():
    global colmapping, colmap_thread
    colmapping = False
    if colmap_thread:
        colmap_thread.join()  # Wait for the thread to finish

def on_press(key):
    try:
        if key.char == 't':
            start_ingp()
        elif key.char == 'i':
            start_rendering()
        elif key.char == 'q':
            stop_ingp()
            stop_rendering()
            stop_colmap()
        elif key.char == 'c':
            start_colmap()
    except AttributeError:
        pass

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False

def main():
    print("Press 't' to start run_ngp(), 'i' to start ingp_render_online(), 'p' to stop run_ngp(), 'q' to stop ingp_render_online().")
    print("Press 'esc' to exit.")

    # Setup the keyboard listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

if __name__ == "__main__":
    main()
