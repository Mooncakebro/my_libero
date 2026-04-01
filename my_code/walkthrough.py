from libero.libero import benchmark, get_libero_path, set_libero_default_path
import os
from termcolor import colored

benchmark_root_path = get_libero_path("benchmark_root")
init_states_default_path = get_libero_path("init_states")
datasets_default_path = get_libero_path("datasets")
bddl_files_default_path = get_libero_path("bddl_files")
print("Default benchmark root path: ", benchmark_root_path)
print("Default dataset root path: ", datasets_default_path)
print("Default bddl files root path: ", bddl_files_default_path)

benchmark_dict = benchmark.get_benchmark_dict()
print(benchmark_dict)

# initialize a benchmark
benchmark_instance = benchmark_dict["libero_10"]()
num_tasks = benchmark_instance.get_num_tasks()
# see how many tasks involved in the benchmark
print(f"{num_tasks} tasks in the benchmark {benchmark_instance.name}: ")

# Check if all the task names and their bddl file names
task_names = benchmark_instance.get_task_names()
print("The benchmark contains the following tasks:")
for i in range(num_tasks):
    task_name = task_names[i]
    task = benchmark_instance.get_task(i)
    bddl_file = os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file)
    print(f"\t {task_name}, detail definition stored in {bddl_file}")
    if not os.path.exists(bddl_file):
        print(colored(f"[error] bddl file {bddl_file} cannot be found. Check your paths", "red"))

# Check if all the init states files exist for tasks
task_names = benchmark_instance.get_task_names()
print("The benchmark contains the following tasks:")
for i in range(num_tasks):
    task_name = task_names[i]
    task = benchmark_instance.get_task(i)
    init_states_path = os.path.join(init_states_default_path, task.problem_folder, task.init_states_file)
    if not os.path.exists(init_states_path):
        print(colored(f"[error] the init states {init_states_path} cannot be found. Check your paths", "red"))
print(f"An example of init file is named like this: {task.init_states_file}")

# Load torch init files
init_states = benchmark_instance.get_task_init_states(0)
# Init states in the same (num_init_rollouts, num_simulation_states)
print(init_states.shape)


### Not working for now ###

# from libero.libero.envs import OffScreenRenderEnv
# from IPython.display import display
# from PIL import Image

# import torch
# import torchvision

# # task_id is the (task_id + 1)th task in the benchmark
# task_id = 9
# task = benchmark_instance.get_task(task_id)

# env_args = {
#     "bddl_file_name": os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file),
#     "camera_heights": 128,
#     "camera_widths": 128
# }

# env = OffScreenRenderEnv(**env_args)


# init_states = benchmark_instance.get_task_init_states(task_id)

# # Fix random seeds for reproducibility
# env.seed(0)

# def make_grid(images, nrow=8, padding=2, normalize=False, pad_value=0):
#     """Make a grid of images. Make sure images is a 4D tensor in the shape of (B x C x H x W)) or a list of torch tensors."""
#     grid_image = torchvision.utils.make_grid(images, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value).permute(1, 2, 0)
#     return grid_image

# images = []
# env.reset()
# for eval_index in range(len(init_states)):
#     env.set_init_state(init_states[eval_index])

#     for _ in range(5):
#         obs, _, _, _ = env.step([0.] * 7)
#     images.append(torch.from_numpy(obs["agentview_image"]).permute(2, 0, 1))

# # # images = torch.stack(images, dim=0).permute(0, 3, 1, 2)
# # print(images.shape)
# grid_image = make_grid(images, nrow=10, padding=2, pad_value=0)
# display(Image.fromarray(grid_image.numpy()[::-1]))
# env.close()

import libero.libero.utils.download_utils as download_utils

# download_dir = get_libero_path("datasets")
# datasets = "libero_spatial" # Can specify "all", "libero_goal", "libero_spatial", "libero_object", "libero_100"

# libero_datasets_exist = download_utils.check_libero_dataset(download_dir=download_dir)

# if not libero_datasets_exist:
#     download_utils.libero_dataset_download(download_dir=download_dir, datasets=datasets)

# Check if the demo files exist
demo_files = [os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(i)) for i in range(num_tasks)]
for demo_file in demo_files:
    if not os.path.exists(demo_file):
        print(colored(f"[error] demo file {demo_file} cannot be found. Check your paths", "red"))
    else:
        print(colored(f"[good news] demo file {demo_file} is found.", "green"))

import h5py
from libero.libero.utils.dataset_utils import get_dataset_info
from IPython.display import HTML
import imageio

example_demo_file = demo_files[9]
print('showing the example_demo_file')
# Print the dataset info. We have a standalone script for doing the same thing available at `scripts/get_dataset_info.py`
get_dataset_info(example_demo_file)

with h5py.File(example_demo_file, "r") as f:
    images = f["data/demo_0/obs/agentview_rgb"][()]

video_writer = imageio.get_writer("output.mp4", fps=60)
for image in images:
    video_writer.append_data(image[::-1])
video_writer.close()

HTML("""
    <video width="640" height="480" controls>
        <source src="output.mp4" type="video/mp4">
    </video>
    <script>
        var video = document.getElementsByTagName('video')[0];
        video.playbackRate = 2.0; // Increase the playback speed to 2x
        </script>    
""")

from torch.utils.data import ConcatDataset, Dataset
from libero.lifelong.datasets import get_dataset, SequenceVLDataset

num_tasks = benchmark_instance.get_num_tasks()
print(f"Number of tasks in the benchmark {benchmark_instance.name}: {num_tasks}")