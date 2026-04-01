# libero_dataset.py
import os
import h5py
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from libero.libero import benchmark, get_libero_path

# ---------------------------------------------------------
# Helper: Get demo file paths AND language for a task
# ---------------------------------------------------------
def get_task_info(task_id, benchmark_name="libero_10"):
    """
    Returns both the demo path AND the language instruction for a task.
    Language is extracted from HDF5 file's data.attrs["problem_info"].
    """
    benchmark_instance = benchmark.get_benchmark_dict()[benchmark_name]()
    task = benchmark_instance.get_task(task_id)
    
    datasets_path = get_libero_path("datasets")
    demo_filename = benchmark_instance.get_task_demonstration(task_id)
    demo_path = os.path.join(datasets_path, demo_filename)
    
    # Extract language from HDF5 file (the correct way!)
    language = ""
    with h5py.File(demo_path, 'r') as f:
        if "problem_info" in f["data"].attrs:
            problem_info = json.loads(f["data"].attrs["problem_info"])
            if "language_instruction" in problem_info:
                language = "".join(problem_info["language_instruction"]).strip('"')
    
    # Fallback to task.instruction if HDF5 doesn't have it
    if not language and hasattr(task, 'instruction'):
        language = task.instruction
    
    return demo_path, language

# ---------------------------------------------------------
# Custom PyTorch Dataset for a SINGLE LIBERO task
# ---------------------------------------------------------
class LiberoTaskDataset(Dataset):
    def __init__(
        self,
        task_id,
        benchmark_name="libero_10",
        obs_keys=["agentview_rgb", "eye_in_hand_rgb", "joint_states"],
        extra_keys=["robot_states"],
        action_key="actions",
        image_size=(128, 128),
        normalize_images=True
    ):
        """
        Args:
            task_id (int): Which task in the benchmark to load.
            benchmark_name (str): Name of the benchmark (e.g., "libero_10").
            obs_keys (list): Observation keys INSIDE the 'obs/' group.
            extra_keys (list): Keys at the DEMO level (NOT inside obs/).
            action_key (str): Key for actions.
            image_size (tuple): Resize images to (H, W).
            normalize_images (bool): Apply ImageNet normalization.
        """
        self.benchmark_name = benchmark_name
        self.task_id = task_id
        self.obs_keys = obs_keys
        self.extra_keys = extra_keys
        self.action_key = action_key
        self.image_size = image_size
        
        # Get demo path AND language from HDF5 file
        self.hdf5_path, self.language = get_task_info(task_id, benchmark_name)
        
        # Precompute index mapping
        self.demo_map = []
        self.total_frames = 0
        
        with h5py.File(self.hdf5_path, 'r') as f:
            for demo_key in f['data'].keys():
                demo_group = f['data'][demo_key]
                num_frames = len(demo_group[action_key])
                if num_frames > 0:
                    self.demo_map.append({
                        'key': demo_key,
                        'start': self.total_frames,
                        'end': self.total_frames + num_frames,
                        'length': num_frames
                    })
                    self.total_frames += num_frames
        
        # Image preprocessing pipeline
        img_transforms = [transforms.Resize(image_size), transforms.ToTensor()]
        if normalize_images:
            img_transforms.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            )
        self.image_transform = transforms.Compose(img_transforms)
        
        print(f"[LiberoTaskDataset] Task {task_id}: Loaded {self.total_frames} frames from {len(self.demo_map)} demos")
        print(f"[LiberoTaskDataset] Language: '{self.language}'")

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        # 1. Find which demo this index belongs to
        demo_info = None
        for info in self.demo_map:
            if info['start'] <= idx < info['end']:
                demo_info = info
                break
        
        if demo_info is None:
            raise IndexError(f"Index {idx} out of range [0, {self.total_frames})")
        
        local_idx = idx - demo_info['start']
        demo_key = demo_info['key']
        
        # 2. Load data from HDF5
        with h5py.File(self.hdf5_path, 'r') as f:
            demo = f['data'][demo_key]
            sample = {}
            
            # -- Load observations from obs/ group --
            for key in self.obs_keys:
                if 'rgb' in key:
                    img_np = demo['obs'][key][local_idx]
                    img_pil = Image.fromarray(img_np)
                    sample[key] = self.image_transform(img_pil)
                else:
                    val = demo['obs'][key][local_idx]
                    sample[key] = torch.from_numpy(val).float()
            
            # -- Load extra keys from demo level --
            for key in self.extra_keys:
                if key in demo:
                    val = demo[key][local_idx]
                    sample[key] = torch.from_numpy(val).float()
            
            # -- Load action --
            action = demo[self.action_key][local_idx]
            sample['actions'] = torch.from_numpy(action).float()
            
            # -- Load language (from HDF5 data.attrs["problem_info"]) --
            sample['language'] = self.language
        
        return sample

# ---------------------------------------------------------
# Example: Inspect Data and Shapes
# ---------------------------------------------------------
if __name__ == "__main__":
    TASK_ID = 9
    
    print(f"=== Loading Task {TASK_ID} ===")
    
    dataset = LiberoTaskDataset(
        task_id=TASK_ID,
        benchmark_name="libero_10",
        obs_keys=["agentview_rgb", "eye_in_hand_rgb", "joint_states"],
        extra_keys=["robot_states"],
        image_size=(128, 128)
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    
    print("\n=== Batch Shapes ===")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"{key:20s}: {list(val.shape)}")
        elif isinstance(val, list) and len(val) > 0:
            print(f"{key:20s}: List[str], example: '{val[0]}'")
    
    print(f"\n=== Sample Action ===")
    print(batch['actions'][0].numpy())
    
    print(f"\n=== Sample Language ===")
    print(batch['language'][0])
    
    # Save sample images
    import torchvision.utils as vutils
    if 'agentview_rgb' in batch:
        vutils.save_image(batch['agentview_rgb'][:4], "sample_agentview.png", normalize=True)
        print("\n✓ Saved sample images to 'sample_agentview.png'")