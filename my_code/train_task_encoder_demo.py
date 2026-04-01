"""
Example usage of TaskEncoder with LIBERO dataset.
Demonstrates: training loop, loss computation, and inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from libero_dataset import LiberoTaskDataset  # Your provided dataset
from task_encoder import TaskEncoder
from torch.utils.tensorboard import SummaryWriter  
import time  
from datetime import datetime


def train_task_encoder(
    task_id: int = 9,
    batch_size: int = 4,
    epochs: int = 2,
    lr: float = 1e-4,
    device: str = None
):
    """Train the task encoder on a single LIBERO task."""
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Initialize model
    model = TaskEncoder(
        clip_path="../ckpts/clip-vit-base-patch32",
        latent_dim=10,
        robot_state_dim=9,
        action_dim=7
    ).to(device)

    # Create a unique log directory based on task_id and timestamp
    log_dir = f"../runs/task_encoder_task{task_id}_" + datetime.now().strftime("%Y-%m-%d-%H-%M")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📊 TensorBoard logs saved to: {log_dir}")
    
    # Optimizer: only trainable params (CLIP is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr)
    
    # Load dataset - use robot_states (9d) from extra_keys
    dataset = LiberoTaskDataset(
        task_id=task_id,
        benchmark_name="libero_10",
        obs_keys=["joint_states"],  # Optional: not used in this encoder
        extra_keys=["robot_states"],  # This is the 9d state we need
        image_size=(128, 128)
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,  # Set >0 for production
        drop_last=True  # Ensure clean batches for temporal shifting
    )
    
    print(f"📦 Dataset: {len(dataset)} frames, language: '{dataset.language}'")
    
    # Training loop
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # ===== Prepare inputs =====
            text_list = batch['language']  # List[str], same instruction for batch
            robot_state = batch['robot_states'].to(device)  # (B, 9)
            action = batch['actions'].to(device)  # (B, 7)
            
            # ===== Temporal targeting: current → next state =====
            # Shift to create (state_t, action_t) → state_{t+1} pairs
            if len(robot_state) < 2:
                continue  # Skip tiny batches
                
            state_curr = robot_state[:-1]    # (B-1, 9)
            action_curr = action[:-1]         # (B-1, 7)
            state_next = robot_state[1:].clone()  # (B-1, 9) - target
            
            # Adjust text_list if needed (all same instruction, so OK)
            text_list = text_list[:len(state_curr)]
            
            # ===== Forward pass =====
            outputs = model(
                text_list=text_list,
                robot_state=state_curr,
                action=action_curr,
                device=device
            )
            
            # ===== Compute loss =====
            loss, loss_dict = model.compute_loss(
                outputs=outputs,
                next_state_target=state_next,
                weights=None
            )
            
            # ===== Backward pass =====
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss_dict['total'])
            
            # Logging
            if batch_idx % 200 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx} | "
                      f"Loss: {loss_dict['total']:.4f} "
                      f"(recon:{loss_dict['recon']:.3f}, "
                      f"kl:{loss_dict['kl']:.3f}, "
                      f"wkl:{loss_dict['wkl']:.3f}, "
                      f"dyn:{loss_dict['dynamics']:.3f})")

                global_step = epoch * len(dataloader) + batch_idx
            
                # Log individual loss components
                writer.add_scalar('Loss/Total', loss_dict['total'], global_step)
                writer.add_scalar('Loss/Recon', loss_dict['recon'], global_step)
                writer.add_scalar('Loss/Dynamics', loss_dict['dynamics'], global_step)
                writer.add_scalar('Loss/KL', loss_dict['kl'], global_step)
                writer.add_scalar('Loss/WKL', loss_dict['wkl'], global_step)
                
                # Log Learning Rate
                writer.add_scalar('Params/LR', optimizer.param_groups[0]['lr'], global_step)
        
        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"✅ Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = f"../ckpts/results/task_encoder_task{task_id}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  💾 Saved best model to {save_path}")
    
    print("🎉 Training finished!")
    writer.close()
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'])
    parser.add_argument('--task_id', type=int, default=9)
    args = parser.parse_args()
    
    train_task_encoder(task_id=args.task_id)