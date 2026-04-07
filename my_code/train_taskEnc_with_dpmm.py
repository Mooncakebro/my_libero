"""
Alternate training of TaskEncoder and DPMM.

Flow:
1) Optimize TaskEncoder every batch.
2) Every N batches, fit DPMM on buffered latent samples (z), mixed with replay
   samples from existing DPMM components.
3) Use DPMM component anchors to compute weighted KL regularization.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Keep imports stable across environments:
# 1) bnpy should come from LIBERO/bnpy/bnpy/__init__.py
# 2) libero package should be discoverable from LIBERO root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_BNPY_REPO_ROOT = os.path.join(_PROJECT_ROOT, "bnpy")
if os.path.isdir(_BNPY_REPO_ROOT) and _BNPY_REPO_ROOT not in sys.path:
    sys.path.insert(0, _BNPY_REPO_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(1, _PROJECT_ROOT)

from libero_dataset import LiberoTaskDataset
from my_dpmm_model import BNPModel
from task_encoder import TaskEncoder
from utils import convert_tensor_to_list, purge_invalid_values
from weighted_kl_div import compute_soft_labels, compute_weighted_kl_loss


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_task_dataloaders(
    task_ids: List[int],
    batch_size: int,
    num_workers: int,
    benchmark_name: str = "libero_10",
) -> List[DataLoader]:
    dataloaders = []
    for task_id in task_ids:
        dataset = LiberoTaskDataset(
            task_id=task_id,
            benchmark_name=benchmark_name,
            obs_keys=["joint_states"],
            extra_keys=["robot_states"],
            image_size=(128, 128),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )
        dataloaders.append(dataloader)
        print(
            f"Task {task_id}: {len(dataset)} samples | "
            f"language='{dataset.language}' | batches={len(dataloader)}"
        )
    return dataloaders


def _log_dpmm_state(
    dpmm: BNPModel,
    track_cluster_dir: str,
    component_log_dir: str,
    task_idx: int,
    epoch: int,
    batch_idx: int,
) -> None:
    tracked_clusters = sorted(
        [
            {"cluster_id": data["id"], "mu": data["mu"], "var": data["var"]}
            for data in dpmm.current_clusters.values()
        ],
        key=lambda x: x["cluster_id"],
    )
    tracked_clusters = convert_tensor_to_list(tracked_clusters)
    tracked_path = os.path.join(
        track_cluster_dir,
        f"{task_idx}-{epoch}-{batch_idx}-tracked_clusters.json",
    )
    with open(tracked_path, "w", encoding="utf-8") as f:
        json.dump(tracked_clusters, f, indent=2)

    components = sorted(dpmm.components, key=lambda x: x["k"])
    components_path = os.path.join(
        component_log_dir,
        f"{task_idx}-{epoch}-{batch_idx}-components.json",
    )
    with open(components_path, "w", encoding="utf-8") as f:
        json.dump(components, f, indent=2)


def _fit_dpmm_from_buffer(
    dpmm: BNPModel,
    latent_buffer: List[torch.Tensor],
) -> bool:
    if not latent_buffer:
        return False

    z_new = torch.cat(latent_buffer, dim=0)
    z_new = purge_invalid_values(z_new, "latent_z_buffer")
    if z_new.ndim != 2 or z_new.shape[0] == 0:
        return False

    if dpmm.model is not None and len(dpmm.components) > 0:
        k = max(1, len(dpmm.components))
        new_task_data_ratio = 1.0 / (1.0 + k)
        num_replay = int((1.0 - new_task_data_ratio) * z_new.shape[0] / new_task_data_ratio)
        replay = dpmm.sample_all(num_samples=max(1, num_replay)) if num_replay > 0 else None
        z_samples = torch.cat([replay, z_new], dim=0) if replay is not None else z_new
    else:
        z_samples = z_new

    z_samples = purge_invalid_values(z_samples, "dpmm_fit_samples")
    if z_samples.ndim == 2 and z_samples.shape[0] > 0:
        dpmm.fit(z_samples)
        return True
    return False


def _update_task_vis_buffer(
    task_vis_buffer: Dict[str, torch.Tensor],
    task_label: str,
    z_batch: torch.Tensor,
    max_points_per_task: int,
) -> None:
    z_cpu = z_batch.detach().cpu()
    if z_cpu.ndim != 2 or z_cpu.shape[0] == 0:
        return

    if task_label not in task_vis_buffer:
        task_vis_buffer[task_label] = z_cpu
    else:
        task_vis_buffer[task_label] = torch.cat([task_vis_buffer[task_label], z_cpu], dim=0)

    cur = task_vis_buffer[task_label]
    if cur.shape[0] > max_points_per_task:
        idx = torch.randperm(cur.shape[0])[:max_points_per_task]
        task_vis_buffer[task_label] = cur[idx]


def _save_tsne_by_task(
    task_vis_buffer: Dict[str, torch.Tensor],
    save_dir: str,
    step_tag: str,
) -> None:
    if not task_vis_buffer:
        return

    all_z = []
    all_labels = []
    for task_label in sorted(task_vis_buffer.keys()):
        z = task_vis_buffer[task_label]
        if z.ndim != 2 or z.shape[0] == 0:
            continue
        all_z.append(z.numpy())
        all_labels.extend([task_label] * z.shape[0])

    if not all_z:
        return

    X = np.concatenate(all_z, axis=0)
    if X.shape[0] < 3:
        return

    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Skip t-SNE visualization because dependency is missing: {e}")
        return

    perplexity = min(30, max(2, (X.shape[0] - 1) // 3))
    if perplexity >= X.shape[0]:
        perplexity = max(1, X.shape[0] - 1)

    tsne = TSNE(
        n_components=2,
        random_state=42,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    )
    X_2d = tsne.fit_transform(X)

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    unique_tasks = sorted(set(all_labels))
    cmap = plt.cm.get_cmap("tab20", max(1, len(unique_tasks)))
    for idx, tid in enumerate(unique_tasks):
        mask = np.array(all_labels) == tid
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=10,
            alpha=0.7,
            color=cmap(idx),
            label=tid,
        )
    plt.title(f"Latent z t-SNE ({step_tag})")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"tsne_{step_tag}.png")
    plt.savefig(save_path, dpi=180)
    plt.close()
    print(f"Saved t-SNE plot to {save_path}")


def train_task_encoder_with_alternating_dpmm(
    model: TaskEncoder,
    dpmm: BNPModel,
    dataloaders: List[DataLoader],
    train_cfg: Dict,
    out_dirs: Dict[str, str],
    device: torch.device,
) -> None:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=train_cfg["learning_rate"])

    best_loss = float("inf")
    task_vis_buffer: Dict[str, torch.Tensor] = {}

    for task_idx, dataloader in enumerate(dataloaders):
        task_id = train_cfg["task_ids"][task_idx]
        task_label = getattr(dataloader.dataset, "language", f"task_{task_id}")
        print(f"\n=== Training task_id={task_id} ({task_idx + 1}/{len(dataloaders)}) ===")
        dpmm_update_freq = max(1, len(dataloader) // train_cfg["dpmm_update_per_epoch"])
        print(f"DPMM update freq: every {dpmm_update_freq} batches")

        for epoch in range(train_cfg["epochs_per_task"]):
            model.train()
            latent_z_buffer = []
            epoch_losses = []

            for batch_idx, batch in enumerate(dataloader):
                text_list = batch["language"]
                robot_state = batch["robot_states"].to(device)
                action = batch["actions"].to(device)

                if robot_state.shape[0] < 2:
                    continue

                # (state_t, action_t) -> state_(t+1)
                state_curr = robot_state[:-1]
                action_curr = action[:-1]
                state_next = robot_state[1:]
                text_curr = text_list[: state_curr.shape[0]]

                optimizer.zero_grad(set_to_none=True)
                outputs = model(
                    text_list=text_curr,
                    robot_state=state_curr,
                    action=action_curr,
                    device=device,
                )

                # Base losses from TaskEncoder
                weights_without_wkl = {
                    "recon": train_cfg["loss_weights"]["recon"],
                    "kl": 0.0,
                    "dynamics": train_cfg["loss_weights"]["dynamics"],
                    "wkl": 0.0,
                }
                base_loss, base_loss_dict = model.compute_loss(
                    outputs=outputs,
                    next_state_target=state_next,
                    weights=weights_without_wkl,
                )

                # Weighted KL to DPMM anchors (only after first DPMM fit)
                wkl_loss = torch.zeros((), device=device)
                if dpmm.model is not None and len(dpmm.comp_mu) > 0:
                    anchor_mu = torch.stack([mu.to(device) for mu in dpmm.comp_mu], dim=0)
                    anchor_var = torch.stack([var.to(device) for var in dpmm.comp_var], dim=0).clamp_min(1e-6)
                    soft_labels = compute_soft_labels(
                        outputs["mu"],
                        outputs["logvar"],
                        anchor_mu,
                        anchor_var,
                        temperature=train_cfg["wkl_temperature"],
                    )
                    wkl_loss, _ = compute_weighted_kl_loss(
                        outputs["mu"],
                        outputs["logvar"],
                        anchor_mu,
                        anchor_var,
                        soft_labels=soft_labels,
                        temperature=train_cfg["wkl_temperature"],
                    )

                total_loss = base_loss + train_cfg["loss_weights"]["wkl"] * wkl_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, train_cfg["grad_clip_norm"])
                optimizer.step()

                epoch_losses.append(total_loss.item())
                latent_z_buffer.append(outputs["z"].detach())
                _update_task_vis_buffer(
                    task_vis_buffer=task_vis_buffer,
                    task_label=task_label,
                    z_batch=outputs["z"].detach(),
                    max_points_per_task=train_cfg["tsne_max_points_per_task"],
                )

                if (batch_idx + 1) % train_cfg["log_every"] == 0:
                    print(
                        f"task={task_id} epoch={epoch + 1} batch={batch_idx + 1}/{len(dataloader)} "
                        f"total={total_loss.item():.4f} "
                        f"recon={base_loss_dict['recon']:.4f} "
                        f"kl={base_loss_dict['kl']:.4f} "
                        f"dyn={base_loss_dict['dynamics']:.4f} "
                        f"wkl={float(wkl_loss.item()):.4f}"
                    )

                if (batch_idx + 1) % dpmm_update_freq == 0:
                    fitted = _fit_dpmm_from_buffer(dpmm, latent_z_buffer)
                    latent_z_buffer = []
                    if fitted and dpmm.model is not None:
                        _log_dpmm_state(
                            dpmm,
                            out_dirs["track_cluster_dir"],
                            out_dirs["component_log_dir"],
                            task_idx=task_idx,
                            epoch=epoch,
                            batch_idx=batch_idx,
                        )
                        _save_tsne_by_task(
                            task_vis_buffer=task_vis_buffer,
                            save_dir=out_dirs["tsne_dir"],
                            step_tag=f"task{task_id}_epoch{epoch}_batch{batch_idx}",
                        )

            # Flush remaining latent buffer at epoch end
            fitted = _fit_dpmm_from_buffer(dpmm, latent_z_buffer)
            latent_z_buffer = []
            if fitted and dpmm.model is not None:
                _log_dpmm_state(
                    dpmm,
                    out_dirs["track_cluster_dir"],
                    out_dirs["component_log_dir"],
                    task_idx=task_idx,
                    epoch=epoch,
                    batch_idx=len(dataloader),
                )
                _save_tsne_by_task(
                    task_vis_buffer=task_vis_buffer,
                    save_dir=out_dirs["tsne_dir"],
                    step_tag=f"task{task_id}_epoch{epoch}_end",
                )

            if epoch_losses:
                avg_loss = float(np.mean(epoch_losses))
                print(f"Task {task_id} | epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = os.path.join(out_dirs["ckpt_dir"], "task_encoder_best.pt")
                    torch.save(model.state_dict(), best_path)
                    print(f"Saved new best TaskEncoder to {best_path}")

        per_task_path = os.path.join(out_dirs["ckpt_dir"], f"task_encoder_task_{task_id}.pt")
        torch.save(model.state_dict(), per_task_path)
        print(f"Saved TaskEncoder checkpoint for task {task_id} to {per_task_path}")

    final_path = os.path.join(out_dirs["ckpt_dir"], "task_encoder_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final TaskEncoder to {final_path}")
    dpmm.save_model(out_dirs["dpmm_save_dir"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_ids", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--epochs_per_task", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dpmm_update_per_epoch", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--benchmark_name", type=str, default="libero_10")
    parser.add_argument("--clip_path", type=str, default="../ckpts/clip-vit-base-patch32")
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument("--gamma0", type=float, default=5.0)
    parser.add_argument("--num_lap", type=int, default=1000)
    parser.add_argument("--sF", type=float, default=1e-5)
    parser.add_argument("--tsne_max_points_per_task", type=int, default=400)
    args = parser.parse_args()

    exp_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.abspath(os.path.join(script_dir, "../dpmm_results", exp_time))

    out_dirs = {
        "exp_dir": exp_dir,
        "component_log_dir": os.path.join(exp_dir, "component_log"),
        "track_cluster_dir": os.path.join(exp_dir, "track_cluster_log"),
        "ckpt_dir": os.path.join(exp_dir, "ckpt"),
        "dpmm_save_dir": os.path.join(exp_dir, "dpmm_model"),
        "tsne_dir": os.path.join(exp_dir, "tsne"),
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    train_cfg = {
        "task_ids": args.task_ids,
        "epochs_per_task": args.epochs_per_task,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "dpmm_update_per_epoch": args.dpmm_update_per_epoch,
        "grad_clip_norm": 1.0,
        "wkl_temperature": 1.0,
        "log_every": args.log_every,
        "loss_weights": {
            "recon": 1.0,
            "kl": 0.0,
            "dynamics": 1.0,
            "wkl": 1e-3,
        },
        "benchmark_name": args.benchmark_name,
        "clip_path": args.clip_path,
        "latent_dim": args.latent_dim,
        "gamma0": args.gamma0,
        "num_lap": args.num_lap,
        "sF": args.sF,
        "tsne_max_points_per_task": args.tsne_max_points_per_task,
    }

    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(train_cfg, f, indent=2)
    print(f"Saved config to {config_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders = build_task_dataloaders(
        task_ids=train_cfg["task_ids"],
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        benchmark_name=train_cfg["benchmark_name"],
    )

    model = TaskEncoder(
        clip_path=train_cfg["clip_path"],
        latent_dim=train_cfg["latent_dim"],
        robot_state_dim=9,
        action_dim=7,
    ).to(device)

    dpmm = BNPModel(
        save_dir=out_dirs["dpmm_save_dir"],
        gamma0=train_cfg["gamma0"],
        num_lap=train_cfg["num_lap"],
        sF=train_cfg["sF"],
    )

    train_task_encoder_with_alternating_dpmm(
        model=model,
        dpmm=dpmm,
        dataloaders=dataloaders,
        train_cfg=train_cfg,
        out_dirs=out_dirs,
        device=device,
    )


if __name__ == "__main__":
    main()
