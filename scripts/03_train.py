"""Main training entry point.

Usage:
    pixi run train
    python scripts/03_train.py
    python scripts/03_train.py model.n_experts=8 training.batch_size=16
"""

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="daly_music", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Import here to avoid slow imports at CLI startup
    from brainsulcal.data.daly_music import DalyMusicDataset
    from brainsulcal.model import BrainSulcal, BrainSulcalConfig
    from brainsulcal.priors.champollion_wrapper import ChampollionWrapper
    from brainsulcal.training.trainer import Trainer

    # Build Champollion wrapper
    champollion_wrapper = ChampollionWrapper(
        embeddings_dir=cfg.data.champollion_embeddings_dir,
    )

    # Discover subjects
    raw_dir = Path(cfg.data.daly_raw_dir)
    all_subjects = sorted([d.name for d in raw_dir.iterdir() if d.name.startswith("sub-")])
    logger.info("Found %d subjects.", len(all_subjects))

    # Build dataset (simple train/val split for non-LOSO run)
    n_val = max(1, len(all_subjects) // 5)
    train_subjects = all_subjects[:-n_val]
    val_subjects = all_subjects[-n_val:]

    train_dataset = DalyMusicDataset(
        train_subjects, raw_dir, champollion_wrapper,
        cache_dir=cfg.data.cache_dir,
        config=OmegaConf.to_container(cfg.data, resolve=True),
    )
    val_dataset = DalyMusicDataset(
        val_subjects, raw_dir, champollion_wrapper,
        cache_dir=cfg.data.cache_dir,
        config=OmegaConf.to_container(cfg.data, resolve=True),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Build model
    model_cfg = BrainSulcalConfig(
        champollion_input_dim=champollion_wrapper.embedding_dim,
        sulcal_hidden_dim=cfg.model.sulcal_aggregator.hidden_dim,
        sulcal_n_heads=cfg.model.sulcal_aggregator.n_heads,
        sulcal_n_layers=cfg.model.sulcal_aggregator.n_layers,
        sulcal_dropout=cfg.model.sulcal_aggregator.dropout,
        n_experts=cfg.model.get("n_experts"),
        router_bias_hidden_dim=cfg.model.moe_router_bias.hidden_dim,
        router_bias_init_std=cfg.model.moe_router_bias.init_std,
        d_model=cfg.model.prefix_proj_dim,
        n_classes=cfg.model.n_classes,
        use_sulcal_prior=cfg.model.get("use_sulcal_prior", True),
        use_router_bias=cfg.model.get("use_router_bias", True),
        use_prefix_token=cfg.model.get("use_prefix_token", True),
    )

    model = BrainSulcal(model_cfg).to(device)
    logger.info("Model built. Trainable params: %d", sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ))

    # Montage info for BrainOmni
    montage_info = {"type": "eeg", "montage": "standard_1020", "n_channels": 31}

    # Init wandb
    try:
        import wandb
        wandb.init(project=cfg.logging.wandb_project, config=OmegaConf.to_container(cfg))
    except Exception as e:
        logger.warning("wandb init failed: %s", e)

    # Train
    trainer = Trainer(
        model, cfg,
        output_dir=Path("runs") / "default",
        use_wandb=True,
    )
    best_metrics = trainer.train(train_loader, val_loader, montage_info)
    logger.info("Training complete. Best metrics: %s", best_metrics)


if __name__ == "__main__":
    main()
