import os
import yaml
import json
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from loguru import logger
from tqdm import tqdm

from dataset_vessel3d import build_vessel_data
from models import build_model
from losses import SetCriterion
from models.matcher import build_matcher

class RelationformerTrainer:
    def __init__(self, config, device, net, loss_function, optimizer, train_loader, val_loader):
        self.config = config
        self.device = device
        self.net = net
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = config.TRAIN.EPOCHS
        self.best_val_loss = float('inf')

    def train_epoch(self):
        self.net.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc="Training"):
            images, segs, nodes, edges, radii = [item.to(self.device) for item in batch]
            target = {"nodes": nodes, "edges": edges, "radii": radii}

            self.optimizer.zero_grad()
            h, out = self.net(segs.float() if self.config.MODEL.USE_SEGMENTATION else images)
            losses = self.loss_function(h, out, target)
            losses['total'].backward()
            self.optimizer.step()

            total_loss += losses['total'].item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.net.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images, segs, nodes, edges, radii = [item.to(self.device) for item in batch]
                target = {"nodes": nodes, "edges": edges, "radii": radii}

                h, out = self.net(segs.float() if self.config.MODEL.USE_SEGMENTATION else images)
                losses = self.loss_function(h, out, target)
                total_loss += losses['total'].item()

        return total_loss / len(self.val_loader)

    def train(self):
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            logger.info(f"Epoch {epoch+1}/{self.epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f"best_model_epoch_{epoch+1}.pth")

            if (epoch + 1) % self.config.TRAIN.SAVE_INTERVAL == 0:
                self.save_checkpoint(f"model_epoch_{epoch+1}.pth")

    def save_checkpoint(self, filename):
        checkpoint = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epochs,
            'config': self.config
        }
        torch.save(checkpoint, os.path.join(self.config.TRAIN.SAVE_PATH, filename))
        logger.info(f"Checkpoint saved: {filename}")

def image_graph_collate(batch):
    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        elif isinstance(x, list):
            return torch.tensor(x)
        else:
            raise ValueError(f"Unsupported type: {type(x)}")

def image_graph_collate(batch):
    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        elif isinstance(x, list):
            return torch.tensor(x)
        else:
            raise ValueError(f"Unsupported type: {type(x)}")

    # Each item in the batch is a tuple
    images = torch.cat([item[0][0] for item in batch], 0).contiguous()
    segs = torch.stack([to_tensor(item[1][0]) for item in batch], 0).contiguous()
    points = [item[2][0] for item in batch]
    edges = [item[3][0] for item in batch]
    radii = [item[4][0] for item in batch]
    
    return [images, segs, points, edges, radii]

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = json.loads(json.dumps(config), object_hook=lambda d: type('config', (), d))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds = build_vessel_data(config, mode='split')

    train_loader = DataLoader(train_ds, batch_size=config.DATA.BATCH_SIZE, shuffle=True,
                              num_workers=config.DATA.NUM_WORKERS, collate_fn=image_graph_collate)
    val_loader = DataLoader(val_ds, batch_size=config.DATA.BATCH_SIZE, shuffle=False,
                            num_workers=config.DATA.NUM_WORKERS, collate_fn=image_graph_collate)

    net = build_model(config).to(device)
    matcher = build_matcher(config)
    loss = SetCriterion(config, matcher, net.relation_embed, net.radius_embed)
    optimizer = optim.AdamW(net.parameters(), lr=float(config.TRAIN.BASE_LR), weight_decay=float(config.TRAIN.WEIGHT_DECAY))

    trainer = RelationformerTrainer(config, device, net, loss, optimizer, train_loader, val_loader)
    trainer.train()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default='configs/synth_3D.yaml', help='config file (.yml) containing the hyper-parameters for training.')
    args = parser.parse_args()
    main(args)