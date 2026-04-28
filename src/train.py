import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torchvision.models as models

from dataloading import get_data_loaders

CONFIG = dict(
    img_size = 224,
    batch_size = 32,
    accumulation_steps = 2,     
    num_workers = 4,
    max_samples = None,   

    head_lr           = 1e-3,  
    head_epochs       = 3,
    finetune_lr       = 1e-4, 
    finetune_epochs   = 15,

    pos_weight        = None,

    checkpoint_dir    = "checkpoints",
    best_model_path   = "resnet18_best.pth",
)


def build_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(device)


def freeze_backbone(model):
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def run_epoch(model, loader, criterion, optimizer, device, scaler,
              accumulation_steps, is_train):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    correct = total = 0
    steps = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        if is_train:
            optimizer.zero_grad()

        pbar = tqdm(loader, desc="train" if is_train else "val", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

            with torch.autocast(device_type=device.type, enabled=(device.type != "cpu")):
                outputs = model(images)
                loss    = criterion(outputs, labels)
                if is_train:
                    loss = loss / accumulation_steps

            if is_train:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            batch_loss = loss.item() * (accumulation_steps if is_train else 1)
            total_loss += batch_loss
            steps += 1

            preds = (outputs.detach() > 0).int()
            correct += (preds == labels.int()).sum().item()
            total   += labels.size(0)

            pbar.set_postfix(loss=f"{batch_loss:.4f}")

    return total_loss / steps, correct / total


def train_stage(model, train_loader, val_loader, criterion, device, scaler,
                lr, num_epochs, accumulation_steps, scheduler_T, start_epoch=0,
                best_val_acc=0.0, best_path="resnet18_best.pth", stage_name=""):

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_T, eta_min=lr * 0.01)

    for epoch in range(num_epochs):
        ep = start_epoch + epoch + 1
        print(f"\nEpoch {ep}  [{stage_name}]")

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer,
                                          device, scaler, accumulation_steps, is_train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer,
                                          device, scaler, accumulation_steps, is_train=False)
        scheduler.step()

        print(f"Train - loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"Val   - loss: {val_loss:.4f}  acc: {val_acc:.4f}  lr: {scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"New best saved : {best_path}  (val_acc={best_val_acc:.4f})")

    return best_val_acc, start_epoch + num_epochs


def main():
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_data_loaders(
        batch_size   = CONFIG["batch_size"],
        max_samples  = CONFIG["max_samples"],
        num_workers  = CONFIG["num_workers"],
        img_size     = CONFIG["img_size"],
        use_weighted_sampler = (CONFIG["max_samples"] is None),
    )

    model = build_model(device)

    pos_weight = (torch.tensor([CONFIG["pos_weight"]], device=device)
                  if CONFIG["pos_weight"] else None)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    best_val_acc = 0.0

    #train head only
    print("\n" + "="*50)
    print("Stage 1 — Training classification head only")
    print("="*50)
    freeze_backbone(model)
    best_val_acc, epoch_offset = train_stage(
        model, train_loader, val_loader, criterion, device, scaler,
        lr                = CONFIG["head_lr"],
        num_epochs        = CONFIG["head_epochs"],
        accumulation_steps= CONFIG["accumulation_steps"],
        scheduler_T       = CONFIG["head_epochs"],
        start_epoch       = 0,
        best_val_acc      = best_val_acc,
        best_path         = CONFIG["best_model_path"],
        stage_name        = "Head only",
    )

    #fine-tune full network
    print("\n" + "="*50)
    print("Stage 2 — Fine-tuning full network")
    print("="*50)
    unfreeze_all(model)
    best_val_acc, _ = train_stage(
        model, train_loader, val_loader, criterion, device, scaler,
        lr                = CONFIG["finetune_lr"],
        num_epochs        = CONFIG["finetune_epochs"],
        accumulation_steps= CONFIG["accumulation_steps"],
        scheduler_T       = CONFIG["finetune_epochs"],
        start_epoch       = epoch_offset,
        best_val_acc      = best_val_acc,
        best_path         = CONFIG["best_model_path"],
        stage_name        = "Full fine-tune",
    )

    print(f"\n✓ Training complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f" Best model saved to: {CONFIG['best_model_path']}")


if __name__ == "__main__":
    main()