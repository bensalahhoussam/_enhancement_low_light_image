
import copy
from typing import Dict, List
import os
from tqdm import tqdm
import torch
import cv2
import numpy  as np


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn1: torch.nn.Module,loss_fn2: torch.nn.Module,loss_fn3: torch.nn.Module,
               optimizer: torch.optim.Optimizer,scheduler,epoch_num: int,device: torch.device) -> float:



    model.train()
    batch_losses = []


    prog_bar = tqdm(dataloader,desc=f"Train Epoch {epoch_num + 1}",
        unit="batch",bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    for batch, (low_light, ref) in enumerate(prog_bar):
        low,high = low_light.to(device), ref.to(device)

        current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        optimizer.zero_grad()
        img, cr = model(low, high)


        with torch.no_grad():
            f_i = model.encoder(low)
            f_p = model.encoder(img)
            f_cat,cp = model.concat(f_i, f_p)


        loss1 = loss_fn1(img, high)
        loss2 = loss_fn2(cr, cp)
        loss3 = loss_fn3(img, low)

        loss = loss3 + loss1 + 2. * loss2


        loss.backward()
        optimizer.step()

        """scheduler.step()"""

        batch_loss = loss.item()

        batch_losses.append(batch_loss)


        prog_bar.set_postfix({'loss': f'{batch_loss:.4f}',"current_lr":f"{current_lr:.6f}"}, refresh=True)


    final_loss = sum(batch_losses) / len(batch_losses)

    return final_loss


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn1: torch.nn.Module,loss_fn2: torch.nn.Module,loss_fn3: torch.nn.Module,
              epoch_num: int,
              device: torch.device) -> float:
    model.eval()
    batch_losses = []
    prog_bar = tqdm(dataloader,
            desc=f"Test Epoch {epoch_num + 1}",
            unit="batch",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    with torch.inference_mode():
        for batch, (low_light, ref) in enumerate(prog_bar):
            low, high = low_light.to(device), ref.to(device)
            img,cr  = model(low, high)
            with torch.no_grad():
                f_i = model.encoder(low)
                f_p = model.encoder(img)
                f_cat, cp = model.concat(f_i, f_p)
            loss1 = loss_fn1(img, high)
            loss2 = loss_fn2(cr, cp)
            loss3 = loss_fn3(img, low)
            loss = loss3 + loss1+2.*loss2
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            prog_bar.set_postfix({'loss': f'{ batch_loss:.4f}',}, refresh=True)
            pred_out,c = model(low, high)
            img = pred_out[0].permute(1, 2, 0).cpu().numpy()
            cv2.imwrite("pred.png", (img * 255).astype(np.uint8)[:, :, ::-1])
    final_loss = sum(batch_losses) / len(batch_losses)
    return final_loss


def train(model: torch.nn.Module,train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn1: torch.nn.Module,loss_fn2: torch.nn.Module,loss_fn3: torch.nn.Module,
          epochs: int,scheduler,device: torch.device) -> Dict[str, List]:
    results = {"train_loss": [], "test_loss": [], }
    model.to(device)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 80)

        train_loss = train_step(model=model,dataloader=train_dataloader,
            loss_fn1=loss_fn1,loss_fn2=loss_fn2,loss_fn3=loss_fn3,
            optimizer=optimizer,scheduler=scheduler,epoch_num=epoch,device=device)

        test_loss = test_step(model=model,dataloader=test_dataloader,
            loss_fn1=loss_fn1,loss_fn2=loss_fn2,loss_fn3=loss_fn3,
            epoch_num=epoch,device=device)

        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

        print(f"\nEpoch {epoch + 1} Summary (Averages):")
        print(f"  Train - Loss: {train_loss:.4f}")
        print(f"  Test  - Loss: {test_loss:.4f}")

        checkpoint_path = os.path.join("checkpoints", f"epoch_{epoch + 1}.pth")
        torch.save({"epoch": epoch + 1,
            "model_state_dict": model.state_dict(),"optimizer_state_dict": optimizer.state_dict(),
            """"scheduler_state_dict": scheduler.state_dict(),""""loss": test_loss,}, checkpoint_path)

        print(f"Model saved at {checkpoint_path}")


    return results



