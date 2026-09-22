import argparse
import csv
import random
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from Models.AttentionGuidedCyclegan import AttentionGuidedCycleGAN
from Utils.Dataset import create_dataloader
from Utils.Losses import d_lsgan, g_lsgan, cycle_loss, pixel_loss
from Utils.ReplayBuffer import ImagePool
from Utils.ImageUtils import save_triplet


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="Data/C_NMC")
    p.add_argument("--target", default="Data/ALL_IDB")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda_gan", type=float, default=0.5)
    p.add_argument("--lambda_cycle", type=float, default=10.0)
    p.add_argument("--lambda_pixel", type=float, default=1.0)
    p.add_argument("--pool_size", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decay(epoch, total):
    half = total // 2
    return 1.0 if epoch < half else max(0.0, 1.0 - (epoch-half)/(total-half))


def save_ckpt(path, model, opt_g, opt_d, epoch, history):
    torch.save({
        "model": model.state_dict(),
        "optimizer_G": opt_g.state_dict(),
        "optimizer_D": opt_d.state_dict(),
        "epoch": epoch,
        "history": history,
    }, path)


def main():
    a = args()
    set_seed(a.seed)
    device = get_device(a.device)
    print("Device:", device)

    Path("Checkpoints").mkdir(exist_ok=True)
    Path("Outputs/Samples").mkdir(parents=True, exist_ok=True)

    loader = create_dataloader(
        a.source, a.target, a.image_size, a.batch_size, a.num_workers
    )

    model = AttentionGuidedCycleGAN().to(device)

    g_params = list(model.G_ST.parameters()) + list(model.F_TS.parameters())
    d_params = list(model.D_T.parameters()) + list(model.D_S.parameters())

    opt_g = Adam(g_params, lr=a.lr, betas=(0.5, 0.999))
    opt_d = Adam(d_params, lr=a.lr, betas=(0.5, 0.999))

    sch_g = LambdaLR(opt_g, lambda e: decay(e, a.epochs))
    sch_d = LambdaLR(opt_d, lambda e: decay(e, a.epochs))

    pool_t = ImagePool(a.pool_size)
    pool_s = ImagePool(a.pool_size)

    history = {"G": [], "D": [], "GAN": [], "cycle": [], "pixel": []}

    log = Path("Outputs/TrainingLog.csv")
    with log.open("w", newline="", encoding="utf8") as f:
        csv.writer(f).writerow(["epoch", "G", "D", "GAN", "cycle", "pixel"])

    for epoch in range(a.epochs):
        model.train()
        sums = {k: 0.0 for k in history}

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{a.epochs}"):
            s = batch["source"].to(device, non_blocking=True)
            t = batch["target"].to(device, non_blocking=True)

            # ---------------- GENERATORS ----------------
            opt_g.zero_grad(set_to_none=True)

            fake_t, att_s = model.G_ST(s)
            fake_s, att_t = model.F_TS(t)

            rec_s, _ = model.F_TS(fake_t)
            rec_t, _ = model.G_ST(fake_s)

            pred_fake_t = model.D_T(att_s * fake_t)
            pred_fake_s = model.D_S(att_t * fake_s)

            gan = g_lsgan(pred_fake_t) + g_lsgan(pred_fake_s)
            cyc = cycle_loss(s, rec_s) + cycle_loss(t, rec_t)
            pix = pixel_loss(s, fake_t) + pixel_loss(t, fake_s)

            loss_g = a.lambda_gan*gan + a.lambda_cycle*cyc + a.lambda_pixel*pix
            loss_g.backward()
            opt_g.step()

            # ---------------- DISCRIMINATORS ----------------
            opt_d.zero_grad(set_to_none=True)

            fake_t_pool = pool_t.query(fake_t)
            fake_s_pool = pool_s.query(fake_s)

            real_t = model.D_T(att_s.detach() * t)
            fake_t_pred = model.D_T(att_s.detach() * fake_t_pool)

            real_s = model.D_S(att_t.detach() * s)
            fake_s_pred = model.D_S(att_t.detach() * fake_s_pool)

            loss_d = d_lsgan(real_t, fake_t_pred) + d_lsgan(real_s, fake_s_pred)
            loss_d.backward()
            opt_d.step()

            vals = {
                "G": loss_g.item(), "D": loss_d.item(),
                "GAN": gan.item(), "cycle": cyc.item(), "pixel": pix.item()
            }
            for k in history:
                sums[k] += vals[k]

        n = len(loader)
        avg = {k: v/n for k, v in sums.items()}
        for k in history:
            history[k].append(avg[k])

        with log.open("a", newline="", encoding="utf8") as f:
            csv.writer(f).writerow([epoch+1] + [avg[k] for k in history])

        model.eval()
        with torch.no_grad():
            b = next(iter(loader))
            s = b["source"].to(device)[:4]
            fake, _ = model.G_ST(s)
            rec, _ = model.F_TS(fake)
            save_triplet(
                s.cpu(), fake.cpu(), rec.cpu(),
                f"Outputs/Samples/Epoch_{epoch+1:03d}.png"
            )

        if (epoch+1) % a.save_every == 0 or epoch == a.epochs-1:
            save_ckpt(
                f"Checkpoints/Epoch_{epoch+1:03d}.pt",
                model, opt_g, opt_d, epoch+1, history
            )

        save_ckpt("Checkpoints/Latest.pt", model, opt_g, opt_d, epoch+1, history)
        sch_g.step()
        sch_d.step()

    print("Training finished.")
    print("Checkpoint: checkpoints/latest.pt")


if __name__ == "__main__":
    main()
