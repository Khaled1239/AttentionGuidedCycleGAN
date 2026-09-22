import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from Models.AttentionGuidedCyclegan import AttentionGuidedCycleGAN
from Utils.Dataset import ImageDataset
from Utils.ReplayBuffer import ReplayBuffer


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

SOURCE_DIR = "Data/C_NMC"
TARGET_DIR = "Data/ALL_IDB"

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

IMAGE_SIZE = 128

BATCH_SIZE = 1

NUM_EPOCHS = 200

LEARNING_RATE = 1e-4

BETA1 = 0.5
BETA2 = 0.999

# ------------------------------------------------------------
# Loss weights
# ------------------------------------------------------------

LAMBDA_GAN = 0.5

LAMBDA_CYCLE = 10.0

LAMBDA_PIXEL = 1.0

# ------------------------------------------------------------
# Replay buffer
# ------------------------------------------------------------

BUFFER_SIZE = 50

# ------------------------------------------------------------
# Checkpoints
# ------------------------------------------------------------

CHECKPOINT_DIR = "Checkpoints"

OUTPUT_DIR = "Outputs"

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("=" * 70)

print(
    "ATTENTION-GUIDED CYCLEGAN TRAINING"
)

print("=" * 70)

print(
    f"\nDevice          : {DEVICE}"
)

print(
    f"Source          : {SOURCE_DIR}"
)

print(
    f"Target          : {TARGET_DIR}"
)

print(
    f"Image size      : {IMAGE_SIZE}"
)

print(
    f"Batch size      : {BATCH_SIZE}"
)

print(
    f"Epochs          : {NUM_EPOCHS}"
)

print(
    f"Learning rate   : {LEARNING_RATE}"
)

print(
    f"Lambda GAN      : {LAMBDA_GAN}"
)

print(
    f"Lambda Cycle    : {LAMBDA_CYCLE}"
)

print(
    f"Lambda Pixel    : {LAMBDA_PIXEL}"
)

print(
    f"Replay buffer   : {BUFFER_SIZE}"
)


# ============================================================
# DATASET
# ============================================================

print("\nLoading dataset...")


dataset = ImageDataset(
    source_dir=SOURCE_DIR,
    target_dir=TARGET_DIR,
    image_size=IMAGE_SIZE
)


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


print(
    f"Dataset size: {len(dataset)}"
)


# ============================================================
# MODEL
# ============================================================

print("\nCreating model...")


model = AttentionGuidedCycleGAN(
    input_channels=3,
    output_channels=3
)


model = model.to(DEVICE)


# ============================================================
# GAN LOSS
# ============================================================

# Least Squares GAN
#
# Real -> 1
# Fake -> 0

criterion_gan = nn.MSELoss()


# ============================================================
# CYCLE LOSS
# ============================================================

criterion_cycle = nn.L1Loss()


# ============================================================
# PIXEL LOSS
# ============================================================

criterion_pixel = nn.L1Loss()


# ============================================================
# OPTIMIZERS
# ============================================================

optimizer_G = optim.Adam(

    list(model.G_ST.parameters())
    +
    list(model.F_TS.parameters()),

    lr=LEARNING_RATE,

    betas=(
        BETA1,
        BETA2
    )
)


optimizer_D = optim.Adam(

    list(model.D_T.parameters())
    +
    list(model.D_S.parameters()),

    lr=LEARNING_RATE,

    betas=(
        BETA1,
        BETA2
    )
)


# ============================================================
# LR SCHEDULER
# ============================================================

def lambda_lr(epoch):

    """
    Keep LR constant until epoch 100.

    Then linearly decay to zero
    by the final epoch.
    """

    if epoch < 100:

        return 1.0

    return 1.0 - (
        epoch - 100
    ) / (
        NUM_EPOCHS - 100
    )


scheduler_G = optim.lr_scheduler.LambdaLR(
    optimizer_G,
    lr_lambda=lambda_lr
)


scheduler_D = optim.lr_scheduler.LambdaLR(
    optimizer_D,
    lr_lambda=lambda_lr
)


# ============================================================
# REPLAY BUFFERS
# ============================================================

fake_source_buffer = ReplayBuffer(
    max_size=BUFFER_SIZE
)

fake_target_buffer = ReplayBuffer(
    max_size=BUFFER_SIZE
)


# ============================================================
# CHECKPOINT PATH
# ============================================================

LATEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "Latest.pt"
)


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {

    "G_total": [],
    "D_total": [],

    "G_gan": [],
    "G_cycle": [],
    "G_pixel": [],

    "D_T": [],
    "D_S": []
}


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"Epoch {epoch}/{NUM_EPOCHS}"
    )

    print(
        f"{'=' * 70}"
    )


    # --------------------------------------------------------
    # Running losses
    # --------------------------------------------------------

    epoch_G_total = 0.0

    epoch_D_total = 0.0

    epoch_G_gan = 0.0

    epoch_G_cycle = 0.0

    epoch_G_pixel = 0.0

    epoch_D_T = 0.0

    epoch_D_S = 0.0


    # ========================================================
    # BATCH LOOP
    # ========================================================

    progress = tqdm(
        dataloader,
        desc=f"Epoch {epoch}"
    )


    for batch in progress:

        # ----------------------------------------------------
        # LOAD DATA
        # ----------------------------------------------------

        source = batch["source"].to(
            DEVICE,
            non_blocking=True
        )

        target = batch["target"].to(
            DEVICE,
            non_blocking=True
        )


        # ====================================================
        # GENERATOR TRAINING
        # ====================================================

        optimizer_G.zero_grad()


        # ----------------------------------------------------
        # SOURCE -> TARGET
        # ----------------------------------------------------

        fake_target, attention_source = model.G_ST(
            source
        )


        # ----------------------------------------------------
        # TARGET -> SOURCE
        # ----------------------------------------------------

        fake_source, attention_target = model.F_TS(
            target
        )


        # ----------------------------------------------------
        # CYCLE:
        #
        # source -> target -> source
        # ----------------------------------------------------

        cycle_source, _ = model.F_TS(
            fake_target
        )


        # ----------------------------------------------------
        # CYCLE:
        #
        # target -> source -> target
        # ----------------------------------------------------

        cycle_target, _ = model.G_ST(
            fake_source
        )


        # ====================================================
        # GAN LOSS
        # ====================================================

        # ----------------------------------------------------
        # Attention-masked fake target
        # ----------------------------------------------------

        masked_fake_target = (
            attention_source
            * fake_target
        )


        # ----------------------------------------------------
        # D_T prediction
        # ----------------------------------------------------

        pred_fake_target = model.D_T(
            masked_fake_target
        )


        target_real_label = torch.ones_like(
            pred_fake_target
        )


        loss_G_T = criterion_gan(
            pred_fake_target,
            target_real_label
        )


        # ----------------------------------------------------
        # Attention-masked fake source
        # ----------------------------------------------------

        masked_fake_source = (
            attention_target
            * fake_source
        )


        # ----------------------------------------------------
        # D_S prediction
        # ----------------------------------------------------

        pred_fake_source = model.D_S(
            masked_fake_source
        )


        target_real_source = torch.ones_like(
            pred_fake_source
        )


        loss_G_S = criterion_gan(
            pred_fake_source,
            target_real_source
        )


        loss_G_gan = (
            loss_G_T
            +
            loss_G_S
        )


        # ====================================================
        # CYCLE CONSISTENCY LOSS
        # ====================================================

        loss_cycle_source = criterion_cycle(
            cycle_source,
            source
        )


        loss_cycle_target = criterion_cycle(
            cycle_target,
            target
        )


        loss_cycle = (
            loss_cycle_source
            +
            loss_cycle_target
        )


        # ====================================================
        # PIXEL LOSS
        # ====================================================

        # The paper uses pixel loss to encourage
        # the translated image to avoid unnecessary
        # changes.
        #
        # Compare source with translated source
        # and target with translated target.

        loss_pixel_source = criterion_pixel(
            fake_target,
            source
        )


        loss_pixel_target = criterion_pixel(
            fake_source,
            target
        )


        loss_pixel = (
            loss_pixel_source
            +
            loss_pixel_target
        )


        # ====================================================
        # TOTAL GENERATOR LOSS
        # ====================================================

        loss_G = (

            LAMBDA_GAN
            * loss_G_gan

            +

            LAMBDA_CYCLE
            * loss_cycle

            +

            LAMBDA_PIXEL
            * loss_pixel
        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss_G.backward()

        optimizer_G.step()


        # ====================================================
        # DISCRIMINATOR TRAINING
        # ====================================================

        optimizer_D.zero_grad()


        # ----------------------------------------------------
        # D_T
        # ----------------------------------------------------

        # Real target
        #
        # Use the source attention mask to focus D_T
        # on the corresponding relevant region.

        masked_real_target = (
            attention_source.detach()
            * target
        )


        pred_real_target = model.D_T(
            masked_real_target
        )


        loss_D_T_real = criterion_gan(
            pred_real_target,
            torch.ones_like(
                pred_real_target
            )
        )


        # Fake target
        fake_target_buffered = (
            fake_target_buffer
            .push_and_pop(
                masked_fake_target.detach()
            )
        )


        pred_fake_target = model.D_T(
            fake_target_buffered
        )


        loss_D_T_fake = criterion_gan(
            pred_fake_target,
            torch.zeros_like(
                pred_fake_target
            )
        )


        loss_D_T = (
            loss_D_T_real
            +
            loss_D_T_fake
        ) * 0.5


        # ----------------------------------------------------
        # D_S
        # ----------------------------------------------------

        masked_real_source = (
            attention_target.detach()
            * source
        )


        pred_real_source = model.D_S(
            masked_real_source
        )


        loss_D_S_real = criterion_gan(
            pred_real_source,
            torch.ones_like(
                pred_real_source
            )
        )


        fake_source_buffered = (
            fake_source_buffer
            .push_and_pop(
                masked_fake_source.detach()
            )
        )


        pred_fake_source = model.D_S(
            fake_source_buffered
        )


        loss_D_S_fake = criterion_gan(
            pred_fake_source,
            torch.zeros_like(
                pred_fake_source
            )
        )


        loss_D_S = (
            loss_D_S_real
            +
            loss_D_S_fake
        ) * 0.5


        # ====================================================
        # TOTAL DISCRIMINATOR LOSS
        # ====================================================

        loss_D = (
            loss_D_T
            +
            loss_D_S
        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss_D.backward()

        optimizer_D.step()


        # ====================================================
        # RECORD LOSSES
        # ====================================================

        epoch_G_total += loss_G.item()

        epoch_D_total += loss_D.item()

        epoch_G_gan += loss_G_gan.item()

        epoch_G_cycle += loss_cycle.item()

        epoch_G_pixel += loss_pixel.item()

        epoch_D_T += loss_D_T.item()

        epoch_D_S += loss_D_S.item()


        # ----------------------------------------------------
        # PROGRESS BAR
        # ----------------------------------------------------

        progress.set_postfix({

            "G": f"{loss_G.item():.4f}",

            "D": f"{loss_D.item():.4f}",

            "Cycle": f"{loss_cycle.item():.4f}",

            "Pixel": f"{loss_pixel.item():.4f}"
        })


    # ========================================================
    # AVERAGE EPOCH LOSSES
    # ========================================================

    num_batches = len(dataloader)


    avg_G_total = (
        epoch_G_total
        / num_batches
    )


    avg_D_total = (
        epoch_D_total
        / num_batches
    )


    avg_G_gan = (
        epoch_G_gan
        / num_batches
    )


    avg_G_cycle = (
        epoch_G_cycle
        / num_batches
    )


    avg_G_pixel = (
        epoch_G_pixel
        / num_batches
    )


    avg_D_T = (
        epoch_D_T
        / num_batches
    )


    avg_D_S = (
        epoch_D_S
        / num_batches
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history["G_total"].append(
        avg_G_total
    )

    history["D_total"].append(
        avg_D_total
    )

    history["G_gan"].append(
        avg_G_gan
    )

    history["G_cycle"].append(
        avg_G_cycle
    )

    history["G_pixel"].append(
        avg_G_pixel
    )

    history["D_T"].append(
        avg_D_T
    )

    history["D_S"].append(
        avg_D_S
    )


    # ========================================================
    # UPDATE LR
    # ========================================================

    scheduler_G.step()

    scheduler_D.step()


    # ========================================================
    # PRINT EPOCH RESULT
    # ========================================================

    current_lr = optimizer_G.param_groups[0]["lr"]


    print(
        "\nEpoch summary:"
    )

    print(
        f"  G Total : {avg_G_total:.6f}"
    )

    print(
        f"  D Total : {avg_D_total:.6f}"
    )

    print(
        f"  G GAN   : {avg_G_gan:.6f}"
    )

    print(
        f"  Cycle   : {avg_G_cycle:.6f}"
    )

    print(
        f"  Pixel   : {avg_G_pixel:.6f}"
    )

    print(
        f"  D_T     : {avg_D_T:.6f}"
    )

    print(
        f"  D_S     : {avg_D_S:.6f}"
    )

    print(
        f"  LR      : {current_lr:.8f}"
    )


    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    checkpoint = {

        "model":
            model.state_dict(),

        "optimizer_G":
            optimizer_G.state_dict(),

        "optimizer_D":
            optimizer_D.state_dict(),

        "scheduler_G":
            scheduler_G.state_dict(),

        "scheduler_D":
            scheduler_D.state_dict(),

        "epoch":
            epoch,

        "history":
            history
    }


    torch.save(
        checkpoint,
        LATEST_CHECKPOINT
    )


    # --------------------------------------------------------
    # Epoch-specific checkpoint
    # --------------------------------------------------------

    epoch_checkpoint_path = os.path.join(

        CHECKPOINT_DIR,

        f"Epoch_{epoch:03d}.pt"
    )


    torch.save(
        checkpoint,
        epoch_checkpoint_path
    )


    print(
        f"\nCheckpoint saved:"
    )

    print(
        f"  {LATEST_CHECKPOINT}"
    )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 70)

print(
    "TRAINING COMPLETE"
)

print("=" * 70)

print(
    f"\nFinal checkpoint:"
)

print(
    LATEST_CHECKPOINT
)