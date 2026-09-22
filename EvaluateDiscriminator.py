import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from Models.AttentionGuidedCyclegan import AttentionGuidedCycleGAN


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = "Checkpoints/Latest.pt"

CNMC_DIR = "Data/C_NMC"
ALL_IDB_DIR = "Data/ALL_IDB"

OUTPUT_DIR = "Outputs/DomainEval"

IMAGE_SIZE = 128


# ============================================================
# SETUP
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# ============================================================
# IMAGE FUNCTIONS
# ============================================================

def load_images(folder):

    extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff"
    )

    files = []

    for root, _, filenames in os.walk(folder):

        for filename in filenames:

            if filename.lower().endswith(extensions):

                files.append(
                    os.path.join(root, filename)
                )

    return sorted(files)


def load_image(path):

    image = Image.open(path).convert("RGB")

    image = transform(image)

    return image.unsqueeze(0).to(DEVICE)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("Attention-Guided CycleGAN Discriminator Evaluation")
print("=" * 60)

print(f"\nDevice: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT}")


# ------------------------------------------------------------
# CREATE MODEL
# ------------------------------------------------------------

model = AttentionGuidedCycleGAN()


# ------------------------------------------------------------
# LOAD CHECKPOINT
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

print("\nCheckpoint keys:")

if isinstance(checkpoint, dict):
    print(list(checkpoint.keys()))
else:
    print("Checkpoint is not a dictionary.")


# ------------------------------------------------------------
# CHECKPOINT FORMAT
# ------------------------------------------------------------

if isinstance(checkpoint, dict) and "model" in checkpoint:

    print("\nLoading model from checkpoint['model']...")

    model.load_state_dict(
        checkpoint["model"]
    )

    if "epoch" in checkpoint:

        print(
            f"Training epoch: "
            f"{checkpoint['epoch']}"
        )

else:

    print(
        "\nLoading checkpoint directly "
        "as state_dict..."
    )

    model.load_state_dict(checkpoint)


# ------------------------------------------------------------
# DEVICE + EVAL MODE
# ------------------------------------------------------------

model = model.to(DEVICE)

model.eval()

print("\nCheckpoint loaded successfully.")


# ============================================================
# LOAD DATA
# ============================================================

cnmc_files = load_images(CNMC_DIR)

all_idb_files = load_images(ALL_IDB_DIR)


print("\n" + "=" * 60)
print("DATASET")
print("=" * 60)

print(
    f"CNMC images:    {len(cnmc_files)}"
)

print(
    f"ALL-IDB images: {len(all_idb_files)}"
)


if len(cnmc_files) == 0:

    raise RuntimeError(
        f"No images found in {CNMC_DIR}"
    )


if len(all_idb_files) == 0:

    raise RuntimeError(
        f"No images found in {ALL_IDB_DIR}"
    )


# ============================================================
# EVALUATION
# ============================================================

results = []


print("\n" + "=" * 60)
print("EVALUATING DISCRIMINATOR")
print("=" * 60)


with torch.no_grad():

    # ========================================================
    # REAL TARGET: ALL-IDB
    # ========================================================

    print("\nEvaluating REAL ALL-IDB...")

    for index, target_path in enumerate(
        tqdm(
            all_idb_files,
            desc="Real ALL-IDB"
        )
    ):

        # ----------------------------------------------------
        # Load real target image
        # ----------------------------------------------------

        target_image = load_image(
            target_path
        )


        # ----------------------------------------------------
        # Select CNMC source image
        #
        # The attention mask is generated from
        # the source image and applied to the
        # target image.
        # ----------------------------------------------------

        source_path = cnmc_files[
            index % len(cnmc_files)
        ]

        source_image = load_image(
            source_path
        )


        # ----------------------------------------------------
        # Generate attention mask
        #
        # G_ST returns:
        #
        # translated image
        # attention mask
        # ----------------------------------------------------

        _, attention = model.G_ST(
            source_image
        )


        # ----------------------------------------------------
        # Mask REAL target
        # ----------------------------------------------------

        masked_target = (
            attention * target_image
        )


        # ----------------------------------------------------
        # D_T
        # ----------------------------------------------------

        discriminator_output = model.D_T(
            masked_target
        )


        scores = (
            discriminator_output
            .detach()
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Save statistics
        # ----------------------------------------------------

        results.append({

            "image": os.path.basename(
                target_path
            ),

            "domain": "real_ALL_IDB",

            "mean": float(
                scores.mean()
            ),

            "std": float(
                scores.std()
            ),

            "min": float(
                scores.min()
            ),

            "max": float(
                scores.max()
            ),

            "median": float(
                np.median(scores)
            )
        })


    # ========================================================
    # FAKE TARGET: CNMC -> ALL-IDB
    # ========================================================

    print("\nEvaluating TRANSLATED CNMC...")

    for source_path in tqdm(
        cnmc_files,
        desc="Translated CNMC"
    ):

        # ----------------------------------------------------
        # Load CNMC source
        # ----------------------------------------------------

        source_image = load_image(
            source_path
        )


        # ----------------------------------------------------
        # Translate CNMC -> ALL-IDB
        # ----------------------------------------------------

        fake_target, attention = model.G_ST(
            source_image
        )


        # ----------------------------------------------------
        # Apply attention mask
        # ----------------------------------------------------

        masked_fake = (
            attention * fake_target
        )


        # ----------------------------------------------------
        # D_T
        # ----------------------------------------------------

        discriminator_output = model.D_T(
            masked_fake
        )


        scores = (
            discriminator_output
            .detach()
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Save statistics
        # ----------------------------------------------------

        results.append({

            "image": os.path.basename(
                source_path
            ),

            "domain": "translated_CNMC",

            "mean": float(
                scores.mean()
            ),

            "std": float(
                scores.std()
            ),

            "min": float(
                scores.min()
            ),

            "max": float(
                scores.max()
            ),

            "median": float(
                np.median(scores)
            )
        })


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "discriminator_statistics.csv"
)


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image",
            "domain",
            "mean",
            "std",
            "min",
            "max",
            "median"
        ]
    )

    writer.writeheader()

    writer.writerows(results)


print(
    f"\nStatistics saved to:\n"
    f"{csv_path}"
)


# ============================================================
# SPLIT RESULTS
# ============================================================

real_scores = [

    result["mean"]

    for result in results

    if result["domain"] == "real_ALL_IDB"

]


fake_scores = [

    result["mean"]

    for result in results

    if result["domain"] == "translated_CNMC"

]


# ============================================================
# AGGREGATE STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("DISCRIMINATOR SCORE SUMMARY")
print("=" * 60)


print("\nREAL ALL-IDB")

print(
    f"  Number : {len(real_scores)}"
)

print(
    f"  Mean   : {np.mean(real_scores):.6f}"
)

print(
    f"  Std    : {np.std(real_scores):.6f}"
)

print(
    f"  Median : {np.median(real_scores):.6f}"
)

print(
    f"  Min    : {np.min(real_scores):.6f}"
)

print(
    f"  Max    : {np.max(real_scores):.6f}"
)


print("\nTRANSLATED CNMC")

print(
    f"  Number : {len(fake_scores)}"
)

print(
    f"  Mean   : {np.mean(fake_scores):.6f}"
)

print(
    f"  Std    : {np.std(fake_scores):.6f}"
)

print(
    f"  Median : {np.median(fake_scores):.6f}"
)

print(
    f"  Min    : {np.min(fake_scores):.6f}"
)

print(
    f"  Max    : {np.max(fake_scores):.6f}"
)


# ============================================================
# SCORE DIFFERENCE
# ============================================================

real_mean = np.mean(real_scores)

fake_mean = np.mean(fake_scores)

score_difference = abs(
    real_mean - fake_mean
)


print("\n" + "-" * 60)

print(
    "Absolute mean score difference:"
)

print(
    f"{score_difference:.6f}"
)


# ============================================================
# HISTOGRAM
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.hist(
    real_scores,
    bins=30,
    alpha=0.6,
    label="Real ALL-IDB"
)


plt.hist(
    fake_scores,
    bins=30,
    alpha=0.6,
    label="Translated CNMC"
)


plt.xlabel(
    "Discriminator Score"
)

plt.ylabel(
    "Number of Images"
)

plt.title(
    "D_T Score Distribution"
)

plt.legend()

plt.grid(
    alpha=0.2
)


hist_path = os.path.join(
    OUTPUT_DIR,
    "discriminator_histogram.png"
)


plt.savefig(
    hist_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# BOXPLOT
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.boxplot(
    [
        real_scores,
        fake_scores
    ],

    labels=[
        "Real ALL-IDB",
        "Translated CNMC"
    ]
)


plt.ylabel(
    "Discriminator Score"
)

plt.title(
    "D_T Score Comparison"
)


plt.grid(
    axis="y",
    alpha=0.2
)


plot_path = os.path.join(
    OUTPUT_DIR,
    "discriminator_scores.png"
)


plt.savefig(
    plot_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 60)

print("Saved plots:")

print(
    f"  {hist_path}"
)

print(
    f"  {plot_path}"
)

print("\nEvaluation finished.")

print("=" * 60)