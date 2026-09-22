from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

log = Path("Outputs/TrainingLog.csv")
if not log.exists():
    raise FileNotFoundError("Run train.py first.")

df = pd.read_csv(log)
Path("Outputs/Plots").mkdir(parents=True, exist_ok=True)

for col in ["G", "D", "GAN", "cycle", "pixel"]:
    plt.figure()
    plt.plot(df["epoch"], df[col])
    plt.xlabel("Epoch")
    plt.ylabel(col)
    plt.title(col + " loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Outputs/Plots/{col}Loss.png", dpi=150)
    plt.close()

print("Plots saved to Outputs/Plots/")
