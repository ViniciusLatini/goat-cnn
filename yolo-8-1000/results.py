import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_CSV = './runs/detect/train3/results.csv'

def save_full_analysis(csv_path=RESULTS_CSV):
    if not os.path.exists(csv_path):
        print(f"ERROR: File '{csv_path}' not found.")
        return

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    best_idx = df['metrics/mAP50(B)'].idxmax()
    best_epoch = df.iloc[best_idx]
    
    last_10 = df.tail(10)
    avg_map50 = last_10['metrics/mAP50(B)'].mean()
    
    txt_filename = "training_report_SEMKF.txt"
    lines = []
    lines.append("="*40)
    lines.append("       YOLO TRAINING REPORT")
    lines.append("="*40)
    lines.append(f"Total Epochs Trained: {len(df)}")
    lines.append("-" * 40)
    lines.append(f"BEST EPOCH (Index {int(best_epoch['epoch'])})")
    lines.append(f"  > mAP50 (Precision):      {best_epoch['metrics/mAP50(B)']:.4f} ({best_epoch['metrics/mAP50(B)']*100:.2f}%)")
    lines.append(f"  > mAP50-95 (Strict):      {best_epoch['metrics/mAP50-95(B)']:.4f}")
    lines.append(f"  > Precision:              {best_epoch['metrics/precision(B)']:.4f}")
    lines.append(f"  > Recall:                 {best_epoch['metrics/recall(B)']:.4f}")
    lines.append("-" * 40)
    lines.append("FINAL STABILITY (Average of last 10 epochs)")
    lines.append(f"  > Average mAP50:          {avg_map50:.4f}")
    lines.append("-" * 40)
    
    loss_train = best_epoch['train/box_loss']
    loss_val = best_epoch['val/box_loss']
    lines.append("LOSS DIAGNOSIS")
    lines.append(f"  > Train Loss: {loss_train:.4f}")
    lines.append(f"  > Val Loss:   {loss_val:.4f}")
    
    if loss_val > loss_train * 1.5:
        lines.append("  > STATUS: OVERFITTING ALERT (Validation loss is much higher than training loss)")
    else:
        lines.append("  > STATUS: HEALTHY (Curves are converging well)")

    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Success: '{txt_filename}' saved to disk.")

    img_filename = "analysis_plots_SEMKF.png"
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].plot(df['epoch'], df['train/box_loss'], label='Train Loss', linewidth=2)
    ax[0].plot(df['epoch'], df['val/box_loss'], label='Val Loss', linewidth=2)
    ax[0].set_title('Error Convergence (Loss)')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', color='green', linewidth=2)
    ax[1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', color='orange', linewidth=2)
    ax[1].set_title('Precision Evolution (mAP)')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Score')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(img_filename, dpi=300)
    plt.close()
    
    print(f"Success: '{img_filename}' saved to disk.")

if __name__ == "__main__":
    save_full_analysis()