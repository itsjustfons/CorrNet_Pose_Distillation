import re
import matplotlib.pyplot as plt

def parse_losses(filename):
    train_losses = []
    val_losses = []

    with open(filename, 'r') as f:
        for line in f:
            train_match = re.search(r"loss: tensor\(([\d\.eE+-]+)", line)
            val_match = re.search(r"validation loss: tensor\(([\d\.eE+-]+)", line)

            if train_match and not val_match:
                train_losses.append(float(train_match.group(1)))
            elif val_match:
                val_losses.append(float(val_match.group(1)))

    return train_losses, val_losses

def plot_losses(train_losses, val_losses, output_file='loss_plot.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Step (approx)')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")

if __name__ == '__main__':
    # Change 'your_output_file.out' to your actual file name
    train_losses, val_losses = parse_losses('/data/group1/z40575r/CorrNet_pose_distillation/CorrNet/pjsub_train.sh.2001560.out')
    plot_losses(train_losses, val_losses)
