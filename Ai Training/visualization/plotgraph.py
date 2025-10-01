import matplotlib.pyplot as plt

log_file = r'AI Training/accuracy_log.txt'

epochs, train_accuracies, val_accuracies = [], [], []

global_epoch = 1  # Start from 1 and increment manually

with open(log_file, 'r') as f:
    for line in f:
        line = line.strip()
        print("Processing line:", line)

        if line.startswith("Epoch,") or not line.startswith("Epoch "):
            continue

        try:
            parts = line.split(', ')
            # We ignore the epoch number in the file and use our own counter
            train_accuracy = float(parts[1].split(':')[1].strip().replace('%', ''))
            val_accuracy = float(parts[2].split(':')[1].strip().replace('%', ''))

            epochs.append(global_epoch)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            print(f"Epoch {global_epoch}: Train Accuracy = {train_accuracy}, Validation Accuracy = {val_accuracy}")
            global_epoch += 1
        except (IndexError, ValueError) as e:
            print("Skipping line due to error:", e)

# Plotting
if len(epochs) == 45:
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue', marker='o')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='red', marker='x')

    # Highlight stages
    plt.axvspan(1, 16, color='lightgray', alpha=0.3, label='Stage 1')
    plt.axvspan(16, 31, color='lightgreen', alpha=0.3, label='Stage 2')
    plt.axvspan(31, 45, color='lightyellow', alpha=0.3, label='Stage 3')

    plt.xticks(range(1, 46))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train vs Validation Accuracy Across Curriculum Stages')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print(f"Expected 45 epochs, but found {len(epochs)}. Plotting anyway.")
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue', marker='o')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='red', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train vs Validation Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
