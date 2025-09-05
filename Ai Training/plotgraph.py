import matplotlib.pyplot as plt

log_file = 'accuracy_log.txt'

epochs, train_accuracies, val_accuracies = [], [], []

current_epoch = 0
with open(log_file, 'r') as f:
    for line in f.readlines():
        print("Processing line:", line.strip())
        if 'Epoch' in line and 'Train Accuracy' in line and 'Validation Accuracy' in line:
            try:
                parts = line.split(', ')  # Split the line by commas
                epoch = int(parts[0].split()[1])  # Extract epoch number
                train_accuracy = float(parts[1].split(':')[1].strip().replace('%', ''))  # Extract train accuracy
                val_accuracy = float(parts[2].split(':')[1].strip().replace('%', ''))  # Extract validation accuracy
                epochs.append(epoch)
                train_accuracies.append(train_accuracy)
                val_accuracies.append(val_accuracy)
                print(f"Epoch: {epoch}, Train Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy}")
            except ValueError as e:
                print("Skipping line due to error:", e)

# Debug output to check the collected data
print("Epochs:", epochs)
print("Train Accuracies:", train_accuracies)
print("Validation Accuracies:", val_accuracies)

# Plotting if there is data
if epochs:
    plt.plot(list(range(1, 16)), train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(list(range(1, 16)), val_accuracies, label='Validation Accuracy', color='red')


    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train vs Validation Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
else:
    print("No data available to plot.")
