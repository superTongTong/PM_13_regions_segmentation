import pandas as pd
import matplotlib.pyplot as plt

# Read the contents of the text file
with open('training_log_2024_1_26_17_08_41.txt', 'r') as file:
    lines = file.readlines()

# Initialize lists to store data
epochs = []
train_losses = []
val_losses = []
Epoch = 0
# Iterate through the lines of the file
for line in lines:
    # Extract relevant information using string manipulation or regular expressions
    # if 'Epoch' in line:
    #     epoch = int(line.split()[3])
    #     epochs.append(epoch)

    if 'train_loss' in line:
        Epoch += 1
        epochs.append(Epoch)
        train_loss = float(line.split()[3])
        train_losses.append(train_loss)
    elif 'val_loss' in line:
        val_loss = float(line.split()[3])
        val_losses.append(val_loss)

# Create a DataFrame from the extracted data
data = {'Epoch': epochs, 'Train Loss': train_losses, 'Val Loss': val_losses}
df = pd.DataFrame(data)

# Plot the data
plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss', color='blue')
plt.plot(df['Epoch'], df['Val Loss'], label='Val Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
# Save the plotted image as a PNG file
plt.savefig('loss_plot.png')
# plt.show()

