import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt(
    "outputs/2025-10-08/16-03-29/train.csv",
    delimiter=",",
    names=True,          # Use first row as column names
    dtype=None,          # Auto-detect types
    encoding="utf-8"     # Avoid byte strings
)

epochs = data['Epoch'][1:]
train_loss_unit = data['training_loss'][0]
train_loss = data['training_loss'][1:]
val_loss_unit = data['validation_loss'][0]
val_loss = data['validation_loss'][1:]


print(train_loss)
print(val_loss)
plt.plot(epochs, train_loss, label='train loss')
plt.plot(epochs, val_loss, label='val loss')
plt.legend()
plt.show()