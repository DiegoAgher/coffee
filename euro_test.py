import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset

# Download the EuroSAT dataset
data = tfds.load('eurosat/rgb', split='train', shuffle_files=True)
print(type(data))
#tfds.show_examples(data)

# Iterate over the dataset and print some info
counted = 0
randoms = [0, 1, 3, 7, 10, 37]
images = []
labels = []
for idx, example in enumerate(data):
    if not idx in randoms:
        continue
    image, label = example['image'], example['label']
    print(f"Image shape: {image.shape}, Label: {label}")
    print(type(image))
    #plt.imshow(image)
    #plt.show()
    images.append(image)
    labels.append(label)
    counted += 1
    if counted >= len(randoms):
        break
x = torch.tensor([i.numpy() for i in images], dtype=torch.float32)
y = torch.tensor([i.numpy() for i in labels], dtype=torch.float32)
pytorch_dataset = TensorDataset(x, y)
print(pytorch_dataset)
print("it finished")