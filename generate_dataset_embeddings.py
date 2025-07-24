import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import get_custom_objects
from sklearn.decomposition import PCA
import tensorflow as tf
from PIL import Image

# === Define custom l2_norm function ===
def l2_norm(x):
    return tf.math.l2_normalize(x, axis=-1)

# === Register custom layer ===
get_custom_objects().update({'l2_norm': tf.keras.layers.Lambda(l2_norm)})

# === Load the pre-trained model ===
model_path = r"C:\Users\hp\Desktop\outfitmatchai\outfitmatch_model.h5"
model = load_model(model_path, custom_objects={'l2_norm': l2_norm})
print("✅ Model loaded successfully.")

# === Dataset folder and output folder ===
dataset_folder = r"C:\Users\hp\Desktop\outfitmatchai\data\images2"
output_folder = r"C:\Users\hp\Desktop\outfitmatchai\embeddings"
os.makedirs(output_folder, exist_ok=True)

# === Function to extract embedding from image ===
def extract_embedding(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    embedding = model.predict(img_array, verbose=0)
    return embedding.flatten()

# === Loop through dataset and collect embeddings ===
embeddings = []
filenames = []

for file in os.listdir(dataset_folder):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(dataset_folder, file)
        embedding = extract_embedding(img_path)
        embeddings.append(embedding)
        filenames.append(file)

# === Save embeddings ===
embeddings = np.array(embeddings)
filenames = np.array(filenames)
np.save(os.path.join(output_folder, "dataset_embeddings.npy"), embeddings)
np.save(os.path.join(output_folder, "filenames.npy"), filenames)
print(f"✅ Saved {len(filenames)} embeddings and filenames to: {output_folder}")

# === Visualization with PCA ===
if len(embeddings) >= 2:
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    plt.scatter(reduced[:, 0], reduced[:, 1], c='skyblue', edgecolor='black')

    for i, fname in enumerate(filenames[:30]):  # Only label first 30 points
        plt.text(reduced[i, 0], reduced[i, 1], fname.split('.')[0], fontsize=8)

    plt.title("2D PCA Visualization of Outfit Embeddings")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Not enough data for PCA visualization.")

# === Optional: Show thumbnails ===
plt.figure(figsize=(12, 4))
for i in range(min(10, len(filenames))):
    img_path = os.path.join(dataset_folder, filenames[i])
    img = Image.open(img_path).resize((64, 64))
    plt.subplot(1, 10, i + 1)
    plt.imshow(img)
    plt.title(filenames[i][:6], fontsize=8)
    plt.axis('off')
plt.suptitle("🔍 Sample Images from Dataset", fontsize=14)
plt.tight_layout()
plt.show()
