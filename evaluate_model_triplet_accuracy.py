import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --------- Step 1: Build and train Keras model ---------

def build_fcn_model(input_shape=(224, 224, 3), embedding_dim=128, num_classes=3):
    base_cnn = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_cnn.trainable = False  # Freeze base CNN

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_cnn(x)
    x = layers.Dense(embedding_dim)(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name="FashionCompatibilityModel")
    return model

def train_and_convert_model():
    input_shape = (224, 224, 3)
    num_classes = 3

    model = build_fcn_model(input_shape=input_shape, num_classes=num_classes)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    X_train = np.random.rand(100, 224, 224, 3).astype(np.float32)
    y_train = np.random.randint(0, num_classes, 100)

    print("⏳ Training model...")
    model.fit(X_train, y_train, epochs=5, batch_size=16, validation_split=0.2)

    keras_model_path = "outfitmatchai_model.h5"
    model.save(keras_model_path)
    print(f"✅ Keras model saved to {keras_model_path}")

    print("⏳ Converting model to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_model_path = "outfitmatchai_model.tflite"
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ TFLite model saved to {tflite_model_path}")

    return tflite_model_path

# --------- Step 2: Load and preprocess images ---------

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return img_array.astype(np.float32)

def load_images(image_folder, target_size=(224, 224)):
    print(f"📂 Checking directory: {os.path.abspath(image_folder)}")
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"❌ Directory not found: {image_folder}")

    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if len(image_files) == 0:
        print("⚠️ No image files found with extensions .jpg, .jpeg, or .png.")

    images = []
    valid_paths = []

    for img_path in image_files:
        try:
            img_array = load_and_preprocess_image(img_path, target_size)
            images.append(img_array)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"❌ Failed to load {img_path}: {e}")

    print(f"📸 Loaded {len(images)} valid image(s).")
    return np.array(images), valid_paths

# --------- Step 3: Triplet generation ---------

def generate_triplets(images, num_triplets=100):
    triplets = []
    total = len(images)
    for _ in range(num_triplets):
        anchor_idx = np.random.randint(0, total)
        positive_idx = anchor_idx
        while positive_idx == anchor_idx:
            positive_idx = np.random.randint(0, total)

        negative_idx = anchor_idx
        while negative_idx == anchor_idx or negative_idx == positive_idx:
            negative_idx = np.random.randint(0, total)

        triplets.append((images[anchor_idx], images[positive_idx], images[negative_idx]))
    return triplets

# --------- Step 4: Embedding and Evaluation ---------

def get_embedding_tflite(interpreter, input_details, output_details, input_image):
    input_data = np.expand_dims(input_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def evaluate_model_with_tflite(interpreter, input_details, output_details, triplets, batch_size=10):
    correct = 0
    total = len(triplets)
    accuracies = []
    triplet_counts = []
    true_labels = []
    predicted_labels = []

    for i, (anchor_img, positive_img, negative_img) in enumerate(triplets):
        anchor_emb = get_embedding_tflite(interpreter, input_details, output_details, anchor_img)
        positive_emb = get_embedding_tflite(interpreter, input_details, output_details, positive_img)
        negative_emb = get_embedding_tflite(interpreter, input_details, output_details, negative_img)

        pos_dist = np.linalg.norm(anchor_emb - positive_emb)
        neg_dist = np.linalg.norm(anchor_emb - negative_emb)

        if pos_dist < neg_dist:
            correct += 1
            predicted_labels.append(1)  # match correct
        else:
            predicted_labels.append(0)  # mismatch
        true_labels.append(1)  # Always treat as match expected

        if (i + 1) % batch_size == 0 or (i + 1) == total:
            current_accuracy = correct / (i + 1)
            accuracies.append(current_accuracy)
            triplet_counts.append(i + 1)
            print(f"🌀 Processed {i+1}/{total} triplets - Accuracy: {current_accuracy*100:.2f}%", end='\r')

    print()

    # --- Plotting accuracy ---
    plt.figure(figsize=(10, 5))
    plt.plot(triplet_counts, accuracies, marker='o', color='green', linewidth=2)
    plt.title("📈 Triplet Accuracy Over Evaluation", fontsize=14)
    plt.xlabel("Triplets Evaluated", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

    # --- Confusion Matrix ---
    cm = confusion_matrix(true_labels, predicted_labels, labels=[1, 0])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Match", "Mismatch"])
    disp.plot(cmap='Blues')
    plt.title("🧮 Triplet Matching Confusion Matrix")
    plt.show()

    final_acc = accuracies[-1]
    print(f"\n✅ Final Triplet Accuracy: {final_acc * 100:.2f}%")
    return final_acc

# --------- Main pipeline ---------

if __name__ == "__main__":
    tflite_model_path = train_and_convert_model()

    print("🔄 Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    print("✅ TFLite model loaded.")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Update this path to your local image folder
    image_folder = r'C:\Users\hp\Desktop\outfitmatchai\data\images2'

    val_images, val_paths = load_images(image_folder)

    if len(val_images) < 3:
        raise ValueError("❌ Need at least 3 validation images to generate triplets.")

    print("⚡ Generating triplets...")
    triplets = generate_triplets(val_images, num_triplets=100)

    print("🔍 Evaluating TFLite model with graphical results...")
    final_accuracy = evaluate_model_with_tflite(interpreter, input_details, output_details, triplets)
