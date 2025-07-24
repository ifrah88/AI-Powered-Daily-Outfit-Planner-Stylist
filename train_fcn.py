import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import os
import json

# Build FCN Model using MobileNetV2
def build_fcn_model(input_shape=(224, 224, 3), embedding_dim=128, num_classes=3):
    base_cnn = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_cnn.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_cnn(x)
    x = layers.Dense(embedding_dim)(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs, name="FashionCompatibilityModel")

# Load metadata
def load_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        return json.load(f)

# Load image dataset
def load_images_with_labels(image_dir, metadata, class_map):
    X, y = [], []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            item_id = os.path.splitext(filename)[0]
            item_meta = metadata.get(item_id, {})
            class_name = item_meta.get('semantic_category')

            if class_name in class_map:
                image_path = os.path.join(image_dir, filename)
                try:
                    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
                    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                    X.append(img)
                    y.append(class_map[class_name])
                except Exception as e:
                    print(f"⚠️ Error reading {image_path}: {e}")
    return np.array(X), np.array(y)

# Main training logic
if __name__ == "__main__":
    input_shape = (224, 224, 3)
    num_classes = 3

    class_map = {
        'tops': 0,
        'bottoms': 1,
        'jewellery': 2
    }

    image_dir = "C:/Users/hp/Desktop/outfitmatchai/data/images2"
    metadata_path = "C:/Users/hp/Desktop/outfitmatchai/data/polyvore_item_metadata.json"

    print("📂 Loading metadata and images...")
    metadata = load_metadata(metadata_path)
    X, y = load_images_with_labels(image_dir, metadata, class_map)

    print(f"✅ Loaded {len(X)} images.")

    if len(X) == 0:
        print("❌ No data to train on. Make sure image names match metadata IDs and valid categories exist.")
        exit()

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Show training/testing ratio
    plt.figure()
    plt.bar(['Training', 'Testing'], [len(X_train), len(X_test)], color=['blue', 'orange'])
    plt.title("Training vs Testing Data Ratio")
    plt.ylabel("Number of Images")
    plt.grid(True, axis='y')
    plt.show()

    # Build and compile model
    model = build_fcn_model(input_shape=input_shape, num_classes=num_classes)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    print("🏋️ Training model...")

    # === ✅ New: Train and store history ===
    history = model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_test, y_test))

    # === ✅ New: Plot Accuracy and Loss ===
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("🔍 Evaluating model on test data...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_map.keys()))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Data)")
    plt.show()

    # Accuracy Pie Chart
    correct = np.sum(y_test == y_pred)
    incorrect = len(y_test) - correct
    plt.figure()
    plt.pie([correct, incorrect], labels=['Correct', 'Incorrect'],
            colors=['green', 'red'], autopct='%1.1f%%', startangle=140)
    plt.title("Prediction Accuracy on Test Data")
    plt.axis('equal')
    plt.show()

    # Class Distribution on Predictions
    class_counts = np.bincount(y_pred, minlength=num_classes)
    plt.figure()
    plt.pie(class_counts, labels=[f'{cls}' for cls in class_map.keys()],
            autopct='%1.1f%%', startangle=140)
    plt.title("Predicted Class Distribution (Test Data)")
    plt.axis('equal')
    plt.show()

    # Save the model
    model_path = "outfitmatch_model.h5"
    model.save(model_path)
    print(f"✅ Keras model saved to {model_path}")

    # Convert to TFLite
    tflite_path = "outfitmatch_model.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ TFLite model saved to {tflite_path}")
