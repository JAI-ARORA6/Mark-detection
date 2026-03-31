import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -------------------- SETTINGS --------------------
INIT_LR = 1e-4
EPOCHS = 10   # reduced for faster training
BS = 32

DIRECTORY = r"C:\Users\Jai\OneDrive\Desktop\face\Face-Mask-Detection\dataset"
CATEGORIES = ["with_mask", "without_mask"]

print("[INFO] loading images...")

data = []
labels = []

# -------------------- LOAD DATA --------------------
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)

    if not os.path.exists(path):
        print(f"[ERROR] Path not found: {path}")
        continue

    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)

            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)

            data.append(image)
            labels.append(category)

        except Exception as e:
            print(f"[WARNING] Skipping {img}: {e}")

# convert to numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# -------------------- LABEL ENCODING --------------------
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# -------------------- TRAIN TEST SPLIT --------------------
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.20, stratify=labels, random_state=42
)

# -------------------- DATA AUGMENTATION --------------------
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# -------------------- MODEL --------------------
print("[INFO] loading MobileNetV2...")

baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# freeze base layers
for layer in baseModel.layers:
    layer.trainable = False

# -------------------- COMPILE --------------------
print("[INFO] compiling model...")

opt = Adam(learning_rate=INIT_LR)

model.compile(
    loss="binary_crossentropy",
    optimizer=opt,
    metrics=["accuracy"]
)

# -------------------- TRAIN --------------------
print("[INFO] training model...")

H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=max(1, len(trainX) // BS),
    validation_data=(testX, testY),
    validation_steps=max(1, len(testX) // BS),
    epochs=EPOCHS
)

# -------------------- EVALUATE --------------------
print("[INFO] evaluating model...")

predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(
    testY.argmax(axis=1),
    predIdxs,
    target_names=lb.classes_
))

# -------------------- SAVE MODEL --------------------
print("[INFO] saving model...")

model.save("mask_detector.keras")

# -------------------- PLOT --------------------
N = EPOCHS
plt.style.use("ggplot")
plt.figure()

plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

plt.savefig("plot.png")
plt.show()