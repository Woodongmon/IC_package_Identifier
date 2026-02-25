import numpy as np
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

BATCH_SIZE = 32
EPOCHS = 20

X_train = np.load('./pack_binary_X_train.npy', allow_pickle=True)
X_test = np.load('./pack_binary_X_test.npy', allow_pickle=True)
Y_train = np.load('./pack_binary_Y_train.npy', allow_pickle=True)
Y_test = np.load('./pack_binary_Y_test.npy', allow_pickle=True)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# 데이터 불균형 확인 (TO-220만 너무 많은지 체크)
unique, counts = np.unique(Y_train, return_counts=True)
print("클래스별 데이터 개수:", dict(zip(unique, counts)))

categories = ['tqfp', 'dip', 'to_220']
num_classes = len(categories)

Y_train_cat = to_categorical(Y_train, num_classes)
Y_test_cat = to_categorical(Y_test, num_classes)

base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    Input(shape=(128, 128, 3)),

    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.Rescaling(2.0, offset=-1.0),

    base_model,

    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# ----------------------------------------------------------------------
model.summary()
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0005),
    metrics=['accuracy']
)

# EarlyStopping은 정의만 해두고 fit에는 넣지 않도록 주석 처리 상태 유지
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

fit_hist = model.fit(
    X_train,
    Y_train_cat,
    batch_size=BATCH_SIZE,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

score = model.evaluate(X_test, Y_test_cat)
print('Evaluation loss :', score[0])
print('Evaluation accuracy :', score[1])

# 파일 형식은 .keras로 유지
model.save('./pack_binary_classification_{:.4f}.keras'.format(score[1]))
plt.plot(fit_hist.history['loss'], label='loss')
plt.plot(fit_hist.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

plt.plot(fit_hist.history['accuracy'], label='train accuracy')
plt.plot(fit_hist.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.show()