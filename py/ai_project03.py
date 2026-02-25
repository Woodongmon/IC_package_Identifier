from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import glob

categories = ['tqfp', 'dip', 'to_220']

model = load_model('pack_binary_classification_1.0000.keras')
model.summary()

img_dir = '/media/user23/data/chip_package/'
image_width = 128
image_height = 128

tqfp_files = glob.glob(img_dir + 'tqfp/*.*')
dip_files = glob.glob(img_dir + 'dip/*.*')
to220_files = glob.glob(img_dir + 'to_220/*.*')

sample_files = []
sample_categories = []

if tqfp_files:
    sample_files.append(np.random.choice(tqfp_files))
    sample_categories.append('tqfp')
if dip_files:
    sample_files.append(np.random.choice(dip_files))
    sample_categories.append('dip')
if to220_files:
    sample_files.append(np.random.choice(to220_files))
    sample_categories.append('to_220')

for img_path, actual_cat in zip(sample_files, sample_categories):
    try:
        img = Image.open(img_path)
        img.show()
        img = img.convert('RGB')
        img = img.resize((image_width, image_height))
        data = np.asarray(img) / 255.0
        data = data.reshape(1, image_width, image_height, 3)

        pred = model.predict(data, verbose=0)
        pred_idx = np.argmax(pred[0])
        pred_cat = categories[pred_idx]
        confidence = pred[0][pred_idx] * 100

        print(f'\n파일: {img_path}')
        print(f'실제: {actual_cat}')
        print(f'예측: {pred_cat} ({confidence:.1f}%)')
        print(f'확률: {pred[0]}')

    except Exception as e:
        print(f'Error: {e}')


