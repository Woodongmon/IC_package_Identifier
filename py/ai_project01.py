from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split

img_dir = '/media/user23/data/chip_package/'
categories = ['tqfp', 'dip', 'to_220']
extensions = ['jpg', 'png', 'webp']

image_width = 128
image_height = 128
pixel = image_width * image_height * 3

X = []
Y = []

for idx, category in enumerate(categories):
    files = []

    for ext in extensions:
        files += glob.glob(img_dir + category + '/*.' + ext)

    for i, f in enumerate(files):
        try:
            img = Image.open(f)
            img = img.convert('RGB')
            data = img.resize((image_width, image_height))
            X.append(data)
            Y.append(idx)

        except (IOError, OSError) as e:
            print(category, i, e)

X = np.array(X)
Y = np.array(Y)
X = X / 255.0 # 스케일링

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

np.save('pack_binary_X_train.npy', X_train)
np.save('pack_binary_X_test.npy', X_test)
np.save('pack_binary_Y_train.npy', Y_train)
np.save('pack_binary_Y_test.npy', Y_test)








