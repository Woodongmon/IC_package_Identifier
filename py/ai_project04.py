import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QPixmap
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

form_window = uic.loadUiType('package.ui')[0]

image_width = 128
image_height = 128

class Project(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.path = None
        self.setupUi(self)
        self.model = load_model('pack_binary_classification_1.0000.keras') # 결과 좋은 모델
        self.pushButton.clicked.connect(self.button_slot)

    def button_slot(self):
        self.path = QFileDialog.getOpenFileName(self, 'Open file', '/home/user23/Downloads',
                                                'Image Files(*.jpg *.jpeg *.png *.webp);;All File(*.*)')
        # 사용자가 파일을 선택하지 않고 취소했을 경우를 대비해 경로 유효성 검사
        if not self.path[0]:
            return  # 경로가 없으면 함수 종료

        selected_file_path = self.path[0]

        # 2. UI 레이블에 이미지를 표시합니다.
        pixmap = QPixmap(selected_file_path)
        self.label.setPixmap(pixmap.scaled(self.label.size(), aspectRatioMode=1))  # 라벨 크기에 맞게 스케일링 추가 (권장)

        try:
            img = Image.open(self.path[0])
            img = img.convert('RGB')
            img = img.resize((image_width, image_height))
            data = np.asarray(img)
            data = data / 255.0
            data = data.reshape(1, image_width, image_height, 3)

            predict_value = self.model.predict(data)
            self.result_predict(predict_value)
            print(predict_value)

        except (IOError, OSError) as e:
            print(e)

    def result_predict(self, predict_value):
        categories = ['TQFP', 'DIP', 'TO-220']

        pred_idx = np.argmax(predict_value[0])
        pred_category = categories[pred_idx]
        confidence = predict_value[0][pred_idx] * 100

        result_text = f"예상되는 모델 : {pred_category}\n"
        result_text += f"정확도 : {confidence:.2f}%"

        self.label_2.setText(result_text)

app = QApplication(sys.argv)
mainWindow = Project()
mainWindow.show()
sys.exit(app.exec_())