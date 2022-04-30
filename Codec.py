import cv2
import time
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMessageBox
from qMain3 import Ui_MainWindow
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QPixmap
from Plus import Plus
from winsound import PlaySound, SND_FILENAME, SND_ASYNC, SND_LOOP, SND_PURGE


# Класс для GUI
class mywindow(QtWidgets.QMainWindow):
    _start = False
    _pause = False
    _params = False
    _soundF = False
    _soundSm = False
    _path = 1

    def __init__(self):
        super().__init__()
        self.initUI()

    # Настрока для виджетов в GUI
    def initUI(self):

        self.title = 'FireSmoke'
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Размер выводимых изображений
        self.video_size = QtCore.QSize(int(665 * self.ui.k_width), int(500 * self.ui.k_height))

        # Указание видимости кнопок и указание их связи с методами
        self.ui.pushButton_2.setEnabled(True)
        self.ui.pushButton_2.clicked.connect(self.coef)

        self.ui.pushButton_3.clicked.connect(self.stop)
        self.ui.pushButton_3.setEnabled(False)

        self.ui.pushButton_7.setEnabled(False)
        self.ui.pushButton_7.clicked.connect(self.path)

        self.ui.pushButton_5.clicked.connect(self.play_pause)

        self.ui.pushButton_6.clicked.connect(self.analisys)

        self.ui.radioButton.clicked.connect(self.radButChecked)

        self.ui.label.setFixedSize(self.video_size)
        # Создание потока, в котором буду храниться вводимые пользователем данные
        self.params = Edits()
        # Создание потока для считывания видео с камеры
        self.th = Thread1(self)
        # Указание связи потока с методом
        self.th.changePixmap.connect(self.setImage)
        # Создание потока для анализа пламени
        self.thf = ThreadFire(self)
        self.thf.changePixmap.connect(self.setImageF)
        # Создание потока для анализа дыма
        self.ths = ThreadSmoke(self)
        self.ths.changePixmap.connect(self.setImageSm)
        # Создание промежуточного потока для разбиения исходного видеокадра на каналы R,G,B и применения гауссовской фильтрации
        self.th2 = Thread2(self)

        self.thSoundF = ThreadSoundF(self)
        self.thSoundSm = ThreadSoundSm(self)
        # Создание экземпляра класса области, в которой будут храниться промежуточные для модулей результаты
        self.all = Frames()

    # Разрешить проведение анализа
    def radButChecked(self):
        if self.ui.radioButton.isChecked():
            self.ui.radioButton.setText("Камера")
            self.ui.pushButton_7.setEnabled(True)
            self.ui.pushButton_2.setEnabled(False)

        else:
            self.ui.radioButton.setText("Видео")
            self.ui.pushButton_2.setEnabled(True)
            self.ui.pushButton_7.setEnabled(False)
            self._path = 1

    def path(self):
        self._path = QtWidgets.QFileDialog.getOpenFileName()[0]
        self.ui.pushButton_2.setEnabled(True)

    # Вывод основного видеопотока в Label
    @pyqtSlot(np.ndarray)
    def setImage(self, image):
        p = self.convertToQt(image)
        self.ui.label.setPixmap(QPixmap.fromImage(p))

    # Вывод визуализированного анализа пламени
    @pyqtSlot(np.ndarray)
    def setImageF(self, image):
        p = self.convertToQt(image)
        self.ui.label_2.setPixmap(QPixmap.fromImage(p))
        self._soundF = True

    # Вывод визуализированного анализа дыма
    @pyqtSlot(np.ndarray)
    def setImageSm(self, image):
        p = self.convertToQt(image)
        self.ui.label_10.setPixmap(QPixmap.fromImage(p))
        # self.show()
        self.ui.lineEdit_6.setText("%s " % (time.time() - self.th.start_time))
        self._soundSm = True

    # Преобразование массива numpy в QImage для возможности вывода в Label
    def convertToQt(self, image):
        if len(image.shape) == 2:
            # Если результат внутрикадровой обработки выходит в отрицательную область, то приравнять этот пиксель к нулю
            image = np.where(image < 0, 0, image)
            # Преобразование из типа float64 в uint8
            image = image.astype(np.uint8)
            # Получение размеров изображения
            h, w = image.shape
            # Конвертирование исходной матрицы
            convert_to_Qt_format = QtGui.QImage(image.data, w, h, QtGui.QImage.Format_Grayscale8)
            p = convert_to_Qt_format.scaled(int(665 * self.ui.k_width), int(500 * self.ui.k_height), Qt.KeepAspectRatio)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(int(665 * self.ui.k_width), int(500 * self.ui.k_height), Qt.KeepAspectRatio)
        return p

    # Сохранение введенных данных и одновременная проверка
    def coef(self):
        try:
            fGreen = float(self.ui.lineEdit.text())
            smGreen = float(self.ui.lineEdit_2.text())
            gausG = int(self.ui.lineEdit_3.text())
            resolution = int(self.ui.lineEdit_4.text())
            porogSm = int(self.ui.lineEdit_7.text())
            porogF = int(self.ui.lineEdit_8.text())
            self.params.FGreen = fGreen
            self.params.SmGreen = smGreen
            self.params.gausG = gausG
            self.params.porogSm = porogSm
            self.params.porogF = porogF
            self.params.window(resolution)
            frames = int(self.ui.lineEdit_5.text())
            self.params.frames(frames)
            self._params = True

        except:
            QMessageBox.critical(self, "Ошибка", "В качестве переменных могут выступать только численные значения",
                                 QMessageBox.Ok)
            pass

    # Метод для выставления паузы
    def play_pause(self):
        self._start = True
        self._pause = not self._pause
        if self._pause:
            self.ui.pushButton_5.setText("Пауза")
            self.ui.pushButton_3.setEnabled(True)
            if self._soundF or self._soundSm:
                PlaySound(None, SND_PURGE)
                self._soundF = False
                self._soundSm = False
            self.th.start()

        else:
            self.ui.pushButton_5.setText("Старт")

    def analisys(self):
        application.th.cst = 200

    # Метод для остановки видео
    def stop(self):
        self._start = False
        self.ui.pushButton_5.setText("Старт")
        self.ui.pushButton_3.setEnabled(False)
        self.ui.pushButton_2.setEnabled(True)

    # Закрытие приложения и остановка потока
    def closeEvent(self, event):
        reply = QMessageBox.question(self, "", "Выйти?", QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.stop()

            self.thf.quit()
            self.ths.quit()
            self.thSoundF.quit()
            self.thSoundSm.quit()
            self.th.quit()

            # self.th2.stop()
            # self.ths.stop()
            # self.thf.stop()
            # self.thSoundF.stop()
            # self.thSoundSm.stop()
            event.accept()
        else:
            event.ignore()

    def soundPlayF(self):
        if self._soundF:
            self.thSoundF.run()

    def soundPlaySm(self):
        if self._soundSm:
            self.thSoundSm.run()


# Поток для считывания видео с камеры
class Thread1(QThread):
    _again = False
    changePixmap = pyqtSignal(np.ndarray)
    # Вспомогательный счетчик кадров
    css = 0

    # Метод нацеленный на воспроизведение потока
    def run(self):
        # Путь выводимого видеопотока
        self.cap = cv2.VideoCapture(application._path)
        # Указываем разрешение
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        # Частота кадров для камеры
        self.cap.set(5, 30)
        self.cap.set(10, 1)

        #Отключение автофокуса
        #self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        #self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)

        # Счетчик кадров
        self.cst = 0

        # Цикл для считывания
        while True:
            # Остановка цикла, если установлен флаг _pause
            while not application._pause:
                time.sleep(0)

            # Если нажата кнопка stop, остановка цикла полностью
            if not application._start:
                self.cap.release()
                self.cst = 0
                self.css = 0
                application._pause = False
                break

            ret, self.frame = self.cap.read()

            if ret:
                # Вывод видеосигнала на Label
                # self.test = np.zeros_like(self.frame)
                # self.test[:, :, 0] = self.frame[:, :, 0]
                # self.test[:, :, 2] = self.frame[:, :, 2]
                self.changePixmap.emit(self.frame)
                # self.changePixmap.emit(self.test)
                self.cst = self.cst + 1
                # print("Main: " + str(self.cst))
                # Обновление
                application.show()
                # Если пользователем введенны данные и число накопленных кадров не равно 4, то произвести накопление
                # and self.css != (application.params.count)
                if self.cst == 1:
                    application.all.haveF = True
                    application.all.haveSm = True

                    application.all.out_Sm = []
                    application.all.out_F = []

                    application.all.porogSm = None
                    application.all.porogF = None

                if application.all.haveF or application.all.haveSm:
                    application.all.have = True
                else:
                    application.all.have = False

                # print(application.all.have)

                if self.cst % 200 == 1 and self.cst >= 100:
                    # print("CST: " + str(self.cst))
                    application._params = True

                # Через-кадровая выборка
                if application._params == True and self.cst >= 100:
                    print("Номер кадра: " + str(self.cst))
                    print('Время взятия выборки: ' + str(time.time()))
                    # print("CSS: " + str(self.css))
                    if self.css == 0:
                        self.start_time = time.time()
                        application.all.initOrig(self.frame)
                    elif (self.css % 2 == 0):
                        application.all.orig(self.frame)
                        # print(self.css)

                    # Если число накопленных кадров достигло n, то произвести запуск потока для подготовки материалов к дальнейшей обработке
                    if self.css == (application.params.count):
                        application.all.capture(self.frame)
                        application.all.shape = self.frame.shape
                        application._params = False
                        application.th2.start()
                    self.css = self.css + 1

                cv2.waitKey(20)

    def stop(self):
        self.cap.release()


# Класс с инструкциями для хранения переменных полученных из обработки видеопотока. Путем его внедрения, получаем независимость модулей.
class Frames():
    # Кадр на который производится наложение при визуализации
    def capture(self, input):
        self.frame = input

    # Создание нулевого массива четырех выбранных из потока кадров
    def initOrig(self, input):
        # Разделение исходного кадра на три канал
        col = cv2.split(input)
        self.B = [col[0]]
        self.G = [col[1]]
        self.R = [col[2]]

    # Объединение массивов
    def orig(self, input):
        col = cv2.split(input)
        self.B.append(col[0])
        self.G.append(col[1])
        self.R.append(col[2])

    # Область для хранения кадров с Гауссовской фильтрацией
    def gaus(self):
        self.Bg = []
        self.Gg = []
        self.Rg = []

    # Массивы для кадров с внутрикадровой обработкой
    def sub(self):
        self.subSm = []
        self.subF = []

    # Нормализованные кадры
    def norm(self):
        self.normSm = []
        self.normF = []

    # Максимальные значения яркости пикселей на межкадровых разностях
    def max(self):
        self.maxSm = []
        self.maxF = []

    # Межкадровая разность
    def interSub(self):
        self.interSm = None
        self.interF = None
        self.interF_norm = None
        self.interSm_norm = None
        self.interF_ch = None
        self.interSm_ch = None

    # Бинарные изображения
    def binar(self):
        self.binSm = None
        self.binF = None

    # Номер кадра
    def count(self):
        self.cnt = 0

    # Порог бинаризации
    def thrash(self):
        self.haveF = False
        self.haveSm = False
        self.have = False
        self.haveF_in = False
        self.haveSm_in = False
        self.have_in = False
        self.porogF = None
        self.porogSm = None
        self.origSm = None
        self.origF = None
        self.out_Sm = []
        self.out_F = []


# Поток, необходимый для независимости обработки
class Thread2(QThread):
    def run(self):
        ra = Intra()
        # Выделение памяти для хранения обрабатываемых переменных
        application.all.gaus()
        application.all.sub()
        application.all.norm()
        application.all.interSub()
        application.all.binar()
        application.all.max()
        application.all.haveF_in = application.all.haveF
        application.all.haveSm_in = application.all.haveSm

        if application.all.haveF_in or application.all.haveSm_in:
            application.all.have_in = True
        else:
            application.all.have_in = False

        # Гауссовская фильтрация для каждого канала каждого из кадров
        for i in range(0, application.params.count // 2):
            application.all.cnt = i
            ra.gaussian()
        application.all.cnt = 0
        application.thf.run()

    def change(self):
        application.ths.run()



# Поток для звукового модуля
class ThreadSoundF(QThread):
    def run(self):
        if application._soundSm:
            PlaySound("Signal/Together.wav", SND_FILENAME | SND_LOOP | SND_ASYNC)
        else:
            PlaySound("Signal/Fire.wav", SND_FILENAME | SND_LOOP | SND_ASYNC)


# Поток для звука дыма
class ThreadSoundSm(QThread):
    def run(self):
        PlaySound("Signal/Smoke.wav", SND_FILENAME | SND_LOOP | SND_ASYNC)


# Исходные данные
class Edits:
    def __init__(self):
        # К-ты для обработки дыма и пламени
        self.FGreen = 0.54
        self.FBlue = 0.18
        self.SmGreen = 0.32
        self.SmRed = 0.07

        # Размер окна Гауса
        self.gausR = 3
        self.gausB = 3
        self.gausG = 5

        # Количество кадров в выборке
        self.count = 15

        # Размер окна
        self.resolution = 12

        # Порог бинаризации в процентах
        self.porogSm = 120
        self.porogF = 120

    def frames(self, count):
        self.count = count

    def window(self, length):
        self.resolution = length


# Набор инструкций для внутрикадровой обработки
class Intra:
    # Функция применения гауссовского фильтра
    def gaussian(self):
        cnt = application.all.cnt

        preBg = cv2.GaussianBlur(application.all.B[cnt], (application.params.gausB, application.params.gausB), 1)
        application.all.Bg.append(preBg)

        preGg = cv2.GaussianBlur(application.all.G[cnt], (application.params.gausG, application.params.gausG), 1)
        application.all.Gg.append(preGg)

        preRg = cv2.GaussianBlur(application.all.R[cnt], (application.params.gausR, application.params.gausR), 1)
        application.all.Rg.append(preRg)


# Класс межкадровой обработки
class Inter:
    def subFire(self):
        for i in range(1, application.params.count // 2):
            presub = application.all.normF[i] - application.all.normF[i - 1]
            np.true_divide(presub, -1, out=presub, where=presub < 0)  # Модуль через умножение на -1

            application.all.maxF.append(np.max(presub))

            if i == 1:
                sub = (1 / (application.params.count // 2)) * presub
            else:
                sub = sub + (1 / (application.params.count // 2)) * presub

        sub = np.where(sub > 255, 255, sub)

        subs = np.zeros_like(application.all.frame)
        subs[:, :, 2] = sub
        cv2.imwrite("Output/Inter_Sum/Fire.bmp", subs)
        maxs = np.max(sub)
        mins = np.min(sub)
        delta = maxs - mins
        norm = 255 * (sub - mins) / delta
        application.all.maxF = np.max(sub)
        norms = np.zeros_like(application.all.frame)
        norms[:, :, 2] = norm
        cv2.imwrite("Output/Normalize/NormFire.bmp", norms)
        application.all.interF = sub
        application.all.interF_norm = norms

    def subSmoke(self):
        for i in range(1, application.params.count // 2):
            presub = application.all.normSm[i] - application.all.normSm[i - 1]
            np.true_divide(presub, -1, out=presub, where=presub < 0)
            application.all.maxSm.append(np.max(presub))

            if i == 1:
                sub = (1 / (application.params.count // 2)) * presub
            else:
                sub = sub + (1 / (application.params.count // 2)) * presub
        sub = np.where(sub > 255, 255, sub)

        subs = np.zeros_like(application.all.frame)
        subs[:, :, 0] = sub
        cv2.imwrite("Output/Inter_Sum/Smoke.bmp", subs)
        maxs = np.max(sub)
        mins = np.min(sub)
        delta = maxs - mins
        norm = 255 * (sub - mins) / delta
        application.all.maxSm = np.max(sub)
        norms = np.zeros_like(application.all.frame)
        norms[:, :, 0] = norm
        application.all.interSm_norm = norms
        cv2.imwrite("Output/Normalize/NormSm.bmp", norms)
        application.all.interSm = sub
        application.thf._pause = False


# Поток внутрикадровой обработки для огня
class ThreadFire(QThread):
    changePixmap = pyqtSignal(np.ndarray)
    _pause = True

    def run(self):
        oneSub = OneSub()
        oneSub.fire()

        norm = Normalize()
        norm.fire()

        some = Inter()
        some.subFire()

        application.th2.change()

        # Функция для наложения бинарного изображения межкадровой разности на оригинальный кадр из выборки

        binar = Binar()
        binar.fire()

        self.color = np.copy(application.all.frame)

        scan = Scan()
        scan.fire()
        self.changePixmap.emit(self.color)
        application.show()
        self._pause = True


# Внутрикадровая обработка
class OneSub:
    def smoke(self):
        # Извлечение к-тов для каналов
        kGr = application.params.SmGreen
        kR = application.params.SmRed
        # Для каждого выделенного кадра произведем внутрикадровую обработку
        for i in range(0, application.params.count // 2):
            presub = application.all.B[i] - kGr * application.all.Gg[i] - kR * application.all.Rg[i]
            presub = np.where(presub < 0, 0, presub)
            thrs = np.zeros_like(application.all.frame)
            thrs[:, :, 0] = presub
            cv2.imwrite("Output/Intra_Smoke/SmokeFrame_%04i.bmp" % i, thrs)
            # Сохранение файлов в выделенную область памяти
            application.all.subSm.append(presub)

    def fire(self):
        kGr = application.params.FGreen
        kB = application.params.FBlue
        for i in range(0, application.params.count // 2):
            presub = application.all.R[i] - kGr * application.all.Gg[i] - kB * application.all.Bg[i]
            presub = np.where(presub < 0, 0, presub)
            thrs = np.zeros_like(application.all.frame)
            thrs[:, :, 2] = presub
            cv2.imwrite("Output/Intra_Fire/FireFrame_%04i.bmp" % i, thrs)
            application.all.subF.append(presub)


# Класс нормализации
class Normalize:
    def fire(self):
        for i in range(0, len(application.all.subF)):
            maxs = np.max(application.all.subF[i])
            mins = np.min(application.all.subF[i])
            delta = maxs - mins
            norm = 255 * (application.all.subF[i] - mins) / delta
            application.all.normF.append(norm)

    def smoke(self):
        for i in range(0, len(application.all.subSm)):
            maxs = np.max(application.all.subSm[i])
            mins = np.min(application.all.subSm[i])
            delta = maxs - mins
            norm = 255 * (application.all.subSm[i] - mins) / delta
            application.all.normSm.append(norm)


# Класс внутрикадровой обработки для дыма
class ThreadSmoke(QThread):
    changePixmap = pyqtSignal(np.ndarray)

    def run(self):
        # Создание экземпляра для внутрикадровой обработки
        oneSub = OneSub()
        oneSub.smoke()

        norm = Normalize()
        norm.smoke()

        # Создание экземпляра для межкадровой обработки
        some = Inter()
        some.subSmoke()

        application.all.interF_ch, application.all.interSm_ch = Plus(application.all.interF, application.all.interSm,
                                                                     application.all.interF_norm)

        binar = Binar()
        binar.smoke()

        self.color = np.copy(application.all.frame)

        scan = Scan()
        scan.smoke()
        self.changePixmap.emit(self.color)
        application.show()


# Класс бинаризации
class Binar:
    def smoke(self):
        # Вытягиваем информацию из общей области памяти
        mask = application.all.interSm_ch

        if application.all.have_in:
            pre_porog = np.max(application.all.maxSm) * (1 / (application.params.count // 2))
            porog = pre_porog * (application.params.porogSm / 100)
            if application.all.porogSm == None:
                application.all.porogSm = pre_porog
            #kT = (porog / application.all.porogSm) * 100
            #application.ui.label_14.setText(str(kT) + '%')
            #print(porog)

            application.all.porogSm = porog
            application.all.out_Sm.append(porog)
            application.all.haveSm = False

        # Бинаризация изображения
        val, thr = cv2.threshold(mask, application.all.porogSm, 255, cv2.THRESH_BINARY)
        thrs = np.zeros_like(application.all.frame)
        thrs[:, :, 0] = thr
        thr = thr.astype(np.uint8)
        # Вносим бинарное изображение в общую, выделенную ранее, память
        cv2.imwrite("Output/Threshold/Smoke.bmp", thrs)
        application.all.binSm = thr

    def fire(self):
        mask = application.all.interF_ch
        if application.all.have_in:
            pre_porog = np.max(application.all.maxF) * (1 / (application.params.count // 2))
            porog = pre_porog * (application.params.porogF / 100)
            if application.all.porogF == None:
                application.all.porogF = pre_porog
            #kT = (porog / application.all.porogF) * 100
            #application.ui.label_13.setText(str(kT) + '%')
            application.all.porogF = porog
            application.all.haveF = False

        val, thr = cv2.threshold(mask, application.all.porogF, 255, cv2.THRESH_BINARY)
        thr = thr.astype(np.uint8)
        thrs = np.zeros_like(application.all.frame)
        thrs[:, :, 2] = thr
        thr = thr.astype(np.uint8)
        # Вносим бинарное изображение в общую, выделенную ранее, память
        cv2.imwrite("Output/Threshold/Fire.bmp", thrs)
        application.all.binF = thr


class Scan:
    def draw(self, binar, x1, y1, x2, y2, grad):
        retush = np.zeros_like(binar)
        print(binar.shape)
        for i in range(x1, x2):
            for j in range(y1, y2):
                retush[j, i] = binar[j, i]

        bin = cv2.bitwise_not(retush)
        frame = application.all.frame
        img = np.zeros(frame.shape, frame.dtype)

        img[:, :, 0] = bin
        img[:, :, 1] = bin
        img[:, :, 2] = bin

        stack = img.astype('float32') / 255.0
        orig = frame / 255.0

        if grad == 'smoke':
            masked = (stack * orig) + (1 - stack) * (1.0, 0.0, 0.0)
        else:
            masked = (stack * orig) + (1 - stack) * (0.0, 0.0, 1.0)
        masked = (masked * 255).astype('uint8')
        return masked

    def smoke(self):
        bin = application.all.binSm
        # Вытаскиваем информацию о связных компонентах
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bin, connectivity=4)
        areas = stats[:, 4]
        # Сортировка связных областей в порядке убывания
        ranked = sorted(areas, reverse=True)

        iterators = np.argsort(areas)
        iterators = np.flipud(iterators)

        width = stats[:, 2]
        height = stats[:, 3]

        # Вытаскиваем информацию о размере окна сканирования
        res = application.params.resolution

        # Функция для сканирования 8x8
        def iters(x1, y1, x2, y2, bin):

            xn = x1
            while xn < (x2 - res):
                yn = y1
                while yn < (y2 - res):
                    count = 0
                    for k in range(0, res):
                        y = yn + k
                        for l in range(0, res):
                            x = xn + 1
                            if bin[y, x] == 255:
                                count = count + 1
                    if count == res * res:
                        #application.ths.color = draw(application.all.binSm)
                        #application.ths.color = cv2.rectangle(application.ths.color, (xn, yn), (x, y), (255, 255, 0), 9)
                        application._pause = False
                        application.ui.pushButton_5.setText("Старт")
                        application.ui.pushButton_3.setEnabled(False)
                        application.all.haveSm = True
                        return application.ths.color, 1
                    yn = yn + 1
                xn = xn + 1
            return application.ths.color, 0

        ans = 0
        i = 1
        # Перечисление связных объектов
        if len(ranked) > 1:
            while ranked[i] > (res * res):
                j = iterators[i]
                if width[j] > res and height[j] > res:
                    x1 = stats[:, 0][j]
                    y1 = stats[:, 1][j]
                    x2 = x1 + width[j]
                    y2 = y1 + height[j]

                    img, ans = iters(x1, y1, x2, y2, bin)

                    if ans == 1:
                        #application.ths.color = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        application.ths.color = self.draw(application.all.binSm, x1, y1, x2, y2, 'smoke')
                        application.ths.color = cv2.rectangle(application.ths.color, (x1, y1-1), (x2, y2-1), (0, 255, 0), 2)
                        cv2.imwrite("Output/Layout/Smoke.bmp", application.ths.color)
                        application.thSoundSm.run()
                        break
                    if i == len(ranked):
                        break
                i = i + 1
        # Производим обнуление счетчика для подсчета кадров учавствующих в обработке
        application.th.css = 0

    def fire(self):
        bin = application.all.binF
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bin, connectivity=4)
        areas = stats[:, 4]
        ranked = sorted(areas, reverse=True)
        # (ranked)

        iterators = np.argsort(areas)
        iterators = np.flipud(iterators)

        width = stats[:, 2]
        height = stats[:, 3]

        # Вытаскиваем информацию о размере окна сканирования
        res = application.params.resolution

        def iters(x1, y1, x2, y2, bin):
            xn = x1
            while xn < (x2 - res):
                yn = y1
                while yn < (y2 - res):
                    count = 0
                    for k in range(0, res):
                        y = yn + k
                        for l in range(0, res):
                            x = xn + 1
                            if bin[y, x] == 255:
                                count = count + 1
                    if count == (res * res):
                        #application.thf.color = cv2.rectangle(application.thf.color, (xn, yn), (xn+res, yn+res), (255, 0, 255), 1)
                        application._pause = False

                        application.ui.pushButton_5.setText("Старт")
                        application.ui.pushButton_3.setEnabled(False)
                        application.all.haveF = True
                        return application.thf.color, 1
                    yn = yn + 1
                xn = xn + 1
            return application.thf.color, 0

        ans = 0
        i = 1
        if len(ranked) > 1:
            while ranked[i] > (res * res):
                j = iterators[i]
                if width[j] > res and height[j] > res:
                    x1 = stats[:, 0][j]
                    y1 = stats[:, 1][j]
                    x2 = x1 + width[j]
                    y2 = y1 + height[j]
                    img, ans = iters(x1, y1, x2, y2, bin)

                    if ans == 1:
                        application.thf.color = self.draw(application.all.binF, x1, y1, x2, y2, 'fire')
                        application.thf.color = cv2.rectangle(application.thf.color, (x1, y1), (x2-1, y2-1), (15, 255, 23), 2)
                        cv2.imwrite("Output/Layout/Fire.bmp", application.thf.color)
                        application.thSoundF.run()
                        break
                    if i == len(ranked):
                        break
                i = i + 1


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    application = mywindow()
    application.showMaximized()
    sys.exit(app.exec())