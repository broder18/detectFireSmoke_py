import sys
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGraphicsScene, QGraphicsView,
                             QMainWindow, QApplication, QGraphicsRectItem, QGraphicsItem, QLabel)
from PyQt5.QtCore    import QRectF, QPointF, Qt, QRect
from PyQt5.QtGui     import QTransform

class GraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        super(GraphicsScene, self).__init__(QRectF(-500, -500, 1000, 1000), parent)

        self._start = QPointF()
        self._current_rect_item = None
        self._have = False

    def mousePressEvent(self, event):
        if self.itemAt(event.scenePos(), QTransform()) is None and self._have == False:
            #Создание ссылки на класс рисования прямоугольника
            self._current_rect_item = QGraphicsRectItem()
            # Установить прозрачность
            self._current_rect_item.setOpacity(0.2)
            # Выбор цвета
            self._current_rect_item.setBrush(Qt.red)

            self._current_rect_item.setFlag(QGraphicsItem.ItemIsMovable, True)
            self.addItem(self._current_rect_item)
            self._start = event.scenePos()
            r = QRectF(self._start, self._start)
            self._current_rect_item.setRect(r)
            self._have = True
        super(GraphicsScene, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._current_rect_item is not None:
            r = QRectF(self._start, event.scenePos()).normalized()
            self._current_rect_item.setRect(r)
        super(GraphicsScene, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._current_rect_item = None
        super(GraphicsScene, self).mouseReleaseEvent(event)

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        scene = GraphicsScene(self)
        view  = QGraphicsView(scene)
        self.setCentralWidget(view)

if __name__ == '__main__':
    application = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.setGeometry(100,100,600,600)
    main_window.show()
    sys.exit(application.exec_())