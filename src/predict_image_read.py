import sys
import cv2
from logging import getLogger

from multiprocessing import Process, Queue

class CameraReader:
    def __init__(self, src):
        self.logger = getLogger(__name__)
        self.src = src

    def open(self):
        """
        openCVインスタンスの初期化
        """
        self.cap = cv2.VideoCapture(self.src)

        # マルチプロセスで、フレームを格納するキューを作成
        self.queue = Queue(maxsize=1)
        # マルチプロセスで、フレームを取得し続ける
        self.p = Process(target=get_frame, args=(self.cap, self.queue))
        self.p.start()

    def read(self):
        """
        カメラから取得した映像に対して、推論を行う
        """
        while True:
            try:
                # キューを取得（この処理後にキューが空になる）
                bgr_img = self.queue.get()

                return bgr_img

            # キーボードの「Ctrl-C」が押された場合、終了処理を行う
            except (KeyboardInterrupt):
                self.logger.warning('pushed Ctrl-C')
                self.close()
    
    def close(self):
        """
        処理を終了する
        """
        self.cap.release()
        self.queue.close()
        self.p.terminate()
    
    def __dell__(self):
        self.cap.release()
        self.queue.close()
        self.p.terminate()


def get_frame(cap, queue):
    """
    フレームを取得し、キューが空の場合、
    フレームをキューに入れ続ける。
    Args:
        cap:VideoCaptureオブジェクト
        queue:キュー
    """
    while True:
        # フレームを取得する
        _, frame = cap.read()

        # キューが空の場合、キューにフレームを入れる。
        if queue.empty():
            queue.put(frame)


if __name__ == '__main__':
    src = 'http://210.254.207.10:80/-wvhttp-01-/GetOneShot?image_size=640x480&frame_count=1000000000'
    c = CameraReader(src)
    c.open()
    try:
        while True:
            img = c.read()
            cv2.imshow('test', img)
            cv2.waitKey(0)
    except:
        c.close()
        print('ERROR')
    c.close()
    cv2.destroyAllWindows()

