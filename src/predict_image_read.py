import cv2
from logging import getLogger
from multiprocessing import Process, Queue


class VideoReader:
    def __init__(self, src):
        self.logger = getLogger(__name__)
        self.src = src

        # 幅, 高さ, フレーム数, フレームレートの初期化
        self.width, self.height, self.frame_num, self.fps = None, None, None, None
        self.queue = None
        self.process = None

    def open(self):
        """
        openCVインスタンスの初期化
        """
        # マルチプロセスで、フレームを格納するキューを作成
        self.queue = Queue(maxsize=1)
        # マルチプロセスで、フレームを取得し続ける
        self.process = Process(target=self.get_frame, args=(self.src, self.queue))
        self.process.start()

    def read(self):
        """
        カメラから取得した映像に対して、推論を行う
        """
        while True:
            # キューからイメージを取得（この処理後にキューが空になる）
            bgr_img = self.queue.get()

            yield bgr_img

    def close(self):
        """
        処理を終了
        """
        self.process.terminate()
        self.queue.close()

    def __dell__(self):
        self.close()

    @staticmethod
    def get_frame(src, queue):
        """
        フレームを取得
        キューが空の場合、フレームをキューに入れ続ける
        Args:
            src: opencvで読み込むカメラソース（バイス番号、URL）
            queue:キュー
        """
        cap = cv2.VideoCapture(src)

        try:
            while True:
                # フレームを取得する
                _, frame = cap.read()

                # キューが空の場合、キューにフレームを入れる。
                if queue.empty():
                    queue.put(frame)
        finally:
            cap.release()


if __name__ == '__main__':
    source = 0
    c = VideoReader(source)
    c.open()
    try:
        for img in c.read():
            cv2.imshow('test', img)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        print('pushed Ctrl-C')
    finally:
        c.close()
    cv2.destroyAllWindows()
