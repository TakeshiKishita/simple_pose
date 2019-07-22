import cv2
import time
from logging import getLogger
from multiprocessing import Process, Queue, Array


class VideoReader:
    def __init__(self, src):
        """
        初期化
        Args:
            src: 動画を読み取るソース
        """
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
        # フレーム情報を取得するための、共有メモリを確保する
        info_array = Array('f', 4)
        # マルチプロセスで、フレームを格納するキューを作成
        self.queue = Queue(maxsize=1)
        # マルチプロセスで、フレームを取得し続ける
        self.process = Process(target=self.get_frame, args=(self.src, self.queue, info_array))
        self.process.start()

        # 共有メモリに書き込まれるのを待ち、フレームの情報を取得する
        time.sleep(1)
        self.width, self.height, self.frame_num, self.fps = info_array

        self.logger.info(
            'width:{}, height:{}, frame_num:{}, fps:{}'.format(self.width, self.height, self.frame_num, self.fps))

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

    def get_frame(self, src, queue, info_array):
        """
        フレームを取得
        キューが空の場合、フレームをキューに入れ続ける
        Args:
            src: opencvで読み込むカメラソース（バイス番号、URL）
            queue:キュー
            info_array: フレーム情報を格納する、共有メモリ配列
        """
        # 動画読み込み
        cap = cv2.VideoCapture(src)

        # フレーム情報を、共有メモリ上に保持する
        info_array[0] = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        info_array[1] = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        info_array[2] = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        info_array[3] = cap.get(cv2.CAP_PROP_FPS)

        try:
            while True:
                # フレームを取得する
                ret, frame = cap.read()

                # フレーム取得に失敗した場合、エラーを出力し、処理を抜ける
                if not ret:
                    self.logger.error('Failed to cv2.VideoCapture.read()')
                    break

                # キューが空の場合、キューにフレームを入れる。
                if queue.empty():
                    queue.put(frame)
        finally:
            cap.release()


if __name__ == '__main__':
    """
    ディスプレイで確認
    """
    # infoログを表示させたいので、ログレベルを変更
    from logging import basicConfig, INFO
    basicConfig(level=INFO)

    # 入力を設定（デバイス番号、URL等）
    SOURCE = 0

    # インスタンスの初期化
    c = VideoReader(SOURCE)
    c.open()
    try:
        for frame in c.read():
            # 出力内容をディスプレイに表示
            cv2.imshow('test', frame)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        print('pushed Ctrl-C')
    finally:
        # 終了処理
        c.close()
        cv2.destroyAllWindows()
