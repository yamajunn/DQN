import cv2
import glob
import re

def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else 0

def create_video(image_folder, output_file, frame_duration):
    # 画像ファイルのリストを取得し、数値順にソート
    image_files = sorted(glob.glob(f"{image_folder}/*.png"), key=numerical_sort)
    if not image_files:
        raise ValueError("指定されたフォルダに画像が見つかりません")

    # 最初の画像を読み込み、動画の設定を決定
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    frame_rate = 1 / frame_duration  # フレームレートを表示時間から計算
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

    for image_file in image_files:
        img = cv2.imread(image_file)
        out.write(img)
        print(f"{image_file} を動画に追加しました")

    out.release()
    print("動画ファイルが作成されました")

if __name__ == "__main__":
    image_folder = "Datas/Pong"  # 画像フォルダのパスを指定
    output_file = "pong.mp4"  # 出力動画ファイル名
    frame_duration = 0.2  # 各画像の表示時間（秒）

    create_video(image_folder, output_file, frame_duration)
