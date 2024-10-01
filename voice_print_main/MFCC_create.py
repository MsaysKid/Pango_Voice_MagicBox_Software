import os

import librosa.display
import matplotlib.pyplot as plt

test_list = [26, 70]

if __name__ == "__main__":
    # 加载音频文件
    file_dir = "/Test_Music"
    files = os.listdir(file_dir)
    files.sort(key=lambda x: int(x[:-4]))

    for i in files:
        file_path = '../voice_print_main/Test_Music/' + i  # 替换为你的音频文件路径
        # file_path = '../voice_print_main/1.m4a'  # 替换为你的音频文件路径
        # file_path = '../Class/Music_Test/Echo.wav'  # 替换为你的音频文件路径
        y, sr = librosa.load(file_path)

        # 计算MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # 绘制MFCC
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, x_axis='time')
        plt.colorbar()
        plt.title('voice_print_main')
        plt.tight_layout()

    plt.show()