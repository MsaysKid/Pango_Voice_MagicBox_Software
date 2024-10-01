import os
from pydub import AudioSegment


def convert_m4a_to_wav(folder_path):
    # 确保文件夹路径存在
    if not os.path.isdir(folder_path):
        print(f"文件夹路径不存在: {folder_path}")
        return

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".m4a"):
            # 构建完整的文件路径
            m4a_file_path = os.path.join(folder_path, filename)
            wav_file_path = os.path.join(folder_path, os.path.splitext(filename)[0] + ".wav")

            # 加载 .m4a 文件
            audio = AudioSegment.from_file(m4a_file_path, format="m4a")

            # 导出为 .wav 文件
            audio.export(wav_file_path, format="wav")
            print(f"已转换: {m4a_file_path} -> {wav_file_path}")
            os.remove(m4a_file_path)


if __name__ == "__main__":
    # 指定文件夹路径
    folder_path = r"../Music/emotion/test/"
    convert_m4a_to_wav(folder_path)
