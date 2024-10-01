import os
from pydub import AudioSegment

# 将MP3文件转换为WAV文件
def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    audio.export(wav_file_path, format="wav")

# 找出文件夹中的所有MP3文件并转换为WAV文件
def convert_all_mp3_to_wav(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".mp3"):
            mp3_file_path = os.path.join(directory, filename)
            wav_file_path = os.path.join(directory, os.path.splitext(filename)[0] + ".wav")
            convert_mp3_to_wav(mp3_file_path, wav_file_path)
            os.remove(mp3_file_path)
            print(f"已将 {mp3_file_path} 转换为 {wav_file_path}")

# 主函数
def main():
    directory = '../voice_print_main/Test_Music'  # 替换为你的音频文件目录
    convert_all_mp3_to_wav(directory)

if __name__ == "__main__":
    main()



