# 变声检测
import os
import pickle

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_std = np.std(mfccs, axis=1)
    return np.hstack((mfccs_mean, mfccs_std))

# 检测变声
def detect_voice_transformation(audio_path, model):
    features = extract_features(audio_path)
    prediction = model.predict([features])
    return 'Transformed' if prediction[0] == 1 else 'Original'


if __name__ == "__main__":
    # # 假设你有两个类别的音频文件：原声和变声
    # original_audio_files = []
    # transformed_audio_files = []
    #
    # folder_path_1 = r'../voice_change_dataset/origin_half'
    # folder_path_2 = r'../voice_change_dataset/change'
    #
    # for filename in os.listdir(folder_path_1):
    #     file_path = os.path.join(folder_path_1, filename)
    #     original_audio_files.append(file_path)
    #
    # for filename in os.listdir(folder_path_2):
    #     file_path = os.path.join(folder_path_2, filename)
    #     transformed_audio_files.append(file_path)
    #
    #
    # # 提取特征和标签
    # X = []
    # y = []
    #
    # for file in original_audio_files:
    #     print(0)
    #     features = extract_features(file)
    #     X.append(features)
    #     y.append(0)  # 原声标签为0
    #
    # for file in transformed_audio_files:
    #     print(1)
    #     features = extract_features(file)
    #     X.append(features)
    #     y.append(1)  # 变声标签为1
    #
    # X = np.array(X)
    # y = np.array(y)
    #
    # # 划分训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    # # 训练随机森林分类器
    # clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # clf.fit(X_train, y_train)
    #
    # # 评估模型
    # y_pred = clf.predict(X_test)
    # print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    #
    # with open('random_forest_model.pkl', 'wb') as f:
    #     pickle.dump(clf, f)

    with open('random_forest_model.pkl', 'rb') as f:
        clf = pickle.load(f)

    # 使用训练好的模型检测新的音频
    new_audio_file = r'../voice_change_dataset/female2male.wav'
    result = detect_voice_transformation(new_audio_file, clf)
    print(f'The audio is: {result}')

    new_audio_file = r'../voice_change_dataset/male2female.wav'
    result = detect_voice_transformation(new_audio_file, clf)
    print(f'The audio is: {result}')

    new_audio_file = r'../voice_change_dataset/Male.wav'
    result = detect_voice_transformation(new_audio_file, clf)
    print(f'The audio is: {result}')

    new_audio_file = r'../voice_change_dataset/Female.wav'
    result = detect_voice_transformation(new_audio_file, clf)
    print(f'The audio is: {result}')