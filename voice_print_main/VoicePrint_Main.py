import librosa
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import os

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# 提取音频特征
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean


# 读取音频文件并提取特征
def load_audio_files(directory):
    features = []
    filenames = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        mfccs = extract_features(file_path)
        features.append(mfccs)
        filenames.append(filename)
    return np.array(features), filenames


# 聚类音频特征
def cluster_audio_features(features, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(features)
    return kmeans.labels_


def visualize_clusters(features, labels, method='pca'):
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, n_iter=300)
    else:
        raise ValueError("method should be 'pca' or 'tsne'")

    reduced_features = reducer.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        plt.scatter(reduced_features[labels == label, 0], reduced_features[labels == label, 1])

    plt.title(f'Output ({method.upper()})')
    # plt.xlabel('v 1')
    # plt.ylabel('v 2')
    plt.legend()
    plt.show()

# 主函数
def voice_print_class(path: str):
    # directory = path  # 替换为你的音频文件目录
    directory = '../voice_print_main/Test_Music'
    features, filenames = load_audio_files(directory)
    labels = cluster_audio_features(features, n_clusters=4)

    # 输出聚类结果
    with open("VoicePrintData.txt", "w", encoding="utf-8") as f:
        for filename, label in zip(filenames, labels):
            print(f"文件: {filename} -> 类别: {label}")
            f.writelines(f"{filename} -> {label}\n")

    visualize_clusters(features, labels, method='pca')  # 使用PCA进行可视化

if __name__ == "__main__":
    voice_print_class("")