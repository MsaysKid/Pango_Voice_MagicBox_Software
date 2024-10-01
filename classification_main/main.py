import argparse
import functools

from classification_macls.predict import MAClsPredictor
from classification_macls.utils.utils import add_arguments, print_arguments


def predict(audio_path):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('configs', str, 'classification_configs/cam++.yml', '配置文件')
    add_arg('use_gpu', bool, True, '是否使用GPU预测')
    add_arg('audio_path', str, audio_path, '音频路径')  # 'classification_dataset/7061-6-0-0.wav'
    add_arg('model_path', str, 'classification_models/20240704_48k/CAMPPlus_Fbank/best_model/', '导出的预测模型文件路径')
    args = parser.parse_args()
    # print_arguments(args=args)

    # 获取识别器
    predictor = MAClsPredictor(configs=args.configs,
                               model_path=args.model_path,
                               use_gpu=args.use_gpu)
    while True:
        label, score = predictor.predict(audio_data=args.audio_path)

        print(label + score)

    # print(f'音频：{args.audio_path} 的预测结果标签为：{label}，得分：{score}')