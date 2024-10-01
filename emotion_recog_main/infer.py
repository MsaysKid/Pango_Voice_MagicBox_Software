import argparse
import functools
import time

from mser.predict import MSERPredictor
from mser.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    '../emotion_recog_configs/bi_lstm.yml',   '配置文件')
add_arg('use_ms_model',     str,    'iic/emotion2vec_plus_base',  '使用ModelScope上公开Emotion2vec的模型')
add_arg('use_gpu',          bool,   True,                    '是否使用GPU预测')
add_arg('audio_path',       str,    '../emotion_recog_dataset/test.wav',      '音频路径')
add_arg('model_path',       str,    '../emotion_recog_models/BiLSTM_Emotion2Vec/best_model/',     '导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)

# 获取识别器
predictor = MSERPredictor(configs=args.configs,
                          use_ms_model=args.use_ms_model,
                          model_path=args.model_path,
                          use_gpu=args.use_gpu)

print("准备判别")
label, score = predictor.predict(audio_data=args.audio_path)
print(f'音频：{args.audio_path} 的预测结果标签为：{label}，得分：{score}')
time.sleep(0.5)
label, score = predictor.predict(audio_data=args.audio_path)
print(f'音频：{args.audio_path} 的预测结果标签为：{label}，得分：{score}')
time.sleep(0.5)
label, score = predictor.predict(audio_data=args.audio_path)
print(f'音频：{args.audio_path} 的预测结果标签为：{label}，得分：{score}')
time.sleep(0.5)
label, score = predictor.predict(audio_data=args.audio_path)
print(f'音频：{args.audio_path} 的预测结果标签为：{label}，得分：{score}')
