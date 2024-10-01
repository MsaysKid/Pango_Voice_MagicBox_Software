import pickle
import shutil
import traceback
import wave
import argparse
import functools
import time

import os
from ctypes import cdll, byref, string_at, c_void_p, CFUNCTYPE, c_char_p, c_uint64, c_int64

import librosa
from loguru import logger

import numpy as np
import pyaudio
from PyQt5.QtChart import QChart, QSplineSeries, QLineSeries, QValueAxis, QDateTimeAxis
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from threading import Event, Thread
from scipy.signal import detrend, medfilt

from mser.predict import MSERPredictor
from mser.utils.utils import add_arguments, print_arguments

from AudioProcessGUI import AudioProcess
from AudioProcessGUI import GuiTool as GT

from Class.SerialControl import SerialControl
from Class.UdpControl import UdpControl

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import pyplot as plt

from classification_macls.predict import MAClsPredictor
from classification_macls.utils.utils import add_arguments, print_arguments
from voice_print_main.VoicePrint_Main import load_audio_files, cluster_audio_features, visualize_clusters

SER_PROCESS_FEMALETOMALE = bytearray([0x83, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01])  # 女变男
SER_PROCESS_NOISE = bytearray([0x83, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02])  # 去噪
SER_PROCESS_MALETOFEMALE = bytearray([0x83, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04])  # 男变女
SER_PROCESS_SEPARA_MUSIC = bytearray([0x83, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10])  # 分离音乐
SER_PROCESS_SEPARA_VOCAL = bytearray([0x83, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08])  # 分离人声
SER_PROCESS_ECHO = bytearray([0x83, 0x01, 0x00, 0x00, 0x00, 0x00, 0x03])  # 回声消除
SER_PROCESS_ECHO = bytearray([0x83, 0x01, 0x00, 0x00, 0x00, 0x00, 0x04])  # 回声消除

SER_LEFT_VOCAL_TRACT = bytearray([0x83, 0x01, 0x00, 0x00, 0x00, 0x00, 0x02])  # 左声道
SER_RIGHT_VOCAL_TRACT = bytearray([0x83, 0x01, 0x00, 0x00, 0x00, 0x00, 0x04])  # 右声道

SER_CLEAR_REGISTER_0 = bytearray([0x83, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])  # 寄存器0清零
SER_CLEAR_REGISTER_1 = bytearray([0x83, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00])  # 寄存器1清零

SER_PROCESS_BEGIN = bytearray([0x83, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01])  # 开始处理
SER_ORIGIN_BEGIN = bytearray([0x83, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])  # 原声播放

SER_OPEN_VOICE = bytearray([0x83, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00])  # 打开音响
SER_CLOSE_VOICE = bytearray([0x83, 0x02, 0x00, 0x00, 0x00, 0x00, 0x01])  # 关闭音响

SER_DISPLAY_SPECTRUM = bytearray([0x83, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00])  # 频谱图
SER_DISPLAY_SPECTROGRAM = bytearray([0x83, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01])  # 语谱图

PROCESS_AUDIOCLASS = 0
PROCESS_VOICEPRINTCLASS = 1
PROCESS_EMOTIONRECOG = 2
PROCESS_VOICECHANGETESTING = 3

INPUT_MODULE_SWITCH = 99
DISPLAY_SWITCH = 100
PLAY_MUSIC = 101
PROCESS_MUSIC = 102

CHUNK = 1024

MUSIC_TXT_CACHE_DIR = "./music_cache"
MUSIC_TXT_CACHE_LOCATION = MUSIC_TXT_CACHE_DIR + "/music.txt"
MUSIC_HEX_TXT_CACHE_LOCATION = MUSIC_TXT_CACHE_DIR + "/music_hex.txt"
MUSIC_WAV_MICRO_CACHE_LOCATION = MUSIC_TXT_CACHE_DIR + "/Micro.wav"

CLASS_CH = {'air_conditioner': "空调声",
            'car_horn': "汽车鸣笛声",
            'children_playing': "玩耍声",
            'dog_bark': "动物叫声",
            'drilling': "钻孔",
            'engine_idling': "引擎空转",
            'gun_shot': "枪声",
            'jackhammer': "手提钻",
            'siren': "警笛声",
            'street_music': "街道音乐"}

VOICE_WAKE_UP_DICT = {"qie1-huan4-shu1-ru4-mo2-shi4": INPUT_MODULE_SWITCH,
                      "qie1-huan4-pu3-tu2-xian3-shi4": DISPLAY_SWITCH,
                      "nan2-sheng1-bian4-nv3-sheng1": SER_PROCESS_MALETOFEMALE,
                      "nv3-sheng1-bian4-nan2-sheng1": SER_PROCESS_FEMALETOMALE,
                      "da3-kai1-zao4-sheng1-qu4-chu2": SER_PROCESS_NOISE,
                      "da3-kai1-hui2-sheng1-xiao1-chu2": SER_PROCESS_ECHO,
                      "bo1-fang4-yin1-pin2": PLAY_MUSIC,
                      "chu2-li3-yin1-pin2": PROCESS_MUSIC}


def mkdir(path, clear_sig=False):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
    elif clear_sig:
        shutil.rmtree(path)
        os.makedirs(path)


def write_data(fileName, txt):
    data_file = open(fileName, 'a+', encoding='utf-8')
    data_file.write("[" + GT.GetShortTimeString() + "]" + "\t" + txt + "\n")
    data_file.close()


def uin16_to_int16(input_uint16):
    x = input_uint16
    if x > 32767:
        return x - 65536
    else:
        return x


def hex2dec(input_file_path, output_file_path):
    with open(input_file_path, "r", encoding="utf-8") as input_file, \
            open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.seek(0)
        output_file.truncate()
        lines = input_file.readlines()
        for line in lines:
            line = line.strip().replace(" ", "")
            decimal_str = ""
            length = len(line)
            for i in range(0, length, 4):
                hex_str = line[i:i + 4]
                if len(hex_str) == 4:
                    # hex_str = hex_str[2:4] + hex_str[0:2]  # 高低位转换
                    input_uint16 = int(hex_str, 16)
                    decimal = uin16_to_int16(input_uint16)
                    decimal_str += str(decimal) + '\n'
            output_file.write(decimal_str.strip() + "\n")


def smooth_spikes(input_data, threshold=1000, window_size=10):  # 去除尖锐爆鸣声
    """
    平滑音频信号中的尖峰噪声。
    y: 音频信号数组
    sr: 采样率
    threshold: 用于检测尖峰的阈值
    window_size: 平滑窗口的大小
    """
    # 计算信号的绝对值
    y = input_data.astype(np.int32)
    abs_signal = np.abs(y)
    print(np.mean(abs_signal))
    threshold = 10 * np.mean(abs_signal)
    # 找到超过阈值的尖峰位置
    spike_indices = np.where(abs_signal > threshold)[0]
    y_no_threshold = y.copy()
    for index in spike_indices:
        # 计算平滑窗口的起始和结束位置
        start = max(0, index - window_size // 2)
        end = min(len(y), index + window_size // 2)
        # 将尖峰值替换为周围值的平均值
        y_no_threshold[spike_indices] = 0
        mm = y_no_threshold[start:end]
        y[index] = np.mean(y_no_threshold[start:end])

    return y


# FigureCanvas 对象
class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = plt.figure(figsize=(width, height), dpi=dpi, facecolor='#f0f0f0')  # facecolor 背景色
        self.ax = fig.add_subplot(111, projection='polar')
        self.ax.set_axis_off()
        self.ln, = self.ax.plot([], [])
        FigureCanvas.__init__(self, fig)


def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_std = np.std(mfccs, axis=1)
    return np.hstack((mfccs_mean, mfccs_std))


class MyWindow(QMainWindow, QWidget):
    draw_waveform_signal = pyqtSignal(bool)
    voice_wake_up_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = AudioProcess.Ui_MainWindow()
        self.ui.setupUi(self)

        # 自适应电脑界面大小
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.screenheight = self.screenRect.height()
        self.screenwidth = self.screenRect.width()
        self.height = int(self.screenheight * 0.8)
        self.width = int(self.screenwidth * 0.8)
        self.resize(self.width, self.height)
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)

        # 初始化缓存文件夹：
        mkdir(MUSIC_TXT_CACHE_DIR, clear_sig=True)

        # 界面名字
        self.setWindowTitle("音频智能处理系统")

        # 变声检测初始化
        self.init_voice_change_testing()

        # 神经网络初始化
        self.init_classiciation_single_predict()
        self.init_emotion_recognitions()

        self.__init_parameters()  # 参数初始化
        self.__font()  # 字体初始化
        self.__configs()  # 配置初始化
        self.__init()  # 界面控件初始化
        self.__connect()  # 槽信号连接初始化

        # 语音唤醒
        self.voice_wake_up_event = 0
        self.voice_wake_up_thread = Thread(target=self.ivw_wakeup)
        self.voice_wake_up_thread.start()

    def __init_parameters(self):
        self.process_func = SER_PROCESS_NOISE  # 音频处理默认为噪声处理
        self.SAVEFILE = False  # Log保存判别符，False代表不保存Log文件
        self.voice_intput_path = ""  # 音频文件存储路径
        self.voice_save_path = ""  # 音频文件保存路径

        self.serial_isOpen = False  # 串口打开状态，False表示串口未开启
        self.serial = SerialControl()  # 串口
        self.udp_isOpen = False  # Udp打开状态，False表示网口未开启
        self.udp = UdpControl()  # 网口

        self.class_thread = None  # 识别功能的线程
        self.play_thread = None  # 播放音频的线程
        self.udp_get_thread = None  # udp通信线程
        self.udp_get_state = True  # udp通信的状态，True代表开始，False代表结束
        self.voice_data = []
        self.micro_udp_get_thread = None  # 麦克风状态下udp通信线程
        self.class_cycle_thread = None  # 循环识别线程

        self.btn_play_process_voice_state = False  # 处理状态，当该标志位为False时，代表未处理音频；为True时，代表正在处理音频
        self.btn_play_left_vocal_origin_state = False  # 左声道播放状态，当该标志位为False时，代表未使用左声道。
        self.btn_play_right_vocal_origin_state = False  # 右声道播放状态，当该标志位为False时，代表未使用右声道。
        self.btn_play_voice_state = False  # 声音播放状态，当前标志位为False时，未播放音频
        self.btn_micro_switch_state = False  # 麦克风状态，False代表未开启麦克风

        self.ch = 1  # 默认单通道
        self.depth = 2  # 默认16bit
        self.sr = 48000  # 默认48k采样率

        self.numframes = 120  # 录音块大小
        self.class_cycle_time = 3.  # 循环分类检测间隔时间
        self.micro_data = []

        self.timex_start = 0
        self.timex = 0

        self.voice_wake_up_keyword = None  # 语音唤醒关键字

    def __font(self):  # 字体初始化
        font = QFont("宋体", pointSize=10, weight=1)
        self.ui.edtMissionLogShow.setFont(font)
        self.errorFormat = '<font color="red" size="2">{}<font>'
        self.normalFormat = '<font color="black" size="2">{}<font>'
        self.timeFormat = '<font color="blue" size="2">{}<font>'

    def __configs(self):
        try:
            setting = QSettings("./configs.ini", QSettings.IniFormat)
            # 创建日志文件
            self.ch = int(setting.value("VOICE/ch"))
            self.depth = int(setting.value("VOICE/depth"))
            self.sr = int(setting.value("VOICE/sr"))
            if int(setting.value("SAVE/save_file")) == 1:
                mkdir("./LogOut")
                file_name = "./LogOut/log_" + GT.GetLongTimeString() + ".txt"
                data_file = open(file_name, 'w', encoding='utf-8')
                data_file.close()
                self.data_file = file_name
                self.SAVEFILE = True
        except Exception as e:
            self.log_show("读取配置文件参数异常，原因：{}[{}]".format(repr(e), traceback.format_exc()), True)

    def __init(self):
        self.__timer_init()  # 定时器初始化
        self.__controls_init()  # 控件初始化
        self.__chart_init()  # 图表初始化

    def __timer_init(self):  # 定时器初始化
        self.timer_draw_waveform = QTimer()
        self.timer_draw_waveform_intensity = 10

    def __controls_init(self):  # 控件初始化
        # groupBox初始化
        self.ui.gbFuncSelect.setEnabled(False)
        self.ui.gbAudioControl.setEnabled(False)

        # PushButton按键情况自定义
        self.ui.btnPlayVoice.setEnabled(False)
        self.ui.btnProcessVoice.setEnabled(False)
        self.ui.btnPlayLeftVocalOrigin.setEnabled(False)
        self.ui.btnPlayRightVocalOrigin.setEnabled(False)
        self.ui.btnMicroSwitch.setEnabled(False)
        self.ui.btnSaveVoice.setEnabled(False)
        self.ui.btnPlayVocal.setEnabled(False)
        self.ui.btnPlayMusic.setEnabled(False)

        # lineEdit初始化
        self.ui.edtMissionLogShow.setReadOnly(True)  # 设置日志显示为只读模式
        self.ui.edtInputUdpPort.setValidator(QIntValidator())  # 仅允许输入整数
        self.ui.edtInputUdpBuffer.setValidator(QIntValidator())
        self.ui.edtInputUdpIp.setText("192.168.1.105")
        self.ui.edtInputUdpPort.setText("8080")
        self.ui.edtInputUdpBuffer.setText("1024")

        # spinBox初始化
        self.ui.spbSetClassTime.setValue(self.class_cycle_time)

    def __connect(self):  # 槽信号连接
        # PushButton信号槽连接
        self.ui.btnVocalSepara.clicked.connect(self.btn_process_func_select)  # 功能控制按钮状态选择
        self.ui.btnMaleToFemale.clicked.connect(self.btn_process_func_select)
        self.ui.btnFemaleToMale.clicked.connect(self.btn_process_func_select)
        self.ui.btnNoiseRemove.clicked.connect(self.btn_process_func_select)
        self.ui.btnEchoRemove.clicked.connect(self.btn_process_func_select)
        self.ui.btnAudioClassification.clicked.connect(self.btn_process_func_select)
        self.ui.btnPlayMusic.clicked.connect(self.btn_process_func_select)
        self.ui.btnPlayVocal.clicked.connect(self.btn_process_func_select)
        self.ui.btnEmotionRecog.clicked.connect(self.btn_process_func_select)
        self.ui.btnVoiceprintClass.clicked.connect(self.btn_process_func_select)
        self.ui.btnVoiceChangeTesting.clicked.connect(self.btn_process_func_select)
        self.ui.btnVoiceWakeUpSwitch.clicked.connect(self.btn_voice_wake_up_switch)  # 设置语音唤醒
        self.ui.btnDisplaySwitchFPGA.clicked.connect(self.btn_display_fpga_switch)

        self.ui.btnMicroSwitch.clicked.connect(self.micro_switch)  # 麦克风开关函数
        self.ui.btnFileInput.clicked.connect(self.file_input_output)  # 文件读取保存函数
        self.ui.btnSaveVoice.clicked.connect(self.file_input_output)  # 文件读取保存函数
        self.ui.btnProcessVoice.clicked.connect(self.process_voice)  # 音频处理播放函数
        self.ui.btnPlayVoice.clicked.connect(self.play_voice)  # 原声播放函数
        self.ui.btnPlayLeftVocalOrigin.clicked.connect(self.play_left_origin_voice)  # 左声道原声
        self.ui.btnPlayRightVocalOrigin.clicked.connect(self.play_right_origin_voice)  # 右声道原声

        self.ui.btnSerialConfirm.clicked.connect(self.serial_connect)  # 打开串口
        self.ui.btnUdpConfirm.clicked.connect(self.udp_connect)  # 打开网口

        self.ui.btnClearLog.clicked.connect(self.clear_log)  # 清除界面日志

        # RadioButton槽连接
        self.ui.rbtnMicroSelect.clicked.connect(self.voice_input_func_switch)
        self.ui.rbtnFileSelect.clicked.connect(self.voice_input_func_switch)

        # QTextEdit
        self.ui.edtMissionLogShow.textChanged.connect(lambda: self.ui.edtMissionLogShow.moveCursor(11))

        # 自定义信号
        self.draw_waveform_signal[bool].connect(self.start_draw_waveform)
        self.timer_draw_waveform.timeout.connect(self.draw_waveform)
        self.voice_wake_up_signal.connect(self.voice_wake_up_process_select)

    # 图表初始化
    def __chart_init(self):
        self.chart_waveform = QChart()
        self.ui.cvFigureProcessTime.setChart(self.chart_waveform)
        self.series_waveform = QSplineSeries()
        self.chart_waveform.addSeries(self.series_waveform)
        self.chart_waveform.legend().hide()

        self.axisX = QValueAxis()
        self.axisY = QValueAxis()

        self.axisX.setRange(0, 10)
        self.axisY.setRange(-1, 1)

        self.axisX.setTickCount(11)
        self.axisY.setTickCount(11)

        self.axisY.setTitleText("幅度")

        self.chart_waveform.addAxis(self.axisX, Qt.AlignBottom)
        self.chart_waveform.addAxis(self.axisY, Qt.AlignLeft)
        self.series_waveform.attachAxis(self.axisX)
        self.series_waveform.attachAxis(self.axisY)

    # 波形绘制开关
    def start_draw_waveform(self, state):
        if state:
            self.timer_draw_waveform.start(self.timer_draw_waveform_intensity)
        else:
            self.timer_draw_waveform.stop()

    # 绘制波形
    def draw_waveform(self):
        # print("timer open")
        timex = self.timex
        value = self.voice_point_data

        self.timex = 0

        if self.draw_waveform_min > value:
            self.draw_waveform_min = value
        if self.draw_waveform_max < value:
            self.draw_waveform_max = value

        if self.timex_start >= 10:
            self.axisX.setRange(self.timex_start - 10, self.timex_start)

        if self.series_waveform.count() >= 400 * 10:
            self.series_waveform.removePoints(0, self.series_waveform.count() - 400 * 10)

        # print(self.timex_start, value)
        self.series_waveform.append(self.timex_start, value)
        self.timex_start += timex

    # 绘图重置
    def draw_waveform_reset(self):
        self.series_waveform.removePoints(0, self.series_waveform.count())
        self.axisX.setRange(0, 10)
        self.axisY.setRange(-1, 1)
        self.series_waveform.clear()
        self.timex_start = 0
        self.draw_waveform_min = 0
        self.draw_waveform_max = 0

    # 绘制纵坐标范围
    def draw_waveform_chart_range(self, axis: QValueAxis, value_list: QLineSeries):
        axis.setRange(0.9 * self.draw_waveform_min if self.draw_waveform_min >= 0 else 1.1 * self.draw_waveform_min,
                      1.1 * self.draw_waveform_max)
        self.draw_waveform_range_thread = None

    def closeEvent(self, QCloseEvent=None):
        # 创建一个消息盒子（提示框）
        quitMsgBox = QMessageBox()
        # 设置提示框的标题
        quitMsgBox.setWindowTitle('提示')
        # 设置提示框的内容
        quitMsgBox.setText('确认退出音频处理系统吗？')
        # 设置按钮标准，一个yes一个no
        quitMsgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        # 获取两个按钮并且修改显示文本
        buttonY = quitMsgBox.button(QMessageBox.Yes)
        buttonY.setText('确定')
        buttonN = quitMsgBox.button(QMessageBox.No)
        buttonN.setText('取消')
        quitMsgBox.exec_()
        # 判断返回值，如果点击的是Yes按钮，我们就关闭组件和应用，否则就忽略关闭事件
        if quitMsgBox.clickedButton() == buttonY:
            self.voice_wake_up_event = 1
            self.voice_wake_up_thread.join()
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()

    # 串口连接
    def serial_connect(self):
        serial_port = self.ui.cmbSerialPortSelect.currentText()
        if not self.serial_isOpen:
            try:
                self.serial = SerialControl(portx=str(serial_port), bps=115200, timex=0.1)
                self.serial.OpenPort()
            except Exception as e:
                self.log_show("串口连接错误，请重试。" + "错误原因:" + repr(e), True)
            else:
                self.serial_isOpen = True
                self.log_show("串口连接成功！portx：{0}\tbps：{1}\ttimex:{2}".format(serial_port, 115200, 0.1))
                self.serial.WritePort(SER_CLEAR_REGISTER_0)  # 寄存器初始化
                self.serial.WritePort(SER_CLEAR_REGISTER_1)  # 寄存器初始化
                self.serial.WritePort(SER_OPEN_VOICE)
                self.ui.btnProcessVoice.setEnabled(True)
                # if self.udp_isOpen:
                #     self.ui.gbFuncSelect.setEnabled(True)
                #     self.ui.gbAudioControl.setEnabled(True)
                self.ui.gbFuncSelect.setEnabled(True)
                self.ui.gbAudioControl.setEnabled(True)
        else:
            self.log_show("串口已连接。")

    # FPGA寄存器初始化
    def serial_clear(self):  # 清除寄存器
        self.serial.WritePort(SER_CLEAR_REGISTER_0)
        self.serial.WritePort(SER_CLEAR_REGISTER_1)

    # UDP连接
    def udp_connect(self):
        if not self.udp_isOpen:
            try:
                ip = self.ui.edtInputUdpIp.text()
                port = int(self.ui.edtInputUdpPort.text())
                buffer_time = int(self.ui.edtInputUdpBuffer.text())
                self.udp = UdpControl(ip, port, buffer_time)
                self.udp.OpenUdp()
            except Exception as e:
                self.log_show("网口连接错误，请重试。" + "错误原因:" + repr(e), True)
            else:
                self.udp_isOpen = True
                self.log_show("网口连接成功！ip：{0}\tport：{1}\tbuffer time:{2}".format(ip, port, buffer_time))
                # if self.serial_isOpen:
                #     self.ui.gbFuncSelect.setEnabled(True)
                #     self.ui.gbAudioControl.setEnabled(True)
        else:
            self.log_show("网口已连接。")

    # 信息日志显示
    def log_show(self, information: str, error_state=False):  # 日志显示
        if error_state:
            self.ui.edtMissionLogShow.append(self.timeFormat.format(
                "[" + GT.GetShortDateString() + " " + GT.GetShortTimeString() + "]") + "\t" +
                                             self.errorFormat.format(information))
        else:
            self.ui.edtMissionLogShow.append(self.timeFormat.format(
                "[" + GT.GetShortDateString() + " " + GT.GetShortTimeString() + "]") + "\t" +
                                             self.normalFormat.format(information))
        if self.SAVEFILE:
            write_data(self.data_file, information)

    # 清楚日志
    def clear_log(self):
        self.ui.edtMissionLogShow.clear()

    # 按钮功能选择
    def btn_process_func_select(self):
        try:
            self.btn_process_reset()
            self.sender().setEnabled(False)
            if self.sender() == self.ui.btnAudioClassification:
                self.process_func = PROCESS_AUDIOCLASS  # 音频分类
                self.log_show("当前模式：音频分类")
                self.ui.btnProcessVoice.setEnabled(True)
                return None

            if self.sender() == self.ui.btnVoiceprintClass:  # 声纹分类
                self.process_func = PROCESS_VOICEPRINTCLASS
                self.log_show("当前模式：声纹分类")
                self.sender().setEnabled(True)

                self.voiceprint_input_path = \
                    QFileDialog.getExistingDirectory(self, "读取声纹分类文件夹", "./")
                try:
                    if self.voiceprint_input_path == "":
                        self.log_show("读取文件夹异常", True)
                    else:
                        self.log_show("已读取声纹分类文件夹：" + self.voiceprint_input_path)
                        self.ui.btnProcessVoice.setEnabled(True)
                except Exception as e:
                    self.log_show("读取文件夹异常，原因：{}[{}]".format(repr(e), traceback.format_exc()), True)

                return None

            if self.sender() == self.ui.btnEmotionRecog:  # 情感识别
                self.process_func = PROCESS_EMOTIONRECOG
                self.log_show("当前模式：情感识别")
                self.ui.btnProcessVoice.setEnabled(True)
                return None

            if self.sender() == self.ui.btnVoiceChangeTesting:  # 变声检测
                self.process_func = PROCESS_VOICECHANGETESTING
                self.log_show("当前模式，变声检测")
                self.ui.btnProcessVoice.setEnabled(True)
                return None

            if self.sender() == self.ui.btnVocalSepara:  # 人声分离
                self.log_show("当前模式：人声分离，请选择分离内容")
                self.ui.btnPlayVocal.setEnabled(True)
                self.ui.btnPlayMusic.setEnabled(True)

            if self.sender() == self.ui.btnPlayMusic:
                self.process_func = SER_PROCESS_SEPARA_MUSIC
                self.ui.btnVocalSepara.setEnabled(False)
                self.ui.btnPlayVocal.setEnabled(True)
                self.log_show("当前模式：分离音乐")
            elif self.sender() == self.ui.btnPlayVocal:
                self.process_func = SER_PROCESS_SEPARA_VOCAL
                self.ui.btnVocalSepara.setEnabled(False)
                self.ui.btnPlayMusic.setEnabled(True)
                self.log_show("当前模式：分离人声")
            elif self.sender() == self.ui.btnMaleToFemale:
                self.process_func = SER_PROCESS_MALETOFEMALE  # 男变女
                self.log_show("当前模式：男声变女声")
            elif self.sender() == self.ui.btnFemaleToMale:
                self.process_func = SER_PROCESS_FEMALETOMALE  # 女变男
                self.log_show("当前模式：女声变男声")
            elif self.sender() == self.ui.btnNoiseRemove:
                self.process_func = SER_PROCESS_NOISE  # 去噪
                self.log_show("当前模式：音频去噪")
            elif self.sender() == self.ui.btnEchoRemove:
                self.process_func = SER_PROCESS_ECHO  # 回声消除
                self.log_show("当前模式：回声消除")

            self.serial.WritePort(self.process_func)  # 写入串口
            self.ui.btnProcessVoice.setEnabled(True)
        except Exception as e:
            self.log_show("模式切换错误，请重试。" + "错误原因:" + repr(e), True)
            self.btn_process_reset()

    def btn_process_reset(self):  # 重置按键情况
        self.ui.btnVocalSepara.setEnabled(True)
        self.ui.btnMaleToFemale.setEnabled(True)
        self.ui.btnFemaleToMale.setEnabled(True)
        self.ui.btnNoiseRemove.setEnabled(True)
        self.ui.btnEchoRemove.setEnabled(True)
        self.ui.btnAudioClassification.setEnabled(True)
        self.ui.btnVoiceChangeTesting.setEnabled(True)
        self.ui.btnVoiceprintClass.setEnabled(True)
        self.ui.btnEmotionRecog.setEnabled(True)
        self.ui.btnPlayVocal.setEnabled(False)
        self.ui.btnPlayMusic.setEnabled(False)

    def voice_input_func_switch(self):
        try:
            self.btn_process_reset()
            self.ui.btnProcessVoice.setEnabled(False)
            self.serial_clear()

            self.ui.btnMicroSwitch.setEnabled(False)
            self.ui.btnFileInput.setEnabled(False)
            self.ui.btnPlayLeftVocalOrigin.setEnabled(False)
            self.ui.btnPlayRightVocalOrigin.setEnabled(False)
            self.ui.btnPlayVoice.setEnabled(False)
            self.ui.btnProcessVoice.setEnabled(False)
            if self.ui.rbtnMicroSelect.isChecked():
                self.serial.WritePort(SER_CLOSE_VOICE)  # 关闭音响
                self.ui.btnMicroSwitch.setEnabled(True)
                self.ui.btnFileInput.setEnabled(False)
                self.log_show("当前输入模式切换为：麦克风")
            elif self.ui.rbtnFileSelect.isChecked():
                self.serial.WritePort(SER_CLOSE_VOICE)  # 打开音响
                self.ui.btnMicroSwitch.setEnabled(False)
                self.ui.btnFileInput.setEnabled(True)
                self.log_show("当前输入模式切换为：文件输入")
                self.voice_intput_path = ''
        except Exception as e:
            self.log_show("输入切换异常，原因：{}[{}]".format(repr(e), traceback.format_exc()), True)
        else:
            pass

    def save_music_from_txt(self, path):
        data_list = []
        with open(MUSIC_TXT_CACHE_LOCATION, "r", encoding="utf-8") as f:  # 打开音频TXT缓存文件
            data = f.readline()
            while data:
                data_list.append(int(data.strip()))
                data = f.readline()
        f.close()
        data_array = np.array(data_list, dtype=np.int16)

        data_array = smooth_spikes(data_array, 1000, 10)  # 消除爆鸣
        data_array = medfilt(data_array, kernel_size=11)  # 中值滤波

        with wave.open(path, "w") as wave_file:
            output_data = data_array.astype(np.int16)
            wave_file.setparams((self.ch, self.depth, self.sr, 0, "NONE", "not compressed"))
            wave_file.writeframes(output_data.tobytes())

    def file_input_output(self):
        if self.sender() == self.ui.btnFileInput:  # 读取文件
            self.voice_intput_path = \
                QFileDialog.getOpenFileName(self, "读取音频文件", "./",
                                            "WAV Files (*.wav);;M4A Files (*.m4a);;All files (*)")[0]
            try:
                f = open(self.voice_intput_path)
            except Exception as e:
                self.log_show("读取音频文件异常，原因：{}[{}]".format(repr(e), traceback.format_exc()), True)
            else:
                self.log_show("已读取音频文件：" + self.voice_intput_path)
                self.ui.btnPlayVoice.setEnabled(True)
                self.ui.btnPlayLeftVocalOrigin.setEnabled(True)
                self.ui.btnPlayRightVocalOrigin.setEnabled(True)
        elif self.sender() == self.ui.btnSaveVoice:  # 保存文件
            self.voice_save_path = QFileDialog.getSaveFileName(self, "保存音频文件", "./", "WAV Files (*.wav)")[0]
            try:
                self.save_music_from_txt(self.voice_save_path)
            except Exception as e:
                self.log_show("保存失败，请重新保存，原因：{}".format(repr(e), traceback.format_exc()), True)
            else:
                self.log_show("已保存音频文件，路径：" + self.voice_save_path)

    def init_classiciation_single_predict(self):  # 单个音频识别初始化
        parser = argparse.ArgumentParser(description=__doc__)
        add_arg = functools.partial(add_arguments, argparser=parser)
        add_arg('configs', str, 'classification_configs/cam++.yml', '配置文件')
        add_arg('use_gpu', bool, True, '是否使用GPU预测')
        add_arg('model_path', str, 'classification_models/20240704_48k/CAMPPlus_Fbank/best_model/',
                '导出的预测模型文件路径')
        args = parser.parse_args()
        # print_arguments(args=args)

        # 获取识别器
        self.classiciation_predictor = MAClsPredictor(configs=args.configs,
                                                      model_path=args.model_path,
                                                      use_gpu=args.use_gpu)

    def classiciation_single_predict(self, audio_path: str, event: Event):  # 单个音频识别
        try:
            label, score = self.classiciation_predictor.predict(audio_data=audio_path)
            self.log_show("当前音频文件分类类别为：{}，得分{}。".format(CLASS_CH[label], score))
            if self.ui.rbtnMicroSelect.isChecked():
                self.log_show("音频分类任务结束，请重新选择功能")
        except Exception as e:
            self.log_show("音频分类失败，原因：{}".format(repr(e), traceback.format_exc()), True)

    def init_emotion_recognitions(self):  # 情感识别初始化
        parser = argparse.ArgumentParser(description=__doc__)
        add_arg = functools.partial(add_arguments, argparser=parser)
        add_arg('configs', str, 'emotion_recog_configs/bi_lstm.yml', '配置文件')
        add_arg('use_ms_model', str, 'iic/emotion2vec_plus_base', '使用ModelScope上公开Emotion2vec的模型')
        add_arg('use_gpu', bool, True, '是否使用GPU预测')
        add_arg('audio_path', str, 'emotion_recog_dataset/test.wav', '音频路径')
        add_arg('model_path', str, 'emotion_recog_models/BiLSTM_Emotion2Vec/best_model/', '导出的预测模型文件路径')
        args = parser.parse_args()
        print_arguments(args=args)

        # 获取识别器
        self.emotion_recog_predictor = MSERPredictor(configs=args.configs,
                                                     use_ms_model=args.use_ms_model,
                                                     model_path=args.model_path,
                                                     use_gpu=args.use_gpu)

    def emotion_recognitions(self, audio_path: str, event: Event):  # 情感识别
        try:
            label, score = self.emotion_recog_predictor.predict(audio_data=audio_path)
            self.log_show("当前音频文件分类类别为：{}。".format(label))
            self.ui.btnProcessVoice.setText("处理音频")
            self.log_show("结束音频分类任务")
        except Exception as e:
            self.log_show("情感识别失败，原因：{}".format(repr(e), traceback.format_exc()), True)

    def voice_print_class(self, audio_path: str, event: Event):
        directory = audio_path  # 替换为你的音频文件目录
        # directory = '../voice_print_main/Test_Music'
        self.features, filenames = load_audio_files(directory)
        self.labels = cluster_audio_features(self.features, n_clusters=4)

        # 输出聚类结果
        # with open("VoicePrintData.txt", "w", encoding="utf-8") as f:
        for filename, label in zip(filenames, self.labels):
            time.sleep(0.1)
            if event.isSet():
                return None
            self.log_show(f"文件: {filename} -> 类别: {label}")
            # f.writelines(f"{filename} -> {label}\n")

        # visualize_clusters(self.features, self.labels, method='pca')  # 使用PCA进行可视化

        self.ui.btnProcessVoice.setText("生成分类图")
        # self.log_show("结束声纹分类任务")

    def process_voice(self):
        try:
            if self.process_func == PROCESS_AUDIOCLASS:  # 当前为声音识别模式
                if not self.btn_play_process_voice_state:
                    self.ui.btnProcessVoice.setText("停止处理")
                    self.log_show("开始音频分类任务")
                    if self.ui.rbtnFileSelect.isChecked():
                        self.class_thread_event = Event()
                        self.class_thread = Thread(target=self.classiciation_single_predict,
                                                   args=(self.voice_intput_path, self.class_thread_event))
                        self.btn_play_process_voice_state = True
                        self.class_thread.start()
                    elif self.ui.rbtnMicroSelect.isChecked():
                        self.class_cycle_thread_event = Event()
                        self.class_cycle_time = self.ui.spbSetClassTime.value()
                        self.class_cycle_thread = Thread(target=self.classiciation_cycle_predict,
                                                         args=(self.class_cycle_thread_event,
                                                               self.sr,
                                                               self.numframes,
                                                               self.class_cycle_time))
                        self.btn_play_process_voice_state = True
                        self.class_cycle_thread.start()
                else:
                    self.ui.btnProcessVoice.setText("处理音频")
                    self.log_show("结束音频分类任务")
                    if self.ui.rbtnFileSelect.isChecked():
                        self.btn_play_process_voice_state = False
                        self.class_thread_event.set()
                        self.class_thread.join()
                        self.class_thread = None
                    elif self.ui.rbtnMicroSelect.isChecked():
                        self.btn_play_process_voice_state = False
                        self.class_cycle_thread_event.set()
                        self.class_cycle_thread.join()
                        self.class_cycle_thread = None
                    self.btn_process_reset()  # 重置功能选择

            elif self.process_func == PROCESS_EMOTIONRECOG:  # 当前为情感识别模式
                if not self.btn_play_process_voice_state:
                    if self.ui.rbtnFileSelect.isChecked():
                        self.ui.btnProcessVoice.setText("停止处理")
                        self.log_show("开始情感识别任务")
                        self.emotion_recog_event = Event()
                        self.emotion_recog_thread = Thread(target=self.emotion_recognitions,
                                                           args=(self.voice_intput_path,
                                                                 self.emotion_recog_event))
                        self.btn_play_process_voice_state = True
                        self.emotion_recog_thread.start()
                    elif self.ui.rbtnMicroSelect.isChecked():
                        if self.ui.btnMicroSwitch.text() == "打开麦克风" and os.path.exists(
                                MUSIC_WAV_MICRO_CACHE_LOCATION):
                            self.ui.btnProcessVoice.setText("停止处理")
                            self.log_show("开始情感识别任务")
                            self.emotion_recog_event = Event()
                            self.emotion_recog_thread = Thread(target=self.emotion_recognitions,
                                                               args=(MUSIC_WAV_MICRO_CACHE_LOCATION,
                                                                     self.emotion_recog_event))
                            self.btn_play_process_voice_state = True
                            self.emotion_recog_thread.start()
                        elif self.ui.btnMicroSwitch.text() == "关闭麦克风":
                            self.log_show("请等待录音完毕")
                else:
                    if self.ui.rbtnFileSelect.isChecked():
                        self.btn_play_process_voice_state = False
                        self.emotion_recog_event.set()
                        self.emotion_recog_thread.join()
                        self.emotion_recog_thread = None
                    elif self.ui.rbtnMicroSelect.isChecked():
                        self.btn_play_process_voice_state = False
                        self.emotion_recog_event.set()
                        self.emotion_recog_thread.join()
                        self.emotion_recog_thread = None
                    self.btn_process_reset()  # 重置功能选择

            elif self.process_func == PROCESS_VOICEPRINTCLASS:  # 声纹分类
                if self.ui.btnProcessVoice.text() == "生成分类图":
                    visualize_clusters(self.features, self.labels, method='pca')  # 使用PCA进行可视化
                    self.ui.btnProcessVoice.setText("处理音频")
                    self.log_show("结束声纹分类任务")
                    self.btn_play_process_voice_state = False
                    self.voiceprint_class_event.set()
                    self.voiceprint_class_thread.join()
                    self.voiceprint_class_thread = None

                    self.btn_process_reset()  # 重置功能选择
                    return None

                if not self.btn_play_process_voice_state:
                    self.ui.btnProcessVoice.setText("停止处理")
                    self.log_show("正在对声纹进行分类任务")
                    self.voiceprint_class_event = Event()
                    self.voiceprint_class_thread = Thread(target=self.voice_print_class,
                                                          args=(self.voiceprint_input_path,
                                                                self.voiceprint_class_event))
                    self.btn_play_process_voice_state = True
                    self.voiceprint_class_thread.start()
                else:
                    self.ui.btnProcessVoice.setText("处理音频")
                    self.log_show("结束声纹分类任务")
                    self.btn_play_process_voice_state = False
                    self.voiceprint_class_event.set()
                    self.voiceprint_class_thread.join()
                    self.voiceprint_class_thread = None

                    self.btn_process_reset()  # 重置功能选择
            elif self.process_func == PROCESS_VOICECHANGETESTING:  # 变声检测模式
                result = self.detect_voice_transformation()
                self.log_show("变声检测结果：" + result)

            elif self.process_func == SER_PROCESS_ECHO:  # 回声消除模式
                if not self.btn_play_process_voice_state:
                    self.ui.btnProcessVoice.setText("停止处理")
                    self.log_show("音频处理已启动，请点击播放音频")
                    self.btn_play_process_voice_state = True
                else:
                    self.ui.btnProcessVoice.setText("处理音频")
                    self.log_show("音频处理结束，请重新选择功能")
                    self.btn_play_process_voice_state = False
                    self.serial_clear()
                    self.btn_process_reset()  # 重置功能选择
            else:  # 其他模式
                if not self.btn_play_process_voice_state:
                    self.ui.btnProcessVoice.setText("停止处理")
                    self.log_show("音频处理已启动，请点击播放音频")
                    self.btn_play_process_voice_state = True
                    self.serial.WritePort(SER_PROCESS_BEGIN)
                else:
                    self.ui.btnProcessVoice.setText("处理音频")
                    self.log_show("音频处理结束，请重新选择功能")
                    self.btn_play_process_voice_state = False
                    self.serial_clear()
                    self.btn_process_reset()  # 重置功能选择
        except Exception as e:
            self.log_show("处理失败：原因：{}".format(repr(e)), True)

    def voice_play(self, path, event: Event):
        wf = wave.open(path, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                        rate=wf.getframerate(), output=True)
        data = wf.readframes(CHUNK)  # 读取数据
        print(data)
        while data != b'':  # 播放
            if event.isSet():
                break
            stream.write(data)
            data = wf.readframes(CHUNK)
        stream.stop_stream()  # 停止数据流
        stream.close()
        p.terminate()  # 关闭 PyAudio

        self.log_show("音频播放结束")
        self.ui.btnPlayVoice.setText("播放音频")

        # udp通信控制
        try:
            if self.ui.rbtnMicroSelect.isChecked():
                self.serial.WritePort(SER_CLOSE_VOICE)

            if self.ui.rbtnFileSelect.isChecked() and self.udp_isOpen:
                self.udp_thread_event.set()  # udp停止通信
                self.udp_get_thread.join()
                self.udp_get_thread = None
        except Exception as e:
            print(e)

        self.serial.WritePort(SER_CLOSE_VOICE)
        self.btn_play_voice_state = False
        self.play_thread = None

    def play_voice_udp_get(self, event: Event):
        """
        启动服务端UDP Socket
        :return:
        """
        # 不断循环，接受客户端发来的消息
        try:
            new_text = ''
            self.voice_point_data = 0
            self.voice_data = []
            s = time.perf_counter()
            self.draw_waveform_reset()
            state = True
            self.timex = 0
            with open(MUSIC_HEX_TXT_CACHE_LOCATION, 'a') as f:
                f.seek(0)
                f.truncate()
                while True:
                    if event.isSet():
                        break

                    data = self.udp.ReadData()

                    data_str = data.strip().replace(" ", "")
                    data_dec = data_str[0:4]

                    if len(data_dec) == 4:
                        decimal = np.array(int(data_dec, 16), dtype=np.uint16)
                        decimal = np.array(decimal, dtype=np.int16).astype(np.float32)

                    self.timex = self.timex + float(time.perf_counter() - s)
                    s = time.perf_counter()
                    self.voice_point_data = decimal * 1. / 2 ** 15

                    if state:
                        self.draw_waveform_signal.emit(state)
                        state = False

                    new_text = new_text + data

                f.write(new_text)
                f.close()

            hex2dec(MUSIC_HEX_TXT_CACHE_LOCATION, MUSIC_TXT_CACHE_LOCATION)  # 将十六进制转换为十进制
            self.draw_waveform_signal.emit(state)
            self.draw_waveform_range_thread.start()
            self.log_show("UDP通信结束")
            self.ui.btnSaveVoice.setEnabled(True)
            # self.udp.CloseUdp() # 关闭连接
        except Exception as e:
            self.log_show("通信失败，原因：{}".format(repr(e)), True)

    def play_voice(self):
        self.serial.WritePort(SER_OPEN_VOICE)
        try:
            if self.ui.rbtnFileSelect.isChecked():
                if not self.btn_play_voice_state:
                    self.ui.btnPlayVoice.setText("停止播放")
                    self.log_show("开始播放音频，UDP通信开始")
                    # 播放音频的线程
                    self.play_thread_event = Event()
                    self.play_thread = Thread(target=self.voice_play,
                                              args=(self.voice_intput_path, self.play_thread_event))
                    # udp通信的线程
                    if self.udp_isOpen:
                        self.udp_thread_event = Event()
                        self.udp_get_thread = Thread(target=self.play_voice_udp_get, args=(self.udp_thread_event,))
                        # 图标范围重设置线程
                        self.draw_waveform_range_thread_event = Event()  # 停止动作
                        self.draw_waveform_range_thread = Thread(target=self.draw_waveform_chart_range,
                                                                 args=(self.axisY, self.series_waveform))
                        # 开启线程
                        # self.play_thread.start()
                        self.udp_get_thread.start()

                    self.play_thread.start()
                    self.btn_play_voice_state = True
                else:
                    self.play_thread_event.set()
                    self.play_thread.join()
                    self.play_thread = None
                    self.btn_play_voice_state = False
            elif self.ui.rbtnMicroSelect.isChecked():
                if self.udp_isOpen:
                    if not self.btn_play_voice_state:
                        # 播放音频的线程
                        self.play_thread_event = Event()
                        self.play_thread = Thread(target=self.voice_play,
                                                  args=(MUSIC_WAV_MICRO_CACHE_LOCATION, self.play_thread_event))
                        self.play_thread.start()
                        self.btn_play_voice_state = True
                        self.ui.btnPlayVoice.setText("停止播放")
                        if self.process_func == PROCESS_AUDIOCLASS:
                            self.class_thread_event = Event()
                            self.class_thread = Thread(target=self.classiciation_single_predict,
                                                       args=(MUSIC_WAV_MICRO_CACHE_LOCATION, self.class_thread_event))
                            self.class_thread.start()
                    else:
                        self.play_thread_event.set()
                        self.play_thread.join()
                        self.play_thread = None
                        self.btn_play_voice_state = False
                        if self.process_func == PROCESS_AUDIOCLASS:
                            self.class_thread_event.set()
                            self.class_thread.join()
                            self.class_thread = None
        except Exception as e:
            self.log_show("音频播放器启动错误：原因：{}".format(repr(e)), True)

    def play_left_origin_voice(self):  # 左声道原声
        try:
            if not self.btn_play_left_vocal_origin_state:
                self.serial.WritePort(SER_LEFT_VOCAL_TRACT)  # 开启左声道
                self.ui.btnPlayRightVocalOrigin.setEnabled(False)  # 右声道按钮关闭使能
                self.btn_play_left_vocal_origin_state = True
                self.ui.btnPlayLeftVocalOrigin.setText("停止播放")
                self.log_show("左声道已打开，请点击播放音频")
            else:
                self.serial_clear()
                self.ui.btnPlayRightVocalOrigin.setEnabled(True)  # 右声道按钮使能
                self.btn_play_left_vocal_origin_state = False
                self.ui.btnPlayLeftVocalOrigin.setText("左声道原声")
                self.log_show("左声道音频已经关闭")
        except Exception as e:
            self.log_show("左声道切换错误：原因：{}".format(repr(e)), True)

    def play_right_origin_voice(self):  # 右声道原声
        try:
            if not self.btn_play_right_vocal_origin_state:
                self.serial.WritePort(SER_RIGHT_VOCAL_TRACT)  # 开启右声道
                self.ui.btnPlayRightVocalOrigin.setText("停止播放")
                self.log_show("右声道已打开，请点击播放音频")
                self.ui.btnPlayLeftVocalOrigin.setEnabled(False)  # 左声道按钮关闭使能
                self.btn_play_right_vocal_origin_state = True
            else:
                self.serial_clear()
                self.ui.btnPlayRightVocalOrigin.setText("右声道原声")
                self.log_show("右声道音频已经关闭")
                self.ui.btnPlayLeftVocalOrigin.setEnabled(True)  # 左声道按钮使能
                self.btn_play_right_vocal_origin_state = False
        except Exception as e:
            self.log_show("右声道切换错误：原因：{}".format(repr(e)), True)

    def micro_udp_get(self, event: Event):
        """
        启动服务端UDP Socket
        :return:
        """
        try:
            # 不断循环，接受客户端发来的消息
            self.log_show("UDP通信开始")
            self.micro_data = []
            save_text = ''
            self.voice_point_data = 0
            self.draw_waveform_reset()
            s = time.perf_counter()
            self.timex = 0
            state = True
            with open(MUSIC_HEX_TXT_CACHE_LOCATION, 'a') as f:
                f.seek(0)
                f.truncate()
                while True:
                    if event.isSet():
                        f.write(save_text)
                        f.close()
                        break

                    data = self.udp.ReadData()

                    data_str = data.strip().replace(" ", "")
                    length = len(data_str)
                    data_out = np.array([], dtype=np.float32)
                    for i in range(0, length, 4):
                        data_dec = data_str[i:i + 4]
                        if i == 0:
                            decimal = np.array(int(data_dec, 16), dtype=np.uint16)
                            decimal = np.array(decimal, dtype=np.int16).astype(np.float32)
                            self.timex = self.timex + float(time.perf_counter() - s)
                            s = time.perf_counter()
                            self.voice_point_data = decimal * 1. / 2 ** 15

                            if state:
                                self.draw_waveform_signal.emit(state)
                                state = False

                        if len(data_dec) == 4:
                            decimal = np.array(int(data_dec, 16), dtype=np.uint16)
                            decimal = np.array(decimal, dtype=np.int16).astype(np.float32)
                            data_out = np.append(data_out, decimal)

                    save_text = save_text + data
                    data_out = data_out[:, np.newaxis]
                    self.micro_data.append(data_out)

                    if len(self.micro_data) >= 400 * 300:
                        del self.micro_data[:len(self.micro_data)]

                # f.write(save_text)
                # f.close()

            hex2dec(MUSIC_HEX_TXT_CACHE_LOCATION, MUSIC_TXT_CACHE_LOCATION)  # 将十六进制转换为十进制
            self.draw_waveform_signal.emit(state)
            self.draw_waveform_range_thread.start()
            self.log_show("UDP通信结束")
            self.save_music_from_txt(MUSIC_WAV_MICRO_CACHE_LOCATION)
            # self.udp.CloseUdp() # 关闭连接
        except Exception as e:
            self.log_show("通信失败，原因：{}".format(repr(e)), True)

    def classiciation_cycle_predict(self, event: Event, sr: int = 48000, numframes: int = 1024,
                                    record_seconds: int = 3):
        parser = argparse.ArgumentParser(description=__doc__)
        add_arg = functools.partial(add_arguments, argparser=parser)
        add_arg('configs', str, 'classification_configs/cam++.yml', '配置文件')
        add_arg('use_gpu', bool, True, '是否使用GPU预测')
        add_arg('record_seconds', float, 3, '录音长度')
        add_arg('model_path', str, 'classification_models/20240704_48k/CAMPPlus_Fbank/best_model/',
                '导出的预测模型文件路径')
        args = parser.parse_args()

        # 获取识别器
        predictor = MAClsPredictor(configs=args.configs,
                                   model_path=args.model_path,
                                   use_gpu=args.use_gpu)

        # 模型输入长度
        infer_len = int(sr * record_seconds / numframes)

        s_test = time.perf_counter()
        while True:
            try:
                if event.isSet():
                    break

                print(len(self.micro_data), infer_len)

                if len(self.micro_data) < infer_len:
                    continue

                # 截取最新的音频数据
                seg_data = self.micro_data[-infer_len:]
                d = np.concatenate(seg_data)  # 归一化

                # 去噪
                d_NoNoise = d.flatten()
                d_NoNoise = smooth_spikes(d_NoNoise, 1000, 10)  # 消除爆鸣
                d_NoNoise = medfilt(d_NoNoise, kernel_size=11)  # 中值滤波
                # d_NoNoise = nr.reduce_noise(d_NoNoise, sr=self.sr)
                d = d_NoNoise[:, np.newaxis]

                # 删除旧的音频数据
                del self.micro_data[:len(self.micro_data) - infer_len]
                label, score = predictor.predict(audio_data=d, sample_rate=sr)
                if float(time.perf_counter() - s_test) >= 0.5 and score > 0.95 and self.btn_micro_switch_state:
                    self.log_show(f'预测结果标签为：{CLASS_CH[label]}，得分：{score}')
                    s_test = time.perf_counter()
            except Exception as e:
                pass
            else:
                pass

    def micro_switch(self):
        try:
            if not self.btn_micro_switch_state:
                if self.udp_isOpen:
                    self.micro_udp_get_thread_event = Event()
                    self.draw_waveform_range_thread_event = Event()  # 停止动作
                    self.draw_waveform_range_thread = Thread(target=self.draw_waveform_chart_range,
                                                             args=(self.axisY, self.series_waveform))
                    self.micro_udp_get_thread = Thread(target=self.micro_udp_get, args=(self.micro_udp_get_thread_event,))
                    self.micro_udp_get_thread.start()
                self.serial.WritePort(SER_OPEN_VOICE)
                self.ui.btnMicroSwitch.setText("关闭麦克风")
                self.btn_micro_switch_state = True
                self.log_show("已开启麦克风")
            else:
                if self.udp_isOpen:
                    self.micro_udp_get_thread_event.set()
                    self.micro_udp_get_thread.join()
                    self.micro_udp_get_thread = None
                self.serial.WritePort(SER_CLOSE_VOICE)
                self.ui.btnMicroSwitch.setText("打开麦克风")
                self.btn_micro_switch_state = False
                self.ui.btnPlayVoice.setEnabled(True)
                self.ui.btnSaveVoice.setEnabled(True)
                self.log_show("已关闭麦克风")
        except Exception as e:
            self.log_show("麦克风开启错误：原因：{}".format(repr(e)), True)

    def py_ivw_callback(self, sessionID, msg, param1, param2, info, userDate, id, keyword):


        x = eval(keyword.decode('utf-8'))
        self.voice_wake_up_keyword = x['keyword']
        self.voice_wake_up_signal.emit()
        print(x['keyword'])

    def voice_wake_up_process_select(self):
        try:
            process = VOICE_WAKE_UP_DICT[self.voice_wake_up_keyword]
            if process == INPUT_MODULE_SWITCH:
                if self.ui.rbtnMicroSelect.isChecked():
                    self.ui.rbtnFileSelect.setChecked(True)
                elif self.ui.rbtnFileSelect.isChecked():
                    self.ui.rbtnMicroSelect.setChecked(True)

                self.voice_input_func_switch()
                return None

            if process == DISPLAY_SWITCH:
                self.btn_display_fpga_switch()
                return None

            if process == PLAY_MUSIC:
                self.play_voice()
                return None

            if process == PROCESS_MUSIC:
                self.process_voice()
                return None

            self.process_func = process
            self.btn_process_reset()

            self.serial.WritePort(self.process_func)  # 写入串口
            self.ui.btnProcessVoice.setEnabled(True)

            if process == SER_PROCESS_MALETOFEMALE:
                self.log_show("当前模式：男声变女声")
                self.ui.btnMaleToFemale.setEnabled(False)
            elif process == SER_PROCESS_FEMALETOMALE:
                self.log_show("当前模式：女声变男声")
                self.ui.btnFemaleToMale.setEnabled(False)
            elif process == SER_PROCESS_NOISE:
                self.log_show("当前模式：音频去噪")
                self.ui.btnNoiseRemove.setEnabled(False)
            elif process == SER_PROCESS_ECHO:
                self.log_show("当前模式：回声消除")
                self.ui.btnEchoRemove.setEnabled(False)

        except Exception as e:
            self.log_show("模式切换错误，请重试。" + "错误原因:" + repr(e), True)
            self.btn_process_reset()

    def ivw_wakeup(self):
        try:
            msc_load_library = r'.\bin\msc_x64.dll'
            app_id = '8b978555'  # 填写自己的app_id
            ivw_threshold = '0:1450'
            jet_path = os.getcwd() + r'.\bin\msc\res\ivw\wakeupresource.jet'
            work_dir = 'fo|' + jet_path

            CALLBACKFUNC = CFUNCTYPE(None, c_char_p, c_uint64,
                                     c_uint64, c_uint64, c_void_p, c_void_p, c_uint64, c_char_p)
            pCallbackFunc = CALLBACKFUNC(self.py_ivw_callback)
        except Exception as e:
            return e

        # ret 成功码
        MSP_SUCCESS = 0

        dll = cdll.LoadLibrary(msc_load_library)
        errorCode = c_int64()
        sessionID = c_void_p()
        # MSPLogin
        Login_params = "appid={},engine_start=ivw".format(app_id)
        Login_params = bytes(Login_params, encoding="utf8")
        ret = dll.MSPLogin(None, None, Login_params)
        if MSP_SUCCESS != ret:
            logger.info("MSPLogin failed, error code is: {}".format(ret))
            return

        # QIVWSessionBegin
        Begin_params = "sst=wakeup,ivw_threshold={},ivw_res_path={}".format(
            ivw_threshold, work_dir)
        Begin_params = bytes(Begin_params, encoding="utf8")
        dll.QIVWSessionBegin.restype = c_char_p
        sessionID = dll.QIVWSessionBegin(None, Begin_params, byref(errorCode))
        if MSP_SUCCESS != errorCode.value:
            logger.info("QIVWSessionBegin failed, error code is: {}".format(
                errorCode.value))
            return

        # QIVWRegisterNotify
        dll.QIVWRegisterNotify.argtypes = [c_char_p, c_void_p, c_void_p]
        ret = dll.QIVWRegisterNotify(sessionID, pCallbackFunc, None)
        if MSP_SUCCESS != ret:
            logger.info("QIVWRegisterNotify failed, error code is: {}".format(ret))
            return

        # QIVWAudioWrite
        # 创建PyAudio对象
        pa = pyaudio.PyAudio()

        # 设置音频参数
        sample_rate = 16000
        chunk_size = 1024
        format = pyaudio.paInt16
        channels = 1

        # 打开音频流
        stream = pa.open(format=format,
                         channels=channels,
                         rate=sample_rate,
                         input=True,
                         frames_per_buffer=chunk_size)

        # 开始录制音频
        logger.info("* start recording")
        ret = MSP_SUCCESS
        self.ui.btnVoiceWakeUpSwitch.setText("关闭语音控制")
        self.log_show("已打开语音控制")
        while ret == MSP_SUCCESS:  # 这里会一直进行监听你的唤醒，只要监听到你的唤醒就调用上面的py_ivw_callback函数打印日志
            if self.voice_wake_up_event == 1:
                break
            audio_data = stream.read(chunk_size)
            audio_len = len(audio_data)
            ret = dll.QIVWAudioWrite(sessionID, audio_data, audio_len, 2)
        logger.info('QIVWAudioWrite ret =>{}', ret)
        logger.info("* done recording")

        ret = dll.QIVWSessionEnd(sessionID, "normal end")
        if MSP_SUCCESS != ret:
            logger.info("QIVWSessionEnd failed, error code is: {}".format(ret))

        ret = dll.MSPLogout()
        if MSP_SUCCESS != ret:
            print("MSPLogout failed, error code is: {}".format(ret))

        # 关闭音频流
        stream.stop_stream()
        stream.close()

        # 终止PyAudio对象
        pa.terminate()

    def btn_voice_wake_up_switch(self):
        if self.ui.btnVoiceWakeUpSwitch.text() == "关闭语音控制":
            self.voice_wake_up_event = 1
            self.voice_wake_up_thread.join()
            self.ui.btnVoiceWakeUpSwitch.setText("打开语音控制")
            self.log_show("已关闭语音控制")
        elif self.ui.btnVoiceWakeUpSwitch.text() == "打开语音控制":
            self.voice_wake_up_event = 0
            self.voice_wake_up_thread = Thread(target=self.ivw_wakeup)
            self.voice_wake_up_thread.start()

    def btn_display_fpga_switch(self):
        if self.ui.btnDisplaySwitchFPGA.text() == "显示语谱":
            self.serial.WritePort(SER_DISPLAY_SPECTROGRAM)
            self.log_show("当前显示音频语谱图")
            self.ui.btnDisplaySwitchFPGA.setText("显示频谱")
        elif self.ui.btnDisplaySwitchFPGA.text() == "显示频谱":
            self.serial.WritePort(SER_DISPLAY_SPECTRUM)
            self.log_show("当前显示音频频谱图")
            self.ui.btnDisplaySwitchFPGA.setText("显示语谱")

    def init_voice_change_testing(self):

        # 加载模型
        with open('voice_change_main/random_forest_model.pkl', 'rb') as f:
            self.clf = pickle.load(f)

    def detect_voice_transformation(self):
        features = extract_features(self.voice_intput_path)
        prediction = self.clf.predict([features])
        return '变声音频' if prediction[0] == 1 else '原声音频'
