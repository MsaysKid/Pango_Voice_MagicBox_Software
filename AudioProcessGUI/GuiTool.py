# -*- coding:utf-8 -*-
# @ProjectName  :AudioProcessing
# @FileName     :GuiTool.py
# @Time         :2024/7/4 19:34
# @Author       :Msays
# @Version      :1.0
from datetime import datetime

def GetLongTimeString():
    return datetime.now().strftime("%Y%m%d%H%M%S")

def GetShortTimeString():
    return datetime.now().strftime("%H:%M:%S")

def GetShortDateString():
    return datetime.now().strftime("%Y-%m-%d")
