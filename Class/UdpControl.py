# -*- coding:utf-8 -*-
# @ProjectName  :AudioProcessing
# @FileName     :UdpControl.py
# @Time         :2024/7/5 13:55
# @Author       :Msays
# @Version      :1.0

import socket
import re


class UdpControl:
    def __init__(self, ip=None, port=None, buffer_size=None):
        self.__ip = ip
        self.__port = port
        self.__buffer_size = buffer_size
        self.udp = None

    def OpenUdp(self):
        self.udp = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.udp.bind((self.__ip, self.__port))

    def ReadData(self):
        try:
            socket_data, address = self.udp.recvfrom(self.__buffer_size)
            text_list = re.findall(".{2}", socket_data.hex())
            new_text = " ".join(text_list)
            return new_text
        except Exception as e:
            print(e)

    def CloseUdp(self):
        self.udp.close()


SOCKET_IP = ('192.168.1.105', 8080)
BUFFER_SIZE = 48000 * 32


def start_server_socket():
    """
    启动服务端UDP Socket
    :return:
    """
    ip, port = SOCKET_IP
    server = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)  # 使用UDP方式传输
    server.bind((ip, port))  # 绑定IP与端口
    print(f'服务端 {ip}:{port} 开启')

    # 不断循环，接受客户端发来的消息
    with open("Test/udptext_music.txt", 'a') as f:
        while True:
            socket_data, address = server.recvfrom(BUFFER_SIZE)
            text_list = re.findall(".{2}", socket_data.hex())
            new_text = " ".join(text_list)
            f.write(new_text)
            print(new_text)

if __name__ == '__main__':
    start_server_socket()
    # wav2txt()
