import threading
import serial


class SerialControl:
    def __init__(self, portx=None, bps=None, timex=None):  # 初始化串口
        self.portx = portx
        self.bps = bps
        self.timex = timex
        self.lock = threading.Lock()
        self.ser = None

    def OpenPort(self):
        self.ser = serial.Serial(self.portx, self.bps, timeout=self.timex)


    def isOpen(self):
        return self.ser is not None and self.ser.isOpen()

    def ClosePort(self):
        if self.isOpen():
            self.ser.close()

    def WritePort(self, code):
        result = self.ser.write(code)
        print("send:{}".format(code))



if __name__ == "__main__":
    ser = SerialControl("COM3", 115200, 0.1)
    ser.OpenPort()
    ser.WritePort(bytearray([0x83, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]))
    ser.ClosePort()