import audioop
import sys
import struct
import pyaudio
import math
import numpy as np
import pyqtgraph as pg
import time
import padasip as pa

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout
from PyQt5.QtCore import QTimer


from numpy import array, concatenate, argmax
from numpy import abs as nabs
from scipy.signal import fftconvolve

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
FREQ_MAX = RATE // 2
FREQ_MIN = 20
CHUNK = 2 * RATE // FREQ_MIN
DELAY = 20


class WinForm(QWidget):
    def __init__(self):
        super(WinForm, self).__init__(parent=None)

        self.setWindowTitle("Audio Spectrum Analyzer")
        self.move(
            QApplication.desktop().screen().rect().center() - self.rect().center()
        )

        self.waveform = pg.PlotWidget(name="waveform")
        self.spectrum = pg.PlotWidget(name="spectrum")
        self.waveform_left = pg.PlotWidget(name="waveform_left")
        self.waveform_right = pg.PlotWidget(name="waveform_right")

        layout = QGridLayout()
        layout.addWidget(self.waveform)
        layout.addWidget(self.spectrum)
        layout.addWidget(self.waveform_left)
        layout.addWidget(self.waveform_right)

        self.setLayout(layout)

        self.waveform.setYRange(-5000, 5000)
        self.waveform.setXRange(0, CHUNK)
        self.waveform.showGrid(x=True, y=True, alpha=1)

        self.spectrum.setLogMode(x=True, y=False)
        self.spectrum.setYRange(-100, 1000)
        self.spectrum.setXRange(np.log10(FREQ_MIN), np.log10(FREQ_MAX))
        self.spectrum.showGrid(x=True, y=True, alpha=1)

        self.wv_x_axis = self.waveform.getAxis("bottom")
        self.wv_x_axis.setStyle(tickAlpha=0.5)

        self.wv_y_axis = self.spectrum.getAxis("left")
        self.wv_y_axis.setStyle(tickAlpha=0.5)

        self.sp_x_axis = self.spectrum.getAxis("bottom")
        self.sp_x_axis.setStyle(tickAlpha=0.5)

        self.sp_y_axis = self.spectrum.getAxis("left")
        self.sp_y_axis.setStyle(tickAlpha=0.5)

        sp_x_labels = [
            (np.log10(15), "15"),
            (np.log10(31), "31"),
            (np.log10(62), "62"),
            (np.log10(125), "125"),
            (np.log10(250), "250"),
            (np.log10(500), "500"),
            (np.log10(1000), "1k"),
            (np.log10(2000), "2k"),
            (np.log10(4000), "4k"),
            (np.log10(8000), "8k"),
            (np.log10(16000), "16k"),
        ]

        self.sp_x_axis.setTicks([sp_x_labels])
        self.sp_x_axis.setLabel("Frequency ", units="HZ")

        self.waveform.setTitle("waveform")
        self.spectrum.setTitle("spectrum")

        self.wv_style = {"color": "#FF0", "font-size": "20pt"}
        self.sp_style = {"color": "#0F0", "font-size": "20pt"}

        self.waveform_left.setYRange(-5000, 5000)
        self.waveform_left.setXRange(0, CHUNK)
        self.waveform_right.setYRange(-5000, 5000)
        self.waveform_right.setXRange(0, CHUNK)
        self.left_x_axis = self.waveform_left.getAxis("bottom")
        self.left_x_axis.setStyle(tickAlpha=0.5)
        self.right_x_axis = self.waveform_right.getAxis("bottom")
        self.right_x_axis.setStyle(tickAlpha=0.5)
class AudioStream:
    def __init__(self):
        self.m = property(WinForm)
        self.p = pyaudio.PyAudio()

        try:
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK,
            )
        except Exception as e:
            print("open microphone failed!\nError:", e)
            return

        self.x = np.arange(0, CHUNK)
        self.f = np.linspace(0, 22050, CHUNK)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(DELAY)
        self.timer_cnt = 0
        self.left_max_temp = 0
        self.right_max_temp = 0

    def update(self):
        wf_data = self.stream.read(CHUNK)
        cnt = len(wf_data) // 2
        fmt = "%dh" % cnt
        shorts = struct.unpack(fmt, wf_data)

        cnt = cnt // CHANNELS
        self.x = np.arange(0, cnt)
        lefts = shorts[::2]
        rights = shorts[1::2]
        self.m.waveform.plot(self.x, lefts, pen="c", clear=True)

        v_rms = audioop.rms(wf_data, 2)
        line_vms_y = np.linspace(v_rms, v_rms, cnt)
        self.m.waveform.plot(self.x, line_vms_y, pen="y", clear=False)

        self.f = np.fft.fftfreq(cnt, 1.0 / RATE)
        self.f = np.fft.fftshift(self.f)
        sp_data = np.abs((1.0 / cnt) * np.fft.fft(lefts))
        sp_data = np.fft.fftshift(sp_data)

        v_max = np.amax(sp_data)
        line_max_x = np.fft.fftfreq(cnt, 1.0 / RATE)
        line_max_y = np.linspace(v_max, v_max, cnt)

        self.m.spectrum.plot(self.f, sp_data, pen="m", clear=True)
        self.m.spectrum.plot(line_max_x, line_max_y, pen="g", clear=False)

        rms = 20 * np.log10(v_rms) if v_rms else 0
        self.m.wv_x_axis.setLabel("RMS:{:.0f}dB".format(rms), **self.m.wv_style)
        self.m.sp_x_axis.setLabel("MAX:{:.0f}".format(v_max), **self.m.sp_style)

        self.m.waveform_left.plot(self.x, lefts, pen="c", clear=True)
        self.m.waveform_right.plot(self.x, rights, pen="r", clear=True)

        # left_rms = rmsValue(lefts, cnt)
        # right_rms =  rmsValue(rights, cnt)
        # self.m.left_x_axis.setLabel("RMS:{:.0f}".format(left_rms), **self.m.sp_style)
        # self.m.right_x_axis.setLabel("RMS:{:.0f}".format(right_rms), **self.m.sp_style)


        # if right_rms:
        #     v = left_rms / right_rms
        #     print('{:.6f}'.format(v))

        left_max = np.max(lefts)
        right_max = np.max(rights)

        self.timer_cnt += 1
        if (self.timer_cnt % (3000 // DELAY) == 0):
            self.left_max_temp = 0
            self.right_max_temp = 0
            self.timer_cnt = 0
        if left_max > self.left_max_temp:
            self.left_max_temp = left_max
        if right_max > self.right_max_temp:
            self.right_max_temp = right_max

        self.m.left_x_axis.setLabel(self.left_max_temp, **self.m.sp_style)
        self.m.right_x_axis.setLabel(self.right_max_temp, **self.m.sp_style)

        #
        # if v_rms > 30:
        #     # angle = doa(lefts, rights)
        #     angle = sound_direction(lefts, rights, cnt, 1)
        #     self.m.left_x_axis.setLabel("ANGLE:{:.0f}".format(angle), **self.m.sp_style)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.timer.stop()
        self.stream.close()

def rmsValue(arr, n):
    square = 0
    mean = 0.0
    root = 0.0
    #Calculate square
    for i in range(0,n):
        square += (arr[i]**2)
        #Calculate Mean
        mean = (square / (float)(n))
        #Calculate Root
        root = math.sqrt(mean)
    return root
def crossco(wav):
    """Returns cross correlation function of the left and right audio. It
    uses a convolution of left with the right reversed which is the
    equivalent of a cross-correlation.
    """
    cor = nabs(fftconvolve(wav[0],wav[1][::-1]))
    return cor
def sound_direction(left, right, chunksize, width):
    # zero pad each channel with zeroes as long as the source
    left = concatenate((left, [0] * chunksize))
    right = concatenate((right, [0] * chunksize))

    chunk = (left, right)

    # if the volume is very low (800 or less), assume 0 degrees
    if abs(max(left)) < 800:
        a = 0.0
    else:
        # otherwise computing how many frames delay there are in this chunk
        cor = argmax(crossco(chunk)) - chunksize * 2
        # calculate the time
        t = cor / RATE
        # get the distance assuming v = 340m/s sina=(t*v)/width
        sina = t / width
        if abs(sina) >=1:
            print(sina)
            a = 0.0
        else:
            a = math.asin(sina) * 180 / (3.14159)

    print('angle=', a)
    return a

def doa(lefts, rights):
    fs = RATE #Sampling frequency(Hz)
    d_micro = 0.1 # Distance between microphones(m)
    c = 340 # Speed of sound(m / s)
    muestrasMAX = int(np.ceil(d_micro * fs / c)) #Maximum; number; of; samples; Nmax;
    DESP = int(np.ceil(muestrasMAX * 1.5)) # Delay we insert in the micro 2 We leave 50 of margin of error

    signal = lefts # Signal in MIC; B
    d = rights # Signal in MIC; A % NORMALIZATION PROCESS
    M1 = np.max(np.abs(signal))# Maximum of channel 1
    M2 = np.max(np.abs(d))# Maximum of channel 2
    M3 = max(M1, M2)# Normalization value
    signal = signal / M3 * 2#Normalizing
    d = d / M3 * 2

    # LMS ALGORITHM
    hDESP = np.zeros(DESP)# Filter to delay the signal DESP samples.
    hDESP = np.append(hDESP, 1)
    d1 = np.convolve(hDESP, d, mode='same')

    # Parameters of the algorithm
    P = 50
    mu = 0.0117
    h0 = np.zeros(P)
    h0[0] = 0#Initialazing the adaptative filter

    # h, y, e = f_adap(signal, d1, h0, mu) # Recursive function calculating the coefficients of the filter h(n)
    # print(muestrasMAX, signal.shape, len(d), d1.shape)
    # f = pa.filters.FilterLMS(len(signal), mu=mu)
    # y, e, h = f.run(d1, signal)

    h = lms(signal, d1, N = P, mu = mu)

    # PROCESSING THE FIcfLTER BEFORE THE FREQUENCY ANALYSIS.
    h1 = np.zeros(DESP - muestrasMAX - 3)
    h1 = np.append(h1, h[DESP - muestrasMAX - 3 :len(h)])
    # print(DESP,muestrasMAX,len(h),len(h1))
    h1[DESP + muestrasMAX + 1: len(h1)] = np.zeros(len(h1)-(DESP + muestrasMAX + 1))
    h1[DESP] = h1[DESP] / 2
    # B, I = np.flip(np.sort(h1))
    # H1 = [np.zeros(I[0] - 3), h[I(0) - 3:I(0) + 2], np.zeros(len(h) - (I[0] + 2))]
    #FREQUENCY ANALYSIS TO OBTAIN THE DELAY(IN SAMPLES)

    # 1 - FFT
    lh = 128#Length of the FFT
    H = np.fft.fft(h1, lh)#FFT of the filter h(n)

    #2 - ANGLE(+UNWRAP)
    alpha = np.angle(np.fft.fftshift(H))#Obtaining the phase
    q = np.unwrap(alpha)

    #3 - SLOPE
    M = np.diff(q)#Obtaining the slope of the phase

    #4 - SLOPE 'S AVERAGE
    lM = len(M) + 2#The slope M1 is not a unique value,
    p1 = int(np.floor(lM / 2 - 4))#it 's an array. So we calculate the
    p2 = int(np.ceil(lM / 2 + 4))# average of the values, K.
    K = np.mean(M[p1-1:p2])
    Nprime = (-K * lh / (2 * np.pi))# Number of samples before substracting DESP.

    # 5 - SAMPLES
    if Nprime < 0: #Two possible cases: negative or positive
        N = Nprime + lh
        N = N - DESP
    else:
        N = Nprime
        N = N - DESP


    #CALLING THE FUNCTION WHICH RETURNS THE ANGLE
    angleGRAD1 = get_angle(N, fs, d_micro)

    if not np.isreal(angleGRAD1): # Security measures in case angleGRAD % the number is complex
        angleGRAD1 = np.real(angleGRAD1)

    return angleGRAD1


def lms(x, d, N = 4, mu = 0.05):
    L = min(len(x),len(d))
    h = np.zeros(N)
    e = np.zeros(L-N)
    for n in range(L-N):
        x_n = x[n:n+N][::-1]
        d_n = d[n]
        y_n = np.dot(h, x_n.T)
        e_n = d_n - y_n
        h = h + mu * e_n * x_n
        e[n] = e_n
    return h

'''
#PERFORMS THE CALCULATION OF THE COEFFICIENTS OF THE FILTER h(n).
# Inputs:   - x = Signal in MIC A
#           - d = Signal in MIC B
#           - h0 = Initial filter(equals to 0)
#           - mu = Step - size
# Outputs:  - h = Desired filter
#           - y = Convolution between x and h
#           - e = Error function
def f_adap(x, d, h0, mu):
    # Implements the LMS algorithm.
    # Inputs:   x(n) Original signal
    #           d(n) Delayed signal
    #           h0 Origal filter
    #           mu Constant value
    # Outputs:  h(n) Filter
    #           y(n) = x(n) * h(n)
    #           e(n) Error function(must be zero)

    h = h0
    P = len(h)
    N = len(x)
    y = np.zeros(N)
    e = y#Reserve space for y[] y e[]
    rP = np.arange(0, -P, -1)

    for k in np.arange(P-1, N):
        xx = x[k + rP]#Last P inputs x[k], x[k - 1], ... x[k - P]
        y[k] = xx * h.conjugate()  #Filter output: x*h Convolution
        e[k] = d[k] - y[k] # Error
        h = h + mu * e[k] * xx# We update the filter coefficients.

    return h, y, e
'''

# OBTAIN THE COORDINATES OF THE POSITIONS WHERE THE SPEAKER CAN BE.
# Inputs:   - muestras = Delay between signals in samples
#           - x = Values of the x - axis
#           - xA = x coordinate of MIC A
#           - fs = Sampling frequency
# Outputs:  - y1 = Values of the y coordinate of the speaker
def hiper(muestras, x, xA, fs):
    c = 340#speed of sound(m / sec)
    pot = 2 * np.ones(len(x))
    dist = muestras * c / fs# distance to B prime
    v = dist ** 2 / 4 - xA ** 2 + (4 * xA ** 2 / dist ** 2 - 1) * x ** pot
    # print(['{:.3f}'.format(x) for x in v])
    # print(len([_ for _ in v if _ < 0]))
    y1 = np.sqrt(np.abs(v))
    #formula with following % requisitions:
    # xA = -xB;
    # yA = yB = 0
    return y1


#OBTAINS THE ANGLE BY PERFORMING A CERTAIN NUMBER OF TRIGONOMETRIC
# CALCULATIONS.IT CALLS THE FUNCTION:
# hiper
# Inputs:   - N = Number of samples
#           - fs = Sampling Frequency
#           - d_micro = Distance between microphones
# Outputs:  - angle = Angle in degrees
def get_angle(N, fs, d_micro):
    if N:
        j = 0.1#Steps
        x = np.arange(-20, 20+j, j)# x axis
        y1 = hiper(N, x, -d_micro / 2, fs)# Calling function hiper
        x1 = round(len(x) / 4)
        x2 = round(len(x) / 8)
        pendiente = (y1[x1-1] - y1[x2-1]) / (j * (x1 - x2))#Slope
        if N > 0:
            angulorad = math.atan(pendiente)
            angulo1 = angulorad * 180 / np.pi
            angle = -90 - angulo1
        else:
            angulorad = math.atan(-pendiente)
            angulo1 = angulorad * 180 / np.pi
            angle = 90 - angulo1
    else:
        angle = 0

    return angle


def main():
    app = QtWidgets.QApplication(sys.argv)
    a = AudioStream()
    if hasattr(a, "stream"):
        a.m = WinForm()
        a.m.show()
        sys.exit(app.exec_())
    else:
        app.exit()
        sys.exit()


if __name__ == "__main__":
    main()
