from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import os


try:
    plt.rcParams.update(
        {
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Palatino'],
        }
    )
except:
    pass


def _readList(i):
    AA = AudioAnalyzer()
    AA.read(i)
    return AA


def _PlotTimeDomain(AA, label, alpha):
    AA.PlotTimeDomain(label=label, single=False, alpha=alpha)


def _PlotFFT(AA, label, alpha):
    AA.PlotFFT(label=label, single=False, alpha=alpha)


class AudioAnalyzer:
    LineWidth = 0.5
    Lang = 'en'
    X = []

    def read(self, InputFile):
        import librosa

        self.data, self.fs = librosa.load(InputFile)
        self.x = self.data
        self.DataType = self.x.dtype
        self.nSamples = len(self.x)
        self.tEnd = self.nSamples / self.fs
        self.t = np.linspace(0, self.tEnd, self.nSamples)

    def readList(self, InputFileList):
        self.data = [[]] * len(InputFileList)
        for i, file in enumerate(InputFileList):
            self.data[i] = _readList(file)

    def record(self):
        from playsound import playsound
        from gtts import gTTS

        if self.Lang.lower() == 'en':
            tts = gTTS(
                text=(
                    'We will now record the audio.\n'
                    'Press control-c to stop recording!\n\n'
                    "Let's go! The recording starts now!"
                ),
                lang='en',
            )
        elif self.Lang.lower() == 'de':
            tts = gTTS(
                text=(
                    'Wir werden den Ton nun aufnehmen.\n'
                    'Drücke Steuerung C, um die Aufnahme zu beenden!\n\n'
                    "Los geht's! Die Aufnahme beginnt jetzt!"
                ),
                lang='de',
            )
        elif self.Lang.lower() == 'it':
            tts = gTTS(
                text=(
                    "Ora registreremo l'audio.\n"
                    'Prema control-c per interrompere la registrazione!\n\n'
                    'Andiamo! La registrazione inizia ora!'
                ),
                lang='it',
            )
        else:
            print(
                'No language specified!\n'
                "Choose English 'EN', German 'DE' or Italian 'IT'"
            )
        tts.save('message.wav')
        playsound('message.wav')
        InputFile = 'New.wav'
        os.system('arecord -f dat --file-type wav ' + InputFile)
        playsound(InputFile)
        self.read(InputFile)

    def stereo2mono(self):
        self.x = self.data[:, 0]

    def play(self):
        import sounddevice as sd

        sd.play(self.data, self.fs)
        time.sleep(self.tEnd)

    # Filter data
    def filterLowPass(self, FCutOff=20000):
        FCutOffNorm = FCutOff / (self.fs / 2)
        b, a = signal.butter(10, FCutOffNorm, btype='low', analog=False)
        self.xFilt = signal.filtfilt(b, a, self.x)

    # Filter data
    def filterBandstop(self, x, fL, fU):
        fLNorm = fL / (self.fs / 2)
        fUNorm = fU / (self.fs / 2)
        b, a = signal.butter(
            2, [fLNorm, fUNorm], btype='bandstop', analog=False
        )
        return signal.filtfilt(b, a, x)

    def PlotTimeDomainList(self, tMin=0.0, tMax=[], alpha=1):
        ymax = []
        ymin = []
        for i, datai in enumerate(self.data):
            _PlotTimeDomain(datai, self.nameList[i], alpha)
            ymax.append(datai.x.max())
            ymin.append(datai.x.min())
        ymax = np.max([np.max(ymax), np.abs(np.min(ymin))]) * 1.1
        plt.ylim(-ymax, ymax)
        plt.xlim(
            0,
        )
        plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
        sns.despine()
        plt.show()

    def PlotTimeDomain(
        self, tMin=0.0, tMax=[], label=[], single=True, alpha=1
    ):
        if tMax == []:
            tMax = self.tEnd
        # plot in time domain (original signal)
        plt.plot(
            self.t, self.x, linewidth=self.LineWidth, label=label, alpha=alpha
        )
        plt.xlabel('Time [s]')
        plt.ylabel('amplitude')
        if single:
            ymax = np.max((np.abs(self.x.min()), self.x.max())) * 1.1
            plt.ylim(-ymax, ymax)
            plt.xlim([tMin, tMax])
            sns.despine()
            plt.show()

    def cutData(self, tMin=0.0, tMax=[]):
        if tMax == []:
            tMax = self.tEnd
        self.x = self.x[self.t > tMin]
        self.t = self.t[self.t > tMin]
        self.x = self.x[self.t < tMax]
        self.t = self.t[self.t < tMax]
        self.t = self.t - tMin
        self.tEnd = tMax - tMin
        self.nSamples = len(self.x)

    def PlotFFTList(self, fMin=0.0, fMax=[], alpha=1):
        xmax = []
        for i, datai in enumerate(self.data):
            _PlotFFT(datai, self.nameList[i], alpha)
            xmax.append(datai.fs / 2)
        if fMax == []:
            fMax = np.max(xmax)
        plt.xlim(fMin, fMax)
        plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
        sns.despine()
        plt.show()

    def PlotFFT(self, fMin=0, fMax=[], label=[], single=True, alpha=1):
        if fMax == []:
            fMax = self.fs / 2
        # analysis in frequency domain via fast Fourier transform (fft)
        if self.X == []:
            self.X = np.fft.fft(self.x)
        XVal = np.abs(self.X)
        self.f = np.linspace(0.0, self.fs, self.nSamples)
        plt.plot(
            self.f[1 : int(self.nSamples / 2)],
            XVal[1 : int(self.nSamples / 2)],
            linewidth=self.LineWidth,
            label=label,
            alpha=alpha,
        )
        plt.xlabel('frequency $f$ [Hz]')
        plt.ylabel('amplitude')
        if single:
            plt.xlim([fMin, fMax])
            sns.despine()
            plt.show()

    def PlotPower(self, fMax=[]):
        if fMax == []:
            fMax = self.fs / 2
        # calculate power and plot
        dB = self.X[0 : int(self.nSamples / 2)] / self.nSamples
        plt.plot(
            self.f[0 : int(self.nSamples / 2)],
            10 * np.log10(dB),
            linewidth=self.LineWidth,
        )
        plt.xlabel('frequency [Hz]')
        plt.ylabel('power [dB]')
        # plt.title('Log spectrum in frequency domain')
        sns.despine()
        plt.show()

    def PlotSpectrogram(self, fMin=0, fMax=[], tMin=0, tMax=[]):
        if tMax == []:
            tMax = self.tEnd
        if fMax == []:
            fMax = self.fs / 2
        # calculate and plot spectrogram
        plt.specgram(self.x, Fs=self.fs, NFFT=2560)  # , cmap='BuPu')
        sns.despine()
        plt.ylabel('frequency [Hz]')
        plt.xlabel('time [s]')
        # plt.title('Spectrogram')
        bar = plt.colorbar()
        bar.outline.set_visible(False)
        bar.set_label('power [dB]')
        plt.ylim([fMin, fMax])
        plt.xlim([tMin, tMax])
        plt.show()

    def PlotFFT2TimeDomain(self):
        # conversion back to time domain via inverse fast Fourier transfrom (ifft)
        self.xt = np.fft.ifft(self.X)
        plt.figure()
        plt.plot(self.t, self.xt, linewidth=self.LineWidth)
        plt.ylabel('amplitude')
        plt.xlabel('time $t$ [s]')
        # plt.title('From frequency domain back into time domain')
        sns.despine()
        plt.show()

    def PlotCepstrum(self, tMax=[]):
        if tMax == []:
            tMax = self.tEnd / 100
        # cepstrum
        # self.cept = np.fft.ifft(np.log(np.abs(X)**2.0))**2.0
        self.cept = np.fft.ifft(np.log(np.abs(self.X)))
        plt.plot(
            self.t[self.t <= tMax] * 1000,
            self.cept[self.t <= tMax],
            linewidth=self.LineWidth,
        )
        plt.xlabel('quenfrency [ms]')
        # plt.title('Cepstrum')
        plt.xlim([0, tMax * 100])
        sns.despine()
        plt.show()

    def PlotFFTpyFFTW(self, nthreads=2, fMin=0, fMax=[]):
        if fMax == []:
            fMax = self.fs / 2
        # analysis in frequency domain via fast Fourier transform (fft) with pyFFTW
        import pyfftw

        self.X = pyfftw.builders.fft(
            self.x,
            auto_align_input=True,
            auto_contiguous=True,
            planner_effort='FFTW_ESTIMATE',
            threads=nthreads,
            overwrite_input=True,
        )()
        XVal = np.abs(self.X)
        f = np.linspace(0.0, self.fs, self.nSamples)
        plt.figure()
        plt.plot(
            f[1 : int(self.nSamples / 2)],
            XVal[1 : int(self.nSamples / 2)],
            linewidth=self.LineWidth,
        )
        plt.ylabel('amplitude')
        plt.xlabel('frequency $f$ [Hz]')
        # plt.title('Frequency domain (calculated with pyFFTW)')
        plt.xlim([fMin, fMax])
        sns.despine()
        plt.show()
