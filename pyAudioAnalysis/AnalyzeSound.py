"""
Code to analyze acoustic emission

Conversion from m4a to wav:
    ffmpeg -i filename.m4a filename.wav

Author: Erich Wehrle

TODO:
    variable titles
    save files as tikz, svg, png
    verify cepstrum
"""

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time


class SoundAnalyzer:
    LineWidth = 0.5
    Lang = 'en'

    def read(self, InputFile):
        import librosa

        self.data, self.fs = librosa.load(InputFile)
        self.x = self.data
        self.DataType = self.x.dtype
        self.nSamples = len(self.x)
        self.tEnd = self.nSamples / self.fs
        self.t = np.linspace(0, self.tEnd, self.nSamples)

    def record(self):
        # If file blank, record file. Press crtl-c to stop recording!
        import os

        # import librosa
        from playsound import playsound

        # try:
        # Better speech
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
        # except:
        #     # Computerized speech
        #     os.system("say 'we will now record you'")
        #     os.system("say 'press control-c when you are done'")
        #     os.system("say 'say something now!'")
        InputFile = 'New.wav'
        os.system('arecord -f dat --file-type wav ' + InputFile)
        playsound(InputFile)
        self.read(InputFile)

    def stereo2mono(self):
        # self.x = self.data[:, channel - 1]
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

    def PlotTimeDomain(self, tMin=0.0, tMax=[]):
        if tMax == []:
            tMax = self.tEnd
        # plot in time domain (original signal)
        plt.plot(self.t, self.x, self.LineWidth)
        # plt.title('Time domain')
        plt.xlabel('Time [s]')
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

    def PlotFFT(self, fMin=0, fMax=[]):
        if fMax == []:
            fMax = self.fs / 2
        # analysis in frequency domain via fast Fourier transform (fft)
        self.X = np.fft.fft(self.x)
        XVal = np.abs(self.X)
        self.f = np.linspace(0.0, self.fs, self.nSamples)
        plt.figure()
        plt.plot(
            self.f[1 : int(self.nSamples / 2)],
            XVal[1 : int(self.nSamples / 2)],
            linewidth=self.LineWidth,
        )
        plt.ylabel('amplitude')
        plt.xlabel('frequency $f$ [Hz]')
        # plt.title('Amplitude in frequency domain')
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
        plt.ylabel('frequency [Hz]')
        plt.xlabel('time [s]')
        # plt.title('Spectrogram')
        bar = plt.colorbar()
        bar.set_label('power [dB]')
        plt.ylim([fMin, fMax])
        plt.xlim([tMin, tMax])
        sns.despine()
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


if __name__ == '__main__':
    # Set file to be read
    InputFileList = [
        'GuitarVG/A-Saite.wav',
        'Music/57007r.wav',
        'Coanda/Coanda_20170930_141820.wav',
        'Coanda/Coanda_20170930_141820.mp4',
        'Saw/20200427_091138-NoCut.m4a',
        'Saw/20200427_091720-BoardCut1.m4a',
        'Saw/20200427_092056-BlockCut1.m4a',
        'Saw/20200427_091702-BoardCut2.m4a',
        'Saw/20200427_092145-BlockCut2.m4a',
        'Randoms/Bobbypfeife.wav',
        'Randoms/Bobbypfeife.ogg',
        'Frequency/Sine_wave_440.ogg',
        'MusicalNotes/Middle_C.mid',
        'MusicalNotes/A440.mid',
        'MusicalNotes/A440_violin.mid',
        'WineGlass.wav',
        'Music/01 Fortune, Empress Of The World_.mp3',
    ]

    # InputFileList= ["Gufler_Klappe.mp4"]
    InputFileList = ['../examples/audioFiles/WineGlass.wav']

    for iFile in InputFileList:
        SA = SoundAnalyzer()
        SA.read(iFile)
        SA.PlotTimeDomain()
        SA.PlotFFT()
        SA.PlotFFTpyFFTW()
        SA.PlotSpectrogram()
        SA.PlotCepstrum()
    # SA.play()
    SA.PlotFFT(fMax=2000)
    SA.PlotFFT(fMax=100)

    import simpleaudio as sa

    xFilt1 = SA.filterBandstop(SA.x, 19, 21)
    xFilt2 = SA.filterBandstop(xFilt1, 38, 42)
    xFilt2 = SA.filterBandstop(xFilt2, 58, 60)
    xFilt3 = SA.filterBandstop(xFilt2, 78, 80)
    SA.x = xFilt3
    note = xFilt3
    audio = note * (2 ** 15 - 1) / np.max(np.abs(note))
    # Convert to 16-bit data
    audio = audio.astype(np.int16)

    # Start playback
    # play_obj = sa.play_buffer(audio, 1, 2, SA.fs)

    # Wait for playback to finish before exiting
    # play_obj.wait_done()

    from scipy.io import wavfile

    wavfile.write('KlappeSimFilter.wav', SA.fs, audio)

    # SA.PlotFFTpyFFTW()
    # SA.PlotFFTpyFFTW(fMax=100)
    # SA.play()

    FCutOffNorm = 2000 / (SA.fs / 2)
    b, a = signal.butter(10, FCutOffNorm, btype='high', analog=False)
    xFilt = signal.filtfilt(b, a, SA.x)
    SA.x = xFilt
    SA.PlotFFTpyFFTW()

    note = xFilt
    audio = note * (2 ** 15 - 1) / np.max(np.abs(note))
    # Convert to 16-bit data
    audio = audio.astype(np.int16)

    # Start playback
    # play_obj = sa.play_buffer(audio, 1, 2, SA.fs)

    # Wait for playback to finish before exiting
    # play_obj.wait_done()

    from scipy.io import wavfile

    wavfile.write('KlappeSimFilter2000.wav', SA.fs, audio)
    # # analysis in frequency domain via fast Fourier transform (fft) with reikna on gpu
    # from reikna.fft import FFT
    # X = FFT(x)
    # XVal = np.abs(X)
    # f = np.linspace(0.0, fs, int(nSamples/2))
    # plt.figure()
    # plt.plot(f[1:], XVal[1:int(nSamples/2)], linewidth=self.LineWidth)
    # plt.ylabel("amplitude")
    # plt.xlabel("frequency $f$ [Hz]")
    # plt.title("Frequency domain")
    # plt.xlim([0, fs/2])
    # plt.show()
