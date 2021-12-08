from pyAudioAnalyzer import AudioAnalyzer

AA = AudioAnalyzer()
AA.Lang = 'IT'
AA.record()
AA.PlotTimeDomain()
AA.PlotFFT()
AA.PlotFFTpyFFTW()
AA.PlotSpectrogram()
AA.PlotCepstrum()
