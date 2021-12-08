from pyAudioAnalyzer import AudioAnalyzer

AA = AudioAnalyzer()
AA.Lang = 'DE'
AA.record()
AA.PlotTimeDomain()
AA.PlotFFT()
AA.PlotFFTpyFFTW()
AA.PlotSpectrogram()
AA.PlotCepstrum()
