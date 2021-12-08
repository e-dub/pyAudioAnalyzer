from pyAudioAnalysis import SoundAnalyzer

SA = SoundAnalyzer()
SA.Lang = 'DE'
SA.record()
SA.PlotTimeDomain()
SA.PlotFFT()
SA.PlotFFTpyFFTW()
SA.PlotSpectrogram()
SA.PlotCepstrum()
