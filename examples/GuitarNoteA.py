import pyAudioAnalysis as paa

SA = paa.SoundAnalyzer()
SA.read('audioFiles/A-Saite.wav')
SA.play()
SA.PlotTimeDomain()
SA.PlotTimeDomain(tMin=1.1, tMax=8)
SA.cutData(tMin=1.1, tMax=8)
SA.PlotTimeDomain()
SA.PlotFFT()
SA.PlotFFT(fMax=1000)
SA.PlotFFT(fMin=100, fMax=120)
SA.PlotSpectrogram()
SA.PlotSpectrogram(fMax=500)
