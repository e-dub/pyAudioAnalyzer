import pyAudioAnalyzer as paa

AA = paa.AudioAnalyzer()
AA.read('audioFiles/A-Saite.wav')
AA.play()
AA.PlotTimeDomain()
AA.PlotTimeDomain(tMin=1.1, tMax=8)
AA.cutData(tMin=1.1, tMax=8)
AA.PlotTimeDomain()
AA.PlotFFT()
AA.PlotFFT(fMax=1000)
AA.PlotFFT(fMin=100, fMax=120)
AA.PlotSpectrogram()
AA.PlotSpectrogram(fMax=500)
