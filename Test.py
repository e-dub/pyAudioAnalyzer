#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 09:58:40 2021

@author: wehrle
"""

from pyAudioAnalysis import SoundAnalyzer

SA = SoundAnalyzer()
#SA.read(iFile)
SA.Lang = "EN"
SA.record()
SA.PlotTimeDomain()
SA.PlotFFT()
SA.PlotFFTpyFFTW()
SA.PlotSpectrogram()
SA.PlotCepstrum()
