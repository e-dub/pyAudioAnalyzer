import pyAudioAnalyzer as paa

All = paa.AudioAnalyzer()
All.readList(
    [
        'audioFiles/SawUnloaded.m4a',
        'audioFiles/SawBlockCut1.m4a',
        'audioFiles/SawBlockCut2.m4a',
        'audioFiles/SawBoardCut1.m4a',
        'audioFiles/SawBoardCut2.m4a',
    ]
)
All.nameList = [
    'unlaoded',
    'block cut 1',
    'block cut 2',
    'board cut 1',
    'board cut 2',
]
All.PlotTimeDomainList(alpha=0.5)
All.PlotFFTList(alpha=0.7)
All.PlotFFTList(fMax=2000, alpha=0.7)


BoardComparison = paa.AudioAnalyzer()
BoardComparison.readList(
    ['audioFiles/SawBoardCut1.m4a', 'audioFiles/SawBoardCut2.m4a']
)
BoardComparison.nameList = ['board cut 1', 'board cut 2']
BoardComparison.PlotTimeDomainList(alpha=0.7)
BoardComparison.PlotFFTList(alpha=0.7)
BoardComparison.PlotFFTList(fMax=2000, alpha=0.7)


BlockComparison = paa.AudioAnalyzer()
BlockComparison.readList(
    ['audioFiles/SawBlockCut1.m4a', 'audioFiles/SawBlockCut2.m4a']
)
BlockComparison.nameList = ['block cut 1', 'block cut 2']
BlockComparison.PlotTimeDomainList(alpha=0.7)
BlockComparison.PlotFFTList(alpha=0.7)
BlockComparison(fMax=2000, alpha=0.7)
