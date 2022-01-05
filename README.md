# pyAudioAnalyzer

[![DOI](https://zenodo.org/badge/414517473.svg)](https://zenodo.org/badge/latestdoi/414517473)
[![PyPi Version](https://img.shields.io/pypi/v/pyAudioAnalyzer.svg?style=flat-square)](https://pypi.org/project/pyAudioAnalyzer)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pyAudioAnalyzer.svg?style=flat-square)](https://pypi.org/project/pyAudioAnalyzer/)
[![GitHub stars](https://img.shields.io/github/stars/e-dub/pyAudioAnalyzer.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/e-dub/pyAudioAnalyzer)
[![PyPi downloads](https://img.shields.io/pypi/dm/pyAudioAnalyzer.svg?style=flat-square)](https://pypistats.org/packages/pyAudioAnalyzer)
[![Code style: blue](https://img.shields.io/badge/code%20style-blue-blue.svg)](https://blue.readthedocs.io/)

**Python library for vibrational analysis of audio**\
**Python-Bibliothek für die Schwingungsanalyse von Audio**\
**Libreria Python per l'analisi vibrazionale dell'audio**

## Installation

### Prerequisites
```
pip install scipy
pip install numpy
pip install matplotlib
pip install seaborn
pip install playsound
pip install pyfftw
pip install librosa
pip install gtts
pip install sounddevice
```
Currently, I am using librosa to read audio files.  This requires FFMPEG to work. Unfortunately, there is no installer available for Windows (to my knowledge) and you have to unzip and set the environmental variables.  Several installation guides are available for this. 

sudo apt-get install libsndfile1

### Install

Clone repository and install via PIP using the state of this repository:
```
python -m pip install -U .
```

You can also install via PIP with the last release:
```
pip install pyAudioAnalyzer
```

## Getting started
See Python scripts and iPython notebooks under examples.


## File types
- M4A:  Audio-only MPEG-4 (Moving Picture Experts Group)
- 3GP, 3G2: MPEG-4, for smartphones typically Part 12 and specifically 3GP
- WAV: Waveform Audio File Format
- MP3: MPEG-1 Audio Layer III and MPEG-2 Audio Layer III
