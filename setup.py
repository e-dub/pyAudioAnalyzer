# -*- coding: utf-8 -*-

import os


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


if __name__ == '__main__':
    from distutils.core import setup

    extra_files = package_files('pyAudioAnalyzer')
    setup(
        name='pyAudioAnalyzer',
        version='0.1',
        description='Python library for vibrational analysis of audio',
        author='E. J. Wehrle',
        url='https://github.com/e-dub/pyAudioAnalyzer',
        package_data={'': extra_files},
        license='gpl-3.0',
        packages=['pyAudioAnalyzer'],
        install_requires=[
            'scipy',
            'numpy',
            'matplotlib',
            'playsound',
            'pyfftw',
            'librosa',
            'gtts',
            'sounddevice'
        ],
    )
