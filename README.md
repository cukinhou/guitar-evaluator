===========
Movie Rec
===========

GuitarEvaluator computes onset and pitch estimation
on monophonic guitar audio recordings. Onset detection
is based on Madmom's Convolutional Neural Networcks
algorithm; pitch estimation is based on Essentia's
Melodia extractor. Evaluation can be done by executing
the tests as explained below. Tests generate a log file
with statistical results regarding the performance
of the onset and pitch detectors. Usage::

    import essentia.standard as es
    from guitareval.guitar_evaluator import GuitarEvaluator

    path = 'some_path/audio.wav'
    sr = 22050

    audio = es.MonoLoader(filename=path, sampleRate=sr)

    evaluator = GuitarEvaluator(sr)
    onsets, pitch = evaluator.compute(audio)


Install
=========

First you need to install Essentia MIR library
as explained in http://essentia.upf.edu/documentation/installing.html
Then un the following commands from GuitarEvaluator root folder:

* easy_install pip (only if you don't have pip installed)

* make init

* make install

Running tests:
-------------

* pip install nose (if you don't have installed nose)

* make test
