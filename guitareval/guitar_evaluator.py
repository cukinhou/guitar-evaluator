"""
This script implements the class GuitarEvaluator for
guitar onset and pitch analysis. The onset detection system
uses Sebastian Bock's algorithm, based on pre-trained
Convolutional Neural Networks, implemented in Madmom.
Pitch estimation is addressed using Justin Salomon's
Melodia algorithm implemented in Essentia library.

author: Javier Nistal
"""
import numpy as np
import essentia.standard as es
from madmom.features import onsets
from essentia import array


class GuitarEvaluator(object):
    """
    This class implements an onset and pitch estimation system for
    guitar audio recordings. The
    """
    def __init__(self, sr=22050):
        """
        Class that implements a guitar evaluation system
        :param sr: sampling rate
        """
        self.sr = sr

    @staticmethod
    def onset_detection(audio):
        """
        Onset detection using Convolutional Neural Networks. This algorithm
        is developed by Sebastian Bock and Jan Schluter and implemented
        in Madmom MIR python library.
        :type audio: vector_real
        :param audio: input audio signal

        :rtype onset_times: vector_real
        :return onset_times: onset times in samples
        """
        audio_filt = es.BandPass(
            bandwidth=200, cutoffFrequency=300
        )(array(audio / max(audio)))

        onset_strength = onsets.CNNOnsetProcessor()(audio_filt)
        onset_frames = onsets.peak_picking(
            onset_strength, threshold=0.9, smooth=6
        )

        frame_rate = len(audio)/len(onset_strength)
        onset_times = onset_frames*frame_rate

        return onset_times

    @staticmethod
    def pitch_estimation(audio, sr=22050, frame_size=1024, hop_size=128):
        """
        Pitch detection using pitch contour estimation and candidate grouping.
        This algorithm was designed by Justin Salomon and it is implemented
        in Essentia Standard library.
        :type audio: vector_real
        :param audio: input audio signal
        :type sr: int
        :param sr: sample rate
        :type frame_size: int
        :param frame_size: frame size

        :type hop_size: int
        :param hop_size: hop size

        :rtype pitch: vector_real
        :return pitch: estimated pitch
        """

        pitch, _ = es.PitchMelodia(
            minFrequency=80,
            maxFrequency=700,
            frameSize=frame_size,
            hopSize=hop_size,
            sampleRate=sr,
            filterIterations=3
        )(audio)
        return pitch

    @staticmethod
    def _f_slicer(f, signal, n1, n2):
        """
        Slices the input 'signal' in chunks of length 'n1' to 'n2'
        and applies method 'f' on them.
        :type f: function
        :param f: method

        :type signal: vector_real
        :param signal: input audio signal
        :type n1: vector_int
        :param n1: start indices of each slice
        :type n2: vector_int
        :param n2: end indicies of each slice
        :return: f(slices)
        """
        return map(lambda x, y: f(signal[x:y]), n1, n2)

    def mean_pitch_in_slice(self, pitch, i_start, i_end):
        """
        Computes the mean pitch within each slice

        :type pitch: vector_real
        :param pitch: input estimated pitch
        :type i_start: int
        :param i_start: start indices of each slice

        :type i_end: int
        :param i_end:  end indices of each slice

        :rtype: vector_real
        :return: mean pitch value per slice
        """
        assert len(i_start) == len(i_end),\
            'Slicer error: different number of start and end times'
        assert type(i_start[0]) and type(i_end[0]) is np.int64, \
            'Start and end index must be type int. Type: %r' % type(i_start[0])

        return self._f_slicer(np.mean, pitch, i_start, i_end)

    def effective_note_duration(self, audio, onset_times):
        """
        Calculates the effective duration of each guitar note based on
        Essentia's EffectiveDuration algorithm. This is measured as the time
        the note envelope is above or equal to a certain threshold.

        :type audio: vector_real
        :param audio: input audio signal
        :type onset_times: vector_real
        :param onset_times: estimated onset times
        :rtype: vector_real
        :return:
        """
        duration = es.EffectiveDuration(thresholdRatio=0.01)
        t_end = len(audio)
        offsets = np.append(onset_times[1:], t_end)

        return self._f_slicer(duration, audio, onset_times, offsets)

    def effective_note_offset(self, audio, onset_times):
        """
        Computes the effective duration of a note given the onset
        times
        :type audio: vector_real
        :param audio: input audio signal
        :type onset_times: vector_int
        :param onset_times: onset time indices

        :rtype: vector_int
        :return: offset time indices
        """
        durations = self.effective_note_duration(audio, onset_times)
        durations = [int(i*self.sr) for i in durations]

        return onset_times+durations

    def compute(
            self, audio, pframe_size=2048, phop_size=128
    ):
        """
        Computes onset detection and pitch estimation on an input
        audio recording. Parameters are tunned for guitar audio
        recordings.
        :type audio: vector_real
        :param audio: input audio recording
        :type pframe_size: int
        :param pframe_size: frame size for pitch estimation
        :type phop_size: int
        :param phop_size: hop size for pitch estimation
        :return: onset and mean pitch estimations
        """
        onset_times = self.onset_detection(audio)
        effective_offset_times = self.effective_note_offset(audio, onset_times)

        pitch = self.pitch_estimation(
            audio, frame_size=pframe_size, hop_size=phop_size
        )

        mean_pitch = self.mean_pitch_in_slice(
            pitch,
            np.ceil(onset_times/phop_size).astype(int),
            np.ceil(effective_offset_times/phop_size).astype(int)
        )

        return onset_times, mean_pitch
