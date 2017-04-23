
"""
This class implements the evaluation of the GuitarEvaluator.
Estimated onsets and pitch values are compared with the annotated
data contained in the folder 'guitar_data'. The results of the
evaluation are stored in a log file in 'log' folder. This results
contain statistical measurements of the overall performance.
"""
import numpy as np
import pandas as pd
import essentia.standard as es
import os
import logging
from datetime import datetime
from unittest import TestCase
from guitareval.guitar_evaluator import GuitarEvaluator
import xml.etree.ElementTree as ET


def xml_parser(xml_file):
    """
    Reads xml annotated data

    :param xml_file:
    :return: data
    """
    x = ET.parse(xml_file).getroot()
    data = pd.DataFrame(
        index=range(1, len(x) + 1), columns=['pitch', 'onsetSec', 'offsetSec']
    )

    n_notes = 0
    for event in x.findall('event'):
        for e in event:
            tag = e.tag
            data.iloc[n_notes][tag] = float(event.find(tag).text)

        n_notes += 1

    return data


def midi2hz(midi_note):
    """
    Converts MIDI notes to hertz values

    :param midi_note:
    :return:
    """
    C1 = 8.1757989156
    return C1*2.**(midi_note/12.)


def onset_evaluation(annot, estim, thres=5, sr=22050):
    """
    Computes the evaluation of the onset detection system. 'thres'
    determines the tolerance to timing differences between
    annotations and estimation times.

    :param annot: original onset times in samples
    :param estim: estimated onset times in samples
    :param thres: time tolerance [ms]
    :param sr: sample rate
    :return:
    """
    fp = 0
    n_correct = 0
    n_thres = thres/1000.*sr
    correct_indx = []
    correct_index_estim = []

    for i in range(len(estim)):
        argmin = (np.abs(annot-estim[i])).argmin()

        t1 = annot[argmin]-n_thres
        t2 = annot[argmin]+n_thres

        if t1 < estim[i] < t2:
            n_correct += 1
            correct_indx = np.append(correct_indx, argmin)
            correct_index_estim = np.append(correct_index_estim, i)
        else:
            fp += 1

    fn = len(annot)-n_correct
    n_detected = len(estim)
    return fp, fn, n_correct, correct_indx, correct_index_estim, n_detected


def pitch_evaluation(annot, estim, thres, idx1, idx2):
    """
    Computes the evaluation of the pitch estimation system. 'thres'
    determines the tolerated deviation of the estimated pitch

    :param annot: original pitch annotations [hz]
    :param estim: estimated pitch [hz]
    :param thres: frequency tolerance [hz]
    :param idx1: index positions of correct estimated notes in the
    annotations vector
    :param idx2: index positions of correct estimated notes in the
    estimated vector
    :return:
    """

    pitch_error = []
    n_correct_notes = 0
    incorrect_notes = 0

    for i1, i2 in zip(idx1, idx2):
        err = abs(annot[int(i1)] - estim[int(i2)])

        if err > thres:
            pitch_error = np.append(pitch_error, err)
            incorrect_notes += 1
        else:
            n_correct_notes += 1

    if incorrect_notes == 0:
        pitch_error = 0

    mean_pitch_err = np.mean(pitch_error)
    var_pitch_err = np.var(pitch_error)
    return n_correct_notes, mean_pitch_err, var_pitch_err


class TestGuitarEvaluator(TestCase):

    DEFAULT_DATA_PATH = 'guitar_data/'
    DEFAULT_SAMPLE_RATE = 22050

    @classmethod
    def setUpClass(
            cls,
            path=DEFAULT_DATA_PATH,
            sr=DEFAULT_SAMPLE_RATE,
            timedev=10,
            pitchdev=10
    ):
            cls.evaluator = GuitarEvaluator()
            cls.sr = sr

    def test_compute(self):
        false_positives = 0
        false_negatives = 0
        n_correct_onsets = 0
        total_correct_notes = 0
        n_detected_tot = 0
        total_onsets = 0
        t = 100

        mean_perror = []
        var_perror = []

        log_name = datetime.now().strftime('guitar_eval_%H_%M_%d_%m_%Y.log')
        logger = logging.getLogger('GuitarEvaluator')
        fhand = logging.FileHandler('log/' + log_name)
        logger.addHandler(fhand)
        logger.info('Processing files...')

        for root, dirnames, filenames in os.walk('guitar_data/'):
            for filename in filenames:
                if filename.endswith('.wav'):

                    annotations = xml_parser(root+filename[:-4]+'.xml')
                    audio = es.MonoLoader(
                        filename=root+filename, sampleRate=self.sr
                    )()
                    estimated_onsets, estimated_pitch = \
                        self.evaluator.compute(audio)

                    onsets = annotations['onsetSec'].values*self.sr
                    pitch = annotations['pitch'].values

                    # convert midi notes to hertz
                    pitch_hz = map(lambda x: midi2hz(x), pitch)

                    fp, fn, ncorr, corr_an_idx, corr_es_idx, n_detected \
                        = onset_evaluation(onsets, estimated_onsets, t)

                    n_correct_notes, mean_pitch_err, var_pitch_err = \
                        pitch_evaluation(
                            pitch_hz,
                            estimated_pitch,
                            thres=t,
                            idx1=corr_an_idx,
                            idx2=corr_es_idx
                        )

                    total_onsets += len(onsets)
                    n_correct_onsets += ncorr
                    total_correct_notes += n_correct_notes

                    n_detected_tot += n_detected
                    false_positives += fp
                    false_negatives += fn

                    mean_perror = np.append(mean_perror, mean_pitch_err)
                    var_perror = np.append(var_perror, var_pitch_err)

        logger.info('Total number of detected onsets = %i', n_detected_tot)
        logger.info('Original number of onsets = %i', total_onsets)
        logger.info(
            'Onset precision = %f', n_correct_onsets * 100. / n_detected_tot
        )
        logger.info(
            'Onset recall = %f', n_correct_onsets * 100. / total_onsets
        )
        logger.info('Undetected onsets = %i', false_negatives)
        logger.info('correct onsets = %i', n_correct_onsets)
        logger.info('pitch error mean mean = %f hz', np.mean(mean_perror))
        logger.info('pitch error var mean = %f hz', np.var(mean_perror))

        logger.info(
            'Correct notes (pitch and onset) = %i', total_correct_notes
        )
