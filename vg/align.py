import json
import multiprocessing
nthreads = multiprocessing.cpu_count()
import logging

import vg.activations
import numpy as np


resources = None

def from_audio(modelpath, audiopaths, transcripts):
    # Align audio files
    alignments = [ align(audiopath, transcript) for audiopath, transcript in zip(audiopaths, transcripts) ]
    # Get activations
    activations = vg.activations.from_audio(modelpath, audiopaths)
    # Return data
    return phoneme_activations(activations, alignments)

def phoneme_activations(activations, alignments, mfcc=False):
    """Return array of phoneme labels and array of corresponding mean-pooled activation states."""
    labels = []
    states = []
    index = (lambda ms: ms//10) if mfcc else vg.activations.index
    for activation, alignment in zip(activations, alignments):
        # extract phoneme labels and activations for current utterance
        y, X = zip(*list(slices(alignment, activation, index=index)))
        y = np.array(y)
        X = np.stack(X)
        labels.append(y)
        states.append(X)
    return np.concatenate(labels), np.concatenate(states)

    
def on_progress(p):
    for k,v in p.items():
        logging.debug("%s: %s" % (k, v))


def align(audiopath, transcript):
    import gentle
    global resources
    if resources is None:
        resources = gentle.Resources()
    logging.info("converting audio to 8K sampled wav")
    with gentle.resampled(audiopath) as wavfile:
        logging.info("starting alignment")
        aligner = gentle.ForcedAligner(resources, transcript, nthreads=nthreads, disfluency=False, 
                                   conservative=False)
        return json.loads(aliger.transcribe(wavfile, progress_cb=on_progress, logging=logging).to_json())


def slices(utt, rep, index=lambda ms: ms//10, aggregate=lambda x: x.mean(axis=0)):
    """Return sequence of slices associated with phoneme labels, given an
       alignment object `utt`, a representation array `rep`, and
       indexing function `index`, and an aggregating function\
       `aggregate`.
    """
    for phoneme in phones(utt):
        phone, start, end = phoneme
        assert index(start)<index(end)+1, "Something funny: {} {} {} {}".format(start, end, index(start), index(end))
        yield (phone, aggregate(rep[index(start):index(end)+1]))

def phones(utt):
    """Return sequence of phoneme labels associated with start and end
     time corresponding to the alignment JSON object `utt`.
    
    """
    for word in utt['words']:
        pos = word['start']
        for phone in word['phones']:
            start = pos
            end = pos + phone['duration']
            pos = end
            label = phone['phone'].split('_')[0]
            if label != 'oov':
                yield (label, int(start*1000), int(end*1000))

def phoneme_data(nets, alignment_path="./data/flickr8k/dataset.val.fa.json", batch_size=64):
    """Generate data for training a phoneme decoding model."""
    import vg.flickr8k_provider as dp
    logging.getLogger().setLevel('INFO')
    logging.info("Loading alignments")
    data = {}
    for line in open(alignment_path):
        item = json.loads(line)
        data[item['sentid']] = item
    logging.info("Loading audio features")
    prov = dp.getDataProvider('flickr8k', root='.', audio_kind='mfcc')
    val = list(prov.iterSentences(split='val'))
    alignments_all = [ data[sent['sentid']] for sent in val ]
    alignments = [ item for item in alignments_all if np.all([word.get('start', False) for word in item['words']]) ]
    sentids = set(item['sentid'] for item in alignments)
    audio = [ sent['audio'] for sent in val if sent['sentid'] in sentids ]
    result = {}
    logging.info("Computing data for MFCC")
    y, X = phoneme_activations(audio, alignments, mfcc=True)
    result['mfcc'] = fa_data(y, X)
    for name, net in nets:
        logging.info("Computing data for {}".format(name))
        activations = vg.activations.get_state_stack(net, audio, batch_size=batch_size)
        y, X = phoneme_activations(activations, alignments, mfcc=False)
        result[name] = fa_data(y, X)
    return result

def fa_data(y, X):
    # Get rid of NaNs
    # ix = np.isnan(X.sum(axis=1))
    # X = X[~ix]
    #y = y[~ix]
    return dict(features=X, labels=y)
