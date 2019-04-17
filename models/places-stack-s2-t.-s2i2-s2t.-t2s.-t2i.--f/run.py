import numpy
import random
seed = 102
random.seed(seed)
numpy.random.seed(seed)

import vg.simple_data as sd
import vg.experiment as E
import vg.places_provider as dp

import vg.defn.three_way_stack as D
import vg.scorer

batch_size = 16
epochs=25

prov_places = dp.getDataProvider('places', root='../..', audio_kind='mfcc')

data_places = sd.SimpleData(prov_places, tokenize=sd.characters, min_df=1, scale=False,
                            batch_size=batch_size, shuffle=True)

model_config = dict(
                    SpeechImage=dict(ImageEncoder=dict(size=1024, size_target=4096),
                                     lr=0.0002,
                                     margin_size=0.2,
                                     max_norm=2.0, 
                                     SpeechEncoderTop=dict(size=1024, size_input=1024, depth=2, size_attn=128)),
                                        
                    SpeechEncoderBottom=dict(size=1024, depth=2, size_vocab=13, filter_length=6, filter_size=64, stride=2),
                    
                   )






def audio(sent):
    return sent['audio']

net = D.Net(model_config)
net.batcher = None
net.mapper = None

scorer = vg.scorer.Scorer(prov_places, 
                    dict(split='val', 
                         tokenize=audio, 
                         batch_size=batch_size
                         ))
                  

run_config = dict(epochs=epochs,
                  validate_period=400,
                  tasks=[ ('SpeechImage', net.SpeechImage) ],
                  Scorer=scorer)
D.experiment(net=net, data=dict(SpeechImage=data_places, SpeechText=data_places, TextImage=data_places), run_config=run_config)

