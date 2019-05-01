import vg.flickr8k_provider as dp
import vg.activations
import torch
import numpy as np

def main():
    
    prov = dp.getDataProvider('flickr8k', root='.', audio_kind='mfcc')
    sent = list(prov.iterSentences(split='val'))[:1000]
    net = torch.load("models/stack-s2-t.-s2i2-s2t.-t2s.-t2i.--f/model.23.pkl")
    audio = [ s['audio'] for s in sent ]
    trans = [ s['raw'] for s in sent ]
    stack = vg.activations.get_state_stack(net, audio, batch_size=16)
    np.save("state_stack_flickr8k_val.npy", stack)
    np.save("transcription_flickr8k_val.npy", trans)



main()

