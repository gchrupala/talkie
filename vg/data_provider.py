class Provider:

  def __init__(self, dataset, root='.', audio_kind='mfcc'):
    raise NotImplementedError
    
  def iterImages(self, split='train', shuffle=False):
    """Yield image data. Each datum is a dictionary with the following structure:
       - sentences: list of sentence data
       - imgid: Image ID
       - feat: numpy array with image features
    """
    raise NotImplementedError

  def iterSentences(self, split='train', shuffle=False):
    """Yield sentence data. Each datum is a dictionary with the following structure:
        - tokens: list of strings (words)
        - raw: a string with the text of the utterance
        - imgid: ID of the image which this sentence describes
        - audio: numpy array with utterance MFCC features
        - sentid: utterance ID
        - speaker: speaker ID
    """
    raise NotImplementedError

def getDataProvider(*args, **kwargs):
	return Provider(*args, **kwargs)

