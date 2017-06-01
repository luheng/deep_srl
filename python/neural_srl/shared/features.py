from constants import *
from dictionary import Dictionary
import reader

def get_srl_features(sentences, config, feature_dicts=None):
  ''' TODO: Support adding more features.
  '''
  feature_names = config.features
  feature_sizes = config.feature_sizes
  use_se_marker = config.use_se_marker
  
  features = []
  feature_shapes = []
  for fname, fsize in zip(feature_names, feature_sizes):
    if fname == "predicate":
      offset = int(use_se_marker)
      features.append([[int(i == sent[1] + offset) for i in range(len(sent[0]))] for sent in sentences])
      feature_shapes.append([2, fsize])
 
  return (zip(*features), feature_shapes)

