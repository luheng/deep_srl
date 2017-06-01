''' Framework independent evaluator. Not in use yet.
'''
import numpy
import os
from os.path import join
import subprocess

from constants import *
from conll_utils import *
from measurements import Timer

class TaggerEvaluator(object):
  def __init__(self, data):
    self.data = data
    self.best_accuracy = 0.0
    self.has_best = False
    
  def compute_accuracy(self, predictions):
    _, y, _, weights = self.data
    num_correct = numpy.sum(numpy.equal(predictions, y) * weights)
    num_total = numpy.sum(weights)
    self.accuracy = (100.0 * num_correct) / num_total
    print("Accuracy: {:.3f} ({}/{})".format(self.accuracy, num_correct, num_total))

  def evaluate(self, predictions):
    self.compute_accuracy(predictions)
    self.has_best = self.accuracy > self.best_accuracy
    if self.has_best:
      print("Best accuracy so far: {:.3f}".format(self.accuracy))
      self.best_accuracy = self.accuracy

class PropIdEvaluator(object):
  def __init__(self, data, label_dict, target_label='V', use_se_marker=False):
    self.data = data
    self.label_dict = label_dict
    self.target_label_id = label_dict.str2idx[target_label]
    self.best_accuracy = 0.0
    self.has_best = False
  
  def compute_accuracy(self, predictions):
    _, y, _, weights = self.data
    identified = numpy.equal(predictions, self.target_label_id) 
    num_correct = numpy.sum(numpy.logical_and(numpy.equal(predictions, y), identified) * weights)
    num_identified = numpy.sum(identified * weights)
    num_gold = numpy.sum(numpy.equal(y, self.target_label_id) * weights)
    self.precision = 100.0 * num_correct / num_identified
    self.recall = 100.0 * num_correct / num_gold
    self.accuracy = 2 * self.precision * self.recall / (self.precision + self.recall)
    print("Accuracy: {:.3f} ({:.3f}, {:.3f})".format(self.accuracy, self.precision, self.recall))

  def evaluate(self, predictions):
    self.compute_accuracy(predictions)
    self.has_best = self.accuracy > self.best_accuracy
    if self.has_best:
      print("Best accuracy so far: {:.3f}".format(self.accuracy))
      self.best_accuracy = self.accuracy


class SRLEvaluator(TaggerEvaluator):
  def __init__(self, data, label_dict,
               gold_props_file=None,
               use_se_marker=False,
               pred_props_file=None, 
               word_dict=None):

    self.data = data
    self.best_accuracy = 0.0
    self.has_best = False
    self.label_dict = label_dict
    self.gold_props_file = gold_props_file
    self.pred_props_file = pred_props_file
    self.use_se_marker = use_se_marker
 
    if gold_props_file is None and pred_props_file is None:
      print ('Warning: not using official gold predicates. Not for formal evaluation.')
      ''' Output to mock gold '''
      assert word_dict != None
      conll_output_path = join(ROOT_DIR, 'temp/srl_pred_%d.gold.tmp' % os.getpid())
      print_gold_to_conll(self.data, word_dict, label_dict, conll_output_path)
      self.pred_props_file = conll_output_path

  def compute_accuracy(self, predictions):
    TaggerEvaluator.compute_accuracy(self, predictions)
    _, _, num_tokens, _ = self.data
    
    self.pred_labels = []
    for pred, slen in zip(predictions,num_tokens):
      pred = pred[1:slen-1] if self.use_se_marker else pred[:slen]
      self.pred_labels.append(bio_to_se([self.label_dict.idx2str[l] for l in pred]))

    temp_output = join(ROOT_DIR, "temp/srl_pred_%d.tmp" % os.getpid())
    print("Printing results to temp file: {}".format(temp_output))
   
    if self.pred_props_file is None: 
      print_to_conll(self.pred_labels, self.gold_props_file, temp_output)
    else:
      print_to_conll(self.pred_labels, self.pred_props_file, temp_output)

    using_pred_props = self.gold_props_file is not None and self.pred_props_file is not None
    gold_path = self.gold_props_file or self.pred_props_file
    eval_script = SRL_CONLL_EVAL_SCRIPT 
    if using_pred_props:
      print "Evaluating with predicted predicates."
      child = subprocess.Popen('sh {} {} {}'.format(eval_script, gold_path, temp_output),
                                shell = True, stdout=subprocess.PIPE)
      eval_info = child.communicate()[0]
      child2 = subprocess.Popen('sh {} {} {}'.format(eval_script, temp_output, gold_path),
                                shell = True, stdout=subprocess.PIPE)
      eval_info2 = child2.communicate()[0]
      try:
        Recall = float(eval_info.strip().split("\n")[6].strip().split()[5])
        Precision = float(eval_info2.strip().split("\n")[6].strip().split()[5])
        Fscore = 2 * Recall * Precision / (Recall + Precision)
        self.accuracy = float(Fscore)
        print(eval_info)
        print(eval_info2)
        print("Combined Precision={}, Recall={}, Fscore={}".format(Precision, Recall, self.accuracy))
      except IndexError:
        print("Unable to get FScore. Skipping.")
    else: 
      child = subprocess.Popen('sh {} {} {}'.format(eval_script, gold_path, temp_output),
                                 shell = True, stdout=subprocess.PIPE)
      eval_info = child.communicate()[0]
      try:
        Fscore = eval_info.strip().split("\n")[6]
        Fscore = Fscore.strip().split()[6]
        self.accuracy = float(Fscore)
        print(eval_info)
        print("Fscore={}".format(self.accuracy))
      except IndexError:
        print("Unable to get FScore. Skipping.")
 
