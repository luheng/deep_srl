# Random shuffle by sentences instead of samples (predicates).
import math
import os
import random
import sys
from os.path import join

def get_sent_to_samples(input_file):
  num_samples = 0
  sent2samples = []
  fin = open(input_file, 'r')
  prev_words = ""
  prev_predicate = -1

  for line in fin:
    line = line.strip()
    line_left = line.split('|||')[0]
    pred_id = int(line_left.split()[0])
    words = ' '.join(line_left.split()[1:]
             )
    if not(words == prev_words and pred_id > prev_predicate):
      sent2samples.append([])
      
    prev_predicate = pred_id
    prev_words = words
    
    sent2samples[-1].append(num_samples) 
    num_samples += 1
    
  fin.close()
  return (sent2samples, num_samples)
  
  
def get_sample_to_folds(sent2samples, num_folds, max_num_dev_sents):
  num_sents = len(sent2samples)
  num_sents_per_fold = int(math.ceil(1.0 * num_sents / num_folds))
  print "Read %d training samples and %d sentences. Splitting to %d folds with %d sentences each."\
    % (num_samples, num_sents, num_folds, num_sents_per_fold)
  
  sample2fold_trn = [set() for i in range(num_samples)]
  sample2fold_dev = [set() for i in range(num_samples)] 
  # prd: the entire heldout set.
  sample2fold_prd = [set() for i in range(num_samples)]

  num_dev_sents = [0 for i in range(num_folds)]
  num_trn_samples = [0 for i in range(num_folds)]
  num_dev_samples = [0 for i in range(num_folds)]
  num_prd_samples = [0 for i in range(num_folds)]
  
  for fid in range(num_folds):
    ll = fid * num_sents_per_fold
    rr = min(num_sents, ll + num_sents_per_fold)
    print fid, ll, rr, rr - ll
  
    for i in range(ll, rr):
      sent_id = sent_ids[i]
      for sample_id in sent2samples[sent_id]:
        # Assign training folds to sample.
        for fid2 in range(num_folds):
          if fid2 != fid:
            sample2fold_trn[sample_id].add(fid2)
            num_trn_samples[fid2] += 1
        
        # Assign pred folds to sample.
        sample2fold_prd[sample_id].add(fid)
        num_prd_samples[fid] += 1
        
    prd_sents = range(ll, rr)
    random.shuffle(prd_sents)
    for i in range(min(len(prd_sents), max_num_dev_sents)):
      sent_id = prd_sents[i]
      # Assign dev folds to sample.
      for sample_id in sent2samples[sent_id]:
        sample2fold_dev[sample_id].add(fid)
        num_dev_samples[fid] += 1
      num_dev_sents[fid] += 1
  
  print sample2fold_trn[:10]
  print sample2fold_dev[:10]
  print sample2fold_prd[:10]

  print "Num trn samples:", num_trn_samples
  print "Num prd samples:", num_prd_samples
  print "Num dev samples:", num_dev_samples
  
  return (sample2fold_trn, sample2fold_dev, sample2fold_prd)

def split_file(input_file, output_files, sample2fold):
  fin = open(input_file, 'r')
  fout = [open(fn, 'w') for fn in output_files]
  
  sample_id = 0
  for line in fin:
    for fid in sample2fold[sample_id]:
      fout[fid].write(line.strip() + "\n")
    sample_id += 1
    
  fin.close()  
  for fo in fout:
    fo.close()
    
if __name__ == '__main__':
  RANDOM_SEED = 12345
  NUM_FOLDS = 5
  MAX_NUM_DEV_SENTS = 5000

  input_file = sys.argv[1]
  output_dir = sys.argv[2]
  
  sent2samples, num_samples = get_sent_to_samples(input_file)
  num_sents = len(sent2samples)
  sent_ids = range(num_sents)

  random.seed(RANDOM_SEED)
  random.shuffle(sent_ids)

  sample2fold_trn, sample2fold_dev, sample2fold_prd = get_sample_to_folds(sent2samples,
                                      NUM_FOLDS,
                                      MAX_NUM_DEV_SENTS)

  # Output ids
  fout_trn_ids = [open(join(output_dir, 'train.f%02d.ids'%fid), 'w') for fid in range(NUM_FOLDS)]
  fout_dev_ids = [open(join(output_dir, 'devel.f%02d.ids'%fid), 'w') for fid in range(NUM_FOLDS)]
  fout_prd_ids = [open(join(output_dir + 'pred.f%02d.ids'%fid), 'w') for fid in range(NUM_FOLDS)]
  
  for sid in range(num_samples):
    for fid in sample2fold_trn[sid]:
      fout_trn_ids[fid].write("%d\n" % sid)
    for fid in sample2fold_dev[sid]:
      fout_dev_ids[fid].write("%d\n" % sid)
    for fid in sample2fold_prd[sid]:
      fout_prd_ids[fid].write("%d\n" % sid)
      
  for fo in fout_trn_ids + fout_dev_ids + fout_prd_ids:
    fo.close()
  
  # Generate output files.
  filename = input_file.split('/')[-1].split('.')[0]
  print filename
  
  split_file(input_file,
         [join(output_dir, '%s.train.f%02d.txt'%(filename,fid)) for fid in range(NUM_FOLDS)],
         sample2fold_trn)
  
  split_file(input_file,
         [join(output_dir, '%s.devel.f%02d.txt'%(filename,fid)) for fid in range(NUM_FOLDS)],
         sample2fold_dev)
  
  split_file(input_file,
         [join(output_dir, '%s.pred.f%02d.txt'%(filename,fid)) for fid in range(NUM_FOLDS)],
         sample2fold_prd)
 
