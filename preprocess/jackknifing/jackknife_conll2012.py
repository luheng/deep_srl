# Random shuffle by sentences instead of samples (predicates).
import math
import os
import random
import sys
from os.path import join

def get_sent_to_samples(input_file, domain_file):
  num_samples = 0
  sent2samples = []
  domain2sents = []
  domain_names = []

  fin = open(input_file, 'r')
  fin_domains = open(domain_file, 'r')

  prev_words = ''
  prev_predicate = -1

  for line in fin:
    line = line.strip()
    line_left = line.split('|||')[0]
    pred_id = int(line_left.split()[0])
    words = ' '.join(line_left.split()[1:])

    dname = fin_domains.readline().strip()
    if not (len(domain_names) > 0 and domain_names[-1] == dname):
      domain2sents.append([])
      domain_names.append(dname)

    if not (words == prev_words and pred_id > prev_predicate):
      sent2samples.append([])
      assert (len(domain_names) > 0 and domain_names[-1] == dname)
      num_sents = len(sent2samples) - 1
      domain2sents[-1].append(num_sents)
      
    prev_predicate = pred_id
    prev_words = words
    
    sent2samples[-1].append(num_samples) 
    num_samples += 1
    
  fin.close()
  fin_domains.close()

  return (sent2samples, domain2sents, domain_names)
  
  
def get_sample_to_folds(sent2samples, sent_range, num_folds, dev_sents_pct):
  '''
  '''
  num_sents = sent_range[1] - sent_range[0]
  num_sents_per_fold = int(math.ceil(1.0 * num_sents / num_folds))
  num_samples = sum([len(s) for s in sent2samples[sent_range[0]:sent_range[1]]])
  print "Has %d training samples and %d sentences. Splitting to %d folds with %d sentences each."\
    % (num_samples, num_sents, num_folds, num_sents_per_fold)

  num_dev_sents = int(math.ceil(dev_sents_pct * num_sents_per_fold))
  print "Num. dev sentences: %d." % num_dev_sents
  
  strn = [set() for i in range(num_samples)]
  sdev = [set() for i in range(num_samples)] 
  # prd: the entire heldout set.
  sprd = [set() for i in range(num_samples)]

  num_trn_samples = [0 for i in range(num_folds)]
  num_dev_samples = [0 for i in range(num_folds)]
  num_prd_samples = [0 for i in range(num_folds)]

  train_sents = range(sent_range[0], sent_range[1])
  #random.shuffle(train_sents)
  s0 = sent2samples[sent_range[0]][0]

  print 'Sample id staring at %d' % s0

  for fid in range(num_folds):
    ll = fid * num_sents_per_fold
    rr = min(num_sents, ll + num_sents_per_fold)
    #print fid, ll, rr, rr - ll
  
    for sent_id in train_sents[ll:rr]:
      for sample_id in sent2samples[sent_id]:
        # Assign training folds to sample.
        for fid2 in range(num_folds):
          if fid2 != fid:
            strn[sample_id - s0].add(fid2)
            num_trn_samples[fid2] += 1
        
        # Assign pred folds to sample.
        sprd[sample_id - s0].add(fid)
        num_prd_samples[fid] += 1
      
    prd_sents = train_sents[ll:rr]
    for sent_id in prd_sents[:min(len(prd_sents),num_dev_sents)]:
      # Assign dev folds to sample.
      for sample_id in sent2samples[sent_id]:
        sdev[sample_id - s0].add(fid)
        num_dev_samples[fid] += 1
  
  #print strn[:10]
  #print sdev[:10]
  #print sprd[:10]

  print "Num trn samples:", num_trn_samples
  print "Num prd samples:", num_prd_samples
  print "Num dev samples:", num_dev_samples
  
  return (strn, sdev, sprd)

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
  DEV_SENTS_PCT = 0.3

  input_file = sys.argv[1]
  domain_file = sys.argv[2]
  output_dir = sys.argv[3]
  
  sent2samples, domain2sents, domain_names = get_sent_to_samples(input_file, domain_file)
  print 'Totol samples: ', sum([len(s) for s in sent2samples])
 
  sample2fold_trn = []
  sample2fold_dev = []
  sample2fold_prd = []
  random.seed(RANDOM_SEED)
  for dname, in_domain_sents in zip(domain_names, domain2sents):
    sent_range = [in_domain_sents[0], in_domain_sents[-1] + 1]
    print dname, sent_range
    strn, sdev, sprd = get_sample_to_folds(sent2samples, sent_range, NUM_FOLDS, DEV_SENTS_PCT)  
    sample2fold_trn.extend(strn)
    sample2fold_dev.extend(sdev)
    sample2fold_prd.extend(sprd)

  print len(sample2fold_trn), len(sample2fold_dev), len(sample2fold_prd)

  # Output ids
  fout_trn_ids = [open(join(output_dir, 'train.f%02d.ids'%fid), 'w') for fid in range(NUM_FOLDS)]
  fout_dev_ids = [open(join(output_dir, 'devel.f%02d.ids'%fid), 'w') for fid in range(NUM_FOLDS)]
  fout_prd_ids = [open(join(output_dir + 'pred.f%02d.ids'%fid), 'w') for fid in range(NUM_FOLDS)]
  
  num_samples = len(sample2fold_trn)
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
  

