import sys

domains = [
  ("mz",0,15884),\
  ("nw",15884,164959),\
  ("wb",164959,192499),\
  ("bc",192499,223187),\
  ("bn",223187,248910),\
  ("pt",248910,265323),\
  ("tc",265323,276725)]

line_counter = 0
domain_counter = 0

fin = open(sys.argv[1], 'r')
fout = open(sys.argv[1] + '.' + domains[domain_counter][0], 'w')

for line in fin:
  fout.write(line)
  line_counter += 1
  if line_counter == domains[domain_counter][2]:
    fout.close()
    domain_counter += 1
    if domain_counter < len(domains):
      fout = open(sys.argv[1] + '.' + domains[domain_counter][0], 'w')

