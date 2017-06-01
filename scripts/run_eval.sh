# Should point to the srlconll library.
SRLPATH="./data/srl"

export PERL5LIB="$SRLPATH/srlconll-1.1/lib:$PERL5LIB"
export PATH="$SRLPATH/srlconll-1.1/bin:$PATH"

perl $SRLPATH/srlconll-1.1/bin/srl-eval.pl $1 $2

