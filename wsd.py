#!/usr/bin/env python

# Naive-Bayes word sense disambiguation by Ghalib Suleiman <ghalib@sent.com>

# P(feature_word | sense) is computed using MLE with add-lambda
# smoothing (where lambda = 0.5 after experimentation).  See README
# for other details.

# Scroll down to main() function for usage.

import sys
import string
import subprocess
import os
from itertools import izip
from math import log
from collections import defaultdict

def strip_punctuation(s):
    return s.translate(string.maketrans('', ''), string.punctuation)

def tokenise(context):
    return strip_punctuation(context.strip().lower()).split()

class WSD:
    def __init__(self, countsfile):
        self.item_counts = self._read_word_counts(countsfile)
        self.c = 0.5               # add-lambda smoothing

        # C(w, s), number of occurences of word w in context of sense
        # s
        self.ctxword_sense_counts = defaultdict(lambda: defaultdict(lambda: 0))

        # table of item-senses pairs
        self.item_senses = defaultdict(set)

        # count instances of each sense
        self.sense_counts = defaultdict(int)

        # table of P(ctxword | sense)
        self.prob_ctxword_sense = defaultdict(lambda: defaultdict(float))

        # table of P(sense) = C(sense)/C(item)
        self.sense_probs = defaultdict(float)

        # denominator sum in MLE estimate
        self.total_ctxword_sense_counts = {}

        # results file for classification
        self.outfile = None

    def _read_word_counts(self, countsfile):
        """Read word counts from supplied word counts file (in this
        example we have countsfile = EnglishLS.words)."""
        word_counts = defaultdict(int)
        f = open(countsfile)
        for x in range(3):              # skip first three lines
            f.readline()
            
        for line in f:
            line = line.strip()
            if line.startswith('-'):
                break
            else:
                tokens = line.split()
                word = tokens[0]
                count = int(tokens[1])
                word_counts[word] = count
        return word_counts

    def _sum_ctxword_sense_counts(self, item, vocabulary):
        """Count total number of words that appear in contexts of
        sense.  This is the denominator in our MLE estimation for
        P(feature_word | sense).  Smoothing is applied."""
        for sense in self.item_senses[item]:
            total = sum(self.ctxword_sense_counts[sense].values()) + (self.c * len(vocabulary))
            self.total_ctxword_sense_counts[sense] = total

        return self.total_ctxword_sense_counts
                
    def _start(self, instancefile, train=True):
        """Process instancefile that has been converted from SENSEVAL
        format using tidy.py.  Train if train==True, else classify."""
        item = ''
        instance_id = ''
        left_context = ''
        right_context = ''
        context = ''
        senses = []
        vocabulary = set()

        f = open(instancefile)
        for line in f:
            line_units = [unit.strip() for unit in line.split('=')]
            line_label = line_units[0]

            if (line_label == 'ITEM'):
                item = line_units[1]

            elif (line_label == 'SENSES'):
                senses = line_units[1].split()

            elif (line_label == 'INSTANCE_ID'):
                instance_id = line_units[1]
                
            elif (line_label == 'LEFT_CONTEXT'):
                left_context = tokenise(line_units[1])

            elif (line_label == 'RIGHT_CONTEXT'):
                right_context = tokenise(line_units[1])
                context = left_context + right_context

            elif (line_label == 'END_INSTANCE'):
                # process instance
                if train:
                    sense = senses[0]
                    # maintain list of senses for each word item we
                    # train on.
                    self.item_senses[item].add(sense)
                    self.sense_counts[sense] += 1
                    
                    for ctxword in context:
                        vocabulary.add(ctxword)
                        self.ctxword_sense_counts[sense][ctxword] += 1
                else:
                    # classify
                    score = defaultdict(float)
                    for sense in self.item_senses[item]:
                        score[sense] = log(self.sense_probs[sense])
                        for ctxword in context:
                            # Smooth zero-probabilities
                            if (self.prob_ctxword_sense[ctxword][sense] == 0):
                                self.prob_ctxword_sense[ctxword][sense] = self.c / self.total_ctxword_sense_counts[sense]
                            score[sense] += log(self.prob_ctxword_sense[ctxword][sense])

                    best_sense = max(score.keys(), key=lambda s: score[s])
                    self.outfile.write('%s %s %s\n' % (item, instance_id,
                                                       best_sense))
                    
            elif (line_label == 'END_ITEM'):
                # process item after having seen all its instances
                if train:
                    print 'processing', item
                    # Compute denominator sum for MLE
                    self.total_ctxword_sense_counts = self._sum_ctxword_sense_counts(item, vocabulary)
                    for sense in self.item_senses[item]:
                        # Computer P(sense)
                        self.sense_probs[sense] = float(self.sense_counts[sense]) / self.item_counts[item]
                        for ctxword in vocabulary:
                            # Estimate P(feature_word | sense) using
                            # MLE, with smoothing.
                            self.prob_ctxword_sense[ctxword][sense] = (self.ctxword_sense_counts[sense][ctxword] + self.c)  / self.total_ctxword_sense_counts[sense]
                        
                    
                    vocabulary.clear()

    def train(self, trainfile):
        print 'Training...'
        self._start(trainfile)

    def classify(self, instancefile, outfile):
        sys.stdout.write('Classifying...')
        sys.stdout.flush()
        self.outfile = open(outfile, 'w')
        self._start(instancefile, False)
        self.outfile.flush()
        sys.stdout.write('Done.\n')
        sys.stdout.write('Results written to file \'%s\'\n' % outfile)
        sys.stdout.flush()

if __name__ == "__main__":
    
    countsfile, trainfile, testfile, resultsfile, answerfile = '', '', '', '', ''

    if (len(sys.argv) >= 5):
        countsfile, trainfile, testfile, resultsfile = sys.argv[1:]
    else:
        countsfile = 'EnglishLS.words'
        trainfile = 'EnglishLS.gtrain'
        testfile = 'EnglishLS.gtest'
        resultsfile = 'results'
        answerfile = 'EnglishLS.test.key'

    W = WSD(countsfile)
    W.train(trainfile)
    W.classify(testfile, resultsfile)

    if 'sense_scorer' not in os.listdir('.'):
        raise Exception("Please run make to compile the scorer.")
    args = ('./sense_scorer', resultsfile, answerfile)
    print subprocess.Popen(args, stdout=subprocess.PIPE).communicate()[0]
