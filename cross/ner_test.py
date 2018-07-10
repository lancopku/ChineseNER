import numpy as np
import time
import sys
import subprocess
import os
import random

from load import load_data, load_emb
from blstm import model
from accuracyweibooov import conlleval
from tools import shuffle, minibatch, contextwin
  
if __name__ == '__main__':

    s = {'read_model':True,'seed':345, 'epoch':20, 'lr':0.1, 'decay':0.95, 'wsize':5, 'hnum':100 , 'dnum':100, 'ynum':17, 'wnum':6336, 'L2': 0.000001,
         'f0':33716, 'f1':34238, 'f2':34189, 'f3':33768, 'f4':42828, 'fsize':1, 'kalpha':0.2}
    
    print 'load train data'
    train_word = load_data("data/train/train.word.txt")    
    train_label = load_data("data/train/train.label.txt")
    train_f0 = load_data('data/train/trainfeature0.txt')
    train_f1 = load_data('data/train/trainfeature1.txt')
    train_f2 = load_data('data/train/trainfeature2.txt')
    train_f3 = load_data('data/train/trainfeature3.txt')
    train_f4 = load_data('data/train/trainfeature4.txt')

    print 'load sighan data'
    sighan_word = load_data("data/train_sighan/trainsighan.word.txt")    
    sighan_label = load_data("data/train_sighan/trainsighan.label.txt")
    sighan_f0 = load_data('data/train_sighan/trainsighanfeature0.txt')
    sighan_f1 = load_data('data/train_sighan/trainsighanfeature1.txt')
    sighan_f2 = load_data('data/train_sighan/trainsighanfeature2.txt')
    sighan_f3 = load_data('data/train_sighan/trainsighanfeature3.txt')
    sighan_f4 = load_data('data/train_sighan/trainsighanfeature4.txt')
    sighan_simi = load_emb('data/train_sighan/cos.txt')#cross / cos / poly / gaussian

    print 'load dev data'
    dev_word = load_data("data/test/test.word.txt")
    dev_label = load_data("data/test/test.label.txt")
    dev_f0 = load_data('data/test/testfeature0.txt')
    dev_f1 = load_data('data/test/testfeature1.txt')
    dev_f2 = load_data('data/test/testfeature2.txt')
    dev_f3 = load_data('data/test/testfeature3.txt')
    dev_f4 = load_data('data/test/testfeature4.txt')

    print 'load test data'
    test_word = load_data("data/test/test.word.txt")
    test_label = load_data("data/test/test.label.txt")
    test_f0 = load_data('data/test/testfeature0.txt')
    test_f1 = load_data('data/test/testfeature1.txt')
    test_f2 = load_data('data/test/testfeature2.txt')
    test_f3 = load_data('data/test/testfeature3.txt')
    test_f4 = load_data('data/test/testfeature4.txt')
    sys.stdout.flush()
    
    print 'load baseline predict data'
    baseline_label = load_data("data/semi_test_pred.txt")
    for i in range(len(baseline_label)):
        for j in range(len(baseline_label[i])):
            if int(baseline_label[i][j])<=8:
                baseline_label[i][j]=0
            else:
                baseline_label[i][j]=int(baseline_label[i][j])

    
    nsentences = len(train_word)

    np.random.seed(s['seed'])
    random.seed(s['seed'])

    rnn = model(read_model=s['read_model'],hnum = s['hnum'], ynum = s['ynum'], wnum = s['wnum'], dnum = s['dnum'], wsize = s['wsize'], fsize = s['fsize'], L2 = s['L2'],
                fnum0 = s['f0'], fnum1 = s['f1'], fnum2 = s['f2'], fnum3 = s['f3'], fnum4 = s['f4'], kalpha=s['kalpha'])
    #rnn.emb = load_emb("data/embeddingsall")
    s['cur_lr'] = s['lr']

    
    dev_pred = []
    for words, f0i, f1i, f2i, f3i, f4i in zip(dev_word, dev_f0, dev_f1, dev_f2, dev_f3, dev_f4):
        dev_pred += [rnn.classify(contextwin(words, s['wsize']),
                 contextwin(f0i, s['fsize']), contextwin(f1i, s['fsize']), contextwin(f2i, s['fsize']), contextwin(f3i, s['fsize']), contextwin(f4i, s['fsize'])) ]
    
    for i in range(len(baseline_label)):
        for j in range(len(baseline_label[i])):
            if int(baseline_label[i][j])>8:
                dev_pred[i][j]=int(baseline_label[i][j])
            

    res_dev = conlleval(train_label,train_word, dev_pred, dev_label, dev_word)
    print ""
    for (d,x) in res_dev.items():
        print d + ": " + str(x)
    best_dev = res_dev['namf1score']
   
