import numpy as np
import time
import sys
import subprocess
import os
import random
import math

from load import load_data, load_emb
from blstm import model
from accuracyweibo import conlleval
from tools import shuffle, minibatch, contextwin
  
if __name__ == '__main__':

    s = {'read_model':False,'seed':300, 'epoch':30, 'lr':0.1, 'decay':0.95, 'wsize':5, 'hnum':100 , 'dnum':100, 'ynum':17, 'wnum':6336, 'L2': 0.000001,
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

    
    nsentences = len(train_word)

    np.random.seed(s['seed'])
    random.seed(s['seed'])

    rnn = model(read_model=s['read_model'],hnum = s['hnum'], ynum = s['ynum'], wnum = s['wnum'], dnum = s['dnum'], wsize = s['wsize'], fsize = s['fsize'], L2 = s['L2'],
                fnum0 = s['f0'], fnum1 = s['f1'], fnum2 = s['f2'], fnum3 = s['f3'], fnum4 = s['f4'], kalpha=s['kalpha'])
    rnn.emb = load_emb("data/embeddingsall")
    s['cur_lr'] = s['lr']

    
    dev_pred = []
    for words, f0i, f1i, f2i, f3i, f4i in zip(dev_word, dev_f0, dev_f1, dev_f2, dev_f3, dev_f4):
        dev_pred += [rnn.classify(contextwin(words, s['wsize']),
                 contextwin(f0i, s['fsize']), contextwin(f1i, s['fsize']), contextwin(f2i, s['fsize']), contextwin(f3i, s['fsize']), contextwin(f4i, s['fsize'])) ]
    res_dev = conlleval(dev_pred, dev_label, dev_word)
    print ""
    for (d,x) in res_dev.items():
        print d + ": " + str(x)
    best_dev = res_dev['namf1score']

    train_weight = list([1.0] for i in range(nsentences))

    total_word = []
    total_word[0:0] = train_word
    total_word[0:0] = sighan_word
    total_label = []
    total_label[0:0] = train_label
    total_label[0:0] = sighan_label
    total_f0 = []
    total_f0[0:0] = train_f0
    total_f0[0:0] = sighan_f0
    total_f1 = []
    total_f1[0:0] = train_f1
    total_f1[0:0] = sighan_f1
    total_f2 = []
    total_f2[0:0] = train_f2
    total_f2[0:0] = sighan_f2
    total_f3 = []
    total_f3[0:0] = train_f3
    total_f3[0:0] = sighan_f3
    total_f4 = []
    total_f4[0:0] = train_f4
    total_f4[0:0] = sighan_f4
    total_weight = []
    total_weight[0:0] = train_weight
    total_weight[0:0] = sighan_simi

    total_yp = [[range(s['ynum']) for j in range(len(total_word[i]))] for i in range(len(total_word))]
    for i in range(len(total_word)):
        for j in range(len(total_word[i])):
            for k in range(s['ynum']):
                if k == total_label[i][j]:
                    total_yp[i][j][k] = 0
                else:
                    total_yp[i][j][k] = 1

    print "start train"
    sys.stdout.flush()
    pre_sighan_label = []
    for e in xrange(s['epoch']):
        pre_label=[]
        pre_confid=[]

        for words, f0i, f1i, f2i, f3i, f4i in zip(sighan_word, sighan_f0, sighan_f1, sighan_f2, sighan_f3, sighan_f4):
          pre_label += [rnn.classify(contextwin(words, s['wsize']),
                                     contextwin(f0i, s['fsize']), contextwin(f1i, s['fsize']),
                                     contextwin(f2i, s['fsize']), contextwin(f3i, s['fsize']),
                                     contextwin(f4i, s['fsize']))]
          
        if len(pre_sighan_label)>5:
              del pre_sighan_label[0]
    
        pre_sighan_label.append(pre_label)
        

        new_pre=list(pre_sighan_label)
        
        for i in range(len(sighan_word)):
            senconfidence=0
            for j in range(len(sighan_word[i])):
              pre_sen_list=[]
              for k in range(len(new_pre)):
                if new_pre[k][i][j] not in pre_sen_list:
                  pre_sen_list.append(new_pre[k][i][j])
              word_confidence=(len(pre_sen_list)*1.0)/len(new_pre)
              senconfidence+=(word_confidence*1.0)/len(sighan_word[i])
            pre_confid+=[senconfidence]
        #print (pre_confid)
    
    
        
        #shuffle
        total_word = []
        total_word[0:0] = train_word
        total_word[0:0] = sighan_word
        total_label = []
        total_label[0:0] = train_label
        total_label[0:0] = sighan_label
        total_f0 = []
        total_f0[0:0] = train_f0
        total_f0[0:0] = sighan_f0
        total_f1 = []
        total_f1[0:0] = train_f1
        total_f1[0:0] = sighan_f1
        total_f2 = []
        total_f2[0:0] = train_f2
        total_f2[0:0] = sighan_f2
        total_f3 = []
        total_f3[0:0] = train_f3
        total_f3[0:0] = sighan_f3
        total_f4 = []
        total_f4[0:0] = train_f4
        total_f4[0:0] = sighan_f4
        total_weight=[]
        #total_weight = [ [x[0]*0.95] for x in total_weight]
        total_weight[0:0] = train_weight
        total_weight[0:0] = [ [sighan_simi[i][0]*math.pow(0.9,e)*pre_confid[i]] for i in range(len(sighan_simi))]
        
        total_yp = [[range(s['ynum']) for j in range(len(total_word[i]))] for i in range(len(total_word))]
        for i in range(len(total_word)):
            for j in range(len(total_word[i])):
                for k in range(s['ynum']):
                    if k == total_label[i][j]:
                        total_yp[i][j][k] = 0
                    else:
                        total_yp[i][j][k] = 1

        
        shuffle([total_word, total_f0, total_f1, total_f2, total_f3, total_f4, total_label, total_weight, total_yp], s['seed'])
        s['cur_epoch'] = e
        tic = time.time()
        for i in xrange(len(total_word)):
            cwords = contextwin(total_word[i], s['wsize'])
            labels = total_label[i]
            feature0 = contextwin(total_f0[i], s['fsize'])
            feature1 = contextwin(total_f1[i], s['fsize'])
            feature2 = contextwin(total_f2[i], s['fsize'])
            feature3 = contextwin(total_f3[i], s['fsize'])
            feature4 = contextwin(total_f4[i], s['fsize'])
            ypi = total_yp[i]
            s['cur_lr'] *= (0.95 ** (1.0 / len(total_word)))
            
            rnn.sentence_train(cwords, feature0, feature1, feature2, feature3, feature4, labels, s['cur_lr'] * total_weight[i][0], ypi)
            #print '[learning epoch %i >> %2.2f%%'%(e, (i+1)*100./len(total_word)), 'completed in %.2f (sec) <<\r'%(time.time()-tic),
            if i%5000==0:
                dev_pred = []
                for words, f0i, f1i, f2i, f3i, f4i in zip(dev_word, dev_f0, dev_f1, dev_f2, dev_f3, dev_f4):
                    dev_pred += [rnn.classify(contextwin(words, s['wsize']),
                                               contextwin(f0i, s['fsize']), contextwin(f1i, s['fsize']),
                                               contextwin(f2i, s['fsize']), contextwin(f3i, s['fsize']),
                                               contextwin(f4i, s['fsize']))]
                res_dev = conlleval(dev_pred, dev_label, dev_word)
                print ("")
                for (d,x) in res_dev.items():
                    print (d + ": " + str(x))
                if res_dev['namf1score'] > best_dev:
                    best_dev = res_dev['namf1score']
                    print ('best score: ',best_dev)
                    rnn.save('model')
                sys.stdout.flush()

        dev_pred = []
        for words, f0i, f1i, f2i, f3i, f4i in zip(dev_word, dev_f0, dev_f1, dev_f2, dev_f3, dev_f4):
            dev_pred += [rnn.classify(contextwin(words, s['wsize']),
                                       contextwin(f0i, s['fsize']), contextwin(f1i, s['fsize']),
                                       contextwin(f2i, s['fsize']), contextwin(f3i, s['fsize']),
                                       contextwin(f4i, s['fsize']))]
        res_dev = conlleval(dev_pred, dev_label, dev_word)
        print ""
        for (d,x) in res_dev.items():
            print d + ": " + str(x)
        fp = open("data/dev_result.txt", 'a')
        fp.write("learning epoch: %d"%e)
        for (d,x) in res_dev.items():
            fp.write(d + ": " + str(x) + "\n")
        fp.close()
        if res_dev['namf1score'] > best_dev:
            best_dev = res_dev['namf1score']
            print ('best score: ',best_dev)
            sys.stdout.flush()
            rnn.save('model')
            fp = open("data/dev_pred.txt", 'w')
            for i in range(len(dev_pred)):
                for j in range(len(dev_pred[i])- 1):
                    fp.write(str(dev_pred[i][j]) + " ")
                fp.write(str(dev_pred[i][len(dev_pred[i])-1]) + "\n")    
            fp.close()
            
           

            '''test_pred = []
            for words, f0i, f1i, f2i, f3i, f4i in zip(test_word, test_f0, test_f1, test_f2, test_f3, test_f4):
                test_pred += [rnn.classify(contextwin(words, s['wsize']),
                                           contextwin(f0i, s['fsize']), contextwin(f1i, s['fsize']),
                                           contextwin(f2i, s['fsize']), contextwin(f3i, s['fsize']),
                                           contextwin(f4i, s['fsize']))]
            res_test = conlleval(test_pred, test_label, test_word)
            fp = open("data/test_pred.txt", 'w')
            for i in range(len(test_pred)):
                for j in range(len(test_pred[i])- 1):
                    fp.write(str(test_pred[i][j]) + " ")
                fp.write(str(test_pred[i][len(test_pred[i])-1]) + "\n")    
            fp.close()

            fp = open("data/test_result.txt", 'w')
            for (d,x) in res_dev.items():
                fp.write(d + ": " + str(x) + "\n")
            fp.close()
'''
