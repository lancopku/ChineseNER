import numpy
import pdb
import cPickle
import random
import os
import stat
import subprocess
from os.path import isfile, join
from os import chmod
from load import load_data


def conlleval(tg, tw, p, g, w):
    '''
    INPUT
    p :: predictions
    g :: groundtruth
    w :: corresonding words
    '''
    totallabel = 0.0
    correctlabel = 0.0
    namcorrectnum = 0.0
    nampred_list = []
    namground_list = []
    nomcorrectnum = 0.0
    nompred_list = []
    nomground_list = []
    train_list = []
    oovpred_list = []
    oovground_list = []
    oovcorrectnum = 0.0
    for stg, stw in zip(tg, tw):
        outg = ''
        for wtg, wtw in zip(stg, stw):
            if wtg % 2 == 1:
                if len(outg) != 0:
                    train_list[0:0] = [outg]
                outg = str(wtw)
                outg += '+'
            elif wtg != 0:
                outg += str(wtw)
                outg += '+'
            else:
                if len(outg) != 0:
                    train_list[0:0] = [outg]
                outg = ''
    for sg, sp, sw in zip(g, p, w):
        for wg, wp in zip(sg, sp):
            totallabel += 1
            if wg == wp:
                correctlabel += 1
        outg = '' 
        outp = ''
        wgl = 0
        wpl = 0
        for wg, wp, ww in zip(sg, sp, sw):
            if wg % 2 == 1:
                if len(outg) != 0:
                    if outg not in train_list:
                        oovground_list[0:0] = [outg]
                    if wg <= 8:
                        namground_list[0:0] = [outg]
                    else:
                        nomground_list[0:0] = [outg]
                else:
                    pass
                outg = str(ww)
                outg += '+'
            elif wg != 0:
                outg += str(ww)
                outg += '+'
            else:
                if len(outg) != 0:
                    if outg not in train_list:
                        oovground_list[0:0] = [outg]
                    if wgl <= 8:
                        namground_list[0:0] = [outg]
                    else:
                        nomground_list[0:0] = [outg]
                outg = ''
            if wp % 2 == 1:
                if len(outp) != 0:
                    if outp not in train_list:
                        oovpred_list[0:0] = [outp]
                        if (len(oovground_list) > 0) and (outp == oovground_list[0]):
                            oovcorrectnum += 1
                    if wp <= 8:
                        nampred_list[0:0] = [outp]
                        if (len(namground_list) > 0) and (outp == namground_list[0]):
                            namcorrectnum += 1
                    else:
                        nompred_list[0:0] = [outp]
                        if (len(nomground_list) > 0) and (outp == nomground_list[0]):
                            nomcorrectnum += 1

                else:
                    pass
                outp = str(ww)
                outp += '+'
            elif wp != 0:
                outp += str(ww)
                outp += '+'
            else:
                if len(outp) != 0:
                    if outp not in train_list:
                        oovpred_list[0:0] = [outp]
                        if (len(oovground_list) > 0) and (outp == oovground_list[0]):
                            oovcorrectnum += 1
                    if wpl <= 8:
                        nampred_list[0:0] = [outp]
                        if (len(namground_list)>0) and (outp == namground_list[0]):
                            namcorrectnum += 1
                    else:
                        nompred_list[0:0] = [outp]
                        if (len(nomground_list) > 0) and (outp == nomground_list[0]):
                            nomcorrectnum += 1
                outp = ''
            wgl = wg
            wpl = wp
    
    
    oovprednum = len(oovpred_list)
    oovgroundnum = len(oovground_list)
    oovprecision = (oovcorrectnum / (oovprednum + 0.000001)) * 100
    oovrecall = (oovcorrectnum / (oovgroundnum + 0.0000001)) * 100
    oovf1score = 2 * oovprecision * oovrecall / (oovprecision + oovrecall + 0.000000000001)
    namprednum = len(nampred_list)
    namgroundnum = len(namground_list)
    namprecision = (namcorrectnum / (namprednum + 0.000001)) * 100
    namrecall = (namcorrectnum / (namgroundnum + 0.000001)) * 100
    namf1score = 2 * namprecision * namrecall / (namprecision + namrecall + 0.00000001)
    nomprednum = len(nompred_list)
    nomgroundnum = len(nomground_list)
    nomprecision = (nomcorrectnum / (nomprednum + 0.000001)) * 100
    nomrecall = (nomcorrectnum / (nomgroundnum + 0.000001)) * 100
    nomf1score = 2 * nomprecision * nomrecall / (nomprecision + nomrecall + 0.000000001)
    label_prec = correctlabel / totallabel * 100
    micro_f1score = (namf1score * namgroundnum + nomf1score * nomgroundnum) / (namgroundnum + nomgroundnum)
    return {'namprecision':namprecision, 'namrecall':namrecall, 'namf1score':namf1score,
            'nomprecision': nomprecision, 'nomrecall': nomrecall, 'nomf1score':nomf1score,'label_prec':label_prec,
            'namgroundnum': namgroundnum, 'namprednum': namprednum, 'nomgroundnum': nomgroundnum, 'nomprednum':nomprednum,
            'micro_f1score': micro_f1score, 'oovprednum': oovprednum, 'oovgroundnum': oovgroundnum, 'oovprecision': oovprecision,
            'oovrecall': oovrecall, 'oovf1score':oovf1score}

if __name__ == '__main__':
    train_word = load_data('train.word.txt')
    train_label = load_data('train.label.txt')
    test_pred = load_data('test_pred.txt')
    test_label = load_data('test.label.txt')
    test_word = load_data('test.word.txt')
    res_test = conlleval(train_label, train_word,test_pred, test_label, test_word)
    print ""
    for (d,x) in res_test.items():
        print d + ": " + str(x)
    fp = open('oov_res.txt', 'w')
    for (d,x) in res_test.items():
        fp.write(d + ": " + str(x) + "\n")
    fp.close()
    

