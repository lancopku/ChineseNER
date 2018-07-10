
import theano
import numpy
import os
import theano.typed_list

from theano import tensor as T
from collections import OrderedDict
import pickle
class model(object):
    
    def __init__(self, read_model, hnum, ynum, wnum, dnum, wsize, fsize, L2, fnum0, fnum1, fnum2, fnum3, fnum4, kalpha):
        '''
        hnum :: dimension of the hidden layer
        ynum :: number of classes
        wnum :: number of word embeddings in the vocabulary
        dnum :: dimension of the word embeddings
        wsize :: word window context size 
		L2 :: scale of L2
        '''
        # parameters of the model
       # self.Aij = theano.shared()
        if read_model:
            print ('reading model')
            flags=""
            self.path_model_data="cross/"
            self.emb = pickle.load(open(self.path_model_data+flags+'model/'+'embeddings'+'.pkl', 'r'))
            self.A = pickle.load(open(self.path_model_data+flags+'model/'+'A'+'.pkl', 'r'))
            self.f0 = pickle.load(open(self.path_model_data+flags+'model/'+'f0'+'.pkl', 'r'))
            self.f1 = pickle.load(open(self.path_model_data+flags+'model/'+'f1'+'.pkl', 'r'))
            self.f2 = pickle.load(open(self.path_model_data+flags+'model/'+'f2'+'.pkl', 'r'))
            self.f3 = pickle.load(open(self.path_model_data+flags+'model/'+'f3'+'.pkl', 'r'))
            self.f4 = pickle.load(open(self.path_model_data+flags+'model/'+'f4'+'.pkl', 'r'))
            self.W_xil = pickle.load(open(self.path_model_data+flags+'model/'+'W_xil'+'.pkl', 'r'))
            self.W_hil  = pickle.load(open(self.path_model_data+flags+'model/'+'W_hil'+'.pkl', 'r'))
            self.W_xol = pickle.load(open(self.path_model_data+flags+'model/'+'W_xol'+'.pkl', 'r'))
            self.W_hol  = pickle.load(open(self.path_model_data+flags+'model/'+'W_hol'+'.pkl', 'r'))
            self.W_xcl = pickle.load(open(self.path_model_data+flags+'model/'+'W_xcl'+'.pkl', 'r'))
            self.W_hcl  = pickle.load(open(self.path_model_data+flags+'model/'+'W_hcl'+'.pkl', 'r'))
            self.W_xfl = pickle.load(open(self.path_model_data+flags+'model/'+'W_xfl'+'.pkl', 'r'))
            self.W_hfl  = pickle.load(open(self.path_model_data+flags+'model/'+'W_hfl'+'.pkl', 'r'))
            self.W_xir = pickle.load(open(self.path_model_data+flags+'model/'+'W_xir'+'.pkl', 'r'))
            self.W_hir  = pickle.load(open(self.path_model_data+flags+'model/'+'W_hir'+'.pkl', 'r'))
            self.W_xor = pickle.load(open(self.path_model_data+flags+'model/'+'W_xor'+'.pkl', 'r'))
            self.W_hor  = pickle.load(open(self.path_model_data+flags+'model/'+'W_hor'+'.pkl', 'r'))
            self.W_xcr = pickle.load(open(self.path_model_data+flags+'model/'+'W_xcr'+'.pkl', 'r'))
            self.W_hcr  = pickle.load(open(self.path_model_data+flags+'model/'+'W_hcr'+'.pkl', 'r'))
            self.W_xfr = pickle.load(open(self.path_model_data+flags+'model/'+'W_xfr'+'.pkl', 'r'))
            self.W_hfr  = pickle.load(open(self.path_model_data+flags+'model/'+'W_hfr'+'.pkl', 'r'))
            self.W_hy   = pickle.load(open(self.path_model_data+flags+'model/'+'W_hy'+'.pkl', 'r'))
            self.bil  = pickle.load(open(self.path_model_data+flags+'model/'+'bil'+'.pkl', 'r'))
            self.bol  = pickle.load(open(self.path_model_data+flags+'model/'+'bol'+'.pkl', 'r'))
            self.bcl  = pickle.load(open(self.path_model_data+flags+'model/'+'bcl'+'.pkl', 'r'))
            self.bfl  = pickle.load(open(self.path_model_data+flags+'model/'+'bfl'+'.pkl', 'r'))
            self.bir  = pickle.load(open(self.path_model_data+flags+'model/'+'bir'+'.pkl', 'r'))
            self.bor  = pickle.load(open(self.path_model_data+flags+'model/'+'bor'+'.pkl', 'r'))
            self.bcr  = pickle.load(open(self.path_model_data+flags+'model/'+'bcr'+'.pkl', 'r'))
            self.bfr  = pickle.load(open(self.path_model_data+flags+'model/'+'bfr'+'.pkl', 'r'))
            self.by   = pickle.load(open(self.path_model_data+flags+'model/'+'by'+'.pkl', 'r'))
            self.h0l  = pickle.load(open(self.path_model_data+flags+'model/'+'h0l'+'.pkl', 'r'))
            self.s0l  = pickle.load(open(self.path_model_data+flags+'model/'+'s0l'+'.pkl', 'r'))
            self.h0r  = pickle.load(open(self.path_model_data+flags+'model/'+'h0r'+'.pkl', 'r'))
            self.s0r  = pickle.load(open(self.path_model_data+flags+'model/'+'s0r'+'.pkl', 'r'))
        else:
            self.emb = theano.shared(0.1 * numpy.random.normal(0.0, 1.0,\
                       (wnum+1, dnum)).astype(theano.config.floatX)) # add one for PADDING at the end
            self.A = theano.shared(0.1 * numpy.random.normal(0.0, 1.0, \
                        (ynum, ynum)).astype(theano.config.floatX))  # add one for PADDING at the end
            self.f0 = theano.shared(0.1 * numpy.random.normal(0.0, 1.0,\
                       (fnum0+1, dnum)).astype(theano.config.floatX))
            self.f1 = theano.shared(0.1 * numpy.random.normal(0.0, 1.0, \
                       (fnum1 + 1, dnum)).astype(theano.config.floatX))
            self.f2 = theano.shared(0.1 * numpy.random.normal(0.0, 1.0, \
                       (fnum2 + 1, dnum)).astype(theano.config.floatX))
            self.f3 = theano.shared(0.1 * numpy.random.normal(0.0, 1.0, \
                       (fnum3 + 1, dnum)).astype(theano.config.floatX))
            self.f4 = theano.shared(0.1 * numpy.random.normal(0.0, 1.0, \
                       (fnum4 + 1, dnum)).astype(theano.config.floatX))
            self.W_xil = theano.shared(0.01 * numpy.random.normal(0.0, 1.0,\
                       (dnum * (wsize + 5), hnum)).astype(theano.config.floatX))
            self.W_hil  = theano.shared(0.01 * numpy.random.normal(0.0, 1.0,\
                       (hnum, hnum)).astype(theano.config.floatX))
            self.W_xol = theano.shared(0.01 * numpy.random.normal(0.0,1.0,\
                       (dnum * (wsize + 5), hnum)).astype(theano.config.floatX))
            self.W_hol  = theano.shared(0.01 * numpy.random.normal(0.0, 1.0,\
                       (hnum, hnum)).astype(theano.config.floatX))
            self.W_xcl = theano.shared(0.01 * numpy.random.normal(0.0, 1.0,\
                       (dnum * (wsize + 5), hnum)).astype(theano.config.floatX))
            self.W_hcl  = theano.shared(0.01 * numpy.random.normal(0.0, 1.0,\
                       (hnum, hnum)).astype(theano.config.floatX))
            self.W_xfl = theano.shared(0.01 * numpy.random.normal(0.0, 1.0,\
                       (dnum * (wsize + 5), hnum)).astype(theano.config.floatX))
            self.W_hfl  = theano.shared(0.01 * numpy.random.normal(0.0, 1.0,\
                       (hnum, hnum)).astype(theano.config.floatX))
            self.W_xir = theano.shared(0.01 * numpy.random.normal(0.0, 1.0,\
                       (dnum * (wsize + 5), hnum)).astype(theano.config.floatX))
            self.W_hir  = theano.shared(0.01 * numpy.random.normal(0.0, 1.0,\
                       (hnum, hnum)).astype(theano.config.floatX))
            self.W_xor = theano.shared(0.01 * numpy.random.normal(0.0, 1.0,\
                       (dnum * (wsize + 5), hnum)).astype(theano.config.floatX))
            self.W_hor  = theano.shared(0.01 * numpy.random.normal(0.0, 1.0,\
                       (hnum, hnum)).astype(theano.config.floatX))
            self.W_xcr = theano.shared(0.01 * numpy.random.normal(0.0, 1.0,\
                       (dnum * (wsize + 5), hnum)).astype(theano.config.floatX))
            self.W_hcr  = theano.shared(0.01 * numpy.random.normal(0.0, 1.0,\
                       (hnum, hnum)).astype(theano.config.floatX))
            self.W_xfr = theano.shared(0.01 * numpy.random.normal(0.0, 1.0,\
                       (dnum * (wsize + 5), hnum)).astype(theano.config.floatX))
            self.W_hfr  = theano.shared(0.01 * numpy.random.normal(0.0, 1.0,\
                       (hnum, hnum)).astype(theano.config.floatX))
            self.W_hy   = theano.shared(0.01 * numpy.random.normal(0.0, 1.0,\
                       (hnum * 2, ynum)).astype(theano.config.floatX))
            self.bil  = theano.shared(numpy.zeros(hnum, dtype=theano.config.floatX))
            self.bol  = theano.shared(numpy.zeros(hnum, dtype=theano.config.floatX))
            self.bcl  = theano.shared(numpy.zeros(hnum, dtype=theano.config.floatX))
            self.bfl  = theano.shared(numpy.zeros(hnum, dtype=theano.config.floatX))
            self.bir  = theano.shared(numpy.zeros(hnum, dtype=theano.config.floatX))
            self.bor  = theano.shared(numpy.zeros(hnum, dtype=theano.config.floatX))
            self.bcr  = theano.shared(numpy.zeros(hnum, dtype=theano.config.floatX))
            self.bfr  = theano.shared(numpy.zeros(hnum, dtype=theano.config.floatX))
            self.by   = theano.shared(numpy.zeros(ynum, dtype=theano.config.floatX))
            self.h0l  = theano.shared(numpy.zeros(hnum, dtype=theano.config.floatX))
            self.s0l  = theano.shared(numpy.zeros(hnum, dtype=theano.config.floatX))
            self.h0r  = theano.shared(numpy.zeros(hnum, dtype=theano.config.floatX))
            self.s0r  = theano.shared(numpy.zeros(hnum, dtype=theano.config.floatX))

        # bundle
        self.params = [ self.emb, self.f0, self.f1, self.f2, self.f3, self.f4, self.A, self.W_xil, self.W_hil, self.W_xol, self.W_hol, self.W_xcl, self.W_hcl, self.W_xfl, self.W_hfl, self.W_hy,
            self.W_xir, self.W_hir, self.W_xor, self.W_hor, self.W_xcr, self.W_hcr, self.W_xfr, self.W_hfr,
            self.bil, self.bol, self.bcl, self.bfl, self.by, self.h0l, self.s0l,
            self.bir, self.bor, self.bcr, self.bfr, self.h0r, self.s0r]
        self.names  = ['embeddings', 'f0', 'f1', 'f2', 'f3','f4','A','W_xil', 'W_hil', 'W_xol', 'W_hol', 'W_xcl', 'W_hcl', 'W_xfl', 'W_hfl', 'W_hy',
			'W_xir', 'W_hir', 'W_xor', 'W_hor', 'W_xcr', 'W_hcr', 'W_xfr', 'W_hfr', 'bil', 'bol', 'bcl', 'bfl', 'by', 'h0l', 's0l',
			'bir', 'bor', 'bcr', 'bfr', 'h0r', 's0r']
        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        f0idxs = T.imatrix()
        f1idxs = T.imatrix()
        f2idxs = T.imatrix()
        f3idxs = T.imatrix()
        f4idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], dnum * wsize))
        x0 = self.f0[f0idxs].reshape((f0idxs.shape[0], dnum * fsize))
        x1 = self.f1[f1idxs].reshape((f1idxs.shape[0], dnum * fsize))
        x2 = self.f2[f2idxs].reshape((f2idxs.shape[0], dnum * fsize))
        x3 = self.f3[f3idxs].reshape((f3idxs.shape[0], dnum * fsize))
        x4 = self.f4[f4idxs].reshape((f4idxs.shape[0], dnum * fsize))
        x = T.concatenate([x, x0], axis=1)
        x = T.concatenate([x, x1], axis=1)
        x = T.concatenate([x, x2], axis=1)
        x = T.concatenate([x, x3], axis=1)
        x = T.concatenate([x, x4], axis=1)
        y_sentence = T.ivector('y_sentence') # labels
        y_p = T.imatrix('y_p')
        #self.getx = theano.function(inputs=[idxs, f0idxs, f1idxs, f2idxs, f3idxs, f4idxs], outputs=x)

        def recurrencel(x_t, h_tm1, s_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xil) + T.dot(h_tm1, self.W_hil) + self.bil)
            o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xol) + T.dot(h_tm1, self.W_hol) + self.bol)
            c_t = T.tanh(T.dot(x_t, self.W_xcl) + T.dot(h_tm1, self.W_hcl) + self.bcl)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xfl) + T.dot(h_tm1, self.W_hfl) + self.bfl)
            s_t = f_t * s_tm1 + i_t * c_t
            h_t = o_t * T.tanh(s_t)
            #y_t = T.nnet.softmax(T.dot(h_t, self.W_hy) + self.by)
            return [h_t, s_t]

        def recurrencer(x_t, h_tm1, s_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xir) + T.dot(h_tm1, self.W_hir) + self.bir)
            o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xor) + T.dot(h_tm1, self.W_hor) + self.bor)
            c_t = T.tanh(T.dot(x_t, self.W_xcr) + T.dot(h_tm1, self.W_hcr) + self.bcr)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xfr) + T.dot(h_tm1, self.W_hfr) + self.bfr)
            s_t = f_t * s_tm1 + i_t * c_t
            h_t = o_t * T.tanh(s_t)
            #y_t = T.nnet.softmax(T.dot(h_t, self.W_hy) + self.by)
            return [h_t, s_t]

        [hl, sl], _ = theano.scan(fn=recurrencel, \
            sequences=x, outputs_info=[self.h0l, self.s0l], \
            n_steps=x.shape[0])

        x = x[::-1]
        [hr, sr], _ = theano.scan(fn=recurrencer, \
            sequences=x, outputs_info=[self.h0r, self.s0r], \
            n_steps=x.shape[0])

        h_t = T.concatenate([hl, hr], axis = 1)
        y, up = theano.scan(lambda h: T.nnet.softmax(T.dot(h, self.W_hy) + self.by), sequences = h_t)
        #y = T.nnet.softmax(T.dot(h_t, self.W_hy) + self.by)
        p_y = y[:,0,:]
        p_y = -T.log(p_y)


        '''def judgeonenum(a):
            if a == 0:
                res = self.constant0
            else:
                res = self.constant1
            return res

        def judgeonelist(alist):
            res, upd = theano.scan(fn=judgeonenum, sequences=[alist])
            return res

        def judgeone(a, b):
            b = b.dimshuffle(0, 'x')
            c = a - b
            res, upd = theano.scan(fn=judgeonelist, sequences=[c])
            return res'''

        def viterbi(p_y, A):
            def inner_function(obs, prior_result, chain):
                prior_result = prior_result.dimshuffle(0, 'x')
                obs = obs.dimshuffle('x', 0)
                out = (prior_result + obs + chain).max(axis=0)
                argout = (prior_result + obs + chain).argmax(axis = 0)
                return [out, argout]
            obs_potentials = p_y
            initial = obs_potentials[0]
            [scanned, arg], _ = theano.scan(fn=inner_function,
                                     outputs_info=[initial,None],
                                     sequences=[obs_potentials[1:]],
                                     non_sequences=A)
            initialy = scanned[-1].argmax(axis=0)
            arg = arg[::-1]
            y_pred, _ = theano.scan(fn=lambda delta,yn: delta[yn],
                                 outputs_info=initialy,
                                 sequences=[arg])
            y_pred = y_pred[::-1]
            y_pred = T.concatenate((y_pred, [initialy]))
            return y_pred

        y_pred = viterbi(p_y, self.A)




        def dp(p_y, y_p, A):
            def inner_function(obs, prior_result, chain):
                prior_result = prior_result.dimshuffle(0, 'x')
                obs = obs.dimshuffle('x', 0)
                out = (prior_result + obs + chain).max(axis = 0)
                return out
            #y_p = T.arange(p_y.shape[0])
            #y_p = y_p.dimshuffle(0, 'x')
            obs_potentials = p_y + kalpha * y_p
            initial = obs_potentials[0]
            scanned, _ = theano.scan(fn=inner_function,
                                     outputs_info = initial,
                                     sequences=[obs_potentials[1:]],
                                     non_sequences=A)
            maxscore = scanned[-1].max(axis=0)
            #for j in range(m):
                #delta[0][j] = p_y[0][j] + kalpha * judgeone(p_y[0][j], y[0])
            #for i in T.arange(n - 1):
                #delta[i + 1][j] = delta[i][m - 1] + p_y[i + 1][j] + kalpha * judgeone(p_y[i + 1][j], y[i + 1]) + A[m - 1][j]
                #Delta[i + 1][j] = m - 1
                #for k in T.arange(m - 1):
                    #tmp = delta[i][k] + p_y[i + 1][j] + kalpha * judgeone(p_y[i + 1][j], y[i + 1]) + A[k][j]
                    #if tmp > delta[i + 1][j]:
                        #delta[i + 1][j] = tmp
                        #Delta[i + 1][j] = k
            #maxscore = delta[n-1][m-1]
            #besty[n-1] = m-1
            #for j in range(m - 1):
                #tmp = delta[n-1][j]
                #if tmp > maxscore:
                    #maxscore = tmp
                    #besty[n-1] = j
            #for i in range(n - 1):
                #besty[n-2-i] = Delta[n-1-i][besty[n-1-i]]
            return maxscore
			
        def cost(p_y, y_p, y, A):
            costvalue = T.sum(p_y[T.arange(p_y.shape[0]), y])
            y0 = numpy.zeros(1, dtype='int32')
            y1 = T.concatenate([y0, y], axis=0)
            y2 = T.concatenate([y, y0], axis=0)
            costvalue += T.sum(A[y1, y2][1:-1])
            #for i in range(len(p_y_l) - 1):
                #costvalue += A[y[i]][y[i + 1]]
            return  dp(p_y, y_p, A) - costvalue



        # cost and gradients and learning rate
        lr = T.scalar('lr')
        sentence_cost = cost(p_y, y_p, y_sentence, self.A)
        #sentence_cost = -T.mean(T.log(p_y)[T.arange(x.shape[0]), y_sentence])
        regularization = 0.0
        regularization += T.sum(self.W_xil * self.W_xil)
        regularization += T.sum(self.W_hil * self.W_hil)
        regularization += T.sum(self.W_xol * self.W_xol)
        regularization += T.sum(self.W_hol * self.W_hol)
        regularization += T.sum(self.W_xcl * self.W_xcl)
        regularization += T.sum(self.W_hcl * self.W_hcl)
        regularization += T.sum(self.W_xfl * self.W_xfl)
        regularization += T.sum(self.W_hfl * self.W_hfl)
        regularization += T.sum(self.W_hy * self.W_hy)
        regularization += T.sum(self.W_xir * self.W_xir)
        regularization += T.sum(self.W_hir * self.W_hir)
        regularization += T.sum(self.W_xor * self.W_xor)
        regularization += T.sum(self.W_hor * self.W_hor)
        regularization += T.sum(self.W_xcr * self.W_xcr)
        regularization += T.sum(self.W_hcr * self.W_hcr)
        regularization += T.sum(self.W_xfr * self.W_xfr)
        regularization += T.sum(self.W_hfr * self.W_hfr)
        sentence_cost += L2 * regularization
        sentence_gradients = T.grad(sentence_cost, self.params )
        sentence_updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , sentence_gradients))
        
        # theano functions
        self.classify = theano.function(inputs=[idxs, f0idxs, f1idxs, f2idxs, f3idxs, f4idxs], outputs=y_pred)

        self.sentence_train = theano.function( inputs  = [idxs, f0idxs, f1idxs, f2idxs, f3idxs, f4idxs, y_sentence, lr, y_p],
                                      outputs = sentence_cost,
                                      updates = sentence_updates )

    def save(self, flags):
        #numpy.save(os.path.join(folder, 'model' + '.npy'), self.params)
        for param, name in zip(self.params, self.names):
            pickle.dump(param, open('model/'+name+'.pkl', 'wb'))
