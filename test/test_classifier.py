import operator
import unittest
from app.classifier import *
from app.remote import *
from app.sensor import *
from app.tools import *

class TestClassifier(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestClassifier, self).__init__(*args, **kwargs)
        self.norm_func=lambda x:abs(x)           # mean is 0
        self.event_func=lambda x: abs(x)+2     # mean is 0.5
        self.rem=Remote()
        self.s=Sensor(self.norm_func,self.event_func,[],1,1)

    def test_window_size(self):
        end_time=100
        cl=Classifier()
        for t in range(end_time):
            cl.learn(t,False)
        self.assertEqual(len(cl.train),min(cl.window_size,end_time))

    def test_classif_easy(self):
        """
        Train only with normal events
        """
        cl=Classifier()
        cl_inv=Classifier()
        nrep=10
        ans=0
        ans_inv=0
        for i in range(nrep):
            cl=Classifier()
            for _ in range(2):
                cl.update_training(self.norm_func(np.random.normal(loc=0,scale=0.4)))
            ans+=cl.classify(self.event_func(np.random.normal(loc=0,scale=0.4)))[0]
            # inverse
            cl_inv=Classifier()
            for _ in range(2):
                cl_inv.update_training(self.event_func(np.random.normal(loc=0,scale=0.4)))
            ans_inv+=cl_inv.classify(self.norm_func(np.random.normal(loc=0,scale=0.4)))[0]
        ans/=float(nrep)
        ans_inv/=float(nrep)
        print(ans)
        self.assertGreater(ans,0.8)
        print(ans_inv)
        self.assertGreater(ans_inv,0.8)

    def test_classif_difficult(self):
        """
        Train with an outlier
        """
        cl=Classifier()
        cl_inv=Classifier()
        nrep=50
        ans=0
        ans_inv=0
        for i in range(nrep):
            cl=Classifier()
            for _ in range(2):
                cl.update_training(self.norm_func(np.random.normal(loc=0,scale=0.4)))
            cl.update_training(self.event_func(np.random.normal(loc=0,scale=0.4)))
            ans+=cl.classify(self.event_func(np.random.normal(loc=0,scale=0.4)))[0]
            # inverse
            cl_inv=Classifier()
            for _ in range(2):
                cl_inv.update_training(self.event_func(np.random.normal(loc=0,scale=0.4)))
            cl_inv.update_training(self.norm_func(np.random.normal(loc=0,scale=0.4)))
            ans_inv+=cl_inv.classify(self.norm_func(np.random.normal(loc=0,scale=0.4)))[0]
        ans/=float(nrep)
        ans_inv/=float(nrep)
        print(ans)
        self.assertLess(ans,0.8)
        print(ans_inv)
        self.assertLess(ans_inv,0.8)

    def helper_score(self,train,ntest,flip):
        score=0
        cl=Classifier()
        for t in train:
            cl.update_training(t)

        for m in [self.norm_func(np.random.normal(loc=0,scale=0.4)) for _ in range(ntest)]:
            ans=cl.classify(m)
            if ans[0]==flip:
                score+=1
        for m in [self.event_func(np.random.normal(loc=0,scale=0.4)) for _ in range(ntest)]:
            ans=cl.classify(m)
            if ans[0]==(not flip):
                score+=1
        return score/float(2*ntest)

    def test_classif_training_contents(self):
        """
        Look at performance when training with different types of points:
        - only normal points
        - only outliers (classification is flipped)
        - only support vectors (both normal and outlier)
        """
        score_sv_n=0
        score_sv_o=0
        score_o=0
        score_n=0
        score_n_and_sv_n=0
        score_o_and_sv_o=0
        nrep=20
        ntrain=10
        ntest=100
        for _ in range(nrep):
            avg=0
            sd=0.4
            normals=[self.norm_func(np.random.normal(loc=avg,scale=sd)) for _ in range(ntrain)]
            avg_n=np.asarray(normals).mean()
            sd_n=np.asarray(normals).std()
            events=[self.event_func(np.random.normal(loc=avg,scale=sd)) for _ in range(ntrain)]
            avg_e=np.asarray(events).mean()
            sd_e=np.asarray(events).std()

            distances=[abs(a-avg_n) for a in normals]
            norms=[v for v,d in zip(normals,distances) if d<=sd_n]
            #print(norm_avg)
            distances=[abs(a-avg_e) for a in events]
            outliers=[v for v,d in zip(events,distances) if d<=sd_e]
            #print(event_avg)
            distances=[a-avg_n for a in normals] # above the mean
            supp_vecs_norms=[v for v,d in zip(normals,distances) if d>sd_n]
            distances=[avg_e-a for a in events] # below the mean
            supp_vecs_outs=[v for v,d in zip(events,distances) if d>sd_e]

            # Train with support vectors, only normals
            score_sv_n+=self.helper_score(supp_vecs_norms,ntest,False)
            # Train with support vectors, only outliers
            score_sv_o+=self.helper_score(supp_vecs_outs,ntest,True) # training with outliers, flip classification
            # Train with outliers
            score_o+=self.helper_score(outliers,ntest,True) # training with outliers, flip classification
            # Train with normal values
            score_n+=self.helper_score(normals,ntest,False)
            score_n_and_sv_n+=self.helper_score(normals+supp_vecs_norms,ntest,False)
            score_o_and_sv_o+=self.helper_score(outliers+supp_vecs_outs,ntest,True)
        score_sv_n=score_sv_n/float(nrep)
        self.assertGreater(score_sv_n,0.3)
        self.assertLess(score_sv_n,0.6)
        score_sv_o=score_sv_o/float(nrep)
        self.assertGreater(score_sv_o,0.3)
        self.assertLess(score_sv_o,0.6)
        score_n=score_n/float(nrep)
        self.assertGreater(score_n,0.6)
        self.assertLess(score_n,1.0)
        score_o=score_o/float(nrep)
        self.assertGreater(score_o,0.6)
        self.assertLess(score_o,1.0)
        score_o_and_sv_o=score_o_and_sv_o/float(nrep)
        score_n_and_sv_n=score_n_and_sv_n/float(nrep)
        self.assertLess(abs(score_n_and_sv_n-score_n),0.2)
        self.assertLess(abs(score_o_and_sv_o-score_o),0.2)


    def helper_score_self(self,train,test,flip,operator):
        """
        Train classifier online, based on own classification.

        Args:
        train: a vector of measurements
        test: a vector of measurements, pairs containing a number and a label that indicates if it is an outlier.
        flip: indicates the type of points we want to train the classifier with, if False the classifier is trained with (points classified as) normal measurements, if True with (points classified as) outliers.
        operator: indicates the distance to the boundary, if operator.le the classifier is trained with points with an uncertain classification, if operator.gt with points with a certain classification.

        If flip is False, the classifier is trained with normal measurements, so outliers are labeled with True.
        If flip is True, the classifier is trained with outliers, so outliers are labeled with False.
        """
        tp=0
        fp=0
        fn=0
        thresh=0.005
        cl=Classifier()
        for t in train:
            cl.update_training(t)
        self.assertNotEqual(flip,None) # messes up with xor operator
        for v,l in test:
            lab,conf=cl.classify(v)
            cls=(lab != flip) # xor operator, inverts lab if flip is True: If the classifier is trained with outliers, the prediction must be flipped
            if l==True:        # outlier
                if cls==True:
                    tp+=1
                else:
                    fn+=1
            else:
                if cls==True:
                    fp+=1
            if (lab==None or    # not trained
                (lab == False and # if this point is classified as normal (in case of training with outliers, normal would mean outlier)
                 (operator==None or operator(abs(conf),thresh)))): # if distance is correct or ignore confidence
                #print(str(flip)+" and "+str(operator)+" learns "+str(lab)+" which is "+str(l))
                cl.update_training(v)
            # else:
            #     print(str(abs(conf))+" "+str(operator if operator==None else operator(abs(conf),thresh))+" "+str(lab)+" which is "+str(l))
        prec,rec=compute_prec_recall(tp,fp,fn)
        return prec,rec

    def helper_self_training(self,ntrain):
        prec_sv_n=0
        rec_sv_n=0
        prec_sv_o=0
        rec_sv_o=0
        prec_o=0
        rec_o=0
        prec_n=0
        rec_n=0
        nrep=50
        ntest=100
        for _ in range(nrep):
            avg=0
            sd=0.4
            normals=[(self.norm_func(np.random.normal(loc=avg,scale=sd)),False) for _ in range(int(ntest))]
            events=[(self.event_func(np.random.normal(loc=avg,scale=sd)),True) for _ in range(int(ntest))]
            test=normals+events
            np.random.shuffle(test)

            train=[self.norm_func(np.random.normal(loc=avg,scale=sd)) for _ in range(ntrain)]
            train_o=[self.event_func(np.random.normal(loc=avg,scale=sd)) for _ in range(ntrain)]
            p,r=self.helper_score_self(train,test,False,None) # normal
            prec_n+=p
            rec_n+=r
            p,r=self.helper_score_self(train_o,test,True,None) # outlier
            prec_o+=p
            rec_o+=r
            p,r=self.helper_score_self(train,test,False,operator.le) # normal, support vector
            prec_sv_n+=p
            rec_sv_n+=r
            p,r=self.helper_score_self(train_o,test,True,operator.le) # outlier, support vector
            prec_sv_o+=p
            rec_sv_o+=r
        prec_sv_n/=float(nrep)
        rec_sv_n/=float(nrep)
        prec_sv_o/=float(nrep)
        rec_sv_o/=float(nrep)
        prec_n/=float(nrep)
        rec_n/=float(nrep)
        prec_o/=float(nrep)
        rec_o/=float(nrep)
        return prec_sv_n,rec_sv_n,prec_sv_o,rec_sv_o,prec_n,rec_n,prec_o,rec_o

    def test_classif_online_training(self):
        """
        Look at performance when training online with data that is classified by one self
        - N: only normal points
        - O: only outliers (classification is flipped)
        - SV_N: only normal support vectors
        - SV_O: only outlier support vectors

        If training on outliers, precision and recall are inverted because in this case outliers are seen as normal measurements.
        """

        prec_sv_n,rec_sv_n,prec_sv_o,rec_sv_o,prec_n,rec_n,prec_o,rec_o=self.helper_self_training(10)

        self.assertGreater(prec_sv_n,0.7);self.assertLess(prec_sv_n,0.9)
        self.assertGreater(rec_sv_n,0.9);self.assertLessEqual(rec_sv_n,1.0)
        print("sv_n "+str(prec_sv_n)+" "+str(rec_sv_n))
        self.assertGreater(prec_sv_o,0.9);self.assertLessEqual(prec_sv_o,1.0)
        self.assertGreater(rec_sv_o,0.6);self.assertLessEqual(rec_sv_o,0.9)
        print("sv_o "+str(prec_sv_o)+" "+str(rec_sv_o))
        self.assertGreater(prec_n,0.6);self.assertLess(prec_n,0.9)
        self.assertGreater(rec_n,0.9);self.assertLessEqual(rec_n,1.0)
        print("n "+str(prec_n)+" "+str(rec_n))
        self.assertGreater(prec_o,0.9);self.assertLessEqual(prec_o,1.0)
        self.assertGreater(rec_o,0.5);self.assertLessEqual(rec_o,0.9)
        print("o "+str(prec_o)+" "+str(rec_o))

    def test_classif_online_notraining(self):
        """
        Look at performance when training online with data that is classified by one self, without any previous training
        - N: only normal points
        - O: only outliers (classification is flipped)
        - SV_N: only normal support vectors
        - SV_O: only outlier support vectors

        If training on outliers, precision and recall are inverted because in this case outliers are seen as normal measurements.
        """
        prec_sv_n,rec_sv_n,prec_sv_o,rec_sv_o,prec_n,rec_n,prec_o,rec_o=self.helper_self_training(0) # no training
        self.assertGreater(prec_sv_n,0.4);self.assertLess(prec_sv_n,0.6)
        self.assertGreater(rec_sv_n,0.9);self.assertLessEqual(rec_sv_n,1.0)
        print("sv_n "+str(prec_sv_n)+" "+str(rec_sv_n))
        self.assertGreater(prec_o,0.4);self.assertLessEqual(prec_o,0.6)
        self.assertGreater(rec_o,0.0);self.assertLessEqual(rec_o,0.1)
        print("sv_o "+str(prec_sv_o)+" "+str(rec_sv_o))
        self.assertGreater(prec_n,0.4);self.assertLess(prec_n,0.6)
        self.assertGreater(rec_n,0.9);self.assertLessEqual(rec_n,1.0)
        print("n "+str(prec_n)+" "+str(rec_n))
        self.assertGreater(prec_o,0.4);self.assertLessEqual(prec_o,0.6)
        self.assertGreater(rec_o,0.0);self.assertLessEqual(rec_o,0.1)
        print("o "+str(prec_o)+" "+str(rec_o))
