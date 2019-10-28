import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, LeakyReLU
from keras.optimizers import Adam, SGD, Adagrad

from keras.datasets import MNIST

import pickle
import argparse
    
class MNIST_Classifier(object):
    def __init__(self, sv, opt, width):
        # initialize training data, classifier and experiment settings
        
        self.sv = sv
        if opt == 'sgd':
            self.opt = SGD(lr=0.1, decay=1e-6)
        if opt == 'adagrad':
            self.opt = Adagrad(lr=0.01)
        if opt == 'adam':
            self.opt = Adam(lr=0.001)
        
        self.width = width
        
        (self.x_train, self.y_train), (self.x_test, self.y_test) = MNIST.load_data()
        
        self.x_train = self.x_train.reshape(-1, 28, 28, 1).astype(np.float32)/255.0
        self.x_test = self.x_test.reshape(-1, 28, 28, 1).astype(np.float32)/255.0
        
        self.classifier = self.classifier()

    def train(self, train_steps=5000, batch_size=64):
        # train the classifier
        
        for i in range(train_steps):
            ix = np.random.choice(self.x_train.shape[0], batch_size, replace=False)
            x = self.x_train[ix,:,:,:]
            y = self.y_train[ix,:]
            
            c_loss = self.classifier.train_on_batch(x, y)
            log_mesg = "%d (train): [loss: %f, acc: %f]" % (i, c_loss[0], c_loss[1])
            
            print(log_mesg)
            
        self.classifier.save(self.sv+'classifier.h5')
    
    def evaluate_dropout_fixed(self, keep_prob=1.0, batch_size=512):
        # evaluates the loss/accuracy of the classifier under random dropout
        # we fix the non-zero weights by sampling without replacement
        # we evaluate on training set since we care about the landscape of the training loss

        w_init = self.classifier.get_weights()
        w_drop = self.channel_drop_fixed(w_init,keep_prob)
        
        self.classifier.set_weights(w_drop)
        
        ix = np.random.choice(self.x_train.shape[0], batch_size, replace=False)
        x = self.x_train[ix,:,:,:]
        y = self.y_train[ix,:]
        
        result = self.classifier.evaluate(x,y)
        
        self.classifier.set_weights(w_init)
        
        return result, w_drop
    
    def evaluate_dropout(self, keep_prob=1.0, batch_size=512):
        # evaluates the loss/accuracy of the classifier under random dropout
        # we evaluate on training set since we care about the landscape of the training loss

        w_init = self.classifier.get_weights()
        w_drop = self.channel_drop(w_init,keep_prob)
        
        self.classifier.set_weights(w_drop)
        
        ix = np.random.choice(self.x_train.shape[0], batch_size, replace=False)
        x = self.x_train[ix,:,:,:]
        y = self.y_train[ix,:]
        
        result = self.classifier.evaluate(x,y)
        
        self.classifier.set_weights(w_init)
        
        return result, w_drop
    
    def evaluate_classifier(self,batch_size=512):
        # evaluates the loss/accuracy of the classifier
        # we evaluate on training set since we care about the landscape of the training loss
        
        ix = np.random.choice(self.x_train.shape[0], batch_size, replace=False)
        x = self.x_train[ix,:,:,:]
        y = self.y_train[ix,:]
        
        result = self.classifier.evaluate(x,y)
        
        return result
        
    def evaluate_path(self, keep_prob=1.0, t_steps=100, batch_size=64):
        # evaluates loss/accuracy along a linear path between the classifier
        # and a version with 1-keep_prob channels zeroed out
        
        w_init = self.classifier.get_weights()
        w_drop = self.channel_drop(w_init,keep_prob)
        
        for i in range(101):
            t = i/100.0
            w = self.interpolate(w_init,w_drop,t)
            
            self.classifier.set_weights(w)
            
            ix = np.random.choice(self.x_train.shape[0], batch_size, replace=False)
            x = self.x_train[ix,:,:,:]
            y = self.y_train[ix,:]
            
            print(t)
            print(self.classifier.evaluate(x,y))
        
        self.classifier.set_weights(w_init)
    
    def evaluate_custom_path(self,w_init,w_final,t_steps=100,batch_size=512):
        # evaluates loss/accuracy along a linear path between two points
        # in parameter space, w_init and w_final
        
        results = []
        for i in range(101):
            t = i/100.0
            w = self.interpolate(w_init,w_final,t)
            
            self.classifier.set_weights(w)
            
            ix = np.random.choice(self.x_train.shape[0], batch_size, replace=False)
            x = self.x_train[ix,:,:,:]
            y = self.y_train[ix,:]
            
            results.append(self.classifier.evaluate(x,y))
        
        self.classifier.set_weights(w_init)
        
        return results
    
    def copy_weights(self,w):
        # makes a copy of a classifier's weights
        
        w_copy = []
        
        for i in range(len(w)):
            w_copy.append(w[i].copy())
        
        return w_copy
    
    def channel_drop_fixed(self,w,keep_prob=1.0):
        # drops out a fixed proportion (i.e. 1-keep_prob) of channels
        
        w_drop = self.copy_weights(w)
        
        for i in range(len(w_drop) - 1):
            c_to_keep = np.random.choice(w_drop[i].shape[-1],int(keep_prob*w_drop[i].shape[-1]),replace=False)
            
            for c in range(w_drop[i].shape[-1]):
                
                if c not in c_to_keep:
                    w_drop[i][:,:,:,c] *= 0
                
                else:
                    w_drop[i][:,:,:,c] *= (1/keep_prob)
        
        return w_drop
    
    def channel_drop(self,w,keep_prob=1.0):
        # drops out each channel independently with probability 1-keep_prob
        
        w_drop = self.copy_weights(w)
        
        for i in range(len(w_drop) - 1):
            for c in range(w_drop[i].shape[-1]):
                p = np.random.uniform()
                
                if p > keep_prob:
                    w_drop[i][:,:,:,c] *= 0
                
                else:
                    w_drop[i][:,:,:,c] *= (1/keep_prob)
        
        return w_drop
                 
    def interpolate(self,w1,w2,t):
        # interpolates between two points in parameter space, w1 and w2
        
        w = []
        for i in range(len(w1)):
            w.append( (1-t)*w1[i] + t*w2[i] )
        
        return w
    
    def classifier(self):
        # classifier architecture
        
        optimizer = self.opt
        C = Sequential()
        
        C.add(Conv2D(self.width, 3, strides=1, input_shape=(28,28,1),\
            padding='same',
            use_bias=False,
            ))
        C.add(LeakyReLU(alpha=0.0))

        C.add(Conv2D(self.width, 3, strides=1, padding='same',
                    use_bias=False))
        C.add(LeakyReLU(alpha=0.0))
        
        C.add(Conv2D(self.width, 3, strides=1, padding='same',
                    use_bias=False))
        C.add(LeakyReLU(0.0))
        
        C.add(Flatten())
        C.add(Dense(10,use_bias=False))
        C.add(Activation('softmax'))
        
        C.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        C.summary()
        
        return C
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Settings')
    parser.add_argument('-s', '--save', default='')
    parser.add_argument('-o', '--opt', default='sgd')
    args = vars(parser.parse_args())
    
    sv = str(args['save'])
    
    opt = str(args['opt'])
    width = 32
    keep_prob = 0.8
    
    mnist_classifier = MNIST_Classifier(sv, opt, width)
    mnist_classifier.train()

    # This experiment is simply designed to show that we can interpolate between
    # a network and its dropped out version without running into a barrier of high loss
    # (i.e. incur loss much higher than that at the endpoints of the path).
    # Thus we rerun dropout until we end up with a good dropped out network.
    rerun = True
    while rerun:
        trial_result, w_drop = mnist_classifier.evaluate_dropout_fixed(keep_prob,batch_size=4096)
        print(trial_result)
        
        if trial_result[1] > 0.98:
            rerun = False
    
    w_init = mnist_classifier.classifier.get_weights()
    
    results = mnist_classifier.evaluate_custom_path(w_init,w_drop,batch_size=4096)
    pickle.dump(results,open(sv+'results.p','wb'))