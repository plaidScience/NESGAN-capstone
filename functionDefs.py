import pickle
import os
import numpy as np
#import pandas as pd
#import music21 as m21
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
import gc


def chooseOneGPU(chosenGPU):
    '''
    Limits tensorflow to one GPU
    Inputs:
        chosenGPU: the ID of the chosen GPU
    Outputs:
        none
    '''
    # Limit to ONE GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use ONE GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[chosenGPU], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
            
            
            
def openScore (fileName, folder = 'databases/nesmdb24_seprsco/train/'):
    '''
    open a seprsco file, as adapted from nesmdb
    input: 
        fileName: name of .pkl file, 
        folder: folder file is in 
    output: a tuple containing:
        rate: rate of file
        nsamps: number of samps file contains
        seprsco: numpy array of len Nx4, where N is number of notes in the file
    '''
    with open(os.path.join(folder, fileName), 'rb') as f:
        rate, nsamps, seprsco = pickle.load(f)
    return rate, nsamps, seprsco

def saveScore (seprsco, fileName, folder='saved/'):
    '''
    save a seprsco file, as adapted from nesmdb
    input: 
        seprsco: the tuple of rate, nsamps, seprsco to save the file to
        fileName: name of the resulting .pkl file
        folder: folder file to save to
    '''
    with open(os.path.join(folder, fileName), 'wb') as f:
        pickle.dump(seprsco, f, protocol=2)

""" old version using dataframes
def openFolder (folder):
    '''
    open seprsco files from a folder
    input: 
        folder: folder files are in 
    output: a dataframe of score, rate, nsamps, filename and notes, where
        score: seprscore of song
        rate: rate of song,
        nsamps: number of samples in song,
        notes: number of notes in song.
    score, rate and nsamps comes from openScore()
    '''
    i = 0
    df = pd.DataFrame(columns = ("score", "rate", "nsamps", "fileName", "notes"))
    for file in os.listdir(folder):
        rate, nsamps, seprsco = openScore(file, folder)
        notes, dump = seprsco.shape
        df.loc[i] = [seprsco, rate, nsamps, file, notes]
        i+=1

    df['rate'] = pd.to_numeric(df['rate'])
    df['nsamps'] = pd.to_numeric(df['nsamps'])
    df['notes'] = pd.to_numeric(df['notes'])
    return df
"""

""" old version using dataframes
def openFolderSplit (folder, splitOn = 100):
    '''
    open seprsco files from a folder, and split the song on a number so all scores are of the same size
    input: 
        folder: folder files are in 
        splitOn, the size of the split
    output: a dataframe of score, rate, nsamps, filename and notes, where
        score: seprscore of song, split on splitOn
        rate: rate of song,
        origNsamps: number of samples in the original song,
    rate and origNsamps comes from openScore()
    '''
    i = 0
    df = pd.DataFrame(columns = ("score", "rate", "origNsamps", "fileName"))
    for file in os.listdir(folder):
        rate, nsamps, seprsco = openScore(file, folder)
        notes, dump = seprsco.shape
        j = 0;
        while (notes > (j+1)*splitOn):
            #should split on number of notes 100
            #print ("added:", file, "notes:", str(j*splitOn), "-", str((j+1)*splitOn))
            df.loc[i] = [seprsco[j*splitOn:(j+1)*splitOn], rate, nsamps, file+str(j)]
            i+=1
            j+=1
    df['rate'] = pd.to_numeric(df['rate'])
    df['origNsamps'] = pd.to_numeric(df['origNsamps'])
    return df
"""

def openFolderSplit2 (folder, splitOn = 100, normalize=True, splitOverlap=100):
    '''
    open seprsco files from a folder, and split the song on a number so all scores are of the same size
    input: 
        folder: folder files are in 
        splitOn, the size of the split
        normalize: whether to normalize to make all notes between -1 and 1
        splitOverlap: decide the overlap between each split,
            such that songs of size 100 and an overlap of 50, between 0 and 200
            would be 0:100, 50:150, 100:200
    output: a np.array of scores, where array is NxsplitOnx4, where N is number of retrieved scores, and split on is the size
    '''
    channelLengths = [108, 108, 108, 16]
    items = []
    for file in os.listdir(folder):
        rate, nsamps, seprsco = openScore(file, folder)
        if (normalize):
            seprsco = seprsco.astype(float)
            for i in range(4):
                seprsco[:,i] =(seprsco[:, i] - channelLengths[i]/2)/(channelLengths[i]/2)
        j = 0;
        while (len(seprsco) > (j+splitOn)):
            #should split on number of notes 100
            items.append(seprsco[j:j+splitOn])
            print ("added:", file, "notes:", str(j), "-", str((j+splitOn-1)))
            j+=splitOverlap
    return np.array(items)

def saveFolder(seprscos, folder):
    '''
    save seprsco files from to folder
    input: 
        seprscos: the list of seprscos to save
        folder: folder to save the files to
    '''
    directory = os.path.dirname(folder)
    if not os.path.exists(directory):
        os.mkdir(directory)
    i = 0
    for seprsco in seprscos:
        saveScore(seprsco, "saved"+str(i)+".seprsco.pkl", folder)
        i +=1


class NESGAN:
    
    def __init__(self, rows, dirPrefix=""):
        '''
        initialize NESGAN
        inputs: 
            rows: the shape of the input sequence
            dirPrefix: the folder to save logs and checkpoints to (will be saved to ./logs/$dirPrefix/ and ./checkpoints/$dirPrefix)
        '''
        self.seq_length = rows
        self.seq_shape = (self.seq_length, 4)
        self.latent_dim = 1024
        self.disc_loss = []
        self.gen_loss = []
        self.disc_test_loss = []
        self.gen_test_loss = []
        self.disc_val_loss = []
        self.gen_val_loss = []
        
        
        self.logDir = './logs'
        self.discLogDir = os.path.join(self.logDir, dirPrefix, "disc/")
        self.genLogDir = os.path.join(self.logDir, dirPrefix, "gen/")
        
        self.generatorOptimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
        self.discriminatorOptimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
        
        self.discriminator = self.createDiscriminator()
        self.discriminator.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits = True), optimizer=self.discriminatorOptimizer, metrics=['accuracy'])
        
        self.generator = self.createGenerator()
        
        z = tf.keras.layers.Input(shape=(self.latent_dim,))
        generated_seq = self.generator(z)
        
        self.discriminator.trainable = False
        
        validity = self.discriminator(generated_seq)
        
        self.combined = tf.keras.models.Model(z, validity)
        self.combined.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits = True), optimizer=self.discriminatorOptimizer)
        
        self.checkpointDir = './checkpoints'
        self.checkpointPrefix = os.path.join(self.checkpointDir, dirPrefix, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generatorOptimizer = self.generatorOptimizer,
                                              discriminatorOptimizer = self.discriminatorOptimizer,
                                              generator = self.generator,
                                              discriminator = self.discriminator,
                                              combined = self.combined)
        
        self.discCallback = tf.keras.callbacks.TensorBoard(self.discLogDir)
        self.discCallback.set_model(self.discriminator)
        self.genCallback = tf.keras.callbacks.TensorBoard(self.genLogDir)
        self.genCallback.set_model(self.combined)
        self.discMetrics = ['loss', 'real_accuracy', 'fake_accuracy']
        self.genMetrics = ['loss']
        self.discValMetrics = ['val_loss', 'val_real_accuracy', 'val_fake_accuracy']
        self.genValMetrics = ['val_loss']
        self.discTestMetrics = ['test_loss', 'test_real_accuracy', 'test_fake_accuracy']
        self.genTestMetrics = ['test_loss']
        
    
    def createDiscriminator(self):
        '''create discriminator model'''
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(512, input_shape = self.seq_shape, return_sequences=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512)),
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.summary()
        seq = tf.keras.layers.Input(self.seq_shape)
        validity = model(seq)
        
        return tf.keras.models.Model(seq, validity)
    
    def createGenerator(self):
        '''create generator model'''
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, input_dim = self.latent_dim),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.BatchNormalization(momentum=0.8),
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.BatchNormalization(momentum=0.8),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.BatchNormalization(momentum=0.8),
            tf.keras.layers.Dense(np.prod(self.seq_shape), activation='tanh'),
            tf.keras.layers.Reshape(self.seq_shape)
        ])
        
        model.summary()
        
        noise = tf.keras.layers.Input(shape=(self.latent_dim,))
        seq=model(noise)
        
        return tf.keras.models.Model(noise, seq)
    def pretrain_disc_step(self, real_seqs, batch_size):
        '''
        one step of discriminator pretraining
        inputs:
            real_seqs: the seqs to pretrain to
            batch_size: the size of the batch
        returns d_loss, list of discriminator loss, discriminator real accuracy and disc fake accuracy
        '''
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        noise = tf.random.normal((batch_size, self.latent_dim), 0, 1)
        gan_result = self.generator.predict(noise)
        gen_seqs =  np.array(gan_result)
            
        d_loss_real = self.discriminator.train_on_batch(real_seqs, real)
        d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
        d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
        real_acc = d_loss_real[1]
        fake_acc = d_loss_fake[1]
        
        return [d_loss, real_acc, fake_acc]
    
    def pretrain_gen_step(self, batch_size):
        '''
        one step of generator pretraining
        inputs:
            real_seqs: the seqs to pretrain to
            batch_size: the size of the batch
        returns g_lossm generator loss
        '''
        
        real = np.ones((batch_size, 1))
        
        noise = tf.random.normal((batch_size, self.latent_dim), 0, 1)
            
        g_loss = self.combined.train_on_batch(noise, real)
        return g_loss, d_loss
        
                                   
    def train_step(self, real_seqs, batch_size):
        '''
        one step of training,
        inputs:
            real_seqs: the real seqs to train to
            batch_size: the size of the batch, used for generating fake sequences
        returns g_loss and d_loss
            g_loss, generator loss
            d_loss, list of discriminator loss, discriminator real accuracy and disc fake accuracy
        '''
        
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        noise = tf.random.normal((batch_size, self.latent_dim), 0, 1)
        gan_result = self.generator.predict(noise)
        gen_seqs =  np.array(gan_result)
            
        d_loss_real = self.discriminator.train_on_batch(real_seqs, real)
        d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
        d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
        real_acc = d_loss_real[1]
        fake_acc = d_loss_fake[1]
            
        noise = tf.random.normal((batch_size, self.latent_dim), 0, 1)
            
        g_loss = self.combined.train_on_batch(noise, real)
        return g_loss, [d_loss, real_acc, fake_acc]
    
    def test_step(self, real_seqs, batch_size):
        '''
        one step of testing,
        inputs:
            real_seqs: the real seqs to test
            batch_size: the size of the batch, used for generating fake sequences
        returns g_loss and d_loss
            g_loss, generator loss
            d_loss, list of discriminator loss, discriminator real accuracy and disc fake accuracy
        '''
        
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        noise = tf.random.normal((batch_size, self.latent_dim), 0, 1)
        gan_result = self.generator.predict(noise)
        gen_seqs =  np.array(gan_result)
            
        d_loss_real = self.discriminator.test_on_batch(real_seqs, real)
        d_loss_fake = self.discriminator.test_on_batch(gen_seqs, fake)
        d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
        real_acc = d_loss_real[1]
        fake_acc = d_loss_fake[1]
            
        noise = tf.random.normal((batch_size, self.latent_dim), 0, 1)
            
        g_loss = self.combined.test_on_batch(noise, real)
        return g_loss, [d_loss, real_acc, fake_acc]
    
    def writeLog(self, callbackFunc, names, logs, logID):
        '''
        write an output log using callbackFunc, to a specific log ID
        calbackFunc: the function to callback to
        names: the names of the logs (expected to be a list)
        logs: the logs (expected to be a list)
        logID the ID of the log to write to, I.E. the Epoch
        '''
        result = {}
        for nameVal in zip(names, logs):
            result[nameVal[0]] = nameVal[1]
            
        callbackFunc(logID, result)
    def writeTestLog(self, callbackFunc, names, logs):
        '''
        write an output log using callbackFunc, to no specific log ID
        calbackFunc: the function to callback to
        names: the names of the logs (expected to be a list)
        logs: the logs (expected to be a list)
        '''
        result = {}
        for nameVal in zip(names, logs):
            result[nameVal[0]] = nameVal[1]
            
        callbackFunc(result)
    
    
    def pretrain_disc (self, x_train, epochs = 10, start_epoch=0, batch_size=128):
        batchCount = x_train.shape[0]//batch_size
        
        for epoch in range(start_epoch, start_epoch+epochs):
            
            avgDLoss = 0
            
            shuffIDs = np.random.randint(0, x_train.shape[0], x_train.shape[0])
            for batch in range(batchCount):
                now = datetime.now()
                print("{}: Pretrain Epoch {} {:.5%} Done! (Batch {}/{})".format(now.strftime("%H:%M:%S"), epoch, batch/batchCount, batch, batchCount), end="\r", flush=True)
                
                idx = shuffIDs[batch*batch_size : (batch+1)*batch_size]
                songs = x_train[idx]
                discLoss = self.pretrain_disc_step(songs, batch_size)
                avgDLoss = np.add(discLoss, avgDLoss)
                
            avgDLoss = (1/batchCount) * avgDLoss
            
            if (epoch % sample_interval == 0):
                now = datetime.now()
                print("Pretrain {:02d}: {} [D loss: {:.5f}, real_acc.: {:.3%}, , fake_acc.: {:.3%}]".format(
                    epoch, now.strftime("%H:%M:%S"),
                    avgDLoss[0], avgDLoss[1], avgDLoss[2],
                ))
    
    def pretrain_gen (self, epochs = 10, start_epoch=0, batch_size=128):
        
        batchCount = x_train.shape[0]//batch_size
        
        for epoch in range(start_epoch, start_epoch+epochs):
            
            avgGLoss = 0
            
            for batch in range(batchCount):
                now = datetime.now()
                print("{}: Pretrain Epoch {} {:.5%} Done! (Batch {}/{})".format(now.strftime("%H:%M:%S"), epoch, batch/batchCount, batch, batchCount), end="\r", flush=True)
                discLoss = self.pretrain_gen_step(batch_size)
                avgGLoss = np.add(discLoss, avgGLoss)
                
            avgGLoss = (1/batchCount) * avgGLoss
            
            if (epoch % sample_interval == 0):
                now = datetime.now()
                print("Pretrain {:02d}: {} [G loss: {:.5f}]".format(
                    epoch, now.strftime("%H:%M:%S"),
                    avgGLoss
                ))
    
    def train(self, x_train, x_test = None, x_validate = None, epochs = 10, start_epoch=0, batch_size=128, sample_interval=50, save_interval=15):
        '''
        run training
            x_train: training set
            x_test: testing set, does not test if None
            x_validate: validation set, does not validate if None
            epochs: number of epochs to train
            start_epoch: epoch to start training on
            batch_size the size of the training batch
            sample_interval: the interval to sample and write output logs on
            save_interval: the interval to save checkpoint on
        '''
        
        batchCount = x_train.shape[0]//batch_size
        if (x_validate is not None):
            valBatchCount = x_validate.shape[0]//batch_size
        if (x_test is not None):
            testBatchCount = x_test.shape[0]//batch_size
        
        for epoch in range(start_epoch, start_epoch+epochs):
            
            avgDLoss = 0
            avgGLoss = 0
            avgValDLoss = 0
            avgValGLoss = 0
            
            shuffIDs = np.random.randint(0, x_train.shape[0], x_train.shape[0])
            if (x_validate is not None):
                shuffValIDs = np.random.randint(0, x_validate.shape[0], x_validate.shape[0])
            for batch in range(batchCount):
                now = datetime.now()
                print("{}: Epoch {} {:.5%} Done! (Batch {}/{})".format(now.strftime("%H:%M:%S"), epoch, batch/batchCount, batch, batchCount), end="\r", flush=True)
                
                idx = shuffIDs[batch*batch_size : (batch+1)*batch_size]
                songs = x_train[idx]
                genLoss, discLoss = self.train_step(songs, batch_size)
                avgGLoss = np.add(genLoss, avgGLoss)
                avgDLoss = np.add(discLoss, avgDLoss)
                
            avgDLoss = (1/batchCount) * avgDLoss
            avgGLoss = (1/batchCount) * avgGLoss
            
            if(x_validate is not None):
                for batch in range(valBatchCount):
                    now = datetime.now()
                    print("{}: Epoch Validation {} {:.5%} Done! (Batch {}/{})".format(now.strftime("%H:%M:%S"), epoch, batch/valBatchCount, batch, valBatchCount), end="\r", flush=True)
                    
                    idx = shuffValIDs[batch*batch_size : (batch+1)*batch_size]
                    songs = x_validate[idx]
                    valGenLoss, valDiscLoss = self.test_step(songs, batch_size)
                    avgValGLoss = np.add(valGenLoss, avgValGLoss)
                    avgValDLoss = np.add(valDiscLoss, avgValDLoss)
                avgValGLoss = (1/valBatchCount) * avgValGLoss
                avgValDLoss = (1/valBatchCount) * avgValDLoss
                    
                
            
            if (epoch % sample_interval == 0):
                now = datetime.now()
                self.disc_loss.append(avgDLoss[0])
                self.gen_loss.append(avgGLoss)
                self.writeLog(self.discCallback.on_epoch_end, self.discMetrics, avgDLoss, epoch)
                self.writeLog(self.genCallback.on_epoch_end, self.genMetrics, [avgGLoss], epoch)
                
                if (x_validate is not None):
                    self.disc_val_loss.append(avgValDLoss[0])
                    self.gen_val_loss.append(avgValGLoss)
                    self.writeLog(self.discCallback.on_epoch_end, self.discValMetrics, avgValDLoss, epoch)
                    self.writeLog(self.genCallback.on_epoch_end, self.genValMetrics, [avgValGLoss], epoch)
                    print("{:003d} Epoch {}:                                      \n    [D loss: {:.5f}, real_acc.: {:.3%} fake_acc.: {:.3%}, val_loss: {:.5f}, val_real_acc.: {:.3%}, val_fake_acc.: {:.3%}] \n    [G loss: {:.5f}, val_loss: {:.5f}]".format(
                        epoch, now.strftime("%H:%M:%S"),
                        avgDLoss[0], avgDLoss[1], avgDLoss[2], avgValDLoss[0], avgValDLoss[1], avgValDLoss[2],
                        avgGLoss, avgValGLoss
                    ))
                else:
                    print("{}: {:02d} [D loss: {:.5f}, real_acc.: {:.3%}, fake_acc.: {:.3%}] [G loss: {:.5f}]".format(
                        now.strftime("%H:%M:%S"), epoch,
                        avgDLoss[0], avgDLoss[1],  avgDLoss[2],
                        avgGLoss
                    ))
                
                    
            
            if ((epoch % save_interval) == (save_interval-1)):
                self.checkpoint.save(file_prefix = self.checkpointPrefix)
            
        
        if (x_test is not None):
            avgTestGLoss = 0
            avgTestDLoss = 0
            shuffTestIDs = np.random.randint(0, x_test.shape[0], x_test.shape[0])
            for batch in range(testBatchCount):
                now = datetime.now()
                print("{}: Testing {:.5%} Done! (Batch {}/{})".format(now.strftime("%H:%M:%S"), batch/testBatchCount, batch, testBatchCount), end="\r", flush=True)
                    
                
                idx = shuffTestIDs[batch*batch_size : (batch+1)*batch_size]
                songs = x_test[idx]
                testGLoss, testDLoss = self.test_step(songs, batch_size)
                avgTestGLoss = np.add(avgTestGLoss, testGLoss)
                avgTestDLoss = np.add(avgTestDLoss, testDLoss)
            print("{}: Testing: [D loss: {:.5f}, real_acc.: {:.3%}, fake_acc.: {:.3%}] [G loss: {:.5f}]".format(
                        now.strftime("%H:%M:%S"),
                        avgTestDLoss[0], avgTestDLoss[1], avgTestDLoss[2],
                        avgTestGLoss
            ))
            avgTestGLoss = (1/testBatchCount) * avgTestGLoss
            avgTestDLoss = (1/testBatchCount) * avgTestDLoss
                
            self.disc_test_loss = avgTestDLoss
            self.gen_test_loss = avgTestGLoss
            self.writeTestLog(self.discCallback.on_test_end, self.discTestMetrics, avgTestDLoss)
            self.writeTestLog(self.genCallback.on_test_end, self.genTestMetrics, [avgTestGLoss])
        
        self.discCallback.on_train_end(None)
        self.genCallback.on_train_end(None)
            
        #self.generate(notes)
        self.plot_loss()
    
    def generate(self, predictNo, endPadding = 100):
        '''
        generate predictions
            predictNo: the number of predictions to make
            endPadding: the amount of padding to add to nSamps
        returns a list of result tuples where:
            rate: first part of the tuple, the rate of the song
            nsamps: the second part of the tuple, the number of samps in the song
            prediction: the prediction result (between -1 and 1)
        '''
        predictions = []
        rate = 24.0
        for i in range(predictNo):
            noise = tf.random.normal((1, self.latent_dim), 0, 1)
            prediction = self.generator.predict(noise)
            nsamps = prediction.shape[1] + endPadding
            predictions.append((rate, nsamps, prediction))
        return predictions
    
    def generateScaled(self, predictNo, channelLengths = [108, 108, 108, 16], endPadding = 100):
        '''
        generate predictions scaled back to an original length (between 0 and N)
            predictNo: the number of predictions to make
            channelLengsth: the lengths of the challels to scale back to
            endPadding: the amount of padding to add to nSamps
        returns a list of result tuples where:
            rate: first part of the tuple, the rate of the song
            nsamps: the second part of the tuple, the number of samps in the song
            prediction: the prediction result, with each channel scaled back to a it's specific channelLength
        '''
        predictions = []
        rate = 24.0
        for i in range(predictNo):
            noise = tf.random.normal((1, self.latent_dim), 0, 1)
            prediction = self.generator.predict(noise)[0]
            for i in range(4):
                prediction[:,i] = (prediction[:, i] * channelLengths[i]/2) + (channelLengths[i]/2)
            nsamps = prediction.shape[1] + endPadding
            
            predictions.append((rate, nsamps, prediction.astype(int)))
        return predictions
    
    def loadCheckpoint(self, checkpointFolder):
        checkpointsPath = os.path.join(self.checkpointDir, checkpointFolder)
        self.checkpoint.restore(save_path = tf.train.latest_checkpoint(checkpointsPath))
        self.checkpoint.save(file_prefix = self.checkpointPrefix)
        
    
    def plot_loss(self):
        plt.plot(self.disc_loss, c='red')
        plt.plot(self.gen_loss, c='blue')
        plt.title("GAN Loss per Epoch")
        plt.legend(['Discriminator', 'Generator'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('GAN_Loss_per_Epoch_final.png', transparent=True)
        plt.show()