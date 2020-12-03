#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import os
import numpy as np
import pandas as pd
import music21 as m21
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# In[2]:


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


# In[4]:


if (str(input("Do You Want to use one GPU? y/[n]: ")).startswith("y")):
    print(tf.config.experimental.list_physical_devices('GPU'))
    chooseOneGPU(int(input("Choose the ID of the GPU you want to use: ")))


# In[3]:


def openScore (fileName, folder = 'databases/nesmdb24_seprsco/train/'):
    with open(folder+fileName, 'rb') as f:
        rate, nsamps, seprsco = pickle.load(f)
    return rate, nsamps, seprsco


# In[4]:


def openFolder (folder):
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


# In[5]:


def openFolderSplit (folder, splitOn = 100):
    i = 0
    df = pd.DataFrame(columns = ("score", "rate", "origNsamps", "fileName"))
    for file in os.listdir(folder):
        rate, nsamps, seprsco = openScore(file, folder)
        notes, dump = seprsco.shape
        j = 0;
        while (notes > (j+1)*splitOn):
            #should split on number of notes 100
            print ("added:", file, "notes:", str(j*splitOn), "-", str((j+1)*splitOn))
            df.loc[i] = [seprsco[j*splitOn:(j+1)*splitOn], rate, nsamps, file+str(j)]
            i+=1
            j+=1
    df['rate'] = pd.to_numeric(df['rate'])
    df['origNsamps'] = pd.to_numeric(df['origNsamps'])
    return df


# In[6]:


def openFolderSplit2 (folder, splitOn = 100, normalize=True):
    channelLengths = [108, 108, 108, 16]
    items = []
    for file in os.listdir(folder):
        rate, nsamps, seprsco = openScore(file, folder)
        if (normalize):
            seprsco = seprsco.astype(float)
            for i in range(4):
                seprsco[:,i] =(seprsco[:, i] - channelLengths[i]/2)/(channelLengths[i]/2)
        j = 0;
        while (len(seprsco) > (j+1)*splitOn):
            #should split on number of notes 100
            items.append(seprsco[j*splitOn:(j+1)*splitOn])
            print ("added:", file, "notes:", str(j*splitOn), "-", str((j+1)*splitOn-1))
            j+=1
    return np.array(items)


# In[7]:


items = openFolderSplit2('databases/nesmdb24_seprsco/train/', 1024)
items.shape


# In[8]:


np.savetxt("foo.csv", items[0], delimiter=",")


# In[9]:


itemsTest = openFolderSplit2('databases/nesmdb24_seprsco/test/', 1024)
itemsTest.shape


# In[10]:


itemsVal = openFolderSplit2('databases/nesmdb24_seprsco/valid/', 1024)
itemsVal.shape


# In[11]:


itemsAll = np.concatenate([items, itemsTest, itemsVal])


# In[12]:


itemsAll.shape


# In[13]:


class NESGAN:
    
    mus_length = 1024
    
    def __init__(self, rows):
        self.seq_length = rows
        self.seq_shape = (self.seq_length, 4)
        self.latent_dim = 1024
        self.disc_loss = []
        self.gen_loss = []
        
        optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
        
        self.discriminator = self.createDiscriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        self.generator = self.createGenerator()
        
        z = tf.keras.layers.Input(shape=(self.latent_dim,))
        generated_seq = self.generator(z)
        
        self.discriminator.trainable = False
        
        validity = self.discriminator(generated_seq)
        
        self.combined = tf.keras.models.Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    def createDiscriminator(self):
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
    
    def train(self, x_train, epochs, batch_size=128, sample_interval=50):
        
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_seqs = x_train[idx]
            
            noise = tf.random.normal((batch_size, self.latent_dim), 0, 1)
            
            gan_result = self.generator.predict(noise)
            gan_seqs =  np.array(gan_result)
            
            test_seqs = np.concatenate((real_seqs, gan_seqs))
            train_seqs = np.concatenate((real, fake))
            test_seqs, test_labels = shuffle(test_seqs, train_seqs)
            
            #d_loss_real = self.discriminator.train_on_batch(real_seqs, real)
            #d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
            #d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            d_loss = self.discriminator.train_on_batch(x=test_seqs, y=test_labels, sample_weight=None, class_weight=None, reset_metrics=True, return_dict=False)

            
            noise = tf.random.normal((batch_size, self.latent_dim), 0, 1)
            
            g_loss = self.combined.train_on_batch(noise, real)
            
            if (epoch % sample_interval == 0):
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
                self.disc_loss.append(d_loss[0])
                self.gen_loss.append(g_loss)
            
        #self.generate(notes)
        self.plot_loss()
    
    def generate(self, input_notes):
        
        noise = tf.random.normal((1, self.latent_dim), 0, 1)
        predictions = self.generator.predict(noise)
    
    def plot_loss(self):
        plt.plot(self.disc_loss, c='red')
        plt.plot(self.gen_loss, c='blue')
        plt.title("GAN Loss per Epoch")
        plt.legend(['Discriminator', 'Generator'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('GAN_Loss_per_Epoch_final.png', transparent=True)
        plt.close()
        
        
        
        


# In[14]:


gan = NESGAN(rows=1024)


# In[ ]:


gan.train(itemsAll, epochs=5000, batch_size=32, sample_interval=1)


# In[ ]:





# In[ ]:




