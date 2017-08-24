import tensorflow as tf
import numpy as np

class fastTextB:
    def __init__(self,label_size,learning_rate,batch_sieze,
                 decay_steps,decay_rate,num_sampled,sentence_len,
                 vocab_size,embed_size,is_training):
        """set hyperparams here"""
        self.label_size=label_size
        self.batch_size=batch_sieze
        self.num_sampled=num_sampled
        self.sentence_len=sentence_len
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.is_training=is_training
        self.learning_rate=learning_rate

        # add place holder
        self.sentence=tf.placeholder(tf.int32,[None,self.sentence_len],name='setntence')
        self.labels=tf.placeholder(tf.int32,[None],name="Lables")
        self.global_setp=tf.Variable(0,trainable=False,name='Global_step')
        self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
        self.epoch_increment=tf.assign(self.epoch_step
                                       ,tf.add(self.epoch_step,tf.constant(1)))
        self.decay_steps,self.decay_rate=decay_steps,decay_rate

        self.epoch_step=tf.Variable(0,trainable=False,name='Epoch_step')
        self.instantite_weights()
        self.logits=self.inference()
        if not is_training:
            return
        self.loss_val=self.loss()
        self.train_op=self.train()
        self.predictions=tf.argmax(self.logits,axis=1,name='predictions')
        correct_prediction=tf.equal(tf.cast(self.predictions,tf.int32),self.labels)
        self.accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name='Accuracy')

    def instantite_weights(self):
        """define all weights here"""
        self.Embedding=tf.get_variable("Embedding",[self.vocab_size,self.embed_size])
        self.W=tf.get_variable("W",[self.embed_size,self.label_size])
        self.b=tf.get_variable("b",[self.label_size])
    def inference(self):
        sentence_embeddings=tf.nn.embedding_lookup(self.Embedding,self.sentence)
        self.sentence_embeddings=tf.reduce_mean(sentence_embeddings,axis=1)
        logits=tf.matmul(self.sentence_embeddings,self.W)+self.b
        return logits
    def loss(self,l2_lambda=0.01):
        pass
    def train(self):
        pass



