# RNN Based Chatbot using Cornell Movie Dataset

#Dependencies:

tensorflow==1.0.1
<br/> numpy
<br/> re
<br/> time


Important Resources:

Chatbots with Seq2Seq: http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/
<br/> Neural Machine Translation (seq2seq) Tutorial: https://github.com/tensorflow/nmt
<br/> Deep Learning and NLP A-ZTM: How to create a ChatBot: https://www.udemy.com/chatbot/learn/v4/overview


Other Relevant Resources:
<br/> https://www.coursera.org/learn/nlp-sequence-models
<br/> https://machinelearningmastery.com/encoder-decoder-recurrent-neural-network-models-neural-machine-translation/
<br/> https://machinelearningmastery.com/configure-encoder-decoder-model-neural-machine-translation/
<br/> https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/
<br/> https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/
<br/> https://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/
<br/> https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
<br/> http://colah.github.io/posts/2015-08-Understanding-LSTMs/
<br/> https://deeplearning4j.org/lstm.html
<br/> https://deeplearning4j.org/neuralnet-overview
<br/> https://www.tensorflow.org/tutorials/seq2seq

Introduction:
Recurrent Neural Networks (RNNs), a type of Artificial Neural Network (ANN),
make decisions based on the knowledge gained from previous inputs, which
makes them ideal for sequential data i.e. textual sentences or speech. Thus, these
networks have been deployed for machine translation, speech recognition, and
sentiment classification [1]. One of the applications arising from translation
models is the conversation model i.e. chatbot [2]. An RNN Encoder-Decoder
architecture consisting of two recurrent neural networks, where the encoder
prepares a fixed-length vector for the input symbols and the decoder processes it
into another sequence of symbols, is used for such tasks [3].
