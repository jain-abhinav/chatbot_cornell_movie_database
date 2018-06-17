# RNN Based Chatbot using Cornell Movie Dataset

#Dependencies:
<br/> tensorflow==1.0.1
<br/> numpy
<br/> re
<br/> time


Important Resources:
<br/> Chatbots with Seq2Seq: http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/
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


<br/>
Abstract:
<br/>
Recurrent Neural Networks (RNNs), a type of Artificial Neural Network (ANN),
make decisions based on the knowledge gained from previous inputs, which
makes them ideal for sequential data i.e. textual sentences or speech. Thus, these
networks have been deployed for machine translation, speech recognition, and
sentiment classification [1]. One of the applications arising from translation
models is the conversation model i.e. chatbot [2]. An RNN Encoder-Decoder
architecture consisting of two recurrent neural networks, where the encoder
prepares a fixed-length vector for the input symbols and the decoder processes it
into another sequence of symbols, is used for such tasks [3]. Such networks are capable of producing meaningful phrases with
the inclusion of bidirectionality, and long short-term memory (LSTM) cells. An attention mechanism [4], which enables the chatbot to handle very long-range dependencies, is introduced along with the previously mentioned
features.


<br/>
Introduction
<br/>
It has long been the focus for Artificial Intelligence (AI) to build systems that can maintain a
dialog with humans in an intuitive manner. This post discusses the implementation of a chatbot
which will enable users to indulge in an one-to-one multi-turn conversation. The movie-dialog
corpus considered for this task is not specific to any particular domain and provides a brief
explanation of what can be achieved if replicated using other more problem specific datasets. One
such dataset is the Ubuntu corpus, which if considered with the discussed model, can be
instrumental in providing technical support as it includes conversations related to technical
domain. Additionally, the considered dataset contains unstructured conversations, which favors
the development of a chatbot that could lead a general conversation [2].
<br>

The chatbot in discussion leverages deep ANNs to do away with older rule-based models which
are restricted in terms of their interpretation of questions and generation of responses, and thus,
cannot handle complex / unstructured queries. Deep ANNs introduce a generative element which
allows the chatbot to look at each word in the query separately and also the entire query as a
whole to generate new / unfamiliar answers compared to the training dataset, thus, equipping the
chatbot to handle complex queries [7]. However, in case exhaustive training is not done, such
models may be prone to grammatical and spelling errors.
<br>

ANNs are computational and mathematical models that mimic the functioning of a human’s
central nervous system. These networks involve a large number of highly interconnected
processing elements called neurons which work in parallel to solve complex problems by
themselves, leaving them somewhat open to unpredictable outcomes. Each neuron is connected
with every other neuron by weight links through hidden layers, and the network requires training
using examples to perform a specific task [8]. The entire setup is referred to as a multi-layer
perceptron, also know as shallow neural networks. Standard ANNs are feedforward networks i.e.
no node is touched twice. ANNs’ advantages include adaptive learning, self-organization, and
real-time operation. Deep ANNs are an extension to such networks, where there are multiple
hidden layers and are therefore, capable of handling higher complexity [9].
<br>

A special type of ANN is the RNN. While feedforward networks like standard ANNs only
consider the current input, RNNs also consider what they had perceived previously in time i.e.
they also contain feedback loops. This attribute of an RNN makes them suitable for handling
sequential data i.e. the kind of text sequences needed to be processed in a chatbot. These networks
have shown promising results in the field of natural language processing including language
modeling, paraphrase detection, word embedding extraction, and neural machine translation
(NMT) [1, 11]. RNNs, without really understanding the symbol’s meaning, look at text structures
and relative symbol positions to infer the meaning. As highlighted before, they contain loops
which provide feedback and memory to the network periodically, allowing the system to learn
through sequence of inputs and not individual patterns. Its other applications include image
captioning, speech synthesis, and music generation.
<br>

This post discusses the application of RNN-encoder-decoder architecture for building chatbots.
These models, containing two RNNs, can be seen as English to English translation systems,
where “two RNNs are jointly trained to maximize conditional probability of the target sequence
given a source sequence” [11]. In recent years, additional components such as long short-term
memory (LSTM) cells and bidirectionality (BRNN) have shown to improve RNNs performance
for language translation, handwriting recognition, and image captioning. Thus, these systems
along with an attention mechanism, for improved handling of long-range dependencies, have
been incorporated in the chatbot to produce more meaningful responses [4]. This model and its
components are described in the next section.


<br>
Background
<br/>
RNNs have diverse architectures and thus, can be deployed for accomplishing various tasks. The
different types of RNN architectures are one-to-one which is considered as a generic neural
network architecture, one-to-many for music notes generation, many-to-one for sentiment
classification, many-to-many for machine translation where the input and output sequence lengths
are different, and many-to-many for named entity recognition where length of input and output
sequence are same [12]. There are other applications of RNNs as well. This paper’s focus is on
the many-to-many architecture used for translation, and it has two RNNs – encoder and decoder.
This architecture has also proved to be efficient for sequence to sequence (seq2seq) prediction
and allows for the training of a single model directly on source and target sentences, while
working with variable length input and output sequence text [2].

