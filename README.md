# RNN Based Chatbot using Cornell Movie Dataset

#Dependencies:
<br/> tensorflow==1.0.1
<br/> numpy
<br/> re
<br/> time


Important Resources:
<br/> Chatbots with Seq2Seq: http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/
<br/> Neural Machine Translation (seq2seq) Tutorial: https://github.com/tensorflow/nmt
<br/> Sequence Models: https://www.coursera.org/learn/nlp-sequence-models 
<br/> Deep Learning and NLP A-ZTM: How to create a ChatBot: https://www.udemy.com/chatbot/learn/v4/overview


Other Relevant Resources:
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


1. Introduction
<br/>It has long been the focus for Artificial Intelligence (AI) to build systems that can maintain a
dialog with humans in an intuitive manner. This post discusses the implementation of a chatbot
which will enable users to indulge in an one-to-one multi-turn conversation. The movie-dialog
corpus considered for this task is not specific to any particular domain and provides a brief
explanation of what can be achieved if replicated using other more problem specific datasets. One
such dataset is the Ubuntu corpus, which if considered with the discussed model, can be
instrumental in providing technical support as it includes conversations related to technical
domain. Additionally, the considered dataset contains unstructured conversations, which favors
the development of a chatbot that could lead a general conversation [2].
<br/>

The chatbot in discussion leverages deep ANNs to do away with older rule-based models which
are restricted in terms of their interpretation of questions and generation of responses, and thus,
cannot handle complex / unstructured queries. Deep ANNs introduce a generative element which
allows the chatbot to look at each word in the query separately and also the entire query as a
whole to generate new / unfamiliar answers compared to the training dataset, thus, equipping the
chatbot to handle complex queries [7]. However, in case exhaustive training is not done, such
models may be prone to grammatical and spelling errors.
<br/>

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
<br/>

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
<br/>

This post discusses the application of RNN-encoder-decoder architecture for building chatbots.
These models, containing two RNNs, can be seen as English to English translation systems,
where “two RNNs are jointly trained to maximize conditional probability of the target sequence
given a source sequence” [11]. In recent years, additional components such as long short-term
memory (LSTM) cells and bidirectionality (BRNN) have shown to improve RNNs performance
for language translation, handwriting recognition, and image captioning. Thus, these systems
along with an attention mechanism, for improved handling of long-range dependencies, have
been incorporated in the chatbot to produce more meaningful responses [4]. This model and its
components are described in the next section.


2. Background
<br/>
RNNs have diverse architectures and thus, can be deployed for accomplishing various tasks. The
different types of RNN architectures are one-to-one which is considered as a generic neural
network architecture, one-to-many for music notes generation, many-to-one for sentiment
classification, many-to-many for machine translation where the input and output sequence lengths
are different, and many-to-many for named entity recognition where length of input and output
sequence are same [12]. There are other applications of RNNs as well. This post’s focus is on
the many-to-many architecture used for translation, and it has two RNNs – encoder and decoder.
This architecture has also proved to be efficient for sequence to sequence (seq2seq) prediction
and allows for the training of a single model directly on source and target sentences, while
working with variable length input and output sequence text [2].
<br/>

![alt text](https://github.com/jain-abhinav/chatbot_cornell_movie_database/blob/master/RNN%20Architectures.png)
<br/>Figure 1: RNN architectures [12]


2.1. Previous Research
<br/>
The encoder-decoder architecture is relatively new, and was first introduced for neural machine
translation and seq2seq learning in 2014. It was deployed for English to French translation. Each
output sentence was padded with a special end-of-sentence symbol “<EOS>” during training to
mark completion of translated sequence, allowing the model to predict variable length sequences.
The input sequence, read in entirety, was encoded to a fixed-length internal representation or
thought vector. The decoder then used this internal representation to produce an output word
sequence until the sequence token end was reached. Two LSTM networks were used, one for each
encoder and decoder [3]. Google succeeded in implementing this architecture for the company’s
translation services [13]. Another paper discussing a similar architecture [11], used this technique
to score candidate translations, rather than for translations itself. The source and target vocabulary
were limited to most frequent 15000 French and English words, covering 93% of the dataset. A
different research highlighted that the performance of the network decreases quickly when the
number of words outside the vocabulary or the input sequence length increases [14]. To address
these two issues, it was suggested to increase the vocabulary size and introduce an attention
mechanism [4]. This allowed the model to focus on specific parts of the source sentence i.e. ones
containing most relevant information, when generating a word. Thus, the word at decoder is
predicted by looking at the context vector for source positions and previous inputs.
  

2.2. Encoder-Decoder RNN Architecture
<br/>
In this architecture, both the encoder and decoder layers have their distinct weights / parameters
that are required to be optimized while training the model. In the below figure, h 0 , h 1 and so on
are input nodes, and g 0 , g 1 and so on are output nodes; all nodes are considered as time-steps.
Each node represents a word, and the parameters are shared for each time-step across a layer of
an RNN. When considering a deep RNN, every layer has its distinct set of parameters. Activation
value (t 0 ) is initialized as a zero vector, and passed on from one time-step to another. W is the
parameter that controls the connection between the activation value and hidden layer, while U is
the parameter that governs the connection between input sequence and hidden layer. As it can be
observed, information from one time-step is passed on to the next until h 4 , which finally stores the
encoder state or thought vector. The thought vector is processed by the decoder which calculates
the conditional probability of each word for the output, given a certain input. V is the parameter
that governs the connection between hidden layer and output sequence. The output considers the
thought vector and previous output i.e. (figure 2) for g 0 , the word “thank” is also considered and a
conditional probability is calculated for the next word. This procedure is repeated till an EOS
token is predicted, leaving room for variable length outputs [3, 11]. Calculation for activation
value and output prediction involve activation functions tanh and softmax, respectively, which
also help in mitigating vanishing gradient issue. Figure 2 depicts forward propagation in RNNs.
<br/>

![alt text](https://github.com/jain-abhinav/chatbot_cornell_movie_database/blob/master/seq-seq%20architecture.png)
<br/>Figure 2: Single hidden layer encoder-decoder RNN architecture with attention [10, 16]


2.2.1 Backpropagation Through Time (BPTT)
The above discussed parameters are optimized and the network learns through backpropagation
using a loss function. During training, loss value for each time-step is calculated by comparing
the predicted and expected output, and these are summed to get the overall loss. Backpropagation
calculations are carried out in the opposite direction to the forward propagation arrows marked in
the above figure. The parameter values are computed and updated periodically using gradient
descent or other optimizers [1, 16]. The most intensive recursive calculation is the one across
activation values, which is from right to left instead of left to right as in the case of forward
propagation i.e. in decreasing time-steps. Thus, the algorithm is stated as backpropagation
through time.


2.2.2 Vanishing Gradient Problem
<br/>
For certain sequences, language can have very long-term dependencies where a word occurring in
the beginning could impact a word towards the end of the sentence. While carrying out forward
and backpropagation, the error outputs for later time-steps find it hard to propagate back to affect
computations of earlier time-steps. i.e. gradient decreases exponentially or vanishes. Essentially
4meaning that an output word in RNN is mostly affected by the words closer to it. Exploding
gradients, which is the opposite of vanishing gradient is observed in few cases and is easier to
identify as the network jams and outputs NaN values [1, 16]. Gradient clipping is a robust and
easy solution for exploding gradient. Whereas, vanishing gradient is a more complex issue and
can be addressed by incorporating Greater Recurrent Units or LSTMs.


2.2.3 LSTM
<br/>
LSTMs, a special form of RNNs, are equipped with handling long-term dependency issues
discussed above. They memorize a relevant state for a word in the beginning, which if required,
can be applied to a word towards the end i.e. they preserve the error that is backpropagated
through time and layers [1, 15]. Consider an input query such as “Cats, who have eaten (15 words
in between), are they hungry?”. By looking at the word “Cats”, it can be implied that the word
should be represented with “are” by the chatbot. However, the model without LSTM, given the
high number of words in between, will find it difficult to carry on this information from the word
“Cats” to the word “are” due to the vanishing gradient problem, and might instead predict “is”.


2.2.4 Bidirectionality
<br/>
As highlighted before, the network functions from left to right, and therefore words appearing
after a particular time-step are not considered for computing parameters for that time-step. This
limitation results in a lower performance, as there could be an instance where a word occurring in
the beginning is described by words following it i.e. there is a higher chance of missing out on
crucial information, instrumental for correctly interpreting a word [1, 16]. To bridge this gap,
bidirectional RNNs are implemented which consider both previous and upcoming input words for
generating an output response. An additional backward RNN layer with backward activation
values is included for this. For instance, consider two statements “Teddy, President of US, should
be hosted at which location?” and “Teddy bear is placed at what position?”. In case of an
unidirectional RNN, the chatbot might not be able to identify which “Teddy” is being talked
about.


2.2.5 Attention Mechanism
<br/>
Through this framework, each output word has access to input context vectors and the model
searches for input positions that can provide the most relevant information. The information from
context vectors is fed in the decoder cell, along with other parameters which helps in predicting a
more accurate word. In figure 2 it can be observed that when the decoder is trying to predict the
word “she”, it looks at the the context vector, and accordingly uses the contained information to
make the prediction. Attention mechanism is instrumental in handling long-range dependencies as
it is not much impacted by the length of the input query i.e. the context vector is independent of
other parameters used for output prediction [4].
<br>
While LSTMs and attention mechanism seem to address the same problem of long-range
dependencies, there is a subtle difference in how complex a problem they can handle and how
they approach it. LSTMs encode the inputs in an hidden layer using the fixed amount of memory
they have. For the chatbot to generate sentences i.e. complex predictions, different parts of an
input sentence need to be focused while predicting words at various positions, which is achieved
using the attention mechanism [4, 11, 15]. So essentially, LSTMs encode the entire sentence at
once in one or more vectors, and attention mechanism allows the model to additionally access
context vectors for each input word while predicting an output word, making them more effective
for long-range dependencies in sequential data.


<br>
References
<br>
[1] C. Lipton, Z. and Berkowitz, J. (2015). A Critical Review of Recurrent Neural Networks for Sequence Learning. arXiv preprint arXiv:1506.00019. Available from: https://arxiv.org/pdf/1506.00019.pdf [accessed 25 April 2018].
<br>
[2] Lowe, R., Pow, N., V. Serban, I. And Pineau, J. (2016). The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems. arXiv preprint arXiv:1506.08909. Available from: https://arxiv.org/pdf/1506.08909.pdf [accessed 25 April 2018].
<br>
[3] Sutskever, I., Vinyals, O. and V. Le, Q. (2014). Sequence to Sequence Learning with Neural Netowrks. arXiv preprint arXiv:1409.3215. Available from: https://arxiv.org/pdf/1409.3215.pdf [accessed 25 April 2018]
<br>
[4] Bahdanau, D., Cho, K. and Bengio, Y. (2016). Neural Machine Translation by Jointly Learning to Align and Translate. arXiv preprint arXiv:1409.0473. Available from: https://arxiv.org/pdf/1409.0473.pdf [accessed 25 April 2018]
<br>
[5] Radziwill, N. and Benton, M. (2017). Evaluating Quality of Chatbots and Intelligent Conversational Agents. arXiv preprint arXiv:1704.04579. Available from: https://arxiv.org/ftp/arxiv/papers/1704/1704.04579.pdf [accessed 25 April 2018]
<br>
[6] Chatbots Magazine. (2018). Why It’s Important For a Business To Have a Chatbot. [online] Available from: https://chatbotsmagazine.com/why-its-important-for-a-business-to-have-a-chatbot-28e446ce7167 [accessed 25 April 2018].
<br>
[7] V. Serban, I., Sordoni, A., Bengio, Y., Courville, A. and Pineau, J. (2016). Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models. Proc. of AAAI [online]. pp.3776-3783. Available from: https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/11957/12160 [accessed 25 April 2018]
<br>
[8] Kumar, P. and Sharma, P. (2014). Artifical Neural Networks-A Study. International Journal of Emerging Engineering Research and Technology, 2(2), pp.143-148. Available from: http://www.ijeert.org/pdf/v2-i2/24.pdf [accessed 25 April 2018]
<br>
[9] Chris V. Nicholson, S. (2018). Introduction to Deep Neural Networks (Deep Learning) - Deeplearning4j: Open-source, Distributed Deep Learning for the JVM. [online] Deeplearning4j.org. Available at: https://deeplearning4j.org/neuralnet-overview [accessed 25 April 2018].
<br>
[10]  Udemy. (2018). Deep Learning and NLP A-ZTM: How to create a ChatBot. [online] Available at: https://www.udemy.com/chatbot/learn/v4/overview [accessed 26 April 2018]
<br>
[11]  Cho, K., Bahdanau, D., Bougares, F., Schwenk, H. and Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. arXiv preprint arXiv:1406.1078. Available from: https://arxiv.org/pdf/1406.1078.pdf [accessed 25 April 2018]
<br>
[12]  Karpathy, A. (2015). The Unreasonable Effectiveness of Recurrent Neural Networks. GitHub [online], 21 May. Available from: http://karpathy.github.io/2015/05/21/rnn-effectiveness/ [accessed 25 April 2018]
<br>
[13]  Wu, Y., Schuster, M., Chen, Z., V. Le, Q. and Norouzi, M. (2016). Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation. arXiv preprint arXiv:1609.08144. Available from: https://arxiv.org/pdf/1609.08144.pdf [accessed 25 April 2018]
<br>
[14]  Cho, K., Van Merrienboer, B. and Bahdanau, D. (2014). On the Properties of Neural Machine Translation: Encoder–Decoder Approaches. arXiv preprint arXiv:1409.1259. Available from: https://arxiv.org/pdf/1409.1259.pdf [accessed 25 April 2018]
<br>
[15]  Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), pp.1735-1780. Available from: http://www.bioinf.jku.at/publications/older/2604.pdf [accessed 25 April 2018]
<br>
[16]  Coursera. Sequence Models. [online] Available from: https://www.coursera.org/learn/nlp-sequence-models?authMode=login [accessed 25 April 2018]
