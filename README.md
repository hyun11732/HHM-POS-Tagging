# HHM-POS-Tagging

In Natural Language Processing(NLP), a part of speech tagging(POS-TAG) is very important to analyze a sentence structure.
Let's say we have a sentence,

 "I am going to eat a dinner."

If we can know POS-Tagging of this sentence, many NLP work will be easier.
Translation will be one example.
To translate this sentence to another language, we need to know what is pronoun, verb, adjective, noun and etc.
If we know POS-tagging, we can just change the word from a foreign language dictionary and swipe a grammar order based on tagging.
Also, grammar check can be another example.
By checking a pos-tags in sentence and their order, we can detect grammar error.

Then, how do we find pos-tags for each word in a sentence?

We will use Hidden Markov Models(HMM) to attach a pos-tag.

There are 2 assumptions we should make before we use HHM

1. The probability of a current tag is depend only on a previous tag.

2. The probability of a current word is depend only on a current tag.

However, these assumptions are not usually true. We are making this assumptions just to apply HHM.

There are 2 types of probability we need to calculate before apply HHM

1. $P(Tag_{k} | Tag_{k - 1})$ The conditional probability of the current tag depend on the last tag(Transition probability)
2. $P(Word | Tag)$ The conditional probability of the current word depend on the current tags(Emission probability)

Based on these two types of probability we need to build HHM model like this.


![equation](https://github.com/hyun11732/HHM-POS-Tagging/blob/master/img/img2.JPG)

First, we need a network(2d-array) with size of n X m when n  is the number of tags and m is the number of words in a sentence.

To find the current tag we should calculate ARGMAX(𝑣𝑠∈𝑇𝐴𝐺𝑆(𝑘+1,𝑡𝑎𝑔𝑘−1)=𝑣𝑠∈𝑇𝐴𝐺𝑆(𝑘,𝑡𝑎𝑔𝑘−1)∗𝑃(𝑇𝑎𝑔𝑘|𝑇𝑎𝑔𝑘−1)∗𝑃(𝑤𝑜𝑟𝑑|𝑇𝑎𝑔𝑘)).

After we find the argmax  past tag of $V_{s \in TAGS}(k+1, tag_{k-1})$ we should connect the current tag and the argmax past tag to construct network.

After modeling the network, we need find argmax node locating at the last column and we need to backtrack our nodes through connection we built.

The route that backtracked will be our result.

There are brown and masc corpuses in this data and accuracy of both corpuses are over 95%.


### How to use this program?

1. open the terminal
2. move your current dir to the folder
3. type :

python main.py  --train "train dir" --test "test dir" --save "save dir"
