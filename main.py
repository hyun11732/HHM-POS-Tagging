import argparse
import sys
from viterbi import viterbi
import collections

tagset = {'NOUN', 'VERB', 'ADJ', 'ADV',
          'PRON', 'DET', 'IN', 'NUM',
          'PART', 'UH', 'X', 'MODAL',
          'CONJ', 'PERIOD', 'PUNCT', 'TO', 'START'}

def load_dataset(data_file):
    sentences = []
    with open(data_file, 'r', encoding='UTF-8') as f:
        for line in f:
            sentence = [('START', 'START')]
            raw = line.split()
            for pair in raw:
                splitted = pair.split('=')
                if (len(splitted) < 2):
                    continue
                else:
                    # find tag
                    if (splitted[-1] not in tagset):
                        tag = 'X'
                    else:
                        tag = splitted[-1]

                    # find word
                    word = splitted[0]
                    for element in splitted[1:-1]:
                        word += '/' + element
                    sentence.append((word.lower(), tag))
            sentences.append(sentence)
    return sentences

def strip_tags(sentences):
    '''
    Strip tags
    input:  list of sentences
            each sentence is a list of (word,tag) pairs
    output: list of sentences
            each sentence is a list of words (no tags)
    '''

    sentences_without_tags = []

    for sentence in sentences:
        sentence_without_tags = []
        for i in range(len(sentence)):
            pair = sentence[i]
            sentence_without_tags.append(pair[0])
        sentences_without_tags.append(sentence_without_tags)

    return sentences_without_tags

def evaluate_accuracies(predicted_sentences, tag_sentences):
    """
    :param predicted_sentences:
    :param tag_sentences:
    :return: (Accuracy, correct word-tag counter, wrong word-tag counter)
    """
    assert len(predicted_sentences) == len(tag_sentences)

    correct_wordtagcounter = {}
    wrong_wordtagcounter = {}
    correct = 0
    wrong = 0
    for pred_sentence, tag_sentence in zip(predicted_sentences, tag_sentences):
        assert len(pred_sentence) == len(tag_sentence)
        for pred_wordtag, real_wordtag in zip(pred_sentence, tag_sentence):
            assert pred_wordtag[0] == real_wordtag[0]
            word = pred_wordtag[0]
            if pred_wordtag[1] == real_wordtag[1]:
                if word not in correct_wordtagcounter.keys():
                    correct_wordtagcounter[word] = collections.Counter()
                correct_wordtagcounter[word][real_wordtag[1]] += 1
                correct += 1
            else:
                if word not in wrong_wordtagcounter.keys():
                    wrong_wordtagcounter[word] = collections.Counter()
                wrong_wordtagcounter[word][real_wordtag[1]] += 1
                wrong += 1

    accuracy = correct / (correct + wrong)

    return accuracy

if __name__ == "__main__":
    # Get training, testing datasets through parsers
    parser = argparse.ArgumentParser(description='Hidden Markov Models(HHM) enter the training set and testing set directory')
    parser.add_argument('--train', dest='training_file', type=str,
                        help='the file of the training data')
    parser.add_argument('--test', dest='test_file', type=str,
                        help='the file of the testing data')
    parser.add_argument('--save', dest='save_file_name', type = str,
                        help='saved filename for the prediction result')
    args = parser.parse_args()
    if args.training_file == None or args.test_file == None:
        sys.exit('You must specify training file and testing file!')
    print("Loading~~~")
    train_set = load_dataset(args.training_file)
    test_set = load_dataset(args.test_file)
    print("Running HHM right now may take some time depend on how large set is...")
    predictions = viterbi(train_set, strip_tags(test_set))
    accu = evaluate_accuracies(test_set, predictions)
    print("Accuracy: " + str(accu * 100)  + "%")
    if(args.save_file_name != None) :
        saved_file = open(args.save_file_name + ".txt", 'w')
        for sentence in predictions :
            for word_tags in sentence :
                saved_file.write(str(word_tags) + " ")
            saved_file.write('\n')
        saved_file.close()
