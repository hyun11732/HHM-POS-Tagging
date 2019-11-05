
import numpy as np

class node :
    def __init__(self, idx) :
        self.idx = idx
        self.grand = True
    def parent(self, parent):
        self.parent = parent
        self.grand = False


def baseline(train, test):
    dic = dict()
    for sent in train :
        for i in sent :
            word = i[0]
            tag = i[1]
            if word not in dic :
                dic[word] = dict()
                dic[word][tag] = 1
            else :
                if tag not in dic[word] :
                    dic[word][tag] = 1
                else :
                    dic[word][tag] += 1
    predicts = []
    for sent in test :
        sent_list = []
        for word in sent :
            if word not in dic :
                sent_list.append((word, "NOUN"))
            else :
                tags = dic[word].keys()
                MAX = 0
                pred_tag = ""
                for tag in tags :
                    if MAX < dic[word][tag] :
                        MAX = dic[word][tag]
                        pred_tag = tag
                sent_list.append((word, pred_tag))
        predicts.append(sent_list)
    return predicts



def make_dic(train) :
    initial = dict()
    trans = dict()
    emiss = dict()
    words = set()
    tag_count = dict()
    part2 = dict()
    trans_total = 0
    tags = set()
    for sentence in train :
        start = sentence[0]
        tags.add(start[1])
        if start not in part2 :
            part2[start] = 1
        else :
            part2[start] += 1
        if start[1] not in initial :
            initial[start[1]] = 1
        else :
            initial[start[1]] += 1
        for i in range(len(sentence) - 1) :
            word1, tag1 = sentence[i]
            word2, tag2 = sentence[i + 1]
            tags.add(tag1)
            words.add(word1)
            if sentence[i] not in part2:
                part2[sentence[i]] = 1
            else:
                part2[sentence[i]] += 1
            if tag1 not in tag_count :
                tag_count[tag1] = 1
            else :
                tag_count[tag1] += 1
            if tag1 not in trans :
                trans[tag1] = dict()
                trans[tag1][tag2] = 1
                trans_total += 1
            else :
                if tag2 not in trans[tag1] :
                    trans[tag1][tag2] = 1
                    trans_total += 1
                else :
                    trans[tag1][tag2] += 1
                    trans_total += 1
            if tag1 not in emiss :
                emiss[tag1] = dict()
                emiss[tag1][word1] = 1
            else :
                if word1 not in emiss[tag1] :
                    emiss[tag1][word1] = 1
                else :
                    emiss[tag1][word1] += 1
        word, tag = sentence[len(sentence) - 1]
        tags.add(tag)
        words.add(sentence[len(sentence) - 1][0])
        if sentence[len(sentence) - 1] not in part2:
            part2[sentence[len(sentence) - 1]] = 1
        else:
            part2[sentence[len(sentence) - 1]] += 1
        if tag not in emiss:
            emiss[tag] = dict()
            emiss[tag][word] = 1
        else:
            if word not in emiss[tag]:
                emiss[tag][word] = 1
            else:
                emiss[tag][word] += 1
        if tag not in tag_count :
            tag_count[tag] = 1
        else :
            tag_count[tag] += 1
    hapex = dict()
    count = 0
    for pack in part2 :
        if(part2[pack] == 1) :
            count += 1
            if pack[1] not in hapex :
                hapex[pack[1]] = 1
            else :
                hapex[pack[1]] +=1
    for i in hapex :
        hapex[i] /= count
    return initial, trans, emiss, len(words), tag_count, hapex, list(tags)

def viterbi(train, test):
    smoothing_parameter = 0.00001
    initial_table, trans_table, emiss_table, words_len, tag_count, hapex, TAGS = make_dic(train)
    trans_prob_table = dict()
    emiss_prob_table = dict()
    initial_prob_table = dict()
    count = 0
    for tag in TAGS :
        if tag not in hapex :
            hapex[tag] = smoothing_parameter
    for tag in initial_table :
        count += initial_table[tag]
    for tag in initial_table :
        initial_prob_table[tag] = np.log((smoothing_parameter + initial_table[tag]) / (count + smoothing_parameter * (len(TAGS) + 1)))

    # CREATE TRANS TABLE
    for tag1 in TAGS :
        count = tag_count[tag1]
        if tag1 in trans_table :
            for tag2 in TAGS :
                if tag2 in trans_table[tag1] :
                    trans_prob_table[(tag1, tag2)] = np.log((hapex[tag2] + trans_table[tag1][tag2]) / (count + hapex[tag2] * (len(TAGS) + 1)))

                else :
                    trans_prob_table[(tag1, tag2)] = np.log((hapex[tag2] + 0) / (tag_count[tag1] + hapex[tag2] * (len(TAGS) + 1)))
        else :
            for tag2 in TAGS:
                trans_prob_table[(tag1, tag2)] = np.log((hapex[tag2] + 0) / (tag_count[tag1] + hapex[tag2] * (len(TAGS)+ 1)))
    #CREATE EMISSION PROB
    zero_emiss_prob = dict()
    for tag in TAGS :
        if tag in emiss_table :
            count = tag_count[tag]
            for word in emiss_table[tag] :
                emiss_prob_table[(tag, word)] = np.log((hapex[tag] + emiss_table[tag][word]) / (count + hapex[tag] * (len(TAGS) + 1)))
            zero_emiss_prob[tag] = np.log((hapex[tag] + 0) / (count + hapex[tag] * (len(TAGS)+ 1)))
        else :
            zero_emiss_prob[tag] = np.log((hapex[tag] + 0) / (tag_count[tag] + hapex[tag] * (len(TAGS) + 1)))

    initial0 = np.log((smoothing_parameter + 0) / (0 + smoothing_parameter * (len(TAGS) + 1)))
    predicts = []

    for sen in test :
        decoding_table = np.zeros((len(TAGS), (len(sen))))
        node_table = [[0] * len(sen) for i in range(len(TAGS))]
        start_word = sen[0]
        words = [start_word]
        for i, tag in enumerate(TAGS) :
            if tag not in initial_prob_table :
                decoding_table[i, 0] = initial0
            else :
                decoding_table[i, 0] = initial_prob_table[tag]
            if (tag, start_word) not in emiss_prob_table :
                decoding_table[i, 0] += zero_emiss_prob[tag]
            else :
                decoding_table[i, 0] += emiss_prob_table[(tag, start_word)]
            node_table[i][0] = node(i)
        real_argmax =0
        for j in range(1, len(sen)) :
            word = sen[j]
            words.append(word)
            MAX2 = -10000000000
            for i, curr_tag in enumerate(TAGS) :
                MAX = -100000000
                argMAX = 0
                for past_idx, past_tag in enumerate(TAGS) :
                    temp = decoding_table[past_idx, j - 1]
                    temp += trans_prob_table[(past_tag, curr_tag)]
                    if (curr_tag, word) in emiss_prob_table :
                        temp += emiss_prob_table[(curr_tag, word)]
                    else :
                        temp += zero_emiss_prob[curr_tag]
                    if(MAX < temp) :
                        MAX = temp
                        argMAX = past_idx
                decoding_table[i, j] = MAX
                if MAX2 < decoding_table[i, j] :
                    MAX2 = decoding_table[i, j]
                    real_argmax = i
                new_node= node(i)
                new_node.parent(argMAX)
                node_table[i][j] = new_node
        curr_node = node_table[real_argmax][len(sen) - 1]
        result = list()
        j = len(sen) - 1
        while curr_node.grand == False :
            result.append(curr_node.idx)
            j -= 1
            curr_node = node_table[curr_node.parent][j]
        result.append(curr_node.idx)
        result = reversed(result)
        sen_predict = list()
        for word, idx in zip(words, result) :
            sen_predict.append((word, TAGS[idx]))
        predicts.append(sen_predict)
    return predicts
