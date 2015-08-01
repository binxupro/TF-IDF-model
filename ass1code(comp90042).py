# Author: Bin Xu 623087

# preprocess the plaintext read from file
def text_processed(raw):

    # del punctuation
    import string
    import re
    exclude = set(string.punctuation)
    data = raw.replace('\n', ' ')
    temp_withoutpunc = re.sub("[^a-zA-Z0-9]"," ",data)
    # temp_withoutpunc = ''.join(ch for ch in raw if ch not in exclude)

    # tokenization
    from nltk.tokenize import word_tokenize
    temp_tokenization = word_tokenize(temp_withoutpunc.lower())

    #remove stop words
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    temp_withoutstopw = [word for word in temp_tokenization if word not in stop_words]


    # stemming
    from nltk import PorterStemmer
    temp_stemming = []
    for word in temp_withoutstopw:
        temp_stemming.append(PorterStemmer().stem_word(word))
    return temp_stemming

# building inverted index for document contents with TF
def inverted_index_TF(inverted_index,tokens):
    for word in tokens:
        if word not in inverted_index:
            inverted_index[word] = {fname:1}
        else:
            inverted_index[word][fname] = tokens.count(word)
    return inverted_index


# building inverted index with weight
def inverted_index_TFIDF(inverted_index_tf,text_dirs,file_length):
    # cal IDF
    import math
    idf_dic = {}
    for item in inverted_index_tf:
        value_idf = 1 + math.log(len(text_dirs)/len(inverted_index_tf[item]))
        idf_dic[item] = value_idf

    # replace TF by TF*IDF in inverted_index
    for item in inverted_index_tf:
        for docid in inverted_index_tf[item]:
            w = inverted_index_tf[item][docid] * idf_dic[item]
            # nomalised weight
            w = w /file_length
            inverted_index_tf[item][docid] = w
    inverted_index_wight = inverted_index_tf
    return inverted_index_wight


# loading file and building inverted index
import os,sys
path="/Users/binxu/blogs/"
dirs = os.listdir(path)
text_dirs = [item for item in dirs if item.endswith('txt')]#only read text files
inverted_index = {}
for fname in text_dirs:
    full_name = path + fname
    # open file
    raw = open(full_name,'r+').read().decode('utf-8')
    len_file = len(raw)
    tokens = []
    for word in text_processed(raw):
        s = word.encode('utf-8')
        tokens.append(s)
    # invoke TF function
    inverted_indextf = inverted_index_TF(inverted_index,tokens)

# invoke TFIDF function
inverted_indexweight = inverted_index_TFIDF(inverted_indextf,text_dirs,len_file)


#this function is used for building the inverted_index for query
def inverted_index_query(inverted_index,tokens):
    for word in tokens:
        if word not in inverted_index:
            inverted_index[word] = tokens.count(word)
    return inverted_index
import re
q = open('/Users/binxu/blogs(part)/query/querys.txt','r')
query_raw = q.read()
query_compile = re.compile(r'<title>\s(.+)')
query_list = query_compile.findall(query_raw)

# output like this: {'851': ['global', 'warming'], '852' : ['apple', 'juice']}
query_number = 851
query_dic = {}
for item in query_list:
    query_dic[query_number] = item
    query_number +=1
for i in query_dic.keys():
    query_dic[i]=text_processed(query_dic[i])

# read from file and write down to file
# import pickle
# # output = open('output.txt','ab+')
# # pickle.dump(inverted_indexweight,output)
# # output.close()
# output = open('output.txt','r')
# inverted_indexweight = pickle.load(output)



for item in query_dic.keys():
    word_appeared = {}
    for word in query_dic[item]:
        # if inverted_indexweight.has_key(word):
            for i in inverted_indexweight[word].keys():
                if inverted_indexweight[word].get(i) in word_appeared:
                    word_appeared[i] += inverted_indexweight[word][i]
                else:
                    word_appeared[i] = inverted_indexweight[word][i]
            # word_appeared.update(inverted_indexweight[word])
    print item
    import operator
    word_appeared_sorted = sorted(word_appeared.items(),key=operator.itemgetter(1),reverse=True)
    print word_appeared_sorted[:]

