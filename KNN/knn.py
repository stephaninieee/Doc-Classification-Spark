from pyspark import SparkContext
sc = SparkContext()

import re
import numpy as np

# load up all of the 19997 documents in the corpus
corpus = sc.textFile ("s3://chrisjermainebucket/comp330_A6/20_news_same_line.txt")

# each entry in validLines will be a line from the text file
validLines = corpus.filter(lambda x : 'id' in x)

# now we transform it into a bunch of (docID, text) pairs
keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:]))

# now we split the text in each (docID, text) pair into a list of words
# after this, we have a data set with (docID, ["word1", "word2", "word3", ...])
# we have a bit of fancy regular expression stuff here to make sure that we do not
# die on some of the documents
regex = re.compile('[^a-zA-Z]')
keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

# now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", 1) ("word2", 1)...
allWords = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))

# now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
allCounts = allWords.reduceByKey (lambda a, b: a + b)

sortedWords = allCounts.sortBy(lambda x: (-x[1], x[0]))
# and get the top 20,000 words in a local array
# each entry is a ("word1", count) pair
topWords = sortedWords.take(20000)

# and we'll create a RDD that has a bunch of (word, dictNum) pairs
# start by creating an RDD that has the number 0 thru 20000
# 20000 is the number of words that will be in our dictionary
twentyK = sc.parallelize(range(20000))

# now, we transform (0), (1), (2), ... to ("mostcommonword", 1) ("nextmostcommon", 2), ...
# the number will be the spot in the dictionary used to tell us where the word is located
# HINT: make use of topWords in the lambda that you supply
dictionary = twentyK.map (lambda x:(topWords[x][0],x))


# Task 1 
def doc_to_vec(word_list):
    vec = np.zeros(20000)
    for i in word_list:
        vec[i] += 1
    return vec

pairs_word_doc = keyAndListOfWords.flatMap(lambda doc: ((word, doc[0]) for word in doc[1]))
doc_word_pairs = dictionary.join(pairs_word_doc).map(lambda x: (x[1][1], x[1][0]))
final_counts = (doc_word_pairs.groupByKey().map(lambda doc: (doc[0], doc_to_vec(list(doc[1])))))

result_1 = final_counts.lookup('20_newsgroups/comp.graphics/37261')
result_2 = final_counts.lookup('20_newsgroups/talk.politics.mideast/75944')
result_3 = final_counts.lookup('20_newsgroups/sci.med/58763')


print(result_1[0][result_1[0].nonzero()])
print(result_2[0][result_2[0].nonzero()])
print(result_3[0][result_3[0].nonzero()])

# Task 2
from math import log
docFreq = keyAndListOfWords.flatMap(lambda x: set((word, 1) for word in x[1])) \
                           .reduceByKey(lambda x, y: x + y)

corpusSize = 19997
broadcastDF = sc.broadcast(docFreq.collectAsMap())
idf = docFreq.map(lambda x: (x[0], log(corpusSize / x[1]))).collectAsMap()

broadcastIDF = sc.broadcast(idf)
broadcastDict = sc.broadcast(dictionary.collectAsMap())

# Function to compute TF-IDF vector for a document
def TFIDF(doc, broadcastDict, broadcastIDF):
    vec = np.zeros(20000)
    idfMap = broadcastIDF.value
    for word in doc:
        if word in broadcastDict.value:
            tf = doc.count(word) / len(doc)
            idf = idfMap.get(word, 0) 
            vec[broadcastDict.value[word]] = tf * idf
    return vec

# Transform the documents into TF-IDF vectors
doc_to_vec_TFIDF = keyAndListOfWords.map(lambda x: (x[0], TFIDF(x[1], broadcastDict, broadcastIDF)))

result_1 = doc_to_vec_TFIDF.lookup('20_newsgroups/comp.graphics/37261')
result_2 = doc_to_vec_TFIDF.lookup('20_newsgroups/talk.politics.mideast/75944')
result_3 = doc_to_vec_TFIDF.lookup('20_newsgroups/sci.med/58763')

print(result_1[0][result_1[0].nonzero()])
print(result_2[0][result_2[0].nonzero()])
print(result_3[0][result_3[0].nonzero()])

# Task 3
from collections import Counter
def predictLabel(k, text):
    preprocess_text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    input_vec = TFIDF(preprocess_text , broadcastDict, broadcastIDF)
    broadcastVec = sc.broadcast(input_vec)
    distances = doc_to_vec_TFIDF.map(lambda doc: (doc[0], np.linalg.norm(doc[1] - broadcastVec.value)))
    nearest_docs = distances.takeOrdered(k, key=lambda x: x[1])

    nearest_labels = [doc_id.split('/')[1] for doc_id, _ in nearest_docs] 
    most_common_label, _ = Counter(nearest_labels).most_common(1)[0]
    return most_common_label

result_1 = predictLabel (10, 'Graphics are pictures and movies created using computers, usually referring to image data created by a computer specifically with help from specialized graphical hardware and software. It is a vast and recent area in computer science. The phrase was coined by computer graphics researchers Verne Hudson and William Fetter of Boeing in 1960. It is often abbreviated as CG, though sometimes erroneously referred to as CGI. Important topics in computer graphics include user interface design, sprite graphics, vector graphics, 3D modeling, shaders, GPU design, implicit surface visualization with ray tracing, and computer vision, among others. The overall methodology depends heavily on the underlying sciences of geometry, optics, and physics. Computer graphics is responsible for displaying art and image data effectively and meaningfully to the user, and processing image data received from the physical world. The interaction and understanding of computers and interpretation of data has been made easier because of computer graphics. Computer graphic development has had a significant impact on many types of media and has revolutionized animation, movies, advertising, video games, and graphic design generally.')
result_2 = predictLabel (10, 'A deity is conceived in various cultures as either a natural or supernatural being deemed divine or sacred. Monotheistic religions recognize a single deity, known as God, while polytheistic religions worship multiple deities. Henotheistic religions acknowledge one supreme deity without denying the existence of other deities, viewing them as aspects of the same divine principle. Some non-theistic religions do not believe in a supreme creator deity but accept a pantheon of deities that live, die, and are reborn. A male deity is referred to as a god, and a female deity as a goddess. According to the Oxford reference, a deity can be a god or goddess in a polytheistic religion, or anything regarded as divine. C. Scott Littleton defines a deity as a being with powers beyond those of ordinary humans, interacting with humans to elevate their consciousness beyond everyday concerns.')
result_3 = predictLabel (10, "Egypt, officially the Arab Republic of Egypt, is a transcontinental country spanning the northeast corner of Africa and southwest corner of Asia by a land bridge formed by the Sinai Peninsula. Egypt is a Mediterranean country bordered by the Gaza Strip and Israel to the northeast, the Gulf of Aqaba to the east, the Red Sea to the east and south, Sudan to the south, and Libya to the west.  Across the Gulf of Aqaba lies Jordan, and across from the Sinai Peninsula lies Saudi Arabia, although Jordan and Saudi Arabia do not share a land border with Egypt.  It is the worlds only contiguous Eurafrasian nation.  Egypt has among the longest histories of any modern country, emerging as one of the worlds first nation states in the tenth millennium BC. Considered a cradle of civilisation, Ancient Egypt experienced some of the earliest developments of writing, agriculture, urbanisation, organised religion and central government.  Iconic monuments such as the Giza Necropolis and its Great Sphinx, as well the ruins of Memphis, Thebes, Karnak, and the Valley of the Kings, reflect this legacy and remain a significant focus of archaeological study and popular interest worldwide.  Egypts rich cultural heritage is an integral part of its national identity, which has endured, and at times assimilated, various foreign influences, including Greek, Persian, Roman, Arab, Ottoman, and European.  One of the earliest centers of Christianity, Egypt was Islamised in the seventh century and remains a predominantly Muslim country, albeit with a significant Christian minority.")
result_4 = predictLabel (10, "The term atheism, deriving from the Greek 'atheos' meaning 'without god(s)', was initially used pejoratively against those rejecting the societal gods. Its application narrowed following the rise of freethought, skeptical inquiry, and increasing criticism of religion during the Age of Enlightenment. The 18th century saw the first self-identified atheists, with the French Revolution marking a significant political movement advocating human reason's supremacy. Atheist arguments range from philosophical to social and historical, citing lack of empirical evidence, the problem of evil, inconsistent revelations, non-falsifiable concepts, and nonbelief. While atheists may adopt secular philosophies like humanism and skepticism, there is no single ideology that unites all atheists.")
result_5 = predictLabel (10, "Established by President Dwight D. Eisenhower in 1958, NASA was formed with a civilian orientation to encourage peaceful applications in space science. The National Aeronautics and Space Act passed on July 29, 1958, transitioning from its predecessor, the National Advisory Committee for Aeronautics (NACA), to the new operational agency on October 1, 1958. NASA has led U.S. space exploration efforts, including the Apollo moon-landing missions, Skylab space station, and the Space Shuttle program. It currently supports the International Space Station and oversees the Orion Multi-Purpose Crew Vehicle, the Space Launch System, and Commercial Crew vehicles development. NASA is also responsible for the Launch Services Program (LSP), managing launch operations and countdown for unmanned launches.")
result_6 = predictLabel (10, "The transistor is the fundamental building block of modern electronic devices, and is ubiquitous in modern electronic systems.  First conceived by Julius Lilienfeld in 1926 and practically implemented in 1947 by American physicists John Bardeen, Walter Brattain, and William Shockley, the transistor revolutionized the field of electronics, and paved the way for smaller and cheaper radios, calculators, and computers, among other things.  The transistor is on the list of IEEE milestones in electronics, and Bardeen, Brattain, and Shockley shared the 1956 Nobel Prize in Physics for their achievement.")
result_7 = predictLabel (10, "The Colt Single Action Army, also known as the Single Action Army, SAA, Model P, Peacemaker, M1873, and Colt .45, is a single-action revolver with a revolving cylinder holding six metallic cartridges. Designed for the U.S. government service revolver trials of 1872 by Colt's Patent Firearms Manufacturing Company, it was adopted as the standard military service revolver until 1892. Available in over 30 calibers and various barrel lengths, its design has remained consistent since 1873. Despite discontinuation, its production resumed due to popular demand. The revolver, favored by ranchers, lawmen, and outlaws, is now primarily purchased by")
result_8 = predictLabel (10, "Howe was recruited by the Red Wings and made his NHL debut in 1946. He led the league in scoring each year from 1950 to 1954, then again in 1957 and 1963. He ranked among the top ten in league scoring for 21 consecutive years and set a league record for points in a season (95) in 1953. He won the Stanley Cup with the Red Wings four times, won six Hart Trophies as the league's most valuable player, and won six Art Ross Trophies as the leading scorer. Howe retired in 1971 and was inducted into the Hockey Hall of Fame the next year. However, he came back two years later to join his sons Mark and Marty on the Houston Aeros of the WHA. Although in his mid-40s, he scored over 100 points twice in six years. He made a brief return to the NHL in 1979-80, playing one season with the Hartford Whalers, then retired at the age of 52. His involvement with the WHA was central to their brief pre-NHL merger success and forced the NHL to expand their recruitment to European talent and to expand to new markets.")




print(result_1)
print(result_2)
print(result_3)
print(result_4)
print(result_5)
print(result_6)
print(result_7)
print(result_8)
