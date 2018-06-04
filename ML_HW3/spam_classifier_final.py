############################
#                          #
# authors:                 #
# Zixi Huang(zh2313)       # 
# Neil Kumar(nk2739)       # 
# Yichen Pan(yp2450)       #
#                          #
############################

import os
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from functools import reduce
from operator import mul
import csv


def preprocess(text):
    processed_text = []

    # tokenize
    tokens = text.split()

    # detect link
    link = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    # stemming
    porter_stemmer = PorterStemmer()

    # cleaning out the stopwords
    tokens = [token for token in tokens if token not in stopwords.words("english")]

    for token in tokens:
        # remove punctuation
        token = token.translate(str.maketrans('', '', string.punctuation))
        if not token == u'':
            if token.isdigit():  # number
                processed_text.append("NUMBER")
            elif token.upper() == token:  # all caps
                processed_text.append("ALL_CAPS")
                processed_text.append(porter_stemmer.stem(token).lower())
            elif link.match(token):  # link
                processed_text.append("LINK")
            else:
                processed_text.append(porter_stemmer.stem(token).lower())

    return processed_text


def get_email_list(file_path):
    email_list = {}
    for email in os.listdir(file_path):
        # Note: it will miss some files
        try:
            email_file = open(os.path.join(file_path, email), 'r')
            email_text = email_file.read()
            email_file.close()

            processed_text = preprocess(email_text)
            email_list[email] = processed_text
        except:
            continue
    print(len(email_list))
    return email_list


# Note: for test files
def get_email_list_v2(file_path):
    email_list = {}
    for email in os.listdir(file_path):
        try:
            email_file = open(os.path.join(file_path, email), 'r')
            email_text = email_file.read()
            email_file.close()

            processed_text = preprocess(email_text)
            email_list[email] = processed_text
        except:
            email_list[email] = 'bad file'
            continue
    print(len(email_list))
    return email_list


def get_word_probability(email_list):
    word_probability = {}
    number_of_email = len(email_list)

    # Count the number of each word in spam/ham
    for file_name, email in email_list.items():
        # email_word = []
        for word in email:
            # if word in email_word:
            #     continue
            # else:
            #     email_word.append(word)
            if word not in word_probability:
                word_probability[word] = 1
            else:
                word_probability[word] += 1

    # calculate word probability in spam/ham
    for word, number_of_word in word_probability.items():
        word_probability[word] = number_of_word / number_of_email

    return word_probability


def get_spam_probability(email, word_probability_in_spam, word_probability_in_ham, spam_probability):
    # Ignore words that have not been encountered
    word_list = [word for word in email if word in word_probability_in_spam and word in word_probability_in_ham]

    # select top 115 words
    top_word_list = sorted(word_list, key=lambda x: abs(0.5 - word_probability_in_spam[x]), reverse=True)[:115]

    # bayes thereom
    spam_probability = reduce(mul, [word_probability_in_spam[word] for word in top_word_list]) \
                       * spam_probability
    ham_probability = reduce(mul, [word_probability_in_ham[word] for word in top_word_list]) \
                      * (1 - spam_probability)

    return spam_probability / (spam_probability + ham_probability)

# spam and ham
directory = '/Users/cee/PycharmProjects/SpamClassifier/email_classification_data/train_data'
spam_path = os.path.join(directory, 'spam')
ham_path = os.path.join(directory, 'ham')

spam_list = get_email_list(spam_path)
ham_list = get_email_list(ham_path)

# test
test_path = '/Users/cee/PycharmProjects/SpamClassifier/email_classification_data/test_data'
test_list = get_email_list_v2(test_path)

# Get overall spam and ham word probability
word_probability_in_spam = get_word_probability(spam_list)
word_probability_in_ham = get_word_probability(ham_list)
spam_probability = len(spam_list) / (len(spam_list) + len(ham_list))

############################################
#                                          #
#                  predict                 #
#                                          #
############################################
predict = {}
for key in sorted(test_list):
    spam_probability_test = get_spam_probability(
        test_list[key], word_probability_in_spam, word_probability_in_ham, spam_probability
    )
    if spam_probability_test > 0.5:
        predict[key] = 1  # spam
    else:
        predict[key] = 0  # ham

# export csv
writer = csv.writer(open("prediction_final.csv", "w"))
for key, value in predict.items():
    writer.writerow([key[11:-4], value])

############################################
#                                          #
#    verify algorithm with spam and ham    #
#                                          #
############################################
#
# spam_count = 0
# ham_count = 0
#
# for key in sorted(spam_list):
#     spam_probability_test_spam = get_spam_probability(
#         spam_list[key], word_probability_in_spam, word_probability_in_ham, spam_probability
#     )
#     if spam_probability_test_spam > 0.5:
#         spam_count += 1
#
# for key in sorted(ham_list):
#     spam_probability_test_ham = get_spam_probability(
#         ham_list[key], word_probability_in_spam, word_probability_in_ham, spam_probability
#     )
#     if spam_probability_test_ham < 0.5:
#         ham_count += 1
#
# spam_success_rate = spam_count / len(spam_list)
# ham_success_rate = ham_count / len(ham_list)
#
# print("Spam success rate: {0:.2f}%".format(spam_success_rate * 100))
# print("Ham success rate: {0:.2f}%".format(ham_success_rate * 100))