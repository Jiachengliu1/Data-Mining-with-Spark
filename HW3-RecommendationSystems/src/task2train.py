from collections import defaultdict
import json
import math
from pyspark import SparkContext
from pyspark import SparkConf
import sys
import time
import re


def parse(text, stopwords):

	text = re.sub(r'[^\w\s]', ' ', text)
	text = ''.join(char for char in text if not char.isdigit())
	word_list = text.split()
	cleaned_word_list = [word for word in word_list if word not in stopwords]

	return cleaned_word_list


def calculate_tf(word_list):

	word_dict = defaultdict(int)
	for word in word_list:
		word_dict[word] += 1
	max_occ = max(word_dict.values())
	tf = list()
	for word, num in word_dict.items():
		tf.append([word, num / max_occ])

	return tf


def sort(li):

	li.sort(key=lambda x: -x[1])

	return li


if __name__ == '__main__':
	start = time.time()
	f_in = sys.argv[1]
	f_out = sys.argv[2]
	stopwords_in = sys.argv[3]

	conf=SparkConf()
	conf.set("spark.executor.memory", "4g")
	conf.set("spark.driver.memory", "4g")
	sc = SparkContext.getOrCreate(conf)

	review_lines = sc.textFile(f_in).map(lambda row: json.loads(row))
	review_rdd = review_lines.map(lambda row: (row['user_id'], row['business_id'], row['text']))

	# build business profile
	stopwords = list()
	with open(stopwords_in, 'r') as stop_words:
		stopwords = [line.strip() for line in stop_words]

	business_words_rdd = review_rdd.repartition(30) \
	.map(lambda x: (x[1], x[2].lower())) \
	.map(lambda x: (x[0], parse(x[1], stopwords))) \
	.groupByKey().map(lambda x: (x[0], [word for li in x[1] for word in li]))

	# calculate TF: (word, (business, TF))
	tf = business_words_rdd.mapValues(lambda x: calculate_tf(x)) \
	.flatMap(lambda x: [(x[0], li) for li in x[1]]) \
	.map(lambda x:(x[1][0], (x[0], x[1][1])))

	# calculate IDF: (word, IDF)
	business = review_rdd.map(lambda x: x[1]).distinct().collect()
	idf = business_words_rdd.flatMap(lambda x: [(word, x[0]) for word in x[1]]) \
	.groupByKey().mapValues(lambda x: math.log2(len(business) / len(set(x))))

	# extract features (top 200 words each business)
	business_profile = tf.join(idf) \
	.map(lambda x: (x[1][0][0], (x[0], x[1][0][1] * x[1][1]))) \
	.groupByKey().map(lambda x: (x[0], [li for li in sort(list(x[1]))[:200]]))

	# word to index mapping, save memory space
	words = business_profile.flatMap(lambda x: [li[0] for li in x[1]]).distinct().collect()
	word_dict = dict()
	for index, word in enumerate(words):
		word_dict[word] = index
	business_profile = business_profile.map(lambda x:(x[0], [(word_dict[li[0]], li[1]) for li in x[1]])).collect()

	# business profile mapping
	business_profile_dict = dict()
	for li in business_profile:
		business_profile_dict[li[0]] = li[1]

	# build user profile
	user_profile = review_rdd.map(lambda x: (x[0], x[1])) \
	.groupByKey().map(lambda x: (x[0], [business for business in set(x[1])])) \
	.map(lambda x: (x[0], [business_profile_dict[business] for business in x[1]])) \
	.map(lambda x: (x[0], list(set([word for li in x[1] for word in li])))) \
	.map(lambda x: (x[0], [li[0] for li in sort(x[1])[:500]])).collect()

	with open(f_out, 'w+') as fout:
		for element in business_profile:
			fout.write(json.dumps({'business': element[0], 'features': [li[0] for li in element[1]]}) + '\n')
		for element in user_profile:
			fout.write(json.dumps({'user': element[0], 'features': element[1]}) + '\n')

	end = time.time()
	print('Duration: {}'.format(end - start))

