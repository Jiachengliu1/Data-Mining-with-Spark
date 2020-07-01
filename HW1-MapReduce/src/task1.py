import json
from operator import add
from pyspark import SparkContext
import sys


def elimiate_punctuations(row):

	punctuations = ['(', '[', ',', '.', '!', '?', ':', ';', ']', ')']
	for char in row:
		if char in punctuations:
			row = row.replace(char, '')

	return row


if __name__ == '__main__':
	f_in = sys.argv[1]
	f_out = sys.argv[2]
	stopwords_in = sys.argv[3]
	y = sys.argv[4]
	m = sys.argv[5]
	n = sys.argv[6]

	sc = SparkContext.getOrCreate()
	review_lines = sc.textFile(f_in).map(lambda row: json.loads(row)) 

	# A
	num_reviews = review_lines.count()

	# B
	num_reviews_by_year = review_lines.map(lambda row: row['date'][0:4]) \
	.filter(lambda year: int(year) == int(y)).count()

	# C
	num_distinct_users = review_lines.map(lambda row: row['user_id']) \
	.distinct().count()

	# D
	top_users = review_lines.map(lambda row: (row['user_id'], row['review_id'])) \
	.groupByKey() \
	.map(lambda x: [str(x[0]), len(list(x[1]))]) \
	.sortBy(lambda x: (-x[1], x[0])) \
	.take(int(m))

	# E
	with open(stopwords_in, 'r') as stop_words:
		stopwords = [line.strip() for line in stop_words]
	top_words = review_lines.map(lambda x: elimiate_punctuations(x['text'].lower())) \
	.flatMap(lambda x: x.split()) \
	.map(lambda x: (x, 1)) \
	.reduceByKey(add) \
	.filter(lambda x: x[0] not in stopwords) \
	.sortBy(lambda x: (-x[1], x[0])) \
	.take(int(n))

	top_freq_words = [element[0] for element in top_words]

	output = dict()
	output['A'] = num_reviews
	output['B'] = num_reviews_by_year
	output['C'] = num_distinct_users
	output['D'] = top_users
	output['E'] = top_freq_words

	#print(output)

	with open(f_out, 'w+') as fout:
		json.dump(output, fout)

