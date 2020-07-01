from collections import defaultdict
from itertools import combinations
import json
from pyspark import SparkContext
from pyspark import SparkConf
import random
import sys
import time


#n = 50; band = 50; row = 1 # accuracy: 0.9272482543955581, duration: 60 s
n = 100; band = 100; row = 1 # accuracy: 0.9978295617060654, duration: 120 s
#n = 200; band = 200; row = 1 # accuracy: 1.0, duration: 220 s


if __name__ == '__main__':
	start = time.time()
	f_in = sys.argv[1]
	f_out = sys.argv[2]

	conf=SparkConf()
	conf.set("spark.executor.memory", "4g")
	conf.set("spark.driver.memory", "4g")
	sc = SparkContext.getOrCreate(conf)

	review_lines = sc.textFile(f_in).map(lambda row: json.loads(row))
	review_rdd = review_lines.map(lambda row: (row['user_id'], row['business_id']))

	# user dictionary
	user = review_rdd.map(lambda x: x[0]).distinct().collect()
	#print(len(user)) 26184 users
	user_dict = dict()
	for index, user_id in enumerate(user):
		user_dict[user_id] = index
		
	review_index_rdd = review_rdd.map(lambda x: (user_dict[x[0]], x[1]))

	hashed_user = defaultdict(list)
	for i in range(len(user)):
		for j in range(n):
			a = random.randint(1, 100000)
			b = random.randint(1, 100000)
			p = 27077
			m = 26184
			hashed_user[i].append(((a * i + b) % p) % m)
	#print(hashed_user)

	# generate signature matrix in data structure of nested list: 10253 x n
	# [(business_id, [h1, h2, ... , hn])]
	signature_matrix = review_index_rdd.groupByKey() \
	.map(lambda x: (x[0], list(set(x[1])))) \
	.map(lambda x: (x[1], hashed_user[x[0]])) \
	.map(lambda x: ((business_id, x[1]) for business_id in x[0])) \
	.flatMap(lambda x: x) \
	.groupByKey().map(lambda x: (x[0], [list(x) for x in x[1]])) \
	.map(lambda  x: (x[0], [min(col) for col in zip(*x[1])])).collect()
	#print(signature_matrix.collect())

	# generate candidate pairs
	candidates = set()

	for band_num in range(band):
		bucket = defaultdict(set)
		for signature in signature_matrix:
			start_index = band_num * row 
			value = tuple()
			for row_num in range(row):
				value += (signature[1][start_index + row_num],)
			hashed_value = hash(value)
			bucket[hashed_value].add(signature[0])
			#print(bucket)
		for li in bucket.values():
			if len(li) >= 2:
				for pair in combinations(li, 2):
					candidates.add(tuple(sorted(pair)))
	#print(len(candidates))

	# business id to user ids dictionary
	bid_uid = review_rdd.map(lambda x: (x[1], x[0])) \
	.groupByKey().map(lambda x: (x[0], set(x[1]))).collectAsMap()

	# calculate similarity
	with open(f_out, 'w+') as fout:
		for pair in candidates:
			b1 = pair[0]
			b2 = pair[1]
			users1 = bid_uid[b1]
			users2 = bid_uid[b2]
			sim = float(len(users1 & users2) / len(users1 | users2))
			if sim >= 0.05:
				fout.write(json.dumps({'b1': b1, 'b2': b2, 'sim': sim}) + '\n')

	end = time.time()
	print('Duration: {}'.format(end - start))

