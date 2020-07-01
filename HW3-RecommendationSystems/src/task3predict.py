import json
from pyspark import SparkContext
from pyspark import SparkConf
import sys
import time


# N nearest neighboors
N = 10


def sort(li):

	li.sort(key=lambda x: -x[1])

	return li


# input (predicting user, predicting business), [(business1, rating1), (business2, rating2)]
def item_predict(li):

	user = li[0][0]
	b1 = li[0][1]
	N_neighbors = []
	for subli in li[1]:
		b2 = subli[0]
		rating = subli[1]
		#print((b1, b2, rating))
		test_pair = tuple(sorted((b1, b2)))
		if model_dict.get(test_pair):
			N_neighbors.append([rating, model_dict[test_pair]])
	N_neighbors = sort(N_neighbors)[:N]
	numerator = sum([subli[0] * subli[1] for subli in N_neighbors])
	denominator	= sum([abs(subli[1]) for subli in N_neighbors])

	if numerator != 0 and denominator != 0:
		stars = numerator / denominator
	else:
		stars = 0

	return (user, b1, stars)


# input (predicting business, predicting user), [(user1, rating1), (user2, rating2)]
def user_predict(li):

	business = li[0][0]
	u1 = li[0][1]
	N_neighbors = []
	for subli in li[1]:
		u2 = subli[0]
		rating = subli[1]
		#print(u1, u2, rating)
		test_pair = tuple(sorted((u1, u2)))
		if model_dict.get(test_pair):
			N_neighbors.append([(rating, user_avg_dict[u2]), model_dict[test_pair]])
	#print(N_neighbors)
	N_neighbors = sort(N_neighbors)[:N]
	numerator = sum([(subli[0][0] - subli[0][1]) * subli[1] for subli in N_neighbors])
	denominator	= sum([abs(subli[1]) for subli in N_neighbors])

	if numerator != 0 and denominator != 0:
		stars = user_avg_dict[u1] + numerator / denominator
	else:
		stars = 0

	return (u1, business, stars)


if __name__ == '__main__':
	start = time.time()
	train_file = sys.argv[1]
	test_file = sys.argv[2]
	model_file = sys.argv[3]
	output_file = sys.argv[4]
	cf_type = sys.argv[5]

	conf=SparkConf()
	conf.set("spark.executor.memory", "4g")
	conf.set("spark.driver.memory", "4g")
	sc = SparkContext.getOrCreate(conf)

	train_review_lines = sc.textFile(train_file).map(lambda row: json.loads(row))
	train_review_rdd = train_review_lines.map(lambda row: (row['user_id'], row['business_id'], row['stars']))

	test_review_lines = sc.textFile(test_file).map(lambda row: json.loads(row))
	test_review_rdd = test_review_lines.map(lambda row: (row['user_id'], row['business_id']))

	model_lines = sc.textFile(model_file).map(lambda row: json.loads(row))

	if cf_type == 'item_based':
		# generate train file into structure: key=user_id, value=[(business_id, rating)]
		train_rdd = train_review_rdd.map(lambda x: (x[0], (x[1], x[2]))) \
		.groupByKey().map(lambda x: (x[0], list(set(x[1]))))
		# generate test file into structure: key=user_id, value=business_id
		test_rdd = test_review_rdd.map(lambda x: (x[0], x[1]))
		# generate model file into dictionary {(business_id1, business_id2): sim}
		model  = model_lines.map(lambda row: (row['b1'], row['b2'], row['sim'])) \
		.map(lambda x: ((x[0], x[1]), x[2])).collect()
		model_dict = dict()
		for li in model:
			model_dict[tuple(sorted(li[0]))] = li[1]
		# generate predictions
		predict = train_rdd.rightOuterJoin(test_rdd) \
		.filter(lambda x: x[1][0] != None) \
		.map(lambda x: ((x[0], x[1][1]), x[1][0])) \
		.filter(lambda x: x[0][1] != x[1][0]) \
		.groupByKey().map(lambda x: (x[0], [item for li in x[1] for item in li])) \
		.map(lambda x: item_predict(x)) \
		.filter(lambda x: x[2] != 0) \
		.collect()

	if cf_type == 'user_based':
		user_avg_file = train_file[:-17] + 'user_avg.json'
		#user_avg_file = '../resource/asnlib/publicdata/user_avg.json'

		# generate train file into structure: key=business_id, value=[(user_id, rating)]
		train_rdd = train_review_rdd.map(lambda x: (x[1], (x[0], x[2]))) \
		.groupByKey().map(lambda x: (x[0], list(set(x[1]))))
		# generate test file into structure: key=business_id, value=user_id
		test_rdd = test_review_rdd.map(lambda x: (x[1], x[0]))
		# generate model file into dictionary {(user_id1, user_id2): sim}
		model  = model_lines.map(lambda row: (row['u1'], row['u2'], row['sim'])) \
		.map(lambda x: ((x[0], x[1]), x[2])).collect()
		model_dict = dict()
		for li in model:
			model_dict[tuple(sorted(li[0]))] = li[1]
		# generate user_avg file into dictionary {user_id: avg}
		user_avg = sc.textFile(user_avg_file).map(lambda row: json.loads(row)) \
		.map(lambda x: dict(x)) \
		.flatMap(lambda x: [(key, val) for key, val in x.items()]).collect()
		user_avg_dict = dict()
		for li in user_avg:
			user_avg_dict[li[0]] = li[1]
		# generate predictions
		predict = train_rdd.rightOuterJoin(test_rdd) \
		.filter(lambda x: x[1][0] != None) \
		.map(lambda x: ((x[0], x[1][1]), x[1][0])) \
		.filter(lambda x: x[0][1] != x[1][0]) \
		.groupByKey().map(lambda x: (x[0], [item for li in x[1] for item in li])) \
		.map(lambda x: user_predict(x)) \
		.filter(lambda x: x[2] != 0) \
		.collect()

	with open(output_file, 'w+') as fout:
		for li in predict:
			fout.write(json.dumps({'user_id': li[0], 'business_id': li[1], 'stars': li[2]}) + '\n')

	end = time.time()
	print('Duration: {}'.format(end - start))

