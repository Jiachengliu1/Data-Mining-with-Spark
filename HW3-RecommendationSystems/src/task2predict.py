import json
import math
from pyspark import SparkContext
from pyspark import SparkConf
import sys
import time


def cosine_similarity(user_profile, business_profile):

	if user_profile == None or business_profile == None:
		result = 0
	else:
		up = set(user_profile)
		bp = set(business_profile)
		result = len(up & bp) / (math.sqrt(len(up)) * math.sqrt(len(bp)))
	
	return result


if __name__ == '__main__':
	start = time.time()
	f_in = sys.argv[1]
	model_file = sys.argv[2]
	f_out = sys.argv[3]

	conf=SparkConf()
	conf.set("spark.executor.memory", "4g")
	conf.set("spark.driver.memory", "4g")
	sc = SparkContext.getOrCreate(conf)

	review_lines = sc.textFile(f_in).map(lambda row: json.loads(row))
	review_rdd = review_lines.map(lambda row: (row['user_id'], row['business_id']))
		
	model_lines = sc.textFile(model_file).map(lambda row: json.loads(row))
	
	# get business profile and user profile into dictionary
	business_profile = model_lines.filter(lambda row: 'business' in row) \
	.map(lambda row: (row['business'], row['features'])).collect()
	business_profile_dict = dict()
	for li in business_profile:
		business_profile_dict[li[0]] = li[1]

	user_profile = model_lines.filter(lambda row: 'user' in row) \
	.map(lambda row: (row['user'], row['features'])).collect()
	user_profile_dict = dict()
	for li in user_profile:
		user_profile_dict[li[0]] = li[1]
	

	# compute cosine similarity and predict
	predict = review_rdd.map(lambda x: (x[0], x[1])) \
	.map(lambda x: ((x[0], x[1]), cosine_similarity(user_profile_dict.get(x[0]), business_profile_dict.get(x[1])))) \
	.filter(lambda x: x[1] >= 0.01).collect()

	with open(f_out, 'w+') as fout:
		for li in predict:
			fout.write(json.dumps({'user_id': li[0][0], 'business_id': li[0][1], 'sim': li[1]}) + '\n')

	end = time.time()
	print('Duration: {}'.format(end - start))

