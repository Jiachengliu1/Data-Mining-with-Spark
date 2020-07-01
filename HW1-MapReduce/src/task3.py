import json
from operator import add
from pyspark import SparkContext
import sys


if __name__ == '__main__':
	review_in = sys.argv[1]
	f_out = sys.argv[2]
	partition_type = sys.argv[3]
	n_partitions = sys.argv[4]
	n = sys.argv[5]
	
	sc = SparkContext.getOrCreate()
	review_lines = sc.textFile(review_in).map(lambda row: json.loads(row)) \
	.map(lambda x: (str(x['business_id']), 1))

	# default use 27 partitions, around 43000 items each, took around 4.6 s
	if partition_type == 'default':
		review_lines = review_lines
	# when using 20 partitions, took around 0.5 s
	elif partition_type == 'customized':
		review_lines = review_lines.partitionBy(int(n_partitions), lambda x: ord(x[0]) - ord(x[-1]))

	num_partitions = review_lines.getNumPartitions()
	num_items = review_lines.glom().map(len).collect()
	reviews_per_business = review_lines.reduceByKey(add) \
	.filter(lambda x: x[1] > int(n)) \
	.map(lambda x: [x[0], x[1]]).collect()

	output = dict()
	output['n_partitions'] = num_partitions
	output['n_items'] = num_items
	output['result'] = reviews_per_business
	
	#print(output)

	with open(f_out, 'w+') as fout:
		json.dump(output, fout)

