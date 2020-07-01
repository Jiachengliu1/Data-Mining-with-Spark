import csv
import json
from pyspark import SparkContext
import sys


if __name__ == '__main__':
	# review_in = 'review.json'
	review_in = sys.argv[1]
	# business_in = 'business.json'
	business_in = sys.argv[2]
	# f_out = 'user_business.csv'
	f_out = sys.argv[3]
	# state = 'NV'
	state = sys.argv[4]

	sc = SparkContext.getOrCreate()

	review_lines = sc.textFile(review_in).map(lambda row: json.loads(row)) 
	business_lines = sc.textFile(business_in).map(lambda row: json.loads(row))

	filtered_business = business_lines.map(lambda row: (row['business_id'], row['state'])) \
	.filter(lambda x: x[1] == state) \
	.map(lambda x: x[0]).collect()

	user_business = review_lines.map(lambda row: (row['user_id'], row['business_id'])) \
	.filter(lambda x: x[1] in filtered_business).collect()

	with open(f_out, 'w+') as fout:
		writer =  csv.writer(fout)
		writer.writerow(['user_id', 'business_id'])
		for line in user_business:
			writer.writerow(line)

