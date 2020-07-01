import json
from pyspark import SparkContext
import sys


def top_n_categories(review_in, business_in, n):
	
	businessid_to_star = dict()
	businessid_to_category = dict()

	with open(review_in) as review:
	    lines = review.readlines()
	    review_li = [json.loads(line) for line in lines]
	    for review_dict in review_li:
	        if review_dict['business_id'] not in businessid_to_star:
	            businessid_to_star[review_dict['business_id']] = [review_dict['stars']]
	        else: 
	            businessid_to_star[review_dict['business_id']].append(review_dict['stars'])

	with open(business_in) as business:
	    lines = business.readlines()
	    business_li = [json.loads(line) for line in lines]
	    for business_dict in business_li:
	        if business_dict['categories'] != None:
	            businessid_to_category[business_dict['business_id']] = [element.strip() for element in business_dict['categories'].split(',')]

	result = {}
	for business_id, star in businessid_to_star.items():
	    if businessid_to_category.get(business_id) != None:
	        for category in businessid_to_category.get(business_id):
	            if category not in result:
	                result[category] = [star]
	            else:
	                result[category].append(star)

	for key, value in result.items():
	    value = [item for sublist in value for item in sublist]
	    result[key] = sum(value) / len(value)
	sorted_result = sorted(result.items(), key=lambda x: (-x[1],x[0]))
	top_n_categories = [list(element) for element in sorted_result[0: n]]

	return top_n_categories


if __name__ == '__main__':
	review_in = sys.argv[1]
	business_in = sys.argv[2]
	f_out = sys.argv[3]
	if_spark = sys.argv[4]
	n = sys.argv[5]

	output = dict()

	if if_spark == 'spark':
		sc = SparkContext.getOrCreate()
		review_lines = sc.textFile(review_in).map(lambda row: json.loads(row)) 
		business_lines = sc.textFile(business_in).map(lambda row: json.loads(row))

		top_categories = review_lines.map(lambda row: (row['business_id'], row['stars'])) \
		.join(business_lines.map(lambda row: (row['business_id'], row['categories']))) \
		.filter(lambda x: x[1][1] != None) \
		.map(lambda x: ([element.strip() for element in x[1][1].split(',')], x[1][0])) \
		.flatMap(lambda x: [(element, float(x[1])) for element in x[0]]) \
		.groupByKey() \
		.map(lambda x: (str(x[0]), list(x[1]))) \
		.map(lambda x: [x[0], sum(x[1]) / len(x[1])]) \
		.sortBy(lambda x: (-x[1], x[0])) \
		.take(int(n))

		output['result'] = top_categories

	elif if_spark == 'no_spark':
		output['result'] = top_n_categories(review_in, business_in, int(n))

	#print(output)

	with open(f_out, 'w+') as fout:
		json.dump(output, fout)

