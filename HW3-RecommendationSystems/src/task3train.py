import json
from pyspark import SparkContext
from pyspark import SparkConf
import sys
import time
from collections import defaultdict
from itertools import combinations
import math
import random


n = 50; band = 50; row = 1


def pearson_similarity(ur1, ur2):

    corelated_users = set(ur1.keys()) & set(ur2.keys())
    corelated_item1 = [ur1[user] for user in corelated_users]
    corelated_item2 = [ur2[user] for user in corelated_users]
    avg1 = sum(corelated_item1) / len(corelated_users)
    avg2 = sum(corelated_item2) / len(corelated_users)
    new_corelated_item1 = [rating - avg1 for rating in corelated_item1]
    new_corelated_item2 = [rating - avg2 for rating in corelated_item2]
    
    numerator = 0
    for i in range(len(new_corelated_item1)):
        temp = new_corelated_item1[i]*new_corelated_item2[i]
        numerator += temp

    denominator = math.sqrt(sum([rating**2 for rating in new_corelated_item1])) * math.sqrt(sum([rating**2 for rating in new_corelated_item2]))

    if numerator > 0 and denominator > 0:
        result = numerator / denominator
    else:
        result = 0

    return result


def jaccard_similarity(br1, br2):


    user1 = set(br1.keys())
    user2 = set(br2.keys())
    result = float(len(user1 & user2) / len(user1 | user2))

    return result


if __name__ == '__main__':
    start = time.time()
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    cf_type = sys.argv[3]

    conf=SparkConf()
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.driver.memory", "4g")
    sc = SparkContext.getOrCreate(conf)

    review_lines = sc.textFile(train_file).map(lambda row: json.loads(row))
    review_rdd = review_lines.map(lambda row: (row['user_id'], row['business_id'], row['stars']))
    
    business = review_rdd.map(lambda x: x[1]).distinct().collect()

    if cf_type == 'item_based':
        # build structure: (key=business, value=(user, rating))
        business_userrating = review_rdd.map(lambda x: (x[1], (x[0], x[2]))) \
        .groupByKey().map(lambda x: (x[0], list(x[1]))) \
        .filter(lambda x: len(x[1]) >= 3).collect()

        # business into dictionary: {business: {user1:rating1, user2:rating2}}
        business_userrating_dict = defaultdict(dict)
        for li in business_userrating:
            bid = li[0]
            userratings = li[1]
            for userrating in userratings:
                business_userrating_dict[bid][userrating[0]] = userrating[1]

        #print(business_userrating_dict)
        # generate all business pairs with pearson similarity with at least 3 co-related users and similarity > 0
        business = review_rdd.map(lambda x: x[1]).distinct().collect()
        with open(model_file, 'w+') as fout:
            for pair in combinations(business, 2):
                b1 = pair[0]
                b2 = pair[1]
                ur1 = business_userrating_dict[b1]
                ur2 = business_userrating_dict[b2]
                if len(set(ur1.keys()) & set(ur2.keys())) >= 3:
                    pearsonSimilarity = pearson_similarity(ur1, ur2)
                    if pearsonSimilarity > 0:
                        fout.write(json.dumps({'b1': b1, 'b2': b2, 'sim': pearsonSimilarity}) + '\n')

    elif cf_type == 'user_based':
        # build structure: (key=user, value=(business, rating))
        user_businessrating = review_rdd.map(lambda x: (x[0], (x[1], x[2]))) \
        .groupByKey().map(lambda x: (x[0], list(x[1]))) \
        .filter(lambda x: len(x[1]) >= 3).collect()

        # user into dictionary: {user: {business1:rating1, business2:rating2}}
        user_businessrating_dict = defaultdict(dict)
        for li in user_businessrating:
            uid = li[0]
            businessratings = li[1]
            for businessrating in businessratings:
                user_businessrating_dict[uid][businessrating[0]] = businessrating[1]

        # generate signature matrix in data structure of nested list: 26184 x n
        # [(user_id, [h1, h2, ... , hn])]
        business_dict = dict()
        for index, business_id in enumerate(business):
            business_dict[business_id] = index
        
        review_index_rdd = review_rdd.map(lambda x: (business_dict[x[1]], x[0]))

        hashed_business = defaultdict(list)
        for i in range(len(business)):
            for j in range(n):
                a = random.randint(1, 100000)
                b = random.randint(1, 100000)
                p = 10343
                m = 10253
                hashed_business[i].append(((a * i + b) % p) % m)

        signature_matrix = review_index_rdd.groupByKey() \
        .map(lambda x: (x[0], list(set(x[1])))) \
        .map(lambda x: (x[1], hashed_business[x[0]])) \
        .map(lambda x: ((user_id, x[1]) for user_id in x[0])) \
        .flatMap(lambda x: x) \
        .groupByKey().map(lambda x: (x[0], [list(x) for x in x[1]])) \
        .map(lambda  x: (x[0], [min(col) for col in zip(*x[1])])).collect()

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

        # generate all similar user pairs with at least 3 co-related businesses, similarity >0, and jaccard similarity >= 0.01
        with open(model_file, 'w+') as fout:
            for pair in candidates:
                u1 = pair[0]
                u2 = pair[1]
                br1 = user_businessrating_dict[u1]
                br2 = user_businessrating_dict[u2]
                if len(set(br1.keys()) & set(br2.keys())) >= 3:
                    pearsonSimilarity = pearson_similarity(br1, br2)
                    if pearsonSimilarity > 0:
                        if jaccard_similarity(br1, br2) >= 0.01:
                            fout.write(json.dumps({'u1': u1, 'u2': u2, 'sim': pearsonSimilarity}) + '\n')

    end = time.time()
    print('Duration: {}'.format(end - start))

