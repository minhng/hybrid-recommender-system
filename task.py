import sys
import numpy as np
import pandas as pd
from pyspark import SparkContext
import time
import json
import xgboost as xgb


def evaluate_RMSE(final_df, test_set):
    test_len = test_set.shape[0]
    assert test_len == final_df.shape[0]
    SE = 0  # squared error
    for i in range(test_len):
        SE += (final_df['avg_prediction'][i] - test_set['stars'][i]) ** 2
    RMSE = np.sqrt(SE / test_len)
    print('RMSE: ', RMSE)
    return RMSE


def compute_weight(item_1, item_2):
    corated = set(item_1.keys()).intersection(set(item_2.keys()))
    avg_sum1 = 0
    avg_sum2 = 0
    w_i_j = 0
    sum_sq_w_i = 0
    sum_sq_w_j = 0
    if not corated:
        return 0.1
    for user in corated:
        avg_sum1 += item_1[user]
        avg_sum2 += item_2[user]
    avg_1 = avg_sum1 / len(corated)
    avg_2 = avg_sum2 / len(corated)
    for user in corated:
        w_i_j += ((item_1[user] - avg_1) * (item_2[user] - avg_2))
        sum_sq_w_i += (item_1[user] - avg_1) ** 2
        sum_sq_w_j += (item_2[user] - avg_2) ** 2
    return sum_sq_w_j/(pow(sum_sq_w_i, 1/2) * pow(sum_sq_w_j, 1/2)) if w_i_j != 0 else 0.1


def predict(user_id, business_id, user_business_mapping, business_user_mapping, business_mean):
    pred = 0
    total_w = 0
    if user_id not in user_business_mapping and business_id not in business_user_mapping:
        return (user_id, business_id), 3.0
    elif user_id not in user_business_mapping or business_id not in business_user_mapping:
        return (user_id, business_id), 3.0
    for user in user_business_mapping[user_id]:
        similarity_w = compute_weight(business_user_mapping[business_id], business_user_mapping[user[0]])
        if similarity_w != 0:
            pred += ((user[1] - business_mean[user[0]]) * similarity_w)
            total_w += similarity_w
    return (user_id, business_id), float(pred/total_w + business_mean[business_id])


def compute_model_based(X_train_set):
    with open(folder_path + '/user.json') as json_file:
        data = json_file.readlines()
        user_data = list(map(json.loads, data))
    user_data_df = pd.DataFrame(user_data)[['user_id', 'average_stars', 'review_count', 'useful', 'fans', 'compliment_hot', 'compliment_more', 'compliment_note']]
    user_data_df.columns = ['user_id', 'user_stars', 'user_review_count', 'user_useful', 'user_fans', 'user_hot', 'user_more', 'user_note']

    with open(folder_path + '/business.json') as json_file:
        data = json_file.readlines()
        business_data = list(map(json.loads, data))
    business_data_df = pd.DataFrame(business_data)[['business_id', 'stars', 'review_count', 'is_open']]
    business_data_df.columns = ['business_id', 'business_stars', 'business_review_count', 'business_open']

    X_train = pd.merge(X_train_set, user_data_df, on='user_id', how='left')
    X = pd.merge(X_train, business_data_df, on='business_id', how='left')

    train_y = X.stars.values
    train_x = X.drop(["stars"], axis=1)
    train_x = train_x.drop(["user_id"], axis=1)
    train_x = train_x.drop(["business_id"], axis=1)
    train_x = train_x.values

    # Fitting XGB regressor
    model = xgb.XGBRegressor(max_depth=6, n_estimators=350)
    model.fit(train_x, train_y)
    print(model)

    X_test_set = pd.read_csv(test_file_name, sep=',')
    user_id_vals = X_test_set.user_id.values
    business_id_vals = X_test_set.business_id.values
    X_test = pd.merge(X_test_set, user_data_df, on='user_id', how='left')
    X_test = pd.merge(X_test, business_data_df, on='business_id', how='left')

    test_x = X_test.drop(["stars"], axis=1)
    test_x = test_x.drop(["user_id"], axis=1)
    test_x = test_x.drop(["business_id"], axis=1)
    test_x = test_x.values
    # Predict
    output = model.predict(data=test_x)
    final_df = pd.DataFrame()
    final_df["user_id"] = user_id_vals
    final_df["business_id"] = business_id_vals
    final_df["prediction"] = output

    return final_df, X_test_set


def write_file(data):
    output = 'user_id, business_id, prediction'+'\n'
    with open(output_file_name, 'w+') as outputFile:
        for i in range(data.shape[0]):
            output += "{},{},{}\n".format(data['user_id_x'][i], data['business_id_x'][i], data['avg_prediction'][i])
        outputFile.write(output)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("""Usage:
                 spark-submit task.py <folder_path> <test_file_name> <output_file_name>  
                 spark-submit task.py ./dataset yelp_val.csv result.csv
              """)
        sys.exit(1)
    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]
    start = time.time()
    with SparkContext('local[*]') as sc:
        sc.setLogLevel("ERROR")
        trainRDD = sc.textFile(folder_path+'/yelp_train.csv', minPartitions=2).persist()
        cf = trainRDD.filter(lambda x: x != "user_id,business_id,stars")\
            .map(lambda x: ((x.split(',')[0], x.split(',')[1]), float(x.split(',')[2]))).persist()
        valRDD = sc.textFile(test_file_name, minPartitions=2)\
            .filter(lambda x: x != "user_id,business_id,stars")\
            .map(lambda x: (x.split(',')[0], x.split(',')[1])).persist()

        ### Calculate mean
        business_mean = cf.map(lambda x: (x[0][1], [x[1]])).reduceByKey(lambda acc, n: acc + n)\
            .map(lambda x: (x[0], sum(x[1]) / len(x[1]))).collectAsMap()
        business_user_mapping = cf.map(lambda x: (x[0][1], {x[0][0]: x[1]})).collectAsMap()
        user_business_mapping = cf.map(lambda x: (x[0][0], [(x[0][1], x[1])]))\
            .reduceByKey(lambda acc, n: acc + n).collectAsMap()
        ### Predict test set
        ret = valRDD.map(lambda x: predict(x[0], x[1], user_business_mapping, business_user_mapping, business_mean))
        cf_df = pd.DataFrame(ret.map(lambda x: [x[0][0], x[0][1], x[1]]).collect())
        cf_df.columns = ['user_id', 'business_id', 'cf_prediction']
        cf_df['pair'] = cf_df['user_id'] + cf_df['business_id']

        ### Model based
        X = pd.DataFrame(trainRDD.filter(lambda x: x != "user_id,business_id,stars")
                         .map(lambda x: [x.split(',')[0], x.split(',')[1], x.split(',')[2]]).collect())
        X.columns = ['user_id', 'business_id', 'stars']
        rs_df, val_set = compute_model_based(X)
        rs_df['pair'] = rs_df['user_id']+rs_df['business_id']

        combined_df = pd.merge(rs_df, cf_df, on='pair', how='left')
        combined_df['avg_prediction'] = ((0.7*pd.to_numeric(combined_df['prediction'])) + (0.3*pd.to_numeric(combined_df['cf_prediction'])))
        combined_df = combined_df.drop(["prediction", "pair", "user_id_y", "business_id_y", "cf_prediction"], axis=1)

        evaluate_RMSE(combined_df, val_set)

        write_file(combined_df)
        print('time: ', time.time() - start)

