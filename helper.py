##Get time from datetime
import pandas as pd
import math
def get_time_from_datetime(data):
   time = []
   for date in data["datetime"]:
    time.append(date.split(" ")[1].split(":")[0])

   data["time"] = time
   data.drop(["datetime"],axis=1,inplace=True)


def get_dummy_values_for_quant_features(features,data):
    for each in features:
        dummies = pd.get_dummies(data[each],prefix=each,drop_first=False)
        data = pd.concat([data,dummies],axis=1)
    data = data.drop(features,axis=1)
    return data


def get_scaled_features(features,data):
    for each in features:
        std , mean = data[each].mean(), data[each].std()
        data.loc[:,each] = (data[each]-mean)/std
    return data


def get_test_train_data(percentage,data):
    data = data.sample(frac=1).reset_index(drop=True)
    row_count = len(data)
    print (row_count)
    test_row_count  = int(math.ceil((100-percentage)*row_count/100))
    print (test_row_count)
    return data[:test_row_count],data[test_row_count:]

def get_target_features(target_fields,data):
    return data.drop(target_fields,axis=1), data[target_fields]

