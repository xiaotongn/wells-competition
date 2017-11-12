import pandas as pd
import numpy as np
import os, pdb, sys
from tqdm import tqdm
from fuzzywuzzy import fuzz
import collections
from sklearn import preprocessing


def fill_nan(df, col, output_type, value, replace=["0"]):
    """ col, output_type: what is the type of df[col] 
        value: the value to substitute missing values with
        replace: a list with the list of values to consider as missing """
    df[col] = df[col].fillna(value)
    if output_type == str:
        df[col] = df[col].str.lower()
    df[col] = df[col].replace(replace, value)
    return df

def names(df, col):
    fullnames_dic = {x: '' for x in df[col]}
    fullnames_list = [x for x in df[col]]
    for key in fullnames_dic:
        for i in range(len(key.split())):
            fullnames_dic[key] += key.split()[i][0]
    counter = collections.Counter(fullnames_list)
    return fullnames_dic, fullnames_list, counter

def fuzzymatch_one(counter, tol_partial, tol_sort, tol_set):
    """ Find duplicates, starting with the most frequent ones"""
    sorted_keys_desc = [x[0] for x in counter.most_common()]
    sorted_keys_asc = [x[0] for x in sorted(counter.items(), key=lambda x: x[1])]
    matched = {}
    fuzzdic = {}
    for key in tqdm(sorted_keys_desc):
        if (key not in matched) and (key not in fuzzdic):
            matched[key] = [(key, counter[key])]
            fuzzdic[key] = (key, key, counter[key])
            for item in sorted_keys_asc:
                if (item != key) and (item not in fuzzdic):
                    p0 = fuzz.ratio(key, item)
                    p1 = fuzz.partial_ratio(key, item)
                    p2 = fuzz.token_sort_ratio(key, item)
                    p3 = fuzz.token_set_ratio(key, item)
                    if (p1>tol_partial and p3>tol_set) or (p2>tol_sort and p3>tol_set):
                        matched[key].append((item, counter[item]))
                        fuzzdic[item] = (key, item, counter[item])
    return matched, fuzzdic

def hand_match_method1(matched, fuzzdic, key_list, value):
    for key in key_list:
        fuzzdic[key] = (value, key, fuzzdic[key][2]) 
        matched[value].append((key, fuzzdic[key][2]))
        if key in matched:
            for key2 in matched[key]:
                fuzzdic[key2[0]] = (value, key2[0], fuzzdic[key2[0]][2])
            del matched[key]
    return matched, fuzzdic

def hand_match_method2(matched, fuzzdic, value, string):
    for key in fuzzdic:
        if (string in fuzzdic[key][0]) and (key != value):
            fuzzdic[key] = (value, key, fuzzdic[key][2])
            matched[value].append((key, fuzzdic[key][2]))
            if key in matched:
                del matched[key]
    return matched, fuzzdic

def hand_remove(matched, fuzzdic, oldkey, newkey, value):
    fuzzdic[value[0]] = (newkey, value[0], value[1])
    matched[oldkey].remove(value)
    if newkey in matched:
        matched[newkey].append((value[0], value[1]))
    else:
        matched[newkey] = [(value[0], value[1])]
    return matched, fuzzdic

def shrink_list(fuzzdic, matched, tol, value):
    for key in matched:
        sum_ = sum([matched[key][i][1] for i in range(len(matched[key]))])
        if sum_ < tol:
            for item in [x for x in matched[key]]:
                fuzzdic[item[0]] = (value, item[0], item[1])
    return fuzzdic

def installer_cleaning_train(train):
    # clean installer column
    col = "installer"

    train = fill_nan(train, col, str, "-999", ["0", "not know", "not kno"])
    train_names_dic, train_names_list, train_names_counter = names(train, col)
    train_match, train_fuzz = fuzzymatch_one(train_names_counter, 50, 50, 90)

    train_match, train_fuzz = hand_remove(train_match, train_fuzz, "government", "local government", \
                                      ("village government", 10))
    key_list = ['rc ch', 'rc c', 'roman church', 'roman catholic', 'roman cathoric -kilomeni', \
                'roman cathoric -same', 'rc cathoric', 'roman catholic rulenge diocese', \
                'roman ca', 'roman', 'roman cathoric same', \
                'roman catholic rulenge diocese', 'roman ca', 'roman cathoric -kilomeni', \
                'roman cathoric -same', 'roman church', 'roman catholic', 'rcchurch/cefa']
    value = 'rc church'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['central govt', 'centr', 'tanzania government', 'tcrs /government', 'concern /government', \
                'adra /government', 'ministry of water', 'idara ya maji', 'wizara ya maji', \
                'gove', 'wachina', 'kuwait', 'go', 'gover']
    value = 'government'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['finn water']
    value = 'fini water'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['consultant engineer', 'citizen engine', 'howard and humfrey consultant']
    value = 'consulting engineer'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['local te', 'local contract', 'local fundi', 'local technical tec', 'local technical']
    value = 'local  technician'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['sengerema water department', 'halmashauri ya wilaya sikonge', 'region water department', \
                'district water depar', 'distri', 'kigoma municipal', 'dwe', 'dw', 'district water department']
    value = 'local government'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['co', 'commu']
    value = 'community'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['wananchi', 'village', 'villa', 'villagers']
    value = 'village council'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['priva', 'chacha']
    value = 'private'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['finw', 'finwater']
    value = 'fini water'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['unisef']
    value = 'unicef'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['kkt']
    value = 'kkkt'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['would bank', 'mileniam project']
    value = 'world bank'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    string = 'local t'
    value = 'kkkt church'
    train_match, train_fuzz = hand_match_method2(train_match, train_fuzz, value, string)
    string = 'italy'
    value = 'italian government'
    train_match, train_fuzz = hand_match_method2(train_match, train_fuzz, value, string)
    string = 'would vission'
    value = 'world vision'
    train_match, train_fuzz = hand_match_method2(train_match, train_fuzz, value, string)
    string = 'plan int'
    value = 'plan internationa'
    train_match, train_fuzz = hand_match_method2(train_match, train_fuzz, value, string)
    string = 'rotery c'
    value = 'rotary club'
    train_match, train_fuzz = hand_match_method2(train_match, train_fuzz, value, string)
    train_match, train_fuzz = hand_remove(train_match, train_fuzz, "government", "foreign government", \
                                          ('italy government', 3))
    train_match, train_fuzz = hand_remove(train_match, train_fuzz, "government", "foreign government", \
                                        ('italian government', 1))
    train_match, train_fuzz = hand_remove(train_match, train_fuzz, "government", "foreign government", \
                                        ('japan government', 1))
    train_match, train_fuzz = hand_remove(train_match, train_fuzz, "government", "foreign government", \
                                        ('belgiam government', 8))
    train_match, train_fuzz = hand_remove(train_match, train_fuzz, "government", "foreign government", \
                                        ('finland government', 16))
    train_match, train_fuzz = hand_remove(train_match, train_fuzz, "government", "foreign government", \
                                        ('kuwait', 195))
    train_match, train_fuzz = hand_remove(train_match, train_fuzz, "government", "foreign government", \
                                        ('wachina', 60))
    train_match, train_fuzz = hand_remove(train_match, train_fuzz, "government", "foreign government", \
                                        ('egypt government', 2))
    train_match, train_fuzz = hand_remove(train_match, train_fuzz, "government", "foreign government", \
                                        ('british government', 1))
    key_list = ['holland']
    value = 'foreign government'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    train_fuzz = shrink_list(train_fuzz, train_match, 50, "small installer")
    train[col] = [train_fuzz[x][0] for x in train[col]]
    return train

def installer_cleaning_test(train):
    # clean installer column
    col = "installer"

    train = fill_nan(train, col, str, "-999", ["0", "not know", "not kno"])
    train_names_dic, train_names_list, train_names_counter = names(train, col)
    train_match, train_fuzz = fuzzymatch_one(train_names_counter, 50, 50, 90)

    train_match, train_fuzz = hand_remove(train_match, train_fuzz, "government", "local government", \
                                        ("local government", 1))
    train_match, train_fuzz = hand_remove(train_match, train_fuzz, "government", "local government", \
                                        ("village government", 1))
    key_list = ['rc ch', 'rc c', 'roman church', 'roman catholic', 'roman cathoric -kilomeni', \
                'roman cathoric -same', 'rc cathoric', 'rulenge diocese', 'roman catholic rulenge diocese', \
                'roman ca', 'roman cathoric and water board', 'roman', 'roman cathoric same', \
                'roman catholic rulenge diocese', 'roman ca', 'roman cathoric -kilomeni', \
                'roman cathoric -same', 'roman church', 'roman catholic', 'rcchurch/cefa']
    value = 'rc church'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['central govt', 'centr', 'tanzania government', 'tcrs /government', 'concern /government', \
                'adra /government', 'ministry of water', 'ministry of healthy', 'idara ya maji', 'wizara ya maji', \
                'gove', 'wachina', 'kuwait', 'kuwit', 'gover']
    value = 'government'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['consultant engineer', 'citizen engine', 'howard and humfrey consultant']
    value = 'consulting engineer'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['local te', 'local contract', 'local fundi', 'local technical tec', 'local technical']
    value = 'local  technician'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['sengerema water department', 'halmashauri ya wilaya sikonge', 'region water department', \
                'district water depar', 'distri', 'kigoma municipal']
    value = 'local government'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['wananchi', 'village', 'villagers']
    value = 'village council'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['priva', 'mzee shekhe', 'mzee ole', 'mzee maisha', 'mzee chacha', 'chacha', 'mzee kizunda', \
                'mzee matiti edwin']
    value = 'private'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['sengerema water department', 'halmashauri ya wilaya sikonge', 'region water department', \
                'district water depar', 'distri', 'kigoma municipal', 'dwe', 'dw', 'district water department']
    value = 'local government'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['finw', 'finwater', 'finwter']
    value = 'fini water'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['unisef']
    value = 'unicef'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['co', 'commu']
    value = 'community'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['kkt']
    value = 'kkkt'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['would bank', 'mileniam project']
    value = 'world bank'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    string = 'local t'
    value = 'kkkt church'
    train_match, train_fuzz = hand_match_method2(train_match, train_fuzz, value, string)
    string = 'italy'
    value = 'italian government'
    train_match, train_fuzz = hand_match_method2(train_match, train_fuzz, value, string)
    string = 'would vission'
    value = 'world vision'
    train_match, train_fuzz = hand_match_method2(train_match, train_fuzz, value, string)
    string = 'china henan contractor'
    value = 'china henan construction'
    train_match, train_fuzz = hand_match_method2(train_match, train_fuzz, value, string)
    string = 'plan int'
    value = 'plan internationa'
    train_match, train_fuzz = hand_match_method2(train_match, train_fuzz, value, string)
    string = 'rotery c'
    value = 'rotary club'
    train_match, train_fuzz = hand_match_method2(train_match, train_fuzz, value, string)
    train_match, train_fuzz = hand_remove(train_match, train_fuzz, "government", "foreign government", \
                                        ('italy government', 1))
    train_match, train_fuzz = hand_remove(train_match, train_fuzz, "government", "foreign government", \
                                        ('italian government', 2))
    train_match, train_fuzz = hand_remove(train_match, train_fuzz, "government", "foreign government", \
                                        ('japan government', 1))
    train_match, train_fuzz = hand_remove(train_match, train_fuzz, "government", "foreign government", \
                                        ('belgiam government', 2))
    train_match, train_fuzz = hand_remove(train_match, train_fuzz, "government", "foreign government", \
                                        ('finland government', 7))
    train_match, train_fuzz = hand_remove(train_match, train_fuzz, "government", "foreign government", \
                                        ('kuwait', 65))
    train_match, train_fuzz = hand_remove(train_match, train_fuzz, "government", "foreign government", \
                                        ('kuwit', 1))

    train_fuzz = shrink_list(train_fuzz, train_match, 50, "small installer")
    train[col] = [train_fuzz[x][0] for x in train[col]]
    return train

def scheme_name_cleaning_train(train):
    col = "scheme_name"
    train = fill_nan(train, col, str, "-999", ["0", "not know", "not kno"])
    train_names_dic, train_names_list, train_names_counter = names(train, col)
    train_match, train_fuzz = fuzzymatch_one(train_names_counter, 50, 50, 90)    
    train_match['local water supplied scheme'] = train_match['ngana water supplied scheme']
    key_list = ['nasula gravity water supply']
    value = 'local water supplied scheme'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    string = 'supply'
    value = 'local water supplied scheme'
    train_match, train_fuzz = hand_match_method2(train_match, train_fuzz, value, string)
    string = 'supplied'
    value = 'local water supplied scheme'
    train_match, train_fuzz = hand_match_method2(train_match, train_fuzz, value, string)
    string = 'gravity'
    value = 'local water supplied scheme'
    train_match, train_fuzz = hand_match_method2(train_match, train_fuzz, value, string)
    string = 'mradi wa maji'
    value = 'local water supplied scheme'
    train_match, train_fuzz = hand_match_method2(train_match, train_fuzz, value, string)
    string = 'water sup'
    value = 'local water supplied scheme'
    train_match, train_fuzz = hand_match_method2(train_match, train_fuzz, value, string)
    string = 'water project'
    value = 'local water supplied scheme'
    train_match, train_fuzz = hand_match_method2(train_match, train_fuzz, value, string) 
    string = 'pipe line'
    value = 'local water supplied scheme'
    train_match, train_fuzz = hand_match_method2(train_match, train_fuzz, value, string)
    train_fuzz = shrink_list(train_fuzz, train_match, 50, "small scheme")
    train[col] = [train_fuzz[x][0] for x in train[col]]
    return train

def funder_cleaning_train(train):
    # clean installer column
    col = "funder"

    train = fill_nan(train, col, str, "-999", ["0", "not know", "not kno"])
    train_names_dic, train_names_list, train_names_counter = names(train, col)
    train_match, train_fuzz = fuzzymatch_one(train_names_counter, 50, 50, 90)

    key_list = ['rc ch', 'roman church', 'roman catholic', 'roman cathoric -kilomeni', \
                'rc cathoric', 'roman catholic rulenge diocese', 'roman', \
                'roman ca', 'roman', 'roman cathoric same', 'rcchurch/cefa']
    value = 'rc church'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['ministry of water', 'go', 'idara ya maji']
    value = 'government of tanzania'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['finn water', 'finw', 'finwater']
    value = 'fini water'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['co', 'commu']
    value = 'community'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['wananchi', 'village', 'villa', 'villagers']
    value = 'village council'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['priva', 'chacha']
    value = 'private individual'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['mileniam project']
    value = 'world bank'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    string = 'plan int'
    value = 'plan international'
    train_match, train_fuzz = hand_match_method2(train_match, train_fuzz, value, string)

    train_match['foreign government'] = train_match['netherlands']
    key_list = ['netherlands', 'germany republi', 'nethalan','swedish','japan', 'kuwait', \
                'finida german tanzania govt', 'china government', 'european union', \
                'swiss if', 'swidish', 'denish', 'italy', 'italian', 'kuwait']
    value = 'foreign government'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    train_fuzz = shrink_list(train_fuzz, train_match, 50, "small funder")
    train[col] = [train_fuzz[x][0] for x in train[col]]
    return train

def funder_cleaning_test(train):
    # clean installer column
    col = "funder"

    train = fill_nan(train, col, str, "-999", ["0", "not know", "not kno"])
    train_names_dic, train_names_list, train_names_counter = names(train, col)
    train_match, train_fuzz = fuzzymatch_one(train_names_counter, 50, 50, 90)

    train_match['fini water'] = train_match['finw']
    key_list = ['rc ch', 'roman church', 'roman catholic', 'roman cathoric -kilomeni', \
                'rc cathoric', 'roman catholic rulenge diocese', 'roman', \
                'roman ca', 'roman', 'roman cathoric same', 'rcchurch/cefa']
    value = 'rc church'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['ministry of water', 'go', 'idara ya maji', "water"]
    value = 'government of tanzania'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['co', 'commu']
    value = 'community'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['finw']
    value = 'fini water'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['wananchi', 'village', 'villa', 'villagers']
    value = 'village council'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['priva', 'chacha']
    value = 'private individual'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    key_list = ['mileniam project']
    value = 'world bank'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    string = 'plan int'
    value = 'plan international'
    train_match, train_fuzz = hand_match_method2(train_match, train_fuzz, value, string)

    train_match['foreign government'] = train_match['netherlands']
    key_list = ['netherlands', 'germany republi', 'nethalan','swedish','japan', 'kuwait', \
                'finida german tanzania govt', 'china government', 'european union', \
                'swiss if', 'italy', 'kuwait']
    value = 'foreign government'
    train_match, train_fuzz = hand_match_method1(train_match, train_fuzz, key_list, value)
    train_fuzz = shrink_list(train_fuzz, train_match, 50, "small funder")
    train[col] = [train_fuzz[x][0] for x in train[col]]
    return train

def labelencode_to_category(train, test, cols):
    for col in cols:
        train[col] = train[col].fillna(-999)
        train[col] = train[col].astype(str)
        test[col] = test[col].fillna(-999)
        test[col] = test[col].astype(str)
        # le = preprocessing.LabelEncoder()
        # le.fit(train[col])
        # train[col] = le.transform(train[col])
        # test[col] = le.transform(test[col])      
    return train, test

def frequencyencode_to_category(train, test, cols):
    for col in cols:
        train[col] = train[col].fillna(-999)
        train[col] = train[col].astype(str)
        test[col] = test[col].fillna(-999)
        test[col] = test[col].astype(str)

        train_counter = collections.Counter(train[col])
        test_counter = collections.Counter(test[col])
        for key in test_counter:
            if key not in train_counter:
                train_counter[key] = test_counter[key]
        dic = {}
        for key in train_counter:
            if train_counter[key] >= 100:
                dic[key] = key
            elif (train_counter[key] < 100) and (train_counter[key] >= 20):
                dic[key] = "mediumsize_" + col
            elif train_counter[key] < 20:
                dic[key] = "smallsize_" + col
        train[col] = [dic[x] for x in train[col]]
        test[col] = [dic[x] for x in test[col]]
        # le = preprocessing.LabelEncoder()
        # le.fit(train[col])
        # train[col] = le.transform(train[col])
        # test[col] = le.transform(test[col])      
    return train, test

if __name__ == '__main__':
    train = pd.read_csv("training.csv", parse_dates=["date_recorded"])
    label = pd.read_csv("labels.csv")
    train =pd.merge(train, label, on="id", how="left")
    test = pd.read_csv("test.csv", parse_dates=["date_recorded"])

    # train = train.drop('wpt_name', axis=1)
    # test = test.drop('wpt_name', axis=1)

    train = funder_cleaning_train(train)
    test = funder_cleaning_test(test)

    train = installer_cleaning_train(train)
    test = installer_cleaning_test(test)

    train = scheme_name_cleaning_train(train)
    test = scheme_name_cleaning_train(test)

    labelencode_cols = ['funder', 'installer', 'basin', 'region', 'lga',
                   'recorded_by', 'scheme_management', 'scheme_name', 'extraction_type',
                   'extraction_type_group', 'payment', 'payment_type', 'water_quality',
                   'quality_group', 'quantity', 'quantity_group', 'source', 'source_type',
                   'source_class', 'waterpoint_type', 'waterpoint_type_group']
    train, test = labelencode_to_category(train, test, labelencode_cols)

    frequencyencode_cols = ["wpt_name", "subvillage", "ward"]
    train, test = frequencyencode_to_category(train, test, frequencyencode_cols)

    train.to_csv("training_cleaned.csv", index=False)
    test.to_csv("test_cleaned.csv", index=False)