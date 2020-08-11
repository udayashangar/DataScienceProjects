import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,total_sqft,bath,bhk):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    x = np.zeros(len(__data_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return __model.predict([x])[0]

def get_location_names():
    return __locations


def load_saved_artifacts():
    print ("loading of saved artifacts started...")
    global __data_columns
    global __locations
    global __model

    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_column']
        __locations = __data_columns[3:]

    with open("./artifacts/Bangalore_house_price_model.pickle", 'rb') as f:
        __model = pickle.load(f)
    print ("loading of saved artifacts completed...")


if __name__ == '__main__':
    load_saved_artifacts()
    print (get_location_names())
    print(get_estimated_price('1st Block Jayanagar',2850,4,4))
    print(get_estimated_price('1st Phase JP Nagar',1000,3,3))
    print(get_estimated_price('1st Phase JP Nagar',1000,2,2))
    print(get_estimated_price('Indira Nagar',1000,2,2))
    print(get_estimated_price('Indira Nagar',1000,3,3))