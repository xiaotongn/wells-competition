import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import vincenty

# This files generate new features:
# (1) Age of the well
# (2) Distance to the regional capital and the national capital
# (3) Population growth between 2002 and 2012 in each region
# (4) Population density in 2012 in each region

# Read in cleaned data
main_data = pd.read_csv("training_cleaned.csv", parse_dates=["date_recorded"])

# Read in the raw data on regional population and capital city
region_data = pd.read_csv("region_data.csv")

#(1) Calculate age of the well based on the date of record and construction year
# Set to 'Nan' if missing information on construction year
def calc_age(main_data):
    # Create a variable for the age of the well
    main_data['construction_year'] = main_data['construction_year'].replace(to_replace=0, value = np.nan)
    main_data['age'] = main_data['date_recorded'].dt.year-main_data['construction_year']
    # Replace negative values with Nan
    main_data['age'] = main_data['age'].replace(to_replace=[-7,-5,-4,-3,-2,-1], value = np.nan)
    return main_data

#(2) Calculate distance to the regional capital and the national capital
def calc_distance_to_capital(main_data,region_data):
    
    # Get the coorindates of each capital
    region_data['capital']=region_data['capital']+', Tanzania'
    geolocator = Nominatim()
    region_data['region_coord'] = region_data['capital'].apply(geolocator.geocode)
    region_data['region_latitude'] = region_data['region_coord'].apply(lambda x: (x.latitude))
    region_data['region_longitude'] = region_data['region_coord'].apply(lambda x: (x.longitude))
    # Get the coordinates of Tanzania capital
    geolocator = Nominatim()
    location = geolocator.geocode("Dodoma, Tanzania")
    region_data['capital_latitude'] = location.latitude
    region_data['capital_longitude'] = location.longitude
    # Drop unused variables
    region_data = region_data.drop('region_coord', axis=1)
    
    #Merge the regional data with the main data
    df = pd.merge(main_data, region_data, on='region')
    
    #Turn variables of longitude and latitude to tuples
    df['coords'] = list(zip(df['latitude'], df['longitude']))
    df['region_coords'] = list(zip(df['region_latitude'],df['region_longitude']))
    df['capital_coords'] = list(zip(df['capital_latitude'],df['capital_longitude']))
    #Calculate the distance to regional capital and country capital
    df['region_distance'] = [vincenty(df['coords'].iloc[i], df['region_coords'].iloc[i]).meters for i in range(len(df))]
    df['capital_distance'] = [vincenty(df['coords'].iloc[i], df['capital_coords'].iloc[i]).meters for i in range(len(df))]

    #Change the unit from meters to km
    df['region_distance'] = df['region_distance']/1000
    df['capital_distance'] = df['capital_distance']/1000
    
    return df

#(3) Calculate population growth from 2002 to 2012
def calc_pop_growth_2002_2012(region_data):
    region_data['pop_growth_0212'] = region_data['pop2012']/region_data['pop2002']
    return region_data

#(4) Calculate population density
def calc_pop_density_2012(region_data):
    region_data['pop_density_2012'] = region_data['pop2012']/region_data['area']
    return region_data

if __name__ == '__main__':
    
    main_data = calc_age(main_data)
    
    region_data = calc_pop_growth_2002_2012(region_data)
    region_data = calc_pop_density_2012(region_data)
    
    output = calc_distance_to_capital(main_data, region_data)
    
    for col in output.columns:
        print("column: %s is of type %s" %(col, output[col].dtype))
    
    print(output[['id','age','region','region_distance','capital_distance','pop_growth_0212','pop_density_2012']].head(10))