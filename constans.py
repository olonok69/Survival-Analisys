import os

RAIL_ROOT = os.path.dirname(__file__)


#Phase 2 Full
column_json2=columns=[ 'ELR_enc', 'HLC_enc', 'Asset_Class_Grouping_enc',
                       'Grouping_Full_Name_enc', 'Engineering_Suffix_enc',
                       'System_Asset_Type_enc', 'EQUIP_CLASS_DESC_enc',
                       'weather_station_enc', 'Sum', 'Asset_Number',
                       'temp_min', 'temp_max', 'humidity_min', 'humidity_max', 'pressure_min',
                       'pressure_max', 'wind_speed_min', 'wind_speed_max']


json_source=['ELR_enc',	'Grouping_Full_Name_enc',	'weather_station',	'Asset_Number']
json_source_reduced=['Asset_Number']
#phase 2, no weather
column_json3=[ 'berth1', 'berth2','berth3', 'locb2', 't2_stanox','journey_id',
         'train_uid', 'origintiploc', 'desttiploc',  'distance',  'diff',
       'season', 'to_midnigth', 'day_of_week']

TPBS_data_prediction_enc=['t2_stanox',  'distance', 'temp',
       'pressure', 'humidity', 'wind_speed',  'diff',  'to_midnigth', 'day_of_week',
       'b1_enc', 'b2_enc', 'b3_enc', 'locb2_enc', 'journey_id_enc',
       'train_uid_enc', 'origintiploc_enc', 'desttiploc_enc',
       'weather_main_enc', 'weather_description_enc', 'season_enc']

TPBS_data_prediction_enc_nw=['t2_stanox', 'distance',
       'diff', 'to_midnigth', 'day_of_week',  'b1_enc', 'b2_enc', 'b3_enc',
       'locb2_enc', 'journey_id_enc', 'train_uid_enc', 'origintiploc_enc',
       'desttiploc_enc',
       'season_enc']