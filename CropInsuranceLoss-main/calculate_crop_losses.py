import numpy as np
import pandas as pd
import os

# pathway to cause-of-loss data
col_dir = os.path.join('crop_loss_data', 'cost_of_loss')
# ensure data directory exists, remind user to run downloader if missing
if not os.path.isdir(col_dir):
    print(f"data directory '{col_dir}' does not exist.\n" \
          "Please run 'python read_crop_loss_api.py' to download the data.")


# cause-of-loss codes
# individual codes for very specific 
# causes of agricultural loss
col_codes = {}
col_codes['00'] = 'none'
col_codes['01'] = 'price'
col_codes['09'] = 'asian soybean rust'
col_codes['11'] = 'drought'
col_codes['12'] = 'heat'
col_codes['13'] = 'irrigation supply'
col_codes['14'] = 'irrigation equipment'
col_codes['15'] = 'land prep'
col_codes['21'] = 'hail'
col_codes['22'] = 'sun'
col_codes['31'] = 'rain'
col_codes['41'] = 'frost'
col_codes['42'] = 'freeze'
col_codes['43'] = 'cold'
col_codes['44'] = 'cold-wet'
col_codes['45'] = 'lack of chill'
col_codes['51'] = 'flood'
col_codes['55'] = 'grp'
col_codes['61'] = 'wind'
col_codes['62'] = 'hot wind'
col_codes['63'] = 'cyclone'
col_codes['64'] = 'tornado'
col_codes['65'] = 'tsunami'
col_codes['66'] = 'no oxygen'
col_codes['67'] = 'storm surge'
col_codes['71'] = 'insects'
col_codes['73'] = 'predation'
col_codes['74'] = 'ice'
col_codes['76'] = 'salinity'
col_codes['80'] = 'disease'
col_codes['81'] = 'plant disease'
col_codes['82'] = 'mycotoxin'
col_codes['87'] = 'falling'
col_codes['91'] = 'fire'
col_codes['92'] = 'hurricane'
col_codes['93'] = 'wildlife'
col_codes['95'] = 'house burn'
col_codes['97'] = 'earthquake'
col_codes['98'] = 'volcano'
col_codes['99'] = 'other'

# group causes-of-loss by category
# broader groups that reduce the number
# of loss causes to a more managable number
loss_groups = {}
loss_groups['none'] = ['none',]
loss_groups['financial'] = ['price',]
loss_groups['other'] = ['none', 'asian soybean rust', 'land prep',
                        'grp', 'no oxygen', 'falling', 'earthquake', 
                        'volcano', 'other']
loss_groups['cold'] = ['frost', 'freeze', 'cold', 'cold-wet', 'ice']
loss_groups['heat'] = ['heat', 'sun', 'lack of chill']
loss_groups['drought'] = ['drought', 'irrigation supply', 
                          'irrigation equipment']
loss_groups['storm'] = ['hail', 'tornado', 'wind', 'hot wind']
loss_groups['precipiation'] = ['rain', 'flood', 'tsunami', 'cyclone',
                               'storm surge', 'hurricane']
loss_groups['wildlife'] = ['insects', 'predation', 'wildlife']
loss_groups['contamination'] = ['salinity', 'disease', 'plant disease',
                                'mycotoxin']
loss_groups['fire'] = ['fire', 'house burn']                                

# initialize dataframe for crop loss data
full_dataset = pd.DataFrame()
yearList = []
countyList = []
cropList = []
causeList = []
lossValueList = []
totalValueList = []
totalAcreageList = []
stateNameList = []
# each year has its own cause-of-loss file
# loop through years
for year in range(1989, 2025): 
  # filename structure changes in 2014
  print(year)
  if year < 2014:
    scc_filepath = os.path.join(col_dir, 'colsom' + str(year)[-2:] + '.txt')
  else:
    scc_filepath = os.path.join(col_dir, 'colsom_' + str(year) + '.txt')
      
  # read cause-of-loss data line by line
  if not os.path.exists(scc_filepath):
    # warn and skip missing years (download data via read_crop_loss_api.py)
    print(f"warning: file not found {scc_filepath}, skipping year {year}")
    continue
  with open(scc_filepath, 'r') as file:
    for line in file:
      # each data field is separated by '|'
      val_array = line.strip().split('|')
      # extract relevant data
      year = int(val_array[0].strip())
      state_fip = val_array[1].strip().zfill(2)
      state_name = val_array[2].strip()
      county_fip = val_array[3].strip().zfill(3)
      crop_type = val_array[6].strip()
      cause_of_loss = val_array[11].strip()
      # some cause-of-loss codes are out of date
      try:
        cause_of_loss_string = col_codes[cause_of_loss]
        for xxx in loss_groups:
          if cause_of_loss_string in loss_groups[xxx]:
            loss_group_use = xxx
      except:
        loss_group_use = 'unknown'
      # format financial data
      value_of_loss = float(val_array[28].strip())
      try:
        acres_insured = float(val_array[18].strip())
      except:
        acres_insured = np.nan
      try:
        value_insured = float(val_array[20].strip())
      except:
        value_insured = np.nan
      # add to lists
      yearList.append(year)
      countyList.append(state_fip + county_fip)
      cropList.append(crop_type)
      causeList.append(loss_group_use)
      lossValueList.append(value_of_loss)
      totalValueList.append(value_insured)
      totalAcreageList.append(acres_insured)
      stateNameList.append(state_name)
# aggregate lists to dataframe  
full_dataset['year'] = yearList
full_dataset['county'] = countyList
full_dataset['crop'] = cropList
full_dataset['cause'] = causeList
full_dataset['loss_value'] = lossValueList
full_dataset['insured_value'] = totalValueList
full_dataset['insured_acreage'] = totalAcreageList

# disaggregate data by state
# create new dataframe w/ only state names
state_name_df = pd.DataFrame()
state_name_df['states'] = stateNameList
# find unique list of states in dataframe
state_names_all = state_name_df['states'].unique()
os.makedirs(os.path.join('crop_loss_data', 'readable_col_data_by_state'), exist_ok=True)
# slice full dataframe by state name
for state in state_names_all:
  # find the rows that have data from individual state
  this_state = state_name_df['states'] == state
  # slice original dataframe by state-row locations
  state_dataset = full_dataset[this_state]
  # write individual state files
  state_dataset.to_csv(os.path.join('crop_loss_data', 'readable_col_data_by_state', state + '_loss_by_cause.csv'))
full_dataset.to_csv(os.path.join('crop_loss_data', 'readable_col_data_by_state','all_loss_by_cause.csv'))