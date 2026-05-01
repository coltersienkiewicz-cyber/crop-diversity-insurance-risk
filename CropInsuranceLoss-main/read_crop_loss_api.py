import requests
import zipfile
import io
import os

# this code reads hurricane track data from nhc
# unzips the drives and stores the shapefiles
# including points, lines, radius, and windswath
# web address of data directory
type_practice_usage_url = 'https://www.rma.usda.gov/sites/default/files/'
folder1 = 'information-tools/'
folder2 = '2024-09/'
ender = '.zip' # multiple file types in directory

# local directory for extracted data
output_dir = 'crop_loss_data'

# data availability on web directory is 1989-2025
start_year = 1989
end_year = 2025

# create directory for storing shapefile output
os.makedirs(output_dir, exist_ok = True)

file_types = ['type_practice_usage', 'state_county_crop', 'cost_of_loss']
file_names = ['sobtpu_', 'sobcov_', 'colsom_']
# loop through each of the crop insurance file types
for name_of_data, file_ender_name in zip(file_types, file_names):
  # create directory to store each file type
  output_directory_year = os.path.join(output_dir, name_of_data)
  os.makedirs(output_directory_year, exist_ok = True)
  # loop through years
  for year_use in range(start_year, end_year + 1):
    print('Reading ' + name_of_data + ' from year ' + str(year_use))
    # data names/pathways change over the course of the historical period
    # it will be one of these two names, we just need to try reading them both
    # save whichever file produces a download
    for folder_look in [folder1, folder2]:
      try:
        response = requests.get(type_practice_usage_url + folder_look + file_ender_name + str(year_use) + ender, stream=True)
        z = zipfile.ZipFile(io.BytesIO(response.content))
      except:
        pass
    
    # unzip the file that is downloaded into the folder created above
    try:
      z.extractall(output_directory_year)
    except:
      pass
    # delete the file from memory
    # some data types don't start until 1999, 1989-1988 will display 'read fail'
    try:
      del z
    except:
      print('Read Fail ' + name_of_data + ' ' + str(year_use))
    