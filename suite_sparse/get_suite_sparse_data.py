from bs4 import BeautifulSoup
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
import tarfile
import wget

# Warning: This script downloads roughly 4.3GB of sparse matrix data, which is
# then extracted. The total disk usage is ~18GB. Make sure there is enough
# space available on disk before running.

# URL to sparse matrix collection
url = 'https://sparse.tamu.edu/'

# Path to data directory to hold sparse matrices
matrix_data_dir = 'data/matrices/'

# Create necessary directories
os.makedirs(matrix_data_dir, exist_ok=True)

# Firefox web session
driver = webdriver.Firefox()
driver.get(url)
driver.implicitly_wait(100)

# Select size and shape filters
filter_options = driver.find_element_by_id('filter-dropdown').click()
driver.find_element_by_id('structure_checkbox').click()

# Wait for 100ms to avoid detection/server overload
driver.implicitly_wait(100)

# Get filter options 
positive_definite_options      = driver.find_element_by_id('filter-input-positive_definite')
rutherford_boeing_type_options = driver.find_element_by_id('filter-input-rb_type')
sorted_by_options              = driver.find_element_by_id('filterrific_sorted_by')
structure_type_options         = driver.find_element_by_id('filter-input-structure')

# Select options we want for each filter
select_positive_definite       = Select(positive_definite_options)
select_rutherford_boeing_type  = Select(rutherford_boeing_type_options)
select_sorted_by               = Select(sorted_by_options)
select_structure_type          = Select(structure_type_options)

select_positive_definite.select_by_visible_text('Yes')       
select_rutherford_boeing_type.select_by_visible_text('Real') 
select_sorted_by.select_by_visible_text('Rows (High to Low)')               
select_structure_type.select_by_visible_text('Symmetric')          

# Display all matrices following filter on the page
per_page_options = driver.find_element_by_id('per_page_top')
select_per_page  = Select(per_page_options)
select_per_page.select_by_visible_text('All')

# Get a list of all links to matrix market files on this page
elems = driver.find_elements_by_xpath("//a[@href]")
matrix_links = [e.get_attribute('href') for e in elems if 'sparse.tamu.edu/MM/' in e.get_attribute('href')]
matrix_links = matrix_links[5:] # First few matrices are very large (> 1GB)
matrix_tar_names = [m.split('/')[-1] for m in matrix_links]

# Exit web session
driver.quit()

# Download matrices using wget package
print('Downloading data...')
for m, t in zip(matrix_links, matrix_tar_names):
    print('{t}')
    try:
        wget.download(m, os.path.join(matrix_data_dir, t))
    except:
        pass
print('Done.')

# Extract tar files
print('Extracting tar files to mtx files...')
for root, dirs, files in os.walk(matrix_data_dir):
    for _fname in files:
        fname = os.path.join(root, _fname)
        if fname.endswith('.tar.gz'):
            print('Extracting file: {fname}')
            tar = tarfile.open(fname, 'r:gz')
            tar.extractall(path=matrix_data_dir)
            tar.close()

# Move mtx files from directories to data/matrices (basically flat extraction)
for root, dirs, files in os.walk(matrix_data_dir):
    for _dname in dirs:
        dname = os.path.join(root, _dname)
        for root_sub, dirs_sub, files_sub in os.walk(dname):
            for _fname_sub in files_sub:
                fname_sub = os.path.join(dname, _fname_sub)
                fname = os.path.join(matrix_data_dir, _fname_sub)
                os.rename(fname_sub, fname)
        os.rmdir(dname)
print('Done.')
