CAPSTONE PROJECT - Readme
Python Version: 2.7
Author: Radek Chramosil
Date: 26-Nov-2018
Python code stored in: main.py
Tested Platform: Linux 4.19.4-1-MANJARO

CONTENTS OF THIS FILE
---------------------
 
 * Description  
 * Introduction
 * Main Requirements
 * Installation
 * Error Installing SciPy
 * Executing the project
 * Troubleshooting
 * Maintainers


DESCRIPTION
-----------
This is project is the capstone project for the WQU MSc course. It fetches data from crypto-currency exchange called Binance vie their API and python package called `python-binance`.

The environment is specified in 'requirements.txt' file.

I use MySQL server instance installed locally to handle the data storage. The default MySQL implementation for Manjaro (the Linux distribution that I use) is `MariaDB`. So, I had to use MariaDB. The SQL codes should be compatible with MySQL but this was not tested. All SQL codes are stored in `.sql` files and are submittedl
 
MAIN REQUIREMENTS
-----------------
The list of main required modules is:

python==2.7.15 - See https://virtualenv.pypa.io/en/latest/ for installation.
 
pip==18.1 - See https://pip.pypa.io/en/stable/installing/ for more.
 
pandas==0.23.1 - See https://pandas.pydata.org/pandas-docs/version/0.23/index.html  for more.
 
python-binance==0.7.0 - See https://python-binance.readthedocs.io/en/latest/index.html  for more.

mysql==0.0.2 - See https://github.com/valhallasw/virtual-mysql-pypi-package
 
MySQL-python==1.2.5 - See https://github.com/farcepest/MySQLdb1

PyMySQL==0.9.2 - See https://pypi.org/project/PyMySQL/

SQLAlchemy==1.2.14 - see https://www.sqlalchemy.org/
 
 
The code is Python 3 compatible where possible.

INSTALLATION
------------
1. Install Python 2.7.14 - https://www.python.org/downloads/ 
2. Unzip files to local drive in desired folder (example: C:\mini_project_i). 
3. Open cmd prompt/shell or terminal.
4. Navigate to created folder.
5. Install requirements:
   1. Type “pip install -r requirements.txt” in cmd prompt/shell or terminal.
   2. Install all requirements.

You will need MySQL server instance as well.  I was don't provide any installation
 steps for MySQL server as this is platform dependent.
 
Also, the data set was fetched using my Binance account credentials. You will need those
if you wish to make my code fully functional. My credentials are stored in `config.py` file
that is not submitted with the project.

ERROR INSTALLING SCIPY
---------------------------------
If there are any issues installing SciPy library, please following instruction provided at 
the documentation: https://www.scipy.org/install.html#scientific-python-distributions.

EXECUTING THE PROJECT
-------------
You can still run the below command if needed, however you need change the functionality in the `main` function.
 At the moment the code only can download the data and store them or create some plots.

$ python main.py
 
TROUBLESHOOTING
---------------
No log is available at the moment.

 
MAINTAINERS
-----------
* Radek Chramosil - radek@keemail.me
