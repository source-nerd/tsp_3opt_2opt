"""
Author: Sonu Prasad
Email: sonu.prasad@mycit.ie
file: README
"""
README file, contains the description for the project submitted, compiling environment and how to run the files.

----------------------------------------------------------------------------------------
1. COMPILING ENVIRONMENT
This code is tested on Dell Alienware 17 R4 Machine for which the config is as follows:
    - Windows 10 Pro (with latest patches and updates)
    - Intel 7700Hq Processor
    - 24 Gb DDR4 2400Mhz RAM (Memory)
    - W.D Black 512G.B M.2 NVME SSD DRIVE
    - JetBrains PYCharm Professional Edition (Latest version - 2018.3.1)
    - Microsoft Visual Studio Code (For Editing small documents)

N.B: Please make sure you have install Microsoft V.S Code on your machine before opening this README.
Also, initialize the project directory as a virtual environment.

----------------------------------------------------------------------------------------
2.  PROJECT STRUCTURE
The Project folder description is:
    a. source_code
        - tabu_search.py
        - novelty_plus.py
        - tsp.py
        - cities_graph.py
        - utils.py
        - requirements.txt
        - dataset directory
    b. Logs and Graphs
        Contains Logs and Graphs of all the test run. The records are also presented in the report submitted along this project.
        Some of the graphs present in this folder isn't included in the report as it will just increase the size of the report.
    c. r00170510_report.pdf
    d. README.txt

----------------------------------------------------------------------------------------
3. INSTALLING DEPENDENCIES
A `requirements.txt` file has been provided with this project. It contains a list of all the packages used as dependency for this project (only - plotly - for Plotting graphs).
You can install the dependencies using the following code: (This assumes you have Python 3.6 installed on your machine)
    `pip install -r requirements.txt`

----------------------------------------------------------------------------------------
4. RUNNING THE PROJECT
The files that can be run are:
    1. tabu_search.py
    2. novelty_plus.py
    3. tsp.py
Other files are supporting files and needs to be called from above 3 python files.
THe code to run any of the file is:
`python <python_file_name.py>`

If you want to change any input file_name, then you need to go to the end of the code and search for file_name variable in the main function of the file and replace with some other file name which you desire to you.

----------------------------------------------------------------------------------------