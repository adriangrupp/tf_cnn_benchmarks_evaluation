# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under the MIT License
# Please, see the LICENSE file
#
"""
Created on Sat Jun  1 23:58:00 2019

@author: valentin
"""
import os
import json
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

img_extension = ".pdf"

data_bin = 100

def parseFiles(rootDir, plot, plotParams):
    """ Read all metric log files for a certain container used.
        Make the plot of losses in getMetciValues()
    """
    for d in next(os.walk(rootDir))[1]:
        currentDir = os.path.join(rootDir,d)
        print("Processing:", currentDir)
        metricValues = defaultdict(list)
        for root, dirs, files in os.walk(currentDir):
            for f in files:
                if "benchmark.log" in f: 
                    # get benchmark parameters 
                    benchmarkFile = os.path.join(root,f)
                    benchParams = getRunParameters(benchmarkFile)
                    scaling = benchParams['num_epochs']/benchParams['num_batches']
            for f in files:
                if "metric.log" in f:
                    # get metric data
                    metricFile = os.path.join(root,f)
                    getMetricValues(metricFile, metricValues, scaling, plot, plotParams)

def getRunParameters(benchmarkFile):
    """ takes log file with benchmark data in JSON format
        and parses necessary parts to identify benchmark run
    """
    benchParams = {}
    with open(benchmarkFile, "r") as read_file:
        jsonData = json.load(read_file) # dictionary
        for el in jsonData["run_parameters"]: #list
            if el['name'] == "model":
                benchParams[el['name']] = el['string_value']
            if el['name'] == "dataset":
                if 'synthetic' in el['string_value']:
                    benchParams[el['name']] = 'synthetic'
                else:
                    benchParams[el['name']] = el['string_value']
            if el['name'] == "num_epochs":
                benchParams[el['name']] = el["float_value"]
            if el['name'] == "num_batches":
                benchParams[el['name']] = el["long_value"]
            if el['name'] == "batch_size":
                benchParams[el['name']] = el["long_value"]
            if el['name'] == "devices":
                deviceList = el['string_value'].split(',')
                benchParams['num_gpus'] = len(deviceList)
        benchParams['gpu_model'] = jsonData["machine_config"]["gpu_info"]["model"]

    return benchParams

### Parse benchmark performance
def getMetricValues(metricFile, metricValues, scaling, loss_plot, plotParams):
    """ takes log file with benchmark data in JSON format
        and parses necessary parts to identify benchmark run
        into a dict.
        Get the values for examples/sec and accuracy.
    """
    
    epoch_steps = []
    loss_steps  = []

    with open(metricFile, "r") as read_file:
        maxStep = 0 # variable for last iteration 
        i_line = 0
        for line in read_file:
            el = json.loads(line) # dictionary
            if el['name'] == "average_examples_per_sec":
                   metricValues["average_examples_per_sec"].append(el["value"])
            if el['name'] == "eval_average_examples_per_sec":
                    metricValues["eval_average_examples_per_sec"].append(el["value"])
            if el['name'] == "eval_top_1_accuracy":
                    metricValues["eval_top_1_accuracy"].append(el["value"])
            if el['name'] == "eval_top_5_accuracy":
                    metricValues["eval_top_5_accuracy"].append(el["value"])
            if el['name'] == "current_examples_per_sec" and el['global_step'] == 1:
                minTime = el['timestamp']
            if el['name'] == "current_examples_per_sec" and el['global_step'] > maxStep:
                maxTime = el['timestamp']
                maxStep = el['global_step']
            if len(el['extras']) > 0:
                if el['extras'][0]['name'] == "loss":
                    if i_line % data_bin == 0:
                        epoch_steps.append(el['global_step']*scaling)
                        loss_steps.append(el['extras'][0]['value'])
            i_line += 1
        
        # calculate runtime
        execTime = calculateExecTime(minTime, maxTime)
        metricValues['exec_time'].append(execTime)
        
        print(metricValues)
        
        loss_plot = plt.plot(epoch_steps, loss_steps, label=plotParams['label'], 
                             linestyle=plotParams['line_style'],
                             color=plotParams['color'], 
                             linewidth=plotParams['line_size'])

    return loss_plot
        
      
        
def calculateExecTime(minTime, maxTime):
    """ takes two timestamps and calculates the intervall
        of passed time between them.
    """
    fmt = '%Y-%m-%dT%H:%M:%S.%fZ' # Timeformat from benchmark
    tdelta = datetime.strptime(maxTime, fmt) - datetime.strptime(minTime, fmt)
    return tdelta.seconds
    

if __name__ == "__main__":
    #TODO args for directory
    wd = os.getcwd()
    results = "%s/benchmark_results" % wd
    
    parser = argparse.ArgumentParser(description='Plot parameters')
    parser.add_argument('--results_dirs', type=str, default=results,
                        help='directory with results (metric.log)')

    parser.add_argument('--dataset', type=str, default="ImageNet",
                        help='Processed dataset. Default: ImageNet')
                        
    parser.add_argument('--network', type=str, default="ResNet50",
                        help='Used neural network. Default: ResNet50')

    args = parser.parse_args()
    
    if ", " in args.results_dirs:
        data_dirs = args.results_dirs.split(', ')
    else:
        data_dirs = args.results_dirs.split(',')

    print(data_dirs)
    
    plotParams = { 'gpu_1': {'color': 'green', 'label': '1 GPU', 
                             'line_style': "-", 'line_size': 0.9},
                   'gpu_2': {'color': 'blue', 'label': '2 GPU', 
                             'line_style': ":", 'line_size': 3},
                   'gpu_4': {'color': 'k', 'label': '4 GPU', 
                             'line_style': "--", 'line_size': 2},
                 }
    
    loss_plot = plt.figure()

    for dirs in data_dirs:
        if "1_gpu" in dirs:
            loss_plot = parseFiles(data_dirs[0], loss_plot, plotParams['gpu_1'])
        if "2_gpu" in dirs:
            loss_plot = parseFiles(data_dirs[1], loss_plot, plotParams['gpu_2'])
        if "4_gpu" in dirs:
            loss_plot = parseFiles(data_dirs[2], loss_plot, plotParams['gpu_4'])

    loss_plot = plt.xlim(0., 10.5)
    loss_plot = plt.legend(loc="upper right")
    loss_plot = plt.xlabel('epoch')
    loss_plot = plt.ylabel('Loss')
    loss_plot = plt.title('Learning curves - %s - %s' %(args.dataset, args.network))

    #plt.show()
    file_name = "loss" + '-' + args.dataset + '-' + args.network + img_extension
    plt.savefig(file_name)  
    