###############################################################################
# Script to parse the created files of tf_cnn_benchmark during several runs.  #
# Calculates Average results for metric.log as well as standard deviation, if #
# multiple runs were run. (stdev=0 if only 1 run).                            #
# Furthermore extracts important parameters for plotting of the runs.         #
# Author: Adrian Grupp                                                        #
###############################################################################

import os
import json
import statistics
from datetime import datetime
from collections import defaultdict
#-----------------------------------

def parseFiles(rootDir):
    """ Read all metric log files for a certain container used.
        Store the averages values in a new file metric_avg.log.
    """
    for d in next(os.walk(rootDir))[1]:
        currentDir = os.path.join(rootDir,d)
        print("Processing:", currentDir)
        metricValues = defaultdict(list)
        for root, dirs, files in os.walk(currentDir):
            for f in files:
                if "metric.log" in f:
                    # get metric data
                    metricFile = os.path.join(root,f)
                    getMetricValues(metricFile, metricValues)
                if "benchmark.log" in f: 
                    # get benchmark parameters 
                    benchmarkFile = os.path.join(root,f)
                    benchParams = getRunParameters(benchmarkFile)
                if "eval.log" in f: 
                    # get evalution parameters 
                    evalFile = os.path.join(root,f)
                    evalParams = getRunParameters(evalFile)

        averages, stdevs = computeAvgDev(metricValues)
        data = {'averages': averages, 'stdevs': stdevs, 'benchmark_params':
                benchParams, 'evaluation_params': evalParams}

        with open(os.path.join(currentDir,'plotdata.log'), 'w') as write_file:
            json.dump(data, write_file)
    return


### Parse benchmark performance
def getMetricValues(metricFile, metricValues):
    """ takes log file with benchmark data in JSON format
        and parses necessary parts to identify benchmark run
        into a dict.
        Get the values for examples/sec and accuracy.
    """
    with open(metricFile, "r") as read_file:
        maxStep = 0 # variable for last iteration        
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
        
        # calculate runtime
        execTime = calculateExecTime(minTime, maxTime)
        metricValues['exec_time'].append(execTime)


def calculateExecTime(minTime, maxTime):
    """ takes two timestamps and calculates the intervall
        of passed time between them.
    """
    fmt = '%Y-%m-%dT%H:%M:%S.%fZ' # Timeformat from benchmark
    tdelta = datetime.strptime(maxTime, fmt) - datetime.strptime(minTime, fmt)
    return tdelta.seconds


def computeAvgDev(metricValues):
    averages = {}
    stdevs = {}
    for key, val in metricValues.items():
        if len(val) > 1:
            stdevs[key] = statistics.stdev(val) # standard deviation
        else:
            stdevs[key] = 0
        averages[key] = sum(val) / float(len(val))
 
    return averages, stdevs


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


if __name__ == "__main__":
    #TODO args for directory
    wd = os.getcwd()
    results = "%s/benchmark_results" % wd
    for d in next(os.walk(results))[1]:
        currentDir = os.path.join(results,d)
        parseFiles(currentDir)
    print("\n Retrieved data written to",
            os.path.join(results,"xx/xx/plotdata.log"))
