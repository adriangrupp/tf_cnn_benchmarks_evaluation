import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
#----------------------------

def readFiles(rootDir):
    """ Read all benchmark log files for a certain container used.
        Store them accoding to their metadata. Structure is as follows:

            dictionary       (for runs of different parameters: datatype-model-numgpus)
              |-- dictionary (collection of relevant metric and parameter data from evaluation)
    """
    plotData = {}
    for d in next(os.walk(rootDir))[1]:
        currentDir = os.path.join(rootDir,d)
        for root, dirs, files in os.walk(currentDir):
            for f in files:
                if "plotdata.log" in f:
    
                    # get file data for unique identification of run
                    dataFile = os.path.join(root,f)
                    with open(dataFile, "r") as read_file:
                        jsonData = json.load(read_file) # dict of dicts
                        numEpochs, numBatches, numGPUs, model, dataset = getBenchParams(jsonData)
                        runParams = "-".join([dataset, model, str(numGPUs)])
                        print('Created: ' + runParams)
                        plotData[runParams] = jsonData
    return plotData


def getBenchParams(data):
    """ takes log file with benchmark data in JSON format
        and parses necessary parts to identify benchmark run
    """
    dataset = data["benchmark_params"]["dataset"]
    model = data["benchmark_params"]["model"]
    numGPUs = data["benchmark_params"]["num_gpus"]
    numEpochs = data["benchmark_params"]["num_epochs"]
    numBatches = data["benchmark_params"]["num_batches"]

    return numEpochs, numBatches, numGPUs, model, dataset


### Histogramm Images/sec for 1,2,4 gpus, no dataset comaprison
#TODO: coloured bars per gpu
#TODO: Legend for numgpus
#TODO: aesthetics
#TODO: avoid empty plots (dic not in plotData) 
def histImgSec(plotData, outDir):
    """ Function to plot histogams that compare images/sec for 
        different amount of gpus and models. 1 Diagramm per dataset.
    """
    datasets = ['CIFAR10', 'Synthetic', 'ImageNet-']  #TODO better name of imagenet
    models = ['ResNet50', 'ResNet56', 'AlexNet']

    # One plot for each dataset to compare models' performance
    for ds in datasets:
        ax = plt.subplot2grid((1,1),(0,0))
        ax.set_axisbelow(True)
        ax.grid(True, axis='y', linewidth=.4)

        i = 1
        width = 0.25
        labels = []
        labelsPos = []

        # Find all models that were used on the current dataset
        for m in models:    
            gpus = []
            examples_per_sec = [] 
            for key, value in plotData.items():
                if ds.lower() in key.lower() and m.lower() in key.lower():
                    gpus.append(value['benchmark_params'].get('num_gpus'))
                    examples_per_sec.append(value['averages'].get('average_examples_per_sec'))

            if len(examples_per_sec) == 0:
                continue
            print(gpus, examples_per_sec)

            gpus_idx = np.array(gpus).argsort()    
            examples_per_sec_sort = np.array(examples_per_sec)[gpus_idx]

            plotData_x = list(np.arange(i, i+len(examples_per_sec)*0.25, 0.25))
            labelsPos.append(plotData_x[1]) #middle
            labels.append(m)

            # Bar chart
            plt.bar(plotData_x, examples_per_sec_sort, width,
                    align='center', facecolor='g', alpha=0.5, edgecolor='w')
            # Bar labels
            for j, v in enumerate(examples_per_sec_sort):
                plt.text(i + j * width - width/2, v + 3, str("%.2f" % v), color='k')
            i+=1

        if len(labels) == 0:
            continue #don't plot any empty diagrams

        plt.xticks(labelsPos, labels)
        plt.xlabel('Model')
        plt.ylabel('Images/sec')
        plt.title('Training: uDocker - 1,2 and 4 GPUs - %s Dataset' %ds)

        filename = ds + '-Training.png'
        plt.savefig(outDir + '/' + filename)
    return    
    

## Histogram comparison real synthetic per num_gpu
def histCompareRealSynth(plotData, outDir):
#TODO: Nicer bars
#TODO: Nicer text
#TODO: more generic?
    """ Function to compare histogams that compare images/sec for 
        different amount of gpus. Compares real to synth data. 
        Works only for imagenet - cifar10 has diferent models
    """
    datasets = ['ImageNet-', 'Synthetic']  #TODO better names
    models = ['ResNet50', 'AlexNet']

    # One plot for each model to compare individually
    for m in models:    
        real_ex_per_sec = []
        real_gpus = []
        synt_ex_per_sec = []
        synt_gpus = []
        # Find all models that were used on the current dataset
        for key, value in plotData.items():
            if m.lower() in key.lower():
                if 'imagenet-' in key.lower():
                    real_ex_per_sec.append(value['averages'].get('average_examples_per_sec'))
                    real_gpus.append(value['benchmark_params'].get('num_gpus'))
                if 'synthetic' in key.lower():
                    synt_ex_per_sec.append(value['averages'].get('average_examples_per_sec'))
                    synt_gpus.append(value['benchmark_params'].get('num_gpus'))
    
        if len(real_ex_per_sec) == 0 or len(synt_ex_per_sec) == 0:
                continue

        filename = m + '-SynthVsReal.png'
        outPath = outDir + '/' + filename            
        plotComparison(synt_ex_per_sec, synt_gpus, real_ex_per_sec,
                real_gpus, outPath)
    return


## Histogram comparison between two sets of containers, systems, ...
def histCompareTwoSets(plotData1, plotData2, outDir):
    """ Function to plot histogams that compare images/sec for 
        two different sets. Compares per num_gpu.
        Requires: same dataset, same model.
        Currently only doing real datasets!
    """
    #TODO: Legend names
    #TODO: Nicer bars
    #TODO: Nicer text
    datasets = ['ImageNet-', 'CIFAR10']  #TODO better names
    models = ['AlexNet', 'ResNet50', 'ResNet56']

    # One plot for each dataset and model to have all comparisons
    for ds in datasets:
        for m in models:    
            ex_per_sec_1 = []
            gpus_1 = []
            ex_per_sec_2 = []
            gpus_2 = []
            # Find all models that were used on the current dataset
            for key, value in plotData1.items():
                if m.lower() in key.lower():
                    if ds.lower() in key.lower():
                        ex_per_sec_1.append(value['averages'].get('average_examples_per_sec'))
                        gpus_1.append(value['benchmark_params'].get('num_gpus'))
            for key, value in plotData2.items():
                if m.lower() in key.lower():
                    if ds.lower() in key.lower():
                        ex_per_sec_2.append(value['averages'].get('average_examples_per_sec'))
                        gpus_2.append(value['benchmark_params'].get('num_gpus'))
                        
            if len(ex_per_sec_1) == 0 or len(ex_per_sec_2) == 0:
                continue
    
            filename = ds + '-foo.png'
            outPath = outDir + '/' + filename            
            plotComparison(ex_per_sec_1, gpus_1, ex_per_sec_2, gpus_2, outDir)
    return


def plotComparison(ex_per_sec_1, gpus_1, ex_per_sec_2, gpus_2, outPath):
    #TODO: Store plots

    print(ex_per_sec_1, ex_per_sec_2)

    ax = plt.subplot2grid((1,1),(0,0))
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', linewidth=.4)
    
    gpus_1_idx = np.array(gpus_1).argsort()    
    ex_per_sec_1_sort = np.array(ex_per_sec_1)[gpus_1_idx]
    gpus_2_idx = np.array(gpus_2).argsort()    
    ex_per_sec_2_sort = np.array(ex_per_sec_2)[gpus_2_idx]

    width = 0.25
    pos = list(range(len(ex_per_sec_1)))

    # first set polts
    p1 = plt.bar(pos, ex_per_sec_1_sort, width,
            align='center', facecolor='r', alpha=0.5, edgecolor='w')
    for j, v in enumerate(ex_per_sec_1_sort):
        plt.text(j - width/2, v + 3, str("%.2f" % v), color='k')
    # second set plots 
    p2 = plt.bar([p + width for p in pos], ex_per_sec_2_sort, width,
            align='center', facecolor='g', alpha=0.5, edgecolor='w')
    for j, v in enumerate(ex_per_sec_2_sort):
        plt.text(j + width/2, v + 3, str("%.2f" % v), color='k')
    # Bar labels
    labelsPos = [p + width/2 for p in pos]
    labels = sorted(gpus_2)

    print(labelsPos, labels)
    plt.xticks(labelsPos, labels)
    plt.xlabel('#GPUs')
    plt.ylabel('Images/sec')

    plt.legend((p1[0], p2[0]), ('set 1', 'set 2')) #TODO fix names!
    plt.title('Training: uDocker - %s - %s ' %('foo', 'bar'))
    plt.savefig(outPath)
    return


## Speedup

## Sclaing
def plotScaling(plotData, outDir):
    """ Plot the scaling for a dataset and several models for each num_gpu.
    Consider only images/sec. Includes also ideal scaling.
    """
    #TODO: Store plots
    #TODO: nicer grid
    #TODO: store
    datasets = ['Synthetic', 'ImageNet-', 'CIFAR10']  #TODO better names
    models = ['AlexNet', 'ResNet50', 'ResNet56']

    # One plot for each dataset 
    for ds in datasets:
        for m in models:    
            width = 0.25
            labels = []
            labelsPos = []
            ex_per_sec = []
            gpus = []
            # Find all models that were used on the current dataset
            for key, value in plotData.items():
                if ds.lower() in key.lower():
                    if m.lower() in key.lower():
                        ex_per_sec.append(value['averages'].get('average_examples_per_sec'))
                        gpus.append(value['benchmark_params'].get('num_gpus'))
                        
            if len(ex_per_sec) == 0:
                continue

            gpus_idx = np.array(gpus).argsort()    
            gpus_sort = sorted(gpus)
            ex_per_sec_sort = np.array(ex_per_sec)[gpus_idx]
            

            # Ideal number of images is number of images for 1 gpu times num_gpus
            scaling = [x/ex_per_sec_sort[0] for x in ex_per_sec_sort]
            plt.plot(gpus_sort, scaling, label=m)
        
        # ideal scaling
        plt.plot([1,2,4], [1,2,4], label='Ideal')

        plt.xticks([1,2,3,4], [1,2,3,4])
        plt.xlabel('#GPUs')
        plt.ylabel('Scaling')
    
        plt.legend(loc=2, fontsize="small")
        plt.title('Scaling: uDocker - %s dataset' %ds)
        plt.grid()
        plt.show()
    return


## Runtime

## Evaluation accuracy

## Evaluation examples/sec


if __name__ == "__main__":
    #TODO argv for selection of plot mode?

    # Dir for benchmark data
    wd = os.getcwd()
    resultDir = "%s/benchmark_results" % wd
    
    # Target location for plots
    plotDir = "%s/training_plots" %wd    
    compDir = "%s/comparison_plots" %wd

    if not os.path.exists(plotDir):
        os.makedirs(plotDir)
    if not os.path.exists(compDir):
        os.makedirs(compDir)

    folders = ['lsdf_udocker_5epoch',   'lsdf_udocker_500batch',
            'lsdf_singularity_5epoch',  'lsdf_singularity_500batch',
            'fh2_udocker_5epoch',       'fh2_udocker_500batch',
            'fh2_singularity_5epoch',   'fh2_singularity_500batch']

    #--------------------------------------------------
    
    # Plot bar charts for imgs/sec to compare models
    # Per dataset, all settings
    """for f in folders:
        procFiles = os.path.join(resultDir, f)
        print('Processing: ' + procFiles)
        plotData = readFiles(procFiles)
        if plotData:
            outDir = os.path.join(plotDir, f)
            if not os.path.exists(outDir):
                os.makedirs(outDir)
            histImgSec(plotData, outDir)"""
   

    # Plot comparison of synthetic, vs real data, 500 batches, udocker, lsdf
    """for f in folders:
        procFiles = os.path.join(resultDir, f)
        print('Processing: ' + procFiles)
        plotData = readFiles(procFiles)
        if plotData:
            outDir = os.path.join(plotDir, f)
            if not os.path.exists(outDir):
                os.makedirs(outDir)
            histCompareRealSynth(plotData, outDir)"""

    # Plot comparison between lsdf & fh2, udocker, 500 batches, real data
    #TODO: iteration
    procFiles1 = os.path.join(resultDir, folders[1])
    plotData1 = readFiles(procFiles1)
    procFiles2 = os.path.join(resultDir, folders[5])
    plotData2 = readFiles(procFiles2)
    outDir = os.path.join(compDir, 'lsdf_vs_fh2') #TODO avoid overwriting
    histCompareTwoSets(plotData1, plotData2, outDir)

    # Plot scaling 500 batches, udocker lsdf
    #TODO: iteration
    """procFiles = os.path.join(resultDir, folders[1])
    plotData = readFiles(procFiles)
    outDir = os.path.join(plotDir, folders[1])
    plotScaling(plotData, outDir)
    """
