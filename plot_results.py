##############################################################################
# Script to plot diferent diagrams for the tf_cnn_tf_benchmark. 
# Reards the parsed JSON data into a dict. Then produces plots for imgs/sec,
# Speedup comparison and various comparison plots between different configu-
# rations. Further variations possible.

# Author: Adrian Grupp
##############################################################################
#TODO dashed grids?
#TODO bar labels smetimes off for small values

import os
import json
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
#----------------------------

img_extension = ".pdf"
legend_loc="upper left"

google_results = { 'Synthetic': { 'AlexNet' : [656, 1209, 2328],
                                  'ResNet50': [52, 99, 195]
                                 },
                   'ImageNet' : { 'AlexNet' : [639, 1136, 2067],
                                  'ResNet50': [51, 99, 194]
                                 }
                 }

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


### Histogramm Images/sec for 1,2,4 gpus, no dataset comparison
#TODO avoid empty plots (dic not in plotData) 
def histImgSec(plotData, metaData, outDir, std=False):
    """ Function to plot histogams that compare images/sec for 
        different amount of gpus and models. 1 Diagramm per dataset.
    """
    datasets = ['CIFAR10', 'Synthetic', 'ImageNet'] 
    models = ['ResNet50', 'ResNet56', 'AlexNet']
    
    patterns = ('\\', '+', 'x', '-', '*', 'o', 'O', '.') #'-'
    barcolors = ('lightsalmon', 'red', 'indianred')  #('lawngreen', 'forestgreen', 'darkgreen')

    # One plot for each dataset to compare models' performance
    for ds in datasets:
        ax = plt.subplot2grid((1,1),(0,0))
        ax.set_axisbelow(True)
        ax.grid(True, axis='y', linewidth=.4)

        i = 0
        width = 0.9
        labels = []
        labelsPos = []    

        # Find all models that were used on the current dataset
        for m in models:    
            gpus = []
            examples_per_sec = [] 
            stdevs = []
            for key, value in plotData.items():
                if ds.lower() in key.lower() and m.lower() in key.lower():
                    gpus.append(value['benchmark_params'].get('num_gpus'))
                    examples_per_sec.append(value['averages'].get('average_examples_per_sec'))
                    stdevs.append(value['stdevs'].get('average_examples_per_sec'))

            if len(examples_per_sec) == 0:
                continue
            print(gpus, examples_per_sec)

            gpus_idx = np.array(gpus).argsort()    
            examples_per_sec_sort = np.array(examples_per_sec)[gpus_idx].astype(int)
            stdevs_sort = np.array(stdevs)[gpus_idx]

            plotData_x = list(np.arange(5*i, 5*i+len(examples_per_sec)))
            labelsPos.append(plotData_x[1]) #middle
            labels.append(m)

            # Bar chart
            pl = plt.bar(plotData_x, examples_per_sec_sort, width,
                 align='center', color=barcolors,
                 alpha=0.7, edgecolor='w')

            # change style of filling    
            for bar, pattern in zip(pl, patterns):
                bar.set_hatch(pattern)
                
            # Standard deviation
            #TODO style 
            if (std):
                plt.errorbar(plotData_x, examples_per_sec_sort,
                        yerr=stdevs_sort, fmt='k.', ecolor='k', elinewidth=2)
            # Bar labels
            else:
                h_shift = examples_per_sec_sort[-1].astype(float)/25.
                if h_shift < 80:
                    h_shift = 80
                for j, v in enumerate(examples_per_sec_sort):
                    plt.text(plotData_x[j] - 0.2, v + 4, str('{:d}'.format(v)),                  # "%i" % v),             # 25.
                            color='black', fontsize='small')
                    if ds in google_results:
                        plt.text(plotData_x[j] - 0.27, v - h_shift,                     # 8.
                                 str('({:d})'.format(google_results[ds][m][j])),
                                 color='k', fontsize='small')
                    if j > 0:
                        plt.text(plotData_x[j] - 0.2, v + h_shift,                    # 2.2
                                 #str(r"$\times$%.1f" % (v/examples_per_sec_sort[0].astype(float))),
                                 str('{}{:3.1f}'.format(r"$\times$", v/examples_per_sec_sort[0].astype(float))),
                                 color='grey', fontsize='x-small')
            i+=1

        if len(labels) == 0:
            continue #don't plot any empty diagrams

        plt.xticks(labelsPos, labels)
        plt.xlabel('Model')
        plt.ylabel('Images/sec')
        #plt.ylim(bottom=-50)
        
        plt.legend(pl, ["1 GPU", "2 GPUs", "4 GPUs"],  fontsize="small", loc=legend_loc)
        
        plt.title('Training: %s - %s - %s Dataset'
                %(metaData[0], metaData[1], ds))

        filename = ds + '-Training' + img_extension
        plt.savefig(outDir + '/' + filename)
    return    
    

## Histogram comparison real synthetic per num_gpu
def histCompareRealSynth(plotData, metaData, outDir):
    """ Function to compare histogams that compare images/sec for 
        different amount of gpus. Compares real to synth data. 
        Works only for imagenet - cifar10 has diferent models
    """
    datasets = ['ImageNet', 'Synthetic'] 
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

        filename = m + '-SynthVsReal' + img_extension
        outPath = outDir + '/' + filename
        meta = ['Synthetic', 'Real Data', metaData[0], metaData[2], metaData[1], m]
        plotComparison(synt_ex_per_sec, synt_gpus, real_ex_per_sec,
                real_gpus, meta, outPath)
    return


## Histogram comparison between two sets of containers, systems, ...
def histCompareTwoSets(plotData1, plotData2, metaData, outDir):
    """ Function to plot histogams that compare images/sec for 
        two different sets. Compares per num_gpu.
        Requires: same dataset, same model.
    """
    datasets = ['Synthetic', 'ImageNet', 'CIFAR10'] 
    models = ['AlexNet', 'ResNet50', 'ResNet56']

    # One plot for each dataset and model to have all comparisons
    for ds in datasets:
        for m in models:    
            meta = metaData.copy()
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
    
            filename = metaData[2] + '-' + ds + '-' + m + '-' + metaData[3] + img_extension
            outPath = outDir + '/' + filename
            meta.append(ds + ' - ' + m)
            plotComparison(ex_per_sec_1, gpus_1, ex_per_sec_2, gpus_2, meta,
                    outPath)
    return


def plotComparison(ex_per_sec_1, gpus_1, ex_per_sec_2, gpus_2, metaData, outPath):
    print(ex_per_sec_1, ex_per_sec_2)

    patterns = ('+', '\\', 'x', '-', '*', 'o', 'O', '.') #'-'

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
            align='center', facecolor='r', alpha=0.5, edgecolor='w', hatch='+')
    for j, v in enumerate(ex_per_sec_1_sort):
        plt.text(j - width + 0.15, v + 4, str("%i" % v), color='k')
                
    # second set plots 
    p2 = plt.bar([p + width for p in pos], ex_per_sec_2_sort, width,
            align='center', facecolor='g', alpha=0.5, edgecolor='w', hatch='x')
    
    h_shift = ex_per_sec_2_sort[-1].astype(float)/20.
    for j, v in enumerate(ex_per_sec_2_sort):
        plt.text(j + width - 0.1, v + 4, str("%i" % v), color='k')
        if metaData[1] == "Real Data":
            plt.text(j + width - 0.115, v - h_shift, 
                     str("(%i)" % google_results['ImageNet'][metaData[5]][j]),
                                 color='k', fontsize='small')        

    # Bar labels
    labelsPos = [p + width/2 for p in pos]
    labels = sorted(gpus_2)

    print(labelsPos, labels)
    plt.xticks(labelsPos, labels)
    plt.xlabel('#GPUs')
    plt.ylabel('Images/sec')
    plt.ylim(top=1.1*ex_per_sec_2_sort[-1].astype(float))

    plt.legend((p1[0], p2[0]), (metaData[0], metaData[1]), fontsize="small", loc=legend_loc)
    plt.title('%s vs %s - %s - %s ' %(metaData[0], metaData[1], metaData[2],
        metaData[4]))
    plt.savefig(outPath)
    return


## Scaling
def plotScaling(plotData, metaData, outDir):
    """ Plot the scaling for a dataset and several models for each num_gpu.
    Consider only images/sec. Includes also ideal scaling.
    """
    #TODO: nicer grid
    #TODO: avoid empty plots
    datasets = ['Synthetic', 'ImageNet', 'CIFAR10']  
    models = ['AlexNet', 'ResNet50', 'ResNet56']

    linestyles = (':', '--', '--')
    linecolors = ('red', 'blue', 'blue')
    linewidths = (2.5, 1.2, 1.2)
    markers = ('o', '^', '^')
   
    # One plot for each dataset 
    for ds in datasets:
        plt.clf()
        for im, m in enumerate(models): #lstyle, lcolor in zip(models, linestyles, linecolors):
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
            plt.plot(gpus_sort, scaling, label=m, linestyle=linestyles[im], 
                     color=linecolors[im], linewidth=linewidths[im])       
        

        # ideal scaling
        plt.plot([1,2,4], [1,2,4], label='Ideal', linestyle="-", 
                 color="silver", linewidth=1.2)

        plt.xticks([1,2,3,4], [1,2,3,4])
        plt.xlim(0.9, 4.1)
        plt.ylim(0.9, 4.1)
        plt.xlabel('#GPUs')
        plt.ylabel('Speedup')
    
        plt.legend(fontsize="small", loc=legend_loc)  #loc=2
        plt.title('Speedup: %s - %s - %s Dataset' %(metaData[0], metaData[1], ds))
        plt.grid()

        filename = ds + '-Scaling' + img_extension
        outPath = outDir + '/' + filename
        plt.savefig(outPath)
    return


## Runtime
#TODO

## Evaluation accuracy
#TODO

## Evaluation examples/sec
#TODO

##Comparison Table
def createCompTable(fileData1, fileData2, metaData, outDir):
    datasets = ['Synthetic', 'ImageNet']  
    models = ['AlexNet', 'ResNet50']

    head = ['#GPUs', 'LSDF', 'FH2', 'official TF']
    gpus = ['1','2','4']

    # One  table for each dataset and model
    for ds in datasets:
        for m in models:    
            ex_per_sec_1 = []
            gpus_1 = []
            ex_per_sec_2 = []
            gpus_2 = []
            # Find all models that were used on the current dataset
            for key, value in fileData1.items():
                if m.lower() in key.lower():
                    if ds.lower() in key.lower():
                        ex_per_sec_1.append(value['averages'].get('average_examples_per_sec'))
                        gpus_1.append(value['benchmark_params'].get('num_gpus'))
            for key, value in fileData2.items():
                if m.lower() in key.lower():
                    if ds.lower() in key.lower():
                        ex_per_sec_2.append(value['averages'].get('average_examples_per_sec'))
                        gpus_2.append(value['benchmark_params'].get('num_gpus'))
                        
            if len(ex_per_sec_1) == 0 or len(ex_per_sec_2) == 0:
                continue
 
            gpus_1_idx = np.array(gpus_1).argsort()    
            ex_per_sec_1_sort = np.array(ex_per_sec_1)[gpus_1_idx].astype(int)
            gpus_2_idx = np.array(gpus_2).argsort()    
            ex_per_sec_2_sort = np.array(ex_per_sec_2)[gpus_2_idx].astype(int)
            
            lines = []
            lines.append(head)
            for i,v in enumerate(ex_per_sec_1_sort):
                lines.append([gpus[i], ex_per_sec_1_sort[i],
                    ex_per_sec_2_sort[i], ''])

            filename = metaData[0] + '-' + ds + '-' + m + '-' + metaData[1] + '.csv'
            outPath = outDir + '/' + filename

            with open(outPath, 'w') as write_file:
                writer = csv.writer(write_file)
                writer.writerows(lines)
            write_file.close()
    return


#-----------------------------------------------------------------
if __name__ == "__main__":
    #TODO argv for selection of plot mode?
    #TODO useful output for data being processed

    # Dir for benchmark data
    wd = os.getcwd()
    resultDir = "%s/benchmark_results" % wd
    
    # Target location for plots
    plotDir = "%s/training_plots" %wd    
    compDir = "%s/comparison_plots" %wd
    tableDir = "%s/comp_tables" %wd

    if not os.path.exists(plotDir):
        os.makedirs(plotDir)
    if not os.path.exists(compDir):
        os.makedirs(compDir)
    if not os.path.exists(tableDir):
        os.makedirs(tableDir)

#    folders = ['lsdf_udocker_5epoch',   'lsdf_udocker_500batch',
#            'lsdf_singularity_5epoch',  'lsdf_singularity_500batch',
#            'fh2_udocker_5epoch',       'fh2_udocker_500batch',
#            'fh2_singularity_5epoch',   'fh2_singularity_500batch']
#
#    metaData = [['LSDF', 'Udocker', '5_epochs'],        ['LSDF', 'Udocker', '500_batches'],
#                ['LSDF', 'Singularity', '5_epochs'],    ['LSDF', 'Singularity', '500_batches'],
#                ['ForHLR2', 'Udocker', '5_epochs'],     ['ForHLR2', 'Udocker', '500_batches'],
#                ['ForHLR2', 'Singularity', '5_epochs'], ['ForHLR2', 'Singularity', '500_batches']]

    folders = ['lsdf_udocker_5epoch',   'lsdf_udocker_500batch',
               'lsdf_singularity_5epoch',  'lsdf_singularity_500batch',
               'fh2_udocker_5epoch',       'fh2_udocker_500batch']

    metaData = [['LSDF', 'Udocker', '5_epochs'],        ['LSDF', 'Udocker', '500_batches'],
                ['LSDF', 'Singularity', '5_epochs'],    ['LSDF', 'Singularity', '500_batches'],
                ['ForHLR2', 'Udocker', '5_epochs'],     ['ForHLR2', 'Udocker', '500_batches']]

    #-----------------------------------------------------------------
    
    ## Debug ##        
    """
    procFiles = os.path.join(resultDir, folders[1])
    print('Processing: ' + procFiles)
    plotData = readFiles(procFiles)
    outDir = os.path.join(plotDir, folders[1])
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    #histImgSec(plotData, metaData[1], outDir)
    plotScaling(plotData, metaData[1], outDir)  

    """

    # Plot bar charts for all folders:
    # 1) Imgs/sec per num_gpu to compare models, per dataset.
    # 2) Comparison of synthetic vs real data (imagenet).
    # 3) Scaling single folder, real&synth extra0
    for i,f in enumerate(folders):
        procFiles = os.path.join(resultDir, f)
        print('Processing: ' + procFiles)
        plotData = readFiles(procFiles)
        if plotData:
            outDir = os.path.join(plotDir, f)
            if not os.path.exists(outDir):
                os.makedirs(outDir)
            histImgSec(plotData, metaData[i], outDir) 
            histCompareRealSynth(plotData, metaData[i], outDir)
            plotScaling(plotData, metaData[i], outDir)  

    # Plot comparison between singularity & udocker, all machines, all data 
    for i in [0,1,4,5]:
        procFiles1 = os.path.join(resultDir, folders[i])
        plotData1 = readFiles(procFiles1)
        procFiles2 = os.path.join(resultDir, folders[i+2])
        plotData2 = readFiles(procFiles2)
        meta = [metaData[i][1], metaData[i+2][1], metaData[i][0],
            metaData[i][2]]
        if plotData1 and plotData2:
            outDir = os.path.join(compDir, 'udocker_vs_singularity')
            if not os.path.exists(outDir):
                os.makedirs(outDir)
            histCompareTwoSets(plotData1, plotData2, meta, outDir)

    # Plot comparison between lsdf & fh2, all containers, all data
    for i in range(4):
        procFiles1 = os.path.join(resultDir, folders[i])
        plotData1 = readFiles(procFiles1)
        procFiles2 = os.path.join(resultDir, folders[i+4])
        plotData2 = readFiles(procFiles2)
        meta = [metaData[i][0], metaData[i+4][0], metaData[i][1],
            metaData[i][2]]
        if plotData1 and plotData2:
            outDir = os.path.join(compDir, 'lsdf_vs_fh2')
            if not os.path.exists(outDir):
                os.makedirs(outDir)
            histCompareTwoSets(plotData1, plotData2, meta, outDir)

    # Plot scaling 500 batches, real + synth
    #TODO
    # Plot scaling comparison per dataset, udocker vs sing, lsdf vs fh2
    #TODO

    # Produce comparison tables lsdf vs fh2 vs dummy column
    # Dummy column is for extra entries such as official tf results
    for i in range(4):
        procFiles1 = os.path.join(resultDir, folders[i])
        fileData1 = readFiles(procFiles1)
        procFiles2 = os.path.join(resultDir, folders[i+4])
        fileData2 = readFiles(procFiles2)
        meta = [metaData[i][1], metaData[i][2]]
        if fileData1 and fileData2:
            outDir = tableDir
            if not os.path.exists(outDir):
                os.makedirs(outDir)
            createCompTable(fileData1, fileData2, meta, outDir)
