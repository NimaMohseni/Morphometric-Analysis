# Morphometric-Analysis

**A python tool to perform analysis on (morphometric) landmark data after performing General Procrustes Analysis.**

*Nima Mohseni, Eran Elhaik*


## Introduction

  This repository contains a python based tool which can process the results of morphometric analysis of landmark data; it can be used for futher investigation structure of data structure, look for outliers, classify (new) samples, plot the classifiaction boundaries and produce plots.

The tool uses the output file of ***morphologika*** program in '*.txt*' format as its input.

## Usage

### Input Data

  After processing ladmark data in morphologika comprising of General Procrustes Analysis and Principal Component Analysis the output can be saved in '*.txt*' format. This module uses that text file as its input.
  
 There's also a need for a data frame with columns containing the information of the samples:
 
* `Group:` The name of the group to which each sample belongs.
* `Encoded:` The encoded labels of these groups.
* `Sex.`
* `Complete:` The complete name of each sample (the names used in the initial landmark data imported in morphologika).

It is recommended that this data-set should be produced for the intial landmark data before removing any samples because it would act as a refrence, making further comparisons feasible.
This refrence data-set can be created just once and saved to be read and used later.

The first step is to create an instance of the class after importing it amd then passing the refrence data-set to it:

### Reading the data

```python
from Morphometrics import procpca as procpca
prpca = procpca(df)
```

After this, the results of a morphologika analysis can be read using the `.read` function:

```python
dataf, datag, ind, name = prpca.read('without papio cynocephalus_8_remains.txt')
```

This function takes a '*.txt*' as its input and returns two data-sets containing a summary of the analysed samples (*dataf*), the results of the procrustes analysis in tabular format (*datag*), the index of the samples in the intial refrence data frame (*ind*) and their names (*name*).

It should be noted that this function would also automatically run the `post_process()` function at its end.

### Processing the data

The *deletg* arguement of `post_process(deletg = None)` can reduce the data set and remove certain groups. The reason for this could be that one might want to perform a classification task and for any possible reason they might not want to include a group as a refrence. For the papi data set we used for examples, it would also remove *lophocebus aterrimus* since it only has two samples. The reduced data set is only used for training classifiers and does not replace anything.

```python
prpca.post_process(deletg='papio cynocephalus')
```
The 



## Examples

```python
prpca = procpca(df)
dataf, datag, ind, name, data = prpca.read('without papio cynocephalus_8_remains.txt')
```


## Dependencies
