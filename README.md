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

### Initiating

```python
from Morphometrics import procpca as procpca
prpca = procpca(df)
```

This class has two inputs:

```python
def(dataframe, classifier = None)
```
The `dataframe` is mandatory but the `classifier` can be left blank. If any specific classifier is supposed to be used for creating decision boundary plots, it should be passed here, if not, a default KNN (n neighbours = 2) will be used.

After this, the results of a morphologika analysis can be read using the `.read` function:

### Reading the data

```python
dataf, datag, ind, name = prpca.read('without papio cynocephalus_8_remains.txt')
```

This function takes a '*.txt*' as its input and returns two data-sets containing a summary of the analysed samples (*dataf*), the results of the procrustes analysis in tabular format (*datag*), the index of the samples in the intial refrence data frame (*ind*) and their names (*name*).

It should be noted that this function would also automatically run the `post_process()` function at its end.

### Processing the data

The *`deletg`* arguement of `post_process(deletg = None)` can reduce the data set and remove certain groups. The reason for this could be that one might want to perform a classification task and for any possible reason they might not want to include a group as a refrence. For the papi data set we used for examples, it would also remove *lophocebus aterrimus* since it only has two samples. The reduced data set is only used for training classifiers and does not replace anything.

```python
prpca.post_process(deletg='papio cynocephalus')
```

### Creating Plots

Several plots can be created using the available functions; including the PCA plots, the t-Distributed Stochastic Neighbor Embedding (t-SNE) plots and a dendrogram plot. The descision boundary of any arbitrary classifier used for classifying the samples can also be represented in the t-SNE plots.

#### PCA Plots

`PCAplotm` cab be used for creating PCA plots:

```python
PCAplotm(y, x, ind1, ind2,
        sav1=0, sav2='', sav3 = '.svg',
        annote = False, ax = None,
        dlegend = True, index_r = 0)
```

`y` is the label of the samples (encoded), `x` is the PC data, the plots can be saved by setting `sav1=1`, `sav2` can take a string for naming the saved file, `sav3` can change the default save format '*.svg*' (vector quality). Setting `annote` to True will write the index of each sample (accordin to the initial data frame) beside it. in case the plot is meant to be presented in an axes object of a *matplotlib* figure, it should be passed to `ax`. Setting `dlegend` to True will add a legend to the plot and setting `index_r=1` will add a string to the legend, listing all the abscent samples from the initial refrence data-set (in case of removal).

`PCAplotm` is a function to create the first 3 PCA plots together.

```python
PCAplot(y, x, ind1, ind2,
        sav1, sav2, sav3 ,
        annote, ax,
        dlegend, index_r)
```

The function arguements are similar to that of `PCAplot`.

#### t-SNE Plots

The `plot_tsne` function is a multi-purpose function to create t-SNE plots.

```python
plot_tsne(n_r=4, ax= None,
          localo = False, decision_boundary = False,
          cv = False, dlegend = True, index_r=0,
          perplexity=10, n_neighbors=5)
```

This function can be used to create a simple t-SNE plot and more. By setting `localo` to True, a local outlier factor analysis will be performed and the score of the samples (according to this measures) will be presented as circles around them with radius proportional to the score. `n_neighbors=5` is a hyper-parameters of the t-SNE algorithm.

Setting the `decision_boundary` to True will depict the decision boundary of the classifier passed to the class. If `cv` is left as default, a single fit will be performed on the data-set, changing it to True will perform a 5-fold cross-validation (shuffled). `perplexity` is a hyper-parameters of the local outlier factor algorithm.

#### Dendrogram plot

`plot_dendrogram()` will create a denrogram using the results of the GPA analysis.
 

## Examples

A jupyter notebook in Examples folder shows examples of implementing the module.


## Dependencies
