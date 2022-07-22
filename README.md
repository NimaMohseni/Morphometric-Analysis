# Morphometric-Analysis

**A python tool to perform analysis on (morphometric) landmark data after performing General Procrustes Analysis.**

*Nima Mohseni, Eran Elhaik*


## Introduction

  This repository contains a python based tool which can process the results of morphometric analysis of landmark data; it can be used for futher investigation structure of data structure, look for outliers, classify (new) samples, plot the classifiaction boundaries and produce plots.

The tool uses the output file of ***morphologika*** program in '*.txt*' format as its input.

## Usage

### Input Data

  After processing ladmark data in morphologika comprising of General Procrustes Analysis and Principal Component Analysis the output can be saved in '.txt' format. This module uses that text file as its input.
  
 There's also a need for a data frame with columns containing the information of the samples:
 
*Group: The name of the group to which each sample belongs.
*Encoded: The encoded labels of these groups.
*Sex.
*Complete: The complete name of each sample (the names used in the initial landmark data imported in morphologika).

  
  

## Examples

## Dependencies
