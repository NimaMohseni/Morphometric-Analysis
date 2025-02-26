####################################################################################################################################
######################################################### By: Nima Mohseni, Eran Elhaik ############################################
##                                                                                                                                ##
##                                    Lund University - Faculty of Science - Department of Biology                                ##
####################################################################################################################################

# A class to process the saved results of Morphologika

####################################################################################################################################
##                                                          Importing Modules                                                     ##
####################################################################################################################################

# ------------------ Plotting Libraries ------------------

# t-SNE for dimensionality reduction
from sklearn.manifold import TSNE

# Matplotlib for general plotting
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter

# Dendrogram plotting and Agglomerative clustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

# Convex hull computation for geometric shapes
from scipy.spatial import ConvexHull

# ------------------ Data Manipulation Libraries ------------------

# Pandas for data handling
import pandas as pd
import numpy as np

# ------------------ Machine Learning Libraries ------------------

# For cross-validation, model evaluation, and label encoding
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder

# For scaling features and data preprocessing
from sklearn.preprocessing import *

# For model performance metrics (e.g., accuracy, confusion matrix)
from sklearn.metrics import *

# ------------------ Outlier Detection Libraries ------------------

# Local Outlier Factor (LOF) for anomaly detection
from sklearn.neighbors import LocalOutlierFactor

# Isolation Forest for outlier detection
from sklearn.ensemble import IsolationForest

# One-Class SVM for outlier detection
from sklearn.svm import OneClassSVM

# ------------------ K-Nearest Neighbors (KNN) ------------------

# KNN classifier for classification tasks
from sklearn.neighbors import KNeighborsClassifier

# Random module for randomization
from random import shuffle

import warnings


####################################################################################################################################
##                                                           The class itself                                                     ##
####################################################################################################################################

class procpca():
    # Initial parameters
    def __init__(self, dataframe, classifier=None, nn=None, encoder=None):
        """
        Initializes the procpca class with the provided dataset and parameters.

        Args:
            dataframe (pd.DataFrame): The main dataset for analysis.
            classifier (optional): Classifier for classification tasks, defaults to KNeighborsClassifier.
            nn (optional): Number of neighbors for the KNN classifier, defaults to 2.
            encoder (optional): Label encoder for encoding categorical labels, defaults to LabelEncoder.
        """
        self.df = dataframe  # Store the dataset
        self.PCAt = []  # Placeholder for PCA data
        self.PCApt = []  # Placeholder for PCA principal component data
        self.GPAt = []  # Placeholder for GPA data (Procrustes Analysis)

        # Setup the encoder for labels
        if encoder is None:
            self.encoder = LabelEncoder()  # Default label encoder
            self.eflag = 0  # Flag to indicate if custom encoder is used
        else:
            self.encoder = encoder  # Use the provided encoder
            self.eflag = 1  # Flag for custom encoder

        # Fit the encoder based on the dataset labels
        if self.eflag == 0:
            self.encoder.fit(self.df.iloc[:, 0])  # Fit encoder on the first column (assumed labels)
        else:
            self.encoder.fit(np.array(self.df.iloc[:, 0]).reshape(-1, 1))  # Custom encoder fitting

        # Set the number of neighbors for KNN
        n_neighbors = 2  # Default number of neighbors for KNN
        if nn is not None:
            n_neighbors = nn  # Override with user-specified number of neighbors

        # Setup classifier: KNN by default
        if classifier is None:
            self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        else:
            self.classifier = classifier  # Use provided classifier

    def read(self, path):
        """
        Reads PCA and Procrustes Analysis data from the given file path and processes it.

        Args:
            path (str): The path to the text file containing PCA and Procrustes analysis data.

        Returns:
            data (pd.DataFrame): A DataFrame containing PCA scores and sample information.
            datag (pd.DataFrame): A DataFrame containing additional group information and Procrustes data.
            ind (np.ndarray): Indices of the samples from the reference dataset.
            name (list): The names of the samples in the dataset.
        """
        # Initialize lists to store PCA, PCAp, and GPAp data
        PCA = []
        PCAp = []
        GPAp = []
        
        # Read PCA data from the file
        with open(path, 'r') as f:
            flag = 0
            for line in f:
                if 'Between OTU tangent space distances' in line:
                    break
                if 'PC Scores' in line:
                    flag = 1  # Start reading PCA scores after this line
                if flag == 0:
                    continue  # Skip lines until PCA scores are found
                else:
                    PCA.append(line)  # Append the PCA data

        # Read PCA principal component data
        with open(path, 'r') as f:
            flag = 0
            for line in f:
                if 'PC Scores' in line:
                    break
                if ',eigenvalue,percentage of total variance explained,cumulative variance explained' in line:
                    flag = 1  # Start reading eigenvalue and variance data
                if flag == 0:
                    continue
                else:
                    PCAp.append(line)  # Append the PCAp data

        # Read Procrustes analysis data
        with open(path, 'r') as f:
            flag = 0
            for line in f:
                if 'Procrustes mean shape' in line:
                    break
                if 'Procrustes output' in line:
                    flag = 1  # Start reading GPA (Procrustes) data
                if flag == 0:
                    continue
                else:
                    GPAp.append(line)  # Append the Procrustes data

        # Clean up the PCA and PCAp data (remove unwanted lines and extra spaces)
        PCA = PCA[3:]  # Remove initial lines (metadata, headers, etc.)
        PCAp = PCAp[1:]  # Skip the first line of PCAp data
        
        # Clean and store PCA and PCAp data (remove extra spaces and newlines)
        PCAt = [i.replace(' ', '').replace('\n', '') for i in PCA]
        PCApt = [i.replace(' ', '').replace('\n', '') for i in PCAp]

        # Process GPA data (Procrustes analysis results)
        GPApt2 = [i.replace(' ', '').replace('\n', '') for i in GPAp]
        GPApt2 = GPApt2[2:-1]  # Trim the beginning and end
        
        # Convert the GPA data into numerical format
        GPApt = []
        for i in GPApt2:
            tmp = []  # Initialize an empty list to store the numerical values
            for j, word in enumerate(i.split(',')):  # Split each line by commas
                if j > 1:  # Skip the first two columns (if applicable)
                    try:
                        tmp.append(float(word))  # Try to convert the word to a float and append it
                    except:
                        a = 0  # If an error occurs, do nothing (could be a non-numeric value)
            GPApt.append(tmp)  # Append the processed line to GPApt
        GPApt = np.array(GPApt)  # Convert the list of lists to a numpy array

        self.GPApt = GPApt  # Store the GPA data

        # Extract variance explained (varex) from PCApt data
        varex = []
        for i in PCApt:
            for j, word in enumerate(i.split(',')):
                if j == 2:  # The third column contains the variance explained
                    varex.append(float(word))
        varex = np.array(varex)  # Convert to numpy array

        self.varex = varex  # Store the explained variance data
        self.PCAt = PCAt  # Store the PCA scores data
        self.PCApt = PCApt  # Store the PCA principal components data

        # Initialize the name and rows lists to store sample names and PCA scores
        name = []
        rows = []
        
        # Extract sample names and PCA score rows from PCA data
        for i in self.PCAt:
            if len(i) <= 1:  # Skip empty lines
                continue

            row = []
            for j, word in enumerate(i.split(',')):
                if j == 0:
                    try:
                        sn = int(word)  # Sample number
                    except:
                        pass
                    continue
                elif j == 1:
                    name.append(word)  # Store the sample name
                else:
                    try:
                        row.append(float(word))  # Store the PCA score for this sample
                    except:
                        pass
            rows.append(np.array(row))  # Add the row to the PCA data

        rows = np.array(rows)  # Convert to numpy array

        # Compare the sample names with the reference dataset to find their indices
        ind = []
        for k, h in enumerate(name):
            for i, word in enumerate(self.df.iloc[:, 3]):
                if word == h:  # If the name matches the reference, record the index
                    ind.append(i)
                    break
        
        ind = np.array(ind)  # Convert to numpy array

        # Preparing a data frame of indices and names
        data = pd.DataFrame()
        data['index'] = ind
        data['label'] = name

        # Get the labels (encoded) for the samples based on the reference dataset
        y = ind
        self.ind = ind
        self.y = np.array(self.df.iloc[y, 1], dtype=int)
        # print(self.df.iloc[:, 1])

        # Adding PCA components to the data frame
        for i in range(rows.shape[1]):
            n = 'PC' + str(i + 1)  # Naming the PCA components (PC1, PC2, ...)
            data[n] = rows[:, i]  # Add each PCA component as a column

        # Prepare a second data frame (datag) with more information
        datag = pd.DataFrame()
        datag['index'] = ind
        datag['group'] = np.array(self.df.iloc[ind, 0])  # Group info from the reference dataset
        datag['label'] = name
        datag['encoded'] = self.y  # Encoded labels

        self.labels = name  # Store the sample names
        self.datag = datag  # Store the additional group info

        # Add Procrustes analysis results (GPA) to the datag DataFrame
        for i in range(GPApt.shape[1]):
            n = f'lndm{int(i/3)+1}-{int(i%3)+1}'  # Naming convention for Procrustes data
            datag[n] = GPApt[:, i]  # Add each GPA result as a new column

        # Record the indices of removed samples (samples not found in PCA)
        self.removed = ''
        for i in range(len(self.df)):
            if i not in ind:
                if self.removed != '':
                    self.removed += ','  # Separate removed indices with commas
                self.removed += str(i)  # Append the removed index

        self.x = rows  # Store the PCA score data

        # Run the post-processing function automatically (assumed to be defined elsewhere)
        self.post_process()

        # Return the results: PCA data, Procrustes data, sample indices, and names
        return data, datag, ind, name


######################################################################################################################################
##                                                    A function for preparing PCA plots                                          ##
######################################################################################################################################

    def PCAplotm(self,
                y,               # Array of class labels for the samples
                x,               # PCA matrix with principal component values for each sample
                ind1, ind2,      # Indices of the principal components to plot on x and y axes
                sav1=0,          # Flag to save the plot (1 to save, 0 to not)
                sav2='',         # Prefix for the saved file name
                sav3='.svg',     # File format to save the plot (default: .svg)
                annote=False,     # Flag to annotate points with their index (True/False)
                ax=None,         # Axes to plot on (None means a new plot will be created)
                dlegend=True,    # Flag to display legend (True/False)
                ticks=True,      # Flag to show axis ticks (True/False)
                hatch=None,      # If set, creates a hatch for certain groups in the plot
                index_r=0,       # Flag to include removed samples in the plot
                size=20,         # Size of the points in the scatter plot
                legend_font_s=10 # Font size for the legend
                ):
        """
        This function generates a PCA scatter plot for the given data. It supports various customization options
        such as annotations, convex hulls, tick settings, and saving the plot.
        """

        lfs = legend_font_s  # Legend font size
        
        # Predefined color list for scatter plot
        colors = ['blue', 'chocolate', 'green', 'deeppink', 'brown', 'gold', 'black']
        color = []

        ##################################################################
        # Assign colors based on class labels (y)
        ##################################################################
        for i in y:
            color.append(colors[i])
        
        # Create a new figure if no axes were provided
        if ax == None:
            fig, ax = plt.subplots(figsize=(7, 6))

        ##################################################################
        # Creating legend for the scatter plot
        ##################################################################
        legends = []
        for i in np.unique(y):
            # Inverse transform the class label
            if self.eflag == 0:
                label = self.encoder.inverse_transform([i]).tolist()[0]
            else:
                label = self.encoder.inverse_transform(np.array(i).reshape(1, -1))[0][0]
            
            # Standardize class labels for better readability
            label = label.replace('lophocebus albigena', 'Lophocebus albigena')
            label = label.replace('lophocebus aterrimus', 'Lophocebus aterrimus')
            label = label.replace('macaca mulatta', 'Macaca mulatta')
            label = label.replace('mandrillus leucophaeus', 'Mandrillus leucophaeus')
            label = label.replace('papio cynocephalus', 'Papio cynocephalus')

            # Scatter plot for each class group
            scatter = ax.scatter(x[:, ind1-1][y==i], x[:, ind2-1][y==i], c=colors[i],
                                label=label, marker='o', s=size, edgecolors='black')
            legends.append(scatter)

        ##################################################################
        # Handling removed samples (if applicable)
        ##################################################################
        if index_r == 1:
            legends.append(mpatches.Patch(color='none',
                                        label='removed samples: '+self.removed[:6]+"\n"+self.removed[6:]))

        ##################################################################
        # Set axis labels with explained variance percentage
        ##################################################################
        ax.set_xlabel(f'PC{ind1} ({self.varex[ind1-1]:.2f}%)', fontsize=22)
        ax.set_ylabel(f'PC{ind2} ({self.varex[ind2-1]:.2f}%)', fontsize=22)
        plt.setp(ax.get_xticklabels(), fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=10)
        
        # Optionally remove axis labels
        if not ticks:
            ax.set(xlabel=None, ylabel=None)

        ##################################################################
        # Annotate the samples with their indices
        ##################################################################
        if annote:
            e1 = (np.max(x[:, ind1-1]) - np.min(x[:, ind1-1])) / 200
            e2 = (np.max(x[:, ind2-1]) - np.min(x[:, ind2-1])) / 150
            for i, txt in enumerate(self.ind):
                ax.annotate(str(txt), (x[i, ind1-1] + e1, x[i, ind2-1] + e2), size=4)

        ##################################################################
        # Draw convex hulls around each group (if applicable)
        ##################################################################
        ind1 = ind1 - 1
        ind2 = ind2 - 1
        for i in np.unique(y):
            points = x[:, [ind1, ind2]][np.array(y == i).ravel()]
            if points.shape[0] < 3:
                continue  # Skip if the group has fewer than 3 points

            # Check if the group should be hatched
            hatch_f = None
            if hatch and self.encoder.inverse_transform([i])[0] in hatch:
                hatch_f = 'x'

            # Create convex hull for the group
            hull = ConvexHull(points)
            x_hull = np.append(points[hull.vertices, 0], points[hull.vertices, 0][0])
            y_hull = np.append(points[hull.vertices, 1], points[hull.vertices, 1][0])

            # Fill the hull with color and optional hatching
            ax.fill(x_hull, y_hull, alpha=0.3, c=colors[i], hatch=hatch_f)

        ##################################################################
        # Adjust x and y axis ticks
        ##################################################################
        start, end = ax.get_xlim()
        step = (end - start) / 8
        ax.xaxis.set_ticks(np.arange(start + step, end, step))

        start, end = ax.get_ylim()
        step = (end - start) / 12
        ax.yaxis.set_ticks(np.arange(start + step, end, step))

        ##################################################################
        # Format axis ticks (optional)
        ##################################################################
        if ticks:
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(axis='both', which='major', labelsize=12)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

        ##################################################################
        # Display legend if requested
        ##################################################################
        if dlegend:
            matplotlib.rcParams['legend.fontsize'] = lfs
            legend1 = ax.legend(handles=legends, loc="best", handlelength=1, framealpha=0.35)
            ax.add_artist(legend1)

        ##################################################################
        # Save the plot if requested
        ##################################################################
        if sav1 == 1:
            plt.savefig(f'Monkeys_{sav2}_PCA_plot{ind1+1}{ind2+1}{sav3}', transparent=True)
            plt.show()

####################################################################################################################################
##                                       A function to create the first 3 PCA plots together                                      ##
####################################################################################################################################

    def PCAplot(self,
                sav1,            # Flag to save the plots (1 to save, 0 to not)
                sav2='',         # Prefix for the saved file name
                sav3='.pdf',     # File format to save the plot (default: .pdf)
                annote=False,     # Flag to annotate points with their index (True/False)
                ax=None,         # Axes to plot on (None means a new plot will be created)
                index_r=0):      # Flag to include removed samples in the plot

        """
        This function generates the first three PCA plots:
        - PC1 vs PC2
        - PC1 vs PC3
        - PC2 vs PC3
        """
        # Plot the first PCA plot (PC1 vs PC2)
        self.PCAplotm(self.y, self.x, 1, 2, sav1, sav2, sav3, annote, ax, False, index_r)
        
        # Plot the second PCA plot (PC1 vs PC3)
        self.PCAplotm(self.y, self.x, 1, 3, sav1, sav2, sav3, annote, ax, False, index_r)
        
        # Plot the third PCA plot (PC2 vs PC3)
        self.PCAplotm(self.y, self.x, 2, 3, sav1, sav2, sav3, annote, ax, False, index_r)

####################################################################################################################################
##                                       Create linkage matrix for the dendrogram to be plotted                                   ##
####################################################################################################################################

    def dendrogram_GPA(self,
                    model,             # Model used for clustering (e.g., AgglomerativeClustering)
                    **kwargs):         # Additional arguments for customization

        """
        This function creates a linkage matrix for the dendrogram plot.
        It calculates the number of samples under each node and uses that to
        create the linkage matrix which is then plotted as a dendrogram.
        """
        
        # Initialize an array to store counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        
        # Calculate the counts of samples under each node
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    # If it's a leaf node (actual sample)
                    current_count += 1
                else:
                    # If it's a non-leaf node, count from previously computed counts
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        # Create the linkage matrix
        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

####################################################################################################################################
##                                       Plot the dendrogram for hierarchical clustering                                        ##
####################################################################################################################################

    def plot_dendrogram(self,
                        **kwargs):         # Additional arguments for customization

        """
        This function plots a dendrogram using AgglomerativeClustering.
        It computes the full tree and plots the top three levels.
        """
        
        # Perform Agglomerative Clustering to compute the full tree
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
        X = self.GPApt  # Data for clustering
        model = model.fit(X)
        
        # Create a figure for the dendrogram
        fig = plt.figure(figsize=(10, 6))
        
        # Plot the dendrogram and truncate it to show only the top three levels
        self.dendrogram_GPA(model, **kwargs, truncate_mode="level", labels=self.y)

####################################################################################################################################
##                                                A function for post processing the data                                         ##
####################################################################################################################################

    def post_process(self, 
                    deletg=None):   # The value of the group to be deleted (None means no deletion)

        """
        This function processes the data by:
        1. Extracting the relevant columns from the dataframe (`self.datag`).
        2. Applying a mask to remove specific samples, such as 'lophocebus aterrimus' and others based on the `deletg` argument.
        3. Creating a label encoder for encoding the target labels (`ys`).
        """

        # Extracting feature matrix (X) and target vector (y)
        X = np.array(self.datag.iloc[:, 4:])   # Features from column 4 onward
        y = np.array(self.datag.iloc[:, 3])    # Target labels from column 3
        self.y0 = np.array(self.datag.iloc[:, 1])  # The original target labels (first column)

        # Creating a mask to filter the data
        mask = np.ones(len(y), dtype=bool)  # Default mask (all True, meaning keep all data)
        
        # Removing samples where 'lophocebus aterrimus' is present in the target labels (y0)
        delet = np.where(self.y0 == 'lophocebus aterrimus')
        mask[delet] = False

        # If a group name is provided for deletion, remove samples belonging to that group
        if deletg is not None:
            delet = np.where(self.y0 == deletg)
            mask[delet] = False

        # Apply the mask to remove the corresponding samples
        self.Xr = X[mask]  # Filtered feature matrix
        self.ys = y[mask]  # Filtered target vector

        # Store the group name to be deleted (if any)
        self.deletg = deletg

        # Encode the target labels using LabelEncoder
        self.encoder2 = LabelEncoder()
        self.encoder2.fit(self.ys)           # Fit the encoder on the filtered target labels
        self.y2e = self.encoder2.transform(self.ys)  # Transform target labels into encoded values


####################################################################################################################################
##                               A multi-purpose function for performing t-SNE and preparing plots                                ##
####################################################################################################################################    

    def plot_tsne(self,
                n_r=4,
                ax=None,
                method=None,
                decision_boundary=False,
                cv=False,
                dlegend=False,
                index_r=0,
                perplexity=10,
                n_neighbors=5,
                annote=False,
                annote_o=False,
                ind_o=None,
                ticks=True,
                size=20,
                rotate=0):
        """
        Perform t-SNE on the dataset and create various types of plots such as outlier detection, decision boundaries, and scatterplots.
        
        Parameters:
        - n_r: Number of colors for contour (default: 4)
        - ax: Matplotlib axis to plot on (default: None, creates a new plot)
        - method: Outlier detection method ('lof', 'if', 'ocsvm', or any classifier)
        - decision_boundary: Boolean to plot decision boundaries (default: False)
        - cv: Boolean to use cross-validation when plotting decision boundaries (default: False)
        - dlegend: Boolean to display legend (default: False)
        - index_r: Index for specific annotations (default: 0)
        - perplexity: Perplexity for t-SNE (default: 10)
        - n_neighbors: Number of neighbors for outlier detection (default: 5)
        - annote: Boolean to annotate points with their indices (default: False)
        - annote_o: Boolean to annotate specific points with arrows (default: False)
        - ind_o: Indices of points for annotation with arrows (default: None)
        - ticks: Boolean to display ticks on axes (default: True)
        - size: Size of scatterplot markers (default: 20)
        - rotate: Angle of rotation for annotations (default: 0)
        """
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))

        Xs = self.GPApt  # Data for t-SNE
        y = self.y  # Target labels for coloring

        # Perform t-SNE embedding
        X_embedded = TSNE(n_components=2,
                        learning_rate='auto',
                        random_state=1,
                        perplexity=perplexity).fit_transform(Xs)

        # Color settings
        colors = ['blue', 'chocolate', 'green', 'deeppink', 'brown', 'gold', 'black']
        colors2 = ['lightskyblue', 'sandybrown', 'yellowgreen', 'deeppink', 'tomato', 'khaki', 'darkgrey']
        
        legends = []

        # Apply outlier detection methods if specified
        if method == 'lof':  # Local Outlier Factor
            model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto')
            y_pred = model.fit_predict(Xs)
            X_scores = model.negative_outlier_factor_
            radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
            ax.scatter(X_embedded[:, 0], X_embedded[:, 1], s=1000 * radius, edgecolors="b", facecolors="none", label="Outlier scores")
            
        elif method == 'if':  # Isolation Forest
            model = IsolationForest(random_state=0)
            model.fit(Xs)
            y_pred = model.predict(Xs)
            X_scores = model.score_samples(Xs)
            radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
            ax.scatter(X_embedded[:, 0], X_embedded[:, 1], s=1000 * radius, edgecolors="b", facecolors="none", label="Outlier scores")
            
        elif method == 'ocsvm':  # One-Class SVM
            model = OneClassSVM(gamma='auto')
            model.fit(Xs)
            y_pred = model.predict(Xs)
            X_scores = model.score_samples(Xs)
            radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
            ax.scatter(X_embedded[:, 0], X_embedded[:, 1], s=1000 * radius, edgecolors="b", facecolors="none", label="Outlier scores")

        elif method is None:
            pass  # No outlier detection or classifier
        
        else:
            # Apply custom model
            model = method
            y_pred = model.fit_predict(Xs)
            X_scores = model.negative_outlier_factor_
            radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
            ax.scatter(X_embedded[:, 0], X_embedded[:, 1], s=1000 * radius, edgecolors="b", facecolors="none", label="Outlier scores")

        # Plot decision boundaries if requested
        if decision_boundary:
            if not cv:
                self.classifier.fit(self.Xr, self.ys)
                y_predicted = self.classifier.predict(Xs)
                y_predicted = self.encoder2.transform(y_predicted)

            else:
                # Cross-validation for decision boundaries
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=72)
                ycv = np.copy(y)
                try:
                    # Catch warnings from StratifiedKFold
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)  # Ignore the specific warning
                        for train, test in cv.split(Xs, y):
                            mask = np.ones(len(y[train]), dtype=bool)
                            delet = np.where(self.y0[train] == 'lophocebus aterrimus')
                            mask[delet] = False
                            delet = np.where(self.y0[train] == self.deletg)
                            mask[delet] = False
                            self.classifier.fit(Xs[train][mask], self.y[train][mask])
                            y_predicted = self.classifier.predict(Xs[test])
                            ycv[test] = y_predicted

                        y_predicted = self.encoder2.transform(ycv)

                except UserWarning as e:
                    print(f"⚠️ Custom Warning: {str(e)} - 'Lophocebus aterrimus' has only 2 samples, so it will not be used for training but will be classified.")

            # Voronoi tessellation for decision boundary background
            colors3 = [colors2[self.encoder2.inverse_transform([i])[0]] for i in np.unique(y_predicted)]
            resolution = 100
            xd, yd = 0.05 * np.abs(np.min(X_embedded[:, 0]) - np.max(X_embedded[:, 0])), 0.05 * np.abs(np.min(X_embedded[:, 1]) - np.max(X_embedded[:, 1]))
            X2d_xmin, X2d_xmax = np.min(X_embedded[:, 0]) - xd, np.max(X_embedded[:, 0]) + xd
            X2d_ymin, X2d_ymax = np.min(X_embedded[:, 1]) - yd, np.max(X_embedded[:, 1]) + yd
            xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))
            
            background_model = KNeighborsClassifier(n_neighbors=1).fit(X_embedded, y_predicted)
            voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape((resolution, resolution))

            cmap = matplotlib.colors.ListedColormap(colors3)
            cset = ax.contourf(xx, yy, voronoiBackground, n_r, cmap=cmap)
            ax.contour(xx, yy, voronoiBackground, cset.levels, colors='black', linewidths=2)

        # Scatter plot of the embedded data points
        for i in np.unique(y):
            if self.eflag == 0:
                label = self.encoder.inverse_transform([i]).tolist()[0]
            else:
                label = self.encoder.inverse_transform(np.array(i).reshape(1, -1))[0][0]

            s = size
            marker = 'o'
            scatter = ax.scatter(X_embedded[:, 0][y == i], X_embedded[:, 1][y == i], c=colors[int(i)], edgecolors='black',
                                label=label, marker=marker, s=s)
            legends.append(scatter)

        # Convex hull for each class
        for i in np.unique(y):
            points = X_embedded[:, [0, 1]][np.array(y == i).ravel()]
            if points.shape[0] < 3:
                continue
            hull = ConvexHull(points)
            x_hull = np.append(points[hull.vertices, 0], points[hull.vertices, 0][0])
            y_hull = np.append(points[hull.vertices, 1], points[hull.vertices, 1][0])
            ax.fill(x_hull, y_hull, alpha=0.3, c=colors[int(i)])

        # Adjust ticks
        start, end = ax.get_xlim()
        step = (end - start) / 8
        ax.xaxis.set_ticks(np.arange(start + step, end, step))

        start, end = ax.get_ylim()
        step = (end - start) / 12
        ax.yaxis.set_ticks(np.arange(start + step, end, step))

        if ticks:
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(axis='both', which='major', labelsize=12)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

        # Add legend
        if dlegend:
            legend1 = ax.legend(handles=legends, loc="best", handlelength=1, framealpha=0.35)
            legend1.get_title().set_fontsize('6')
            ax.add_artist(legend1)

        # Annotate points with their index
        if annote:
            start, end = ax.get_xlim()
            stepx = (end - start) / 200
            start, end = ax.get_ylim()
            stepy = (end - start) / 200

            for i, txt in enumerate(self.ind):
                ax.annotate(str(txt), (X_embedded[i, 0] + stepx, X_embedded[i, 1] + stepy), size=4, c='black')

        # Annotate specific points with arrows
        if annote_o:
            start, end = ax.get_xlim()
            stepx = (end - start) / 6
            start, end = ax.get_ylim()
            stepy = (end - start) / 6

            xt = X_embedded[ind_o, 0] - stepx
            yt = X_embedded[ind_o, 1] + stepy

            if rotate != 0:
                rotate = rotate * (np.pi) / 180
                s = np.sin(rotate)
                c = np.cos(rotate)

                xt -= X_embedded[ind_o, 0]
                yt -= X_embedded[ind_o, 1]

                xnew = xt * c - yt * s
                ynew = xt * s + yt * c

                xt = xnew + X_embedded[ind_o, 0]
                yt = ynew + X_embedded[ind_o, 1]

            arrow = mpatches.FancyArrowPatch((xt, yt), (X_embedded[ind_o, 0], X_embedded[ind_o, 1]),
                                            color='red', edgecolor='black', mutation_scale=10)
            ax.add_patch(arrow)

        # Label axes
        if ticks:
            ax.set_xlabel('Dimension 1', fontsize=22)
            ax.set_ylabel('Dimension 2', fontsize=22)
        else:
            ax.set(xlabel=None)
            ax.set(ylabel=None)
