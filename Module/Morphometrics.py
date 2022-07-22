####################################################################################################################################
############################################             By; Nima Mohseni, Eran Elhaik       #######################################
##                                                                                                                                ##
##                                                                                                                                ##
###########################################Lund University-Faculty of Science-Department of Biology#################################
####################################################################################################################################

# A class to process the saved results of morphologika

# Importing modules

# Plots

# t-SNE
from sklearn.manifold import TSNE

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter

# PLotting dendrogram
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

#Convex hull
from scipy.spatial import ConvexHull

# Data manipulation modules
import pandas as pd
import numpy as np

# For cross validation
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder

# For scaling and label encoder
from sklearn.preprocessing import *

# Metrics for evaluation of performance
from sklearn.metrics import *

# Outlier detection
# Local Outlier Factor
from sklearn.neighbors import LocalOutlierFactor

# KNN classifier
from sklearn.neighbors import KNeighborsClassifier

from random import shuffle

class procpca():


    # initial parameters
    def __init__(self, dataframe, classifier = None):

        self.df = dataframe
        self.PCAt = []
        self.PCApt = []
        self.GPAt = []
        self.encoder = LabelEncoder()
        self.encoder.fit(self.df.iloc[:, 0])
        if classifier == None:
            self.classifier = KNeighborsClassifier(n_neighbors=2)
        else:
            self.classifier = classifier

    # reading the '.txt' file and storing it in PCA and PCAp
    
    def read(self, path):
        
        # PCA will store the samples data
        f = open(path, 'r')
        PCA = []
        flag = 0
        for line in f:
            if 'Between OTU tangent space distances' in line:
                break
            if 'PC Scores' in line:
                flag = 1
            if flag == 0:
                continue
            else:
                PCA.append(line)
        f.close()

        # PCAp stores the data of the principal components
        f = open(path, 'r')
        PCAp = []
        flag = 0
        for line in f:
            if 'PC Scores' in line:
                break
            if ',eigenvalue,percentage of total variance explained,cumulative variance explained' in line:
                flag = 1
            if flag == 0:
                continue
            else:
                PCAp.append(line)
        f.close()

        # GPAp stores the data of the procrustes analysis
        f = open(path, 'r')
        GPAp = []
        flag = 0
        for line in f:
            if 'Procrustes mean shape' in line:
                break
            if 'Procrustes output' in line:
                flag = 1
            if flag == 0:
                continue
            else:
                GPAp.append(line)
        f.close()
        
        PCA = PCA[3:]
        PCAp = PCAp[1:]
            
        # A bit of necessary text processing
        PCAt = []
        for i in PCA:
            j = i.replace(' ', '')
            j = i.replace('\n', '')
            PCAt.append(i.replace(' ', ''))
            
        PCApt = []
        for i in PCAp:
            j = i.replace(' ', '')
            j = i.replace('\n', '')
            PCApt.append(i.replace(' ', ''))

        GPApt2 = []
        for i in GPAp:
            j = i.replace(' ', '')
            j = i.replace('\n', '')
            GPApt2.append(i.replace(' ', '')) 

        GPApt2 = GPApt2[2:-1]
        GPApt = []
        for i in GPApt2:
            tmp = []
            for j , word in enumerate(i.split(',')):
                if j > 1:
                    try:
                        tmp.append(float(word))
                    except:
                        a = 0
            GPApt.append(tmp)
        GPApt = np.array(GPApt) 

        self.GPApt = GPApt

        # Varex is the explained variance
        varex = []

        for i in PCApt:
            
            for j , word in enumerate(i.split(',')):
                if j == 2: 
                 varex.append(float(word))

        varex = np.array(varex) 

        self.varex = varex
        self.PCAt = PCAt
        self.PCApt = PCApt
        

        # Getting the index of the samples from the initial data-set
        name = []
        rows = []
        flag0 = 0
        flag1 = 0

        for i in self.PCAt:
    
            if len(i) <= 1:
                continue
    
            for j , word in enumerate(i.split(',')):
        
                if j == 0:
                    try:
                        sn = int(word)
                        #print(word)
                    except:
                        a = 1   
                    continue
            
                elif j == 1:
                    #print(word)
                    name.append(word)
                    row = []    
                    continue
            
                else:
                    try:
                        row.append(float(word))
                    except:
                        a = 1

            rows.append(np.array(row))
            
        rows = np.array(rows)

        # Comparing the name of the samples with that of the refrence data-set to find the initial indices
        ind = []
        for k ,h in enumerate(name):
            for i , word in enumerate(self.df.iloc[:, 3]):
                if word == h:              
                    ind.append(i)
                    break

        ind = np.array(ind)
        
        # Preparing a data frame of indices and names
        data = pd.DataFrame()
        data['index'] = ind 
        data['label'] = name
        
        y = ind
        self.ind = ind
        self.y = np.array(self.df.iloc[y, 1])

        for i in range(rows.shape[1]):
            n = 'PC' + str(i+1)
            data[n] = rows[:, i]

        # Preparing a data frame with more information
        datag = pd.DataFrame()
        datag['index'] = ind
        datag['group'] = np.array(self.df.iloc[ind, 0]) #why has to be np.array??
        datag['label'] = name
        datag['encoded'] = self.y

        self.labels = name
        self.datag = datag

        # Adding the results of the procrustes analysis
        for i in range(GPApt.shape[1]):
            n = 'lndm' + str(int((i)/3)+1) + '-' + str(int((i)%3)+1)
            datag[n] = GPApt[:, i]
     
        self.removed = ''
        for i in range(len(self.df)):
            if i not in ind:
                if self.removed != '':
                    self.removed += ','
                self.removed += str(i)

        self.x = rows

        # Running the post process function automatically
        self.post_process()

        # Returning the results of the morphologika analysis in a convenient format
        return(data, datag, ind, name)
    
    # A function for preparing PCA plots
    # ind s are the # of principal components which should be ploted
    def PCAplotm(self, y, x, ind1, ind2, sav1=0, sav2='', sav3 = '.svg', annote = False, ax = None,
        dlegend = True, index_r = 0):
        
        colors = ['blue', 'chocolate', 'green', 'deeppink', 'brown', 'gold']
        color = []

        # Iterating through y
        for i in y:
            color.append(colors[i])
            
        if ax == None:
            fig, ax = plt.subplots(figsize=(7,6))

        # Creating a legend of the PCA plots
        legends = []
        # Iterating over groups
        for i in (np.unique(y)):
            scatter = ax.scatter(x[:, ind1-1][y==i], x[:, ind2-1][y==i], c=colors[i],
                label = self.encoder.inverse_transform([i]).tolist()[0], marker = 'o', s = 20, edgecolors='black')
            legends.append(scatter)

        # If the index of the removed samples should be included as well
        if index_r == 1:
            legends.append(mpatches.Patch(color='none',
                            label='removed samples: '+self.removed[:6]+"\n"+self.removed[6:]))
        
        # The axis labels mentioning the number of the principal component and the explained variance in %
        ax.set_xlabel('PC'+str(ind1)+' (%.2f'%self.varex[ind1-1]+'%)')
        ax.set_ylabel('PC'+str(ind2)+' (%.2f'%self.varex[ind2-1]+'%)')
        plt.setp(ax.get_xticklabels(), fontsize=4)
        plt.setp(ax.get_yticklabels(), fontsize=4)
        # If the plot needs a title
        # ax.set_title(' ')

        # If the index of the samples should be annoted according to their order in the initial data-set
        if annote == True:
            e1 = (np.max(x[:, ind1-1]) - np.min(x[:, ind1-1]))/200
            e2 = (np.max(x[:, ind2-1]) - np.min(x[:, ind2-1]))/150
            for i, txt in enumerate(self.ind):
                ax.annotate(str(txt), (x[i, ind1-1]+e1, x[i, ind2-1]+e2), size = 4)

        # Creating the convex hull        
        ind1 = ind1 - 1
        ind2 = ind2 - 1
        # Iterating over groups
        for i in (np.unique(y)):
            points = x[:,[ind1,ind2]][np.array(y==i).ravel()]
            # There should be more than 3 samples in each group
            if points.shape[0] < 3:
                #print(points.shape)
                continue
            hull = ConvexHull(points)
            x_hull = np.append(points[hull.vertices,0],
                       points[hull.vertices,0][0])
            y_hull = np.append(points[hull.vertices,1],
                       points[hull.vertices,1][0])
            # The hull itself
            ax.fill(x_hull, y_hull, alpha=0.3, c=colors[i])

        # Adjusting x and y ticks
        start, end = ax.get_xlim()
        step = (end-start)/12
        ax.xaxis.set_ticks(np.arange(start+step, end, step))
        start, end = ax.get_ylim()
        step = (end-start)/12
        ax.yaxis.set_ticks(np.arange(start+step, end, step))

        # The number of decimals to be shown on each axis
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(axis='both', which='major', labelsize=10)

        # If a legend should be included in the plot
        if dlegend == True:
            # Produce a legend with the unique colors from the scatter
            legend1 = ax.legend(handles = legends,
                loc="best", handlelength=1, framealpha=0.35)

            # The legend font size
            matplotlib.rcParams['legend.fontsize'] = 10
            ax.add_artist(legend1)
        
        # If the plot needs to be saved
        if sav1 == 1:
            plt.savefig('Monkeys_'+ sav2 + '_PCA_plot' + str(ind1+1) + str(ind2+1) + sav3, transparent=True)
            plt.show()

    # A function to create the first 3 PCA plots together
    def PCAplot(self,sav1, sav2, sav3 = '.pdf', annote = False, ax=None, index_r = 0):

        self.PCAplotm(self.y,self.x,1,2,sav1, sav2, sav3, annote,ax, False, index_r)
        self.PCAplotm(self.y,self.x,1,3,sav1, sav2, sav3, annote,ax, False, index_r)
        self.PCAplotm(self.y,self.x,2,3,sav1, sav2, sav3, annote,ax, False, index_r)

    # Create linkage matrix for the dendrogram to be plotted
    def dendrogram_GPA(self, model, **kwargs):
        
        # The counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    # leaf node
                    current_count += 1  
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs, color_threshold=0.35)

    # The dendrogram function
    def plot_dendrogram(self):

        # setting distance_threshold=0 ensures we compute the full tree.
        # Using agglomerative clustering
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
        X = self.GPApt
        model = model.fit(X)
        fig = plt.figure(figsize=(10, 6))
        plt.title("Hierarchical Clustering")
        # plot the top three levels of the dendrogram
        self.dendrogram_GPA(model, truncate_mode="level", p=10, labels= self.labels)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        #plt.savefig('Dendro.svg', transparent=True, bbox_inches='tight')
        #plt.show()

    # A function for post processing the data
    def post_process(self, deletg = None):

        X = np.array(self.datag.iloc[:, 4:])
        y = np.array(self.datag.iloc[:, 3])
        self.y0 = np.array(self.datag.iloc[:, 1])

        mask = np.ones(len(y), dtype = bool)
        delet = np.where(self.y0 == 'lophocebus aterrimus')
        mask[delet] = False
        delet = np.where(self.y0 == deletg)
        mask[delet] = False
        self.Xr = X[mask]
        self.ys = y[mask]
        self.deletg = deletg

        self.encoder2 = LabelEncoder()
        self.encoder2.fit(self.ys)
        self.y2e = self.encoder2.transform(self.ys)
        #y02 = y0[mask]

    # A multi-purpose function for performing t-SNE and preparing plots
    def plot_tsne(self, n_r=4, ax= None, localo = False, decision_boundary = False, cv = False, dlegend = False, index_r=0, perplexity=10, n_neighbors=5):
        
        if ax == None:
            fig, ax = plt.subplots(figsize=(7,6))

        Xs = self.GPApt
        y = self.y

        # Embedding the procrustes results into 2 dimensions using t-SNE
        X_embedded = TSNE(n_components=2, learning_rate='auto', random_state=1, perplexity=perplexity).fit_transform(Xs)

        colors = ['blue', 'chocolate', 'green', 'deeppink', 'brown', 'gold']
        colors2 = ['lightskyblue', 'sandybrown', 'yellowgreen', 'deeppink', 'tomato', 'khaki']

        legends = []

        if localo == True:
            # Local Outlier Factor
            model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)

            y_pred = model.fit_predict(Xs)
            X_scores = model.negative_outlier_factor_

            # plot circles with radius proportional to the outlier scores
            radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
            ax.scatter(X_embedded[:, 0], X_embedded[:, 1], s=1000 * radius,
                edgecolors="b", facecolors="none", label="Outlier scores",)

        # If the decision boundaries of the classifier should be plotted
        if decision_boundary == True:
            
            # If there is no need for cross-validation
            if cv == False:
                self.classifier.fit(self.Xr, self.ys)
                y_predicted = self.classifier.predict(Xs)
                y_predicted = self.encoder2.transform(y_predicted)

            # If cross-validation is needed
            if cv == True:
                # A 5-fold cross validation
                cv = StratifiedKFold(n_splits=5, shuffle=True) #, random_state=1
                ycv = np.copy(y)
                for train, test in cv.split(Xs, y):

                    mask = np.ones(len(y[train]), dtype = bool)
                    delet = np.where(self.y0[train] == 'lophocebus aterrimus')
                    mask[delet] = False
                    delet = np.where(self.y0[train] == self.deletg)
                    mask[delet] = False

                    self.classifier.fit(Xs[train][mask], self.y[train][mask])
                    y_predicted = self.classifier.predict(Xs[test])
                    ycv[test] = y_predicted
                    
                y_predicted = ycv
                y_predicted = self.encoder2.transform(y_predicted)

            # Colours for voroni background
            colors3 = []
            for i in (np.unique(y_predicted)):
                colors3.append(colors2[self.encoder2.inverse_transform([i])[0]])                

            # create meshgrid
            resolution = 100 # 100x100 background pixels
            xd = 0.05*np.abs(np.min(X_embedded[:,0])-np.max(X_embedded[:,0]))
            yd = 0.05*np.abs(np.min(X_embedded[:,1])-np.max(X_embedded[:,1]))
            X2d_xmin, X2d_xmax = np.min(X_embedded[:,0])-xd, np.max(X_embedded[:,0])+xd
            X2d_ymin, X2d_ymax = np.min(X_embedded[:,1])-yd, np.max(X_embedded[:,1])+yd
            xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))
            # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
            background_model = KNeighborsClassifier(n_neighbors=1).fit(X_embedded, y_predicted) 
            voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
            voronoiBackground = voronoiBackground.reshape((resolution, resolution))
            # Contour plot
            cset = ax.contourf(xx, yy, voronoiBackground, n_r, colors = colors3)
            ax.contour(xx, yy, voronoiBackground, cset.levels, colors='black', linewidths=2)

        # Iterating over groups to creat the scatterplots
        for i in (np.unique(y)):
            s = 20
            marker = 'o'

            scatter = ax.scatter(X_embedded[:, 0][y==i], X_embedded[:, 1][y==i], c=colors[int(i)], edgecolors='black',
            label = self.encoder.inverse_transform([i]).tolist()[0], marker = marker, s = s)
            legends.append(scatter)

        # Iterating over groups for the convex hull
        for i in (np.unique(y)):
            points = X_embedded[:,[0, 1]][np.array(y==i).ravel()]
            if points.shape[0] < 3:
                #print(points.shape)
                continue

            hull = ConvexHull(points)
            x_hull = np.append(points[hull.vertices,0],
                        points[hull.vertices,0][0])
            y_hull = np.append(points[hull.vertices,1],
                        points[hull.vertices,1][0])
            ax.fill(x_hull, y_hull, alpha=0.3, c=colors[int(i)])

            start, end = ax.get_xlim()
            step = (end-start)/6
            ax.xaxis.set_ticks(np.arange(start+step, end, step))
            start, end = ax.get_ylim()
            step = (end-start)/12
            ax.yaxis.set_ticks(np.arange(start+step, end, step))

            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(axis='both', which='major', labelsize=10)



        if index_r == 1:
            legends.append(mpatches.Patch(color='none',
                            label='removed samples: '+self.removed[:6]+"\n"+self.removed[6:]))
        if dlegend == True:    
            # Produce a legend with the unique colors from the scatter
            legend1 = ax.legend(handles = legends,
                loc="best", handlelength=1, framealpha=0.35)
            legend1.get_title().set_fontsize('6')
            ax.add_artist(legend1)
                                
        ax.set_xlabel('D 1')
        ax.set_ylabel('D 2')
        #ax.set_title(' ')
        #plt.savefig(' ')         
        #plt.show() 
