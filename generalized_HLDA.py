# Created by Myong In Oh on February 1, 2021
# Updated by Myong In Oh on March 2, 2021
# Updated by Gabe da Hora on Feb 8, 2022
# This code creates a collective variable (defined as a linear combination of input descriptors)
# that maximizes the separation of two classes (or states) using harmonic linear discriminant analysis.
# (J. Chem. Phys. 149, 194113 (2018); https://doi.org/10.1063/1.5053566)

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

#################### USER INPUT ####################

num_class = 2
num_descriptor = 8
num_eigenvector = 1

####################################################

##### Step 0. Read in the dataset

feature_dict = {i:label for i,label in zip(range(num_descriptor), ('d1','d2','d3','d4','d5','d6','d7','d8',))}  # change the list of descriptors in the feature dictionary if necessary

df = pd.read_csv('mj25_native_CAs_without_d0.csv')                                            # exclude the headers (first row) in the data file
df.columns = [l for i,l in sorted(feature_dict.items())] + ['class']    # add column labels using the values (l) in the dictionary and 'class'
df.dropna(how='all', inplace=True)                                      # drop the empty line at file-end

df.info()
print(df.head(5))
print(df.tail(5))

corr = df.corr()
print("Correlation: \n", corr)
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.savefig("1_correlation.png", dpi=150)
#plt.show()
#plt.close()

X = df.iloc[:,:num_descriptor].values
y = df['class'].values

enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1

print(y)

##### Step 1. Compute the d-dimensional mean vectors
### Here, we calculate #num_class column vectors, each of which contains #num_descriptor elements (means)

np.set_printoptions(precision=4)

mean_vectors = []
for cl in range(1,num_class+1):
    mean_vectors.append(np.mean(X[y==cl], axis=0))                  # X[y==cl] : Boolean indexing/slicing - select lists (rows) in X that satisfy the condition y==cl 
    print(f'Mean Vector class {cl}: {mean_vectors[cl-1]}\n')

##### Step 2. Compute the scatter matrices
### 2-1. Within-class scatter matrix SW

S_W = np.zeros((num_descriptor,num_descriptor))
S_W_int = np.zeros((num_descriptor,num_descriptor))
for cl,mv in zip(range(1,num_class+1), mean_vectors):
    class_sc_mat = np.zeros((num_descriptor,num_descriptor))
    for row in X[y==cl]:
        row, mv = row.reshape(num_descriptor,1), mv.reshape(num_descriptor,1)   # make column vectors
        class_sc_mat += (row-mv).dot((row-mv).T)
    S_W_int += np.linalg.inv(class_sc_mat)                                      # sum class scatter matrices
S_W = np.linalg.inv(S_W_int)

# Alternative way to compute SW (for 3 classes)
#Cov1 = np.cov(np.transpose(X[y==1]))
#Cov2 = np.cov(np.transpose(X[y==2]))
#Cov3 = np.cov(np.transpose(X[y==3]))
#Cov_inv = np.linalg.inv(Cov1) + np.linalg.inv(Cov2) + np.linalg.inv(Cov3)
#S_W = np.linalg.inv(Cov_inv)

print('within-class Scatter Matrix:\n', S_W)

### 2-2. Between-class scatter matrix SB

overall_mean = np.mean(X, axis=0)                               # overall mean of each descriptor # “axis=0” represents rows and “axis=1” represents columns
S_B = np.zeros((num_descriptor,num_descriptor))
for i,mean_vec in enumerate(mean_vectors):                      # enumerate(mean_vectors): generates tuples in the form of (index, 'element') - here index starts from 0
    n = X[y==i+1,:].shape[0]                                    # use X[y==i+1] since i (index) starts from 0 while y (class) starts from 1 # X.shape[0]: returns m when X is an m x n matrix.
    mean_vec = mean_vec.reshape(num_descriptor,1)               # make column vector
    overall_mean = overall_mean.reshape(num_descriptor,1)       # make column vector
    S_B += n*(mean_vec-overall_mean).dot((mean_vec-overall_mean).T)

# Alternative way to compute SB (for 3 classes)
#mutot = np.zeros(num_descriptor)
#mutot = (mean_vectors[0] + mean_vectors[1] + mean_vectors[2])/3.0
#S_B = np.outer((mean_vectors[0]-mutot),(mean_vectors[0]-mutot)) + np.outer((mean_vectors[1]-mutot),(mean_vectors[1]-mutot)) + np.outer((mean_vectors[2]-mutot),(mean_vectors[2]-mutot))

print('between-class Scatter Matrix:\n', S_B)

##### Step 3. Solve the generalized eigenvalue problem for the matrix SW^-1.SB

eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(num_descriptor,1)         # [:,i] = all rows and column i
    print(f'\nEigenvector {i+1}: \n{eigvec_sc.real}')
    print(f'Eigenvalue {i+1}: {eig_vals[i].real:.2e}')

### Check the eigenvector-eigenvalue calculation

for i in range(len(eig_vals)):
    eigv = eig_vecs[:,i].reshape(num_descriptor,1)
    np.testing.assert_array_almost_equal(np.linalg.inv(S_W).dot(S_B).dot(eigv), eig_vals[i] * eigv, decimal=6, err_msg='', verbose=True)
print('ok')

##### Step 4. Select linear discriminants for the new feature subspace
### 4-1. Sort the eigenvectors by decreasing eigenvalues

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]    # make a list of (eigenvalue, eigenvector) tuples
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)                     # sort the (eigenvalue, eigenvector) tuples from high to low

print('Eigenvalues in decreasing order:\n')     # visually confirm that the list is correctly sorted by decreasing eigenvalues
for i in eig_pairs:
    print(i[0])

print('Variance explained:\n')
eigv_sum = sum(eig_vals)
for i,j in enumerate(eig_pairs):
    print(f'eigenvalue {i+1}: {(j[0]/eigv_sum).real:.2%}')

### 4-2. Choosing #num_eigenvectors eigenvectors with the largest eigenvalues

W = np.concatenate([eig_pairs[i][1].reshape(num_descriptor,1) for i in range(num_eigenvector)], axis=1)
print('Matrix W:\n', W.real)

##### Step 5. Transform the samples onto the new subspace

X_lda = X.dot(W)
#print(X_lda.real[y == 1])
#print(X_lda.real[y == 2])
#print(X_lda)
#print(X_lda.shape)

with open('HLDA1.txt', 'w') as myfile1:
    for c1 in X_lda.real[y == 1]:
        myfile1.write(f'{c1}\n')

with open('HLDA2.txt', 'w') as myfile2:
    for c2 in X_lda.real[y == 2]:
        myfile2.write(f'{c2}\n')

with open('matrixW.txt', 'w') as myfile3:
    myfile3.write(f'{W.real}')

##### Step 6. Draw a scatter plot with vector W and decision boundary H

if num_class == 2 and num_descriptor == 2:
    ### vector W
    # Option 1. Using centroid
    centroid = (mean_vectors[0] + mean_vectors[1]) / 2
    c1, c2 = centroid[0], centroid[1]
    w1, w2 = W[0][0], W[1][0]
    # Option 2. Using centre of mass
    #mass1 = X[y==1,:].shape[0]
    #mass2 = X[y==2,:].shape[0]
    #com = ((mass1 * mean_vectors[0]) + (mass2 * mean_vectors[1])) / (mass1 + mass2)
    #c1, c2 = com[0], com[1]
    #w1, w2 = W[0][0], W[1][0]

    ### estimated decision boundary (assuming that all class covariances are equal)
    n1 = X[y==1,:].shape[0]
    n2 = X[y==2,:].shape[0]
    n = n1 + n2

    pi_hat_1 = n1 / n   # class priors
    pi_hat_2 = n2 / n
    
    mu_hat_1 = 1 / n1 * np.sum(X[y==1,:], axis=0) # class means (or simply use 'mean_vectors[0]')
    mu_hat_2 = 1 / n2 * np.sum(X[y==2,:], axis=0)

    cov_hat_1 = 1 / (n1-1) * np.matmul((X[y==1,:]-mu_hat_1).T, (X[y==1,:]-mu_hat_1))
    cov_hat_2 = 1 / (n2-1) * np.matmul((X[y==2,:]-mu_hat_2).T, (X[y==2,:]-mu_hat_2))

    cov_hat = 2 * np.linalg.inv((np.linalg.inv(cov_hat_1) + np.linalg.inv(cov_hat_2)))      # harmonic mean of the covariances
    cov_inv = np.linalg.inv(cov_hat)

    #cov_hat = (cov_hat_1 + cov_hat_2) / 2      # arithmetic mean of the covariances
    #cov_inv = np.linalg.inv(cov_hat)

    ### slope
    slope_vec = np.matmul(cov_inv, (mu_hat_1-mu_hat_2))
    slope = -slope_vec[0] / slope_vec[1]
    print(slope_vec)

    ### intercept
    intercept_partial = np.log(pi_hat_2)-np.log(pi_hat_1) + 0.5 * np.matmul(np.matmul(mu_hat_1.T, cov_inv), mu_hat_1)-0.5 * np.matmul(np.matmul(mu_hat_2.T, cov_inv), mu_hat_2)
    intercept = intercept_partial / slope_vec[1]

    plt.figure(figsize=(7,7))
    scatter = plt.scatter(X[:,0], X[:,1], alpha=0.2, s=30, c=y, cmap="bwr")
    handles, labels = scatter.legend_elements()
    labels = ["y = 1", "y = 2"]
    plt.legend(handles, labels)
    plt.axline((c1, c2), (c1+w1, c2+w2), color='grey', linestyle="dashed")
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '-', color='green')
    plt.xlabel(feature_dict[0])
    plt.ylabel(feature_dict[1])
    #plt.xlim(2, 9);
    #plt.ylim(2, 9);
    plt.savefig("2_scatter_with_vectorW_decision_boundary.png", dpi=150)
    plt.show()

##### Alternative way for step 6

#if num_class == 2 and num_descriptor == 2:
#    ### vector W
#    centroid = (mean_vectors[0] + mean_vectors[1]) / 2
#    c1, c2 = centroid[0], centroid[1]   # centroid
#    w1, w2 = W[0][0], W[1][0]           # vector W
#    h1, h2 = -w2, w1                    # decision boundary H

#    scatter = plt.scatter(X[:,0], X[:,1], alpha=0.2, s=30, c=y, cmap="bwr")
#    handles, labels = scatter.legend_elements()
#    labels = ["y = 1", "y = 2"]
#    plt.legend(handles, labels)
#    plt.axline((c1, c2), (c1+w1, c2+w2), color='grey', linestyle="dashed")
#    plt.axline((c1, c2), (c1+h1, c2+h2), color='green', linestyle="dashed")
#    plt.xlabel(feature_dict[0])
#    plt.ylabel(feature_dict[1])
#    plt.savefig("2_vectorW_hyperplane.png", dpi=150)
#    plt.show()
