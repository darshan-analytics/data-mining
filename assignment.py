import numpy as np                      #importing library for for numpy as np
import pandas as pd                     #importing pandas library as pd
import matplotlib.pyplot as plt         #importing matplot library for plotting graph
from sklearn.decomposition import PCA   #importing sklearn.decomposition library for PCA analysis
from sklearn.preprocessing import StandardScaler #importing sklearn library for preprocessing of data
import statsmodels.api as sm                #importing stats model for processing
from scipy.fftpack import dct               #import DCT function
from sklearn.decomposition import FastICA   #importing ICA function

#Finding distribution in dataset 1############################################################################
df1 = pd.read_csv("dist1_500_1.txt",sep=" ",header=None)    #reading dataset1 part 1
df2 = pd.read_csv("dist1_500_2.txt",sep=" ",header=None)
dt1 = pd.DataFrame(df1)                     #taking data1 part1 as dataframe
dt2 = pd.DataFrame(df2)
dt_1 = dt1.dropna(how='all')                #removing the blank lines form the data
dt_2 = dt2.dropna(how='all')
filenames = [dt_1, dt_2]                      #concatinating of files
data = pd.concat(filenames)
print("\nDATASET ONE AFTER CONCATINATING")
print(data)
data1 = data.sample(n = 10)               #Randomly taking data fromm the dataset
print("\n \n")
print("Random ten samples")
print(data1)
plt.boxplot(data1)                      #to show boxplot
plt.title("BOXPLOT DIAGRAM FOR DATASET 1")
plt.show()
sm.qqplot(data1, line='s')
plt.title("Q-Q Plot for Dataset one")
plt.show()
plt.hist(data1)
plt.title("Histogram of dataset one")
plt.show()
data1.values.tolist()
e = pd.DataFrame(data1)
f = e.values.tolist()                       #converitng the dataset into the list
temp= []
for small_list in f:                           #combining the 10 values into one to get an 10 samples which contain 100
    temp+=small_list
datar1=temp
plt.hist(datar1)
plt.title("Histogram of dataset one with binning")
plt.show()
con = data1.values                       #convert dataframe values into con
a1 = con[0]
b1 = con[1]
c1 = con[2]
d1 = con[3]
e1 = con[4]
f1 = con[5]
g1 = con[6]
h1 = con[7]
i1 = con[8]
j1 = con[9]
con1 = [*a1,*b1,*c1,*d1,*e1,*f1,*g1,*h1,*i1,*j1]    #zipping the elements into the list
print("\n\nLength of random data for dataset one is")
print(len(con1))
d = {}
for num in con1:                            #to create a dictionary of key(value_in_dataset) with value(freq_of_value)
    d[num] = d.get(num, 0) + 1

print("\nCreating dictionary with pair of value and its freq in dataset 1")
print(d)
f = []
for key,value in d.items():
    f.append((key,value))
print("\nConverting dictionary to list")
print(f)
f.sort(reverse=True)
print("\nPrinting sorted list")
print(f)
x, y = zip(*f)                          #converting list into the tuples
plt.plot(x,y)
plt.title("Frequency plot for DATASET ONE ")
plt.show()
#####################Finding Distribution in DATASET 2#################################################
df21 = pd.read_csv("dist2_500_1.txt",sep=" ",header=None)           #reading dataset2
df22 = pd.read_csv("dist2_500_2.txt",sep=" ", header=None)
dt21 = pd.DataFrame(df21)
dt22 = pd.DataFrame(df22)
dt_21 = dt21.dropna(how='all')
dt_22 = dt22.dropna(how='all')
filenames2 = [dt_21, dt_22]
data21 = pd.concat(filenames2)
print("\n \n")
print("Dataset2 two is here")
print(data21)
data2 = data21.sample(n = 10)
print("Ten samples for dataset two")    #taking ten samples from dataset 2
print(data2)
plt.boxplot(data2)
plt.title("BOXPLOT DIAGRAM FOR DATASET 2")
plt.show()
plt.hist(data2)
plt.title("histogram Plot for Dataset two")
plt.show()
sm.qqplot(data2, line='s')
plt.title("Q-Q Plot for Dataset two")
plt.show()
data2.values.tolist()
g = pd.DataFrame(data2)
h = g.values.tolist()               #convert to list
temp= []
for small_list in h:
    temp+=small_list
datar2=temp
plt.hist(datar2)
plt.title("Histogram of dataset two with binning")
plt.show()
con2 = data2.values
a2 = con2[0]
b2 = con2[1]
c2 = con2[2]
d2 = con2[3]
e2 = con2[4]
f2 = con2[5]
g2 = con2[6]
h2 = con2[7]
i2 = con2[8]
j2 = con2[9]
con2 = [*a2,*b2,*c2,*d2,*e2,*f2,*g2,*h2,*i2,*j2]            #zipping the elements into the list
print("\n \nLength of sample data for dataset two")
print(len(con2))
d1 = {}
for num2 in con2:
    d1[num2] = d1.get(num2, 0) + 1
print("\nCreating dictionary with pair of value and its freq in dataset two")
print(d1)
f2 = []
for key,value in d1.items():
    f2.append((key,value))
print("\nConverting dictionary to list")
print(f2)
f2.sort(reverse=True)
print("\n Printing sorted list")
print(f2)
x, y = zip(*f2)
plt.plot(x,y)
plt.title("Frequency plot for DATASET TWO ")
plt.show()

#PCA starts here#######################################################################################
print(1000 * '-')
print("PCA ANALYSIS FOR DATASET 1 STARTS HERE")
X_std1 = StandardScaler().fit_transform(data)            #for pre-processing of data
print("\n \n")
print("NumPy covariance matrix: \n%s" %np.cov(X_std1.T))
cov1 = np.cov(X_std1.T)                               #get covariance of data

eigan_values1, eigan_vectors1 = np.linalg.eig(cov1)

print('Eigenvectors For Dataset One\n%s' %eigan_vectors1)
print('\nEigenvalues For Dataset one\n%s' %eigan_values1)

cor11 = np.corrcoef(X_std1.T)                 #get correaltion between data

eigan_values1, eigan_vectors1 = np.linalg.eig(cor11)

print('Eigenvectors For Dataset One\n%s' %eigan_vectors1)
print('\nEigenvalues For Dataset one\n%s' %eigan_values1)

u,s,v = np.linalg.svd(X_std1.T)          #doing SVD on dataset for singular value decomposition
print("\n Singular directional data for dataset one")
print(u)
for ev in eigan_vectors1:                             #to determine data is single directional
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('\nEverything ok!')
                                                # Make a list of (eigenvalue, eigenvector) tuples
eigan_pairs1 = [(np.abs(eigan_values1[i]), eigan_vectors1[:,i]) for i in range(len(eigan_values1))]

                                                        # Sort the (eigenvalue, eigenvector) tuples from high to low
eigan_pairs1.sort()
eigan_pairs1.reverse()


                                                        # Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('\nEigenvalues in descending order:')
for i in eigan_pairs1:
    print(i[0])

tot1 = sum(eigan_values1)
var_exp1 = [(i / tot1)*100 for i in sorted(eigan_values1, reverse=True)]
cum_var_exp1 = np.cumsum(var_exp1)
print("\nvariance by each component \n", var_exp1)       #finding the variance of data
print(1000 * '_')
print("\nsum of contribution",cum_var_exp1)                 #finding the total contribution for data
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(100), var_exp1, alpha=0.5, align='center',
            label='individual explained variance')



    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title("Principal components vs Variance ratio for dataset one")
    plt.tight_layout()
    plt.show()


# Fit the PCA and transform the data
pca = PCA(n_components=53)              #n_components = 53 to gain 81.70% accuracy for the data
reduced1 = pca.fit_transform(X_std1)      #fitting the standard data
print("\nSIZE of DATASET ONE after PCA feature extraction")
print(reduced1.shape)
print("\nReduced dataset 1 after PCA")
print(reduced1)
plt.figure(figsize=(12,8))
plt.title('PCA Components for DATASET ONE')
j= 1
for i in range(51):
    plt.scatter(reduced1[:, i], reduced1[:, j]) #plot for PCA
    i += 1
    j += 1

plt.scatter(reduced1[:,52],reduced1[:,0])
plt.show()
#############################################################################################################
##PCA FOR DATASET 2###########################################################################################
print(1000 * '-')
print("PCA FOR DATASET 2 STARTS HERE")
X_std2 = StandardScaler().fit_transform(data21)            #for pre-processing of data
print("\n \n")
print("NumPy covariance matrix: \n%s" %np.cov(X_std2.T))
cov2 = np.cov(X_std2.T)                               #get covariance of data

eigan_values2, eigan_vectors2 = np.linalg.eig(cov2)

print('Eigenvectors for dataset 2 \n%s' %eigan_vectors2)
print('\nEigenvalues for dataset 2 \n%s' %eigan_values2)

cor21 = np.corrcoef(X_std2.T)                 #get correaltion between data

eigan_values2, eigan_vectors2 = np.linalg.eig(cor21)

print('Eigenvectors for dataset 2 \n%s' %eigan_vectors2)
print('\nEigenvalues dataset 2\n%s' %eigan_values2)

u,s,v = np.linalg.svd(X_std2.T)
print("\n Singular directional data for dataset two")
print(u)
for ev in eigan_vectors2:                             #to determine data is single directional
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('\nEverything ok!')
                                                # Make a list of (eigenvalue, eigenvector) tuples
eigan_pairs2 = [(np.abs(eigan_values2[i]), eigan_vectors2[:,i]) for i in range(len(eigan_values2))]

                                                        # Sort the (eigenvalue, eigenvector) tuples from high to low
eigan_pairs2.sort()
eigan_pairs2.reverse()


                                                        # Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('\nEigenvalues in descending order:')
for i in eigan_pairs2:
    print(i[0])

tot2 = sum(eigan_values2)
var_exp2 = [(i / tot2)*100 for i in sorted(eigan_values2, reverse=True)]
cum_var_exp2 = np.cumsum(var_exp2)
print("\nvariance by each component \n", var_exp2)       #finding the variance of data
print(1000 * '_')
print("\nsum of contribution",cum_var_exp2)                 #finding the total contribution for data
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(100), var_exp2, alpha=0.5, align='center',
            label='individual explained variance')



    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components for dataset2')
    plt.title("Principal components vs Variance ratio for dataset two")
    plt.tight_layout()
    plt.show()

# Fit the PCA and transform the data
pca = PCA(n_components=72)  # n_components = 72 to gain 91% accuracy for the data
reduced2 = pca.fit_transform(X_std2)  # fitting the standard data
print("\nSIZE of DATASET TWO after PCA feature extraction")
print(reduced2.shape)
print("\nDATASET TWO after PCA feature extraction")
print(reduced2)
plt.figure(figsize=(12, 8))
plt.title('PCA Components for dataset TWO')
j = 1
for i in range(70):
   plt.scatter(reduced2[:, i], reduced2[:, j])  # plot for PCA on dataset 2
   i += 1
   j += 1

plt.scatter(reduced2[:, 71], reduced2[:, 0])
plt.show()
#DCT starts here############################################################################################################################
print(1000* '_')


def dct2(X_std1):
    return dct(dct(X_std1.T, norm='ortho').T, norm='ortho')  #applying orthogonal function

t1 = dct2(X_std1)                 #applying dct filters

print("data one after DCT extraction\n\n")
print(t1)
j= 1
for i in range(98):
    plt.scatter(t1[:,[i]], t1[:, [j]])
    i += 1
    j +=1

plt.scatter(t1[:, 99], t1[:, 0])
plt.title("DCT of DATASET ONE")
plt.show()
print("Dimension of Data one after applying DCT")
print(t1.shape)
##################################################################################################
print(1000* '_')


def dct3(X_std2):
    return dct(dct(X_std2.T, norm='ortho').T, norm='ortho')  #applying orthogonal function

t2 = dct3(X_std2)                 #applying dct filters

print("data two after DCT extraction\n\n")
print(t2)
j= 1
for i in range(98):
    plt.scatter(t2[:,[i]], t2[:, [j]])
    i += 1
    j +=1

plt.scatter(t2[:, 99], t2[:, 0])
plt.title("DCT on DATASET TWO")
plt.show()
print("Dimension of Data two after applying DCT")
print(t2.shape)
################################
print(1000*'-')
# load packages
ic1 = X_std1
ica = FastICA(n_components= 53)
ica.fit(X_std1)
ic11 = ica.fit_transform(ic1)   #applying ICA on standard data
print("Size of dataset one after reduced dimension with ICA is\n")

print(ic11.shape)               #checking size of the ica at dataset 1
# show image to screen
j= 1
for i in range(51):
    plt.scatter(ic11[:, i], ic11[:, j]) #plot for PCA
    i += 1
    j += 1
plt.scatter(ic11[:, 52], ic11[:, 0])

plt.title("ICA FOR DATASET ONE")
plt.ylabel('Co-ordinates at Y-axis ')
plt.xlabel('Co-ordinates at X-axis')
plt.show()
####################################################################################################################################
print(1000*'-')
ic2 = X_std2
ica2 = FastICA(n_components= 72)
ica2.fit(X_std2)
ic22 = ica2.fit_transform(ic2)   #applying ICA on standard data
print("Size of dataset two after reduced dimension with ICA is\n")
print(ic22.shape)               #checking size of the ica at dataset 1
# show image to screen
j= 1
for i in range(70):
    plt.scatter(ic22[:, i], ic22[:, j]) #plot for PCA
    i += 1
    j += 1
plt.scatter(ic22[:, 71], ic22[:, 0])

plt.title("ICA on dataset two")
plt.ylabel('Co-ordinates at Y-axis ')
plt.xlabel('Co-ordinates at X-axis')
plt.show()
#DCT-1 Starts here######################################

print("\nDCT -1 On DATASET One")
# Program to print matrix in Zig-zag pattern

# Create an empty list
print(1000* '-')
print("DATA AFTER ZIG-ZAG Traversing through DATA one\n")
x = data.values.tolist()    #converting data to list
#print(x[0])
#print(\n x[1])

matrix = x                              #creating a matrix
rows = 1000
columns = 100

solution = [[] for i in range(rows + columns - 1)]          #traversing in zizag way of matrix

for i in range(rows):
    for j in range(columns):
        sum = i + j
        if (sum % 2 == 0):

            # add at beginning
            solution[sum].insert(0, matrix[i][j])
        else:

            # add at end of the list
            solution[sum].append(matrix[i][j])

        # print the solution as it as
for i in solution:
    for j in i:
        print(j, end=" ")


import math

print("\n")


                       #converting solution matrix to an array

dct_1=np.zeros([1100])
n=100
vari=0
sum1=0
for i in solution:
    vari+=1
    varj=0
    sum1= 0
    for j in i:
        varj+=1
        x=j
        sum1 = sum1 + (x *(math.cos((((2*varj)+1) * vari * math.pi) / (2*n)) ))         #applying DCT formula
    #print(math.sqrt(2/n) * summ)
    if(vari==0):
        dct_1[varj] = math.sqrt(1/n) * sum1                   #applying DCT formula for U0
    else:
        dct_1[vari]= math.sqrt(2/n) * sum1                 #applying DCT for Uk where k = 1,2,3,...,k-1


print("DATA AFTER APPLYING DCT-I on DATASET 1\n")
print(dct_1)
print(1000*'-')
vari=0
for i in dct_1:
    vari+=1
    if(i==min(dct_1)):
        threshold1=vari                                #finding the threshold for dataset

print("Threshold value is ")
print(threshold1)
kp1 = 0
total1 = 1099
kp1 =((total1 - threshold1) / total1 )* 100
print("\n% of DATA after applying DCT-I for feature extraction on dataset one")
print(kp1)
plt.plot(dct_1,'ro')                                        #plotting the data
plt.title("DCT on DATASET one prior truncating to threshold")
plt.show()
plt.plot(dct_1[threshold1:], 'ro')                          #Plotting after truncation at threshold
plt.title("DCT on dataset one after truncation")
plt.show()

#################################################################################################################3
# Program to print matrix in Zig-zag pattern

print("\n DCT on DATASET 2")
# Create an empty list
print(1000* '-')
print("DATA AFTER ZIG-ZAG Traversing through DATA two\n")
x = data21.values.tolist()    #converting data to list
o = pd.DataFrame(x)
l = o.values.tolist()
#print(x[0])
#print(\n x[1])

matrix = l                              #creating a matrix
rows = 1000
columns = 100

sol2 = [[] for i in range(rows + columns - 1)]          #traversing in zizag way of matrix

for i in range(rows):
    for j in range(columns):
        sum = i + j
        if (sum % 2 == 0):

            # add at beginning
            sol2[sum].insert(0, matrix[i][j])
        else:

            # add at end of the list
            sol2[sum].append(matrix[i][j])

        # print the solution as it as
for i in sol2:
    for j in i:
        print(j, end=" ")


import math
print("\n")
                   #converting solution matrix to an array

dct_2=np.zeros([1100])
n=100
vari=0
sum2=0
for i in sol2:
    vari+=1
    varj=0
    sum2= 0
    for j in i:
        varj+=1
        x=j
        sum2 = sum2 + (x *(math.cos((((2*varj)+1) * vari * math.pi) / (2*n)) ))         #applying DCT formula

    if(vari==0):
        dct_2[vari] = math.sqrt(1/n) * sum2                   #applying DCT formula for U0
    else:
        dct_2[vari]= math.sqrt(2/n) * sum2                    #applying DCT for Uk where k = 1,2,3,...,k-1


print("DATA AFTER APPLYING DCT-I on DATASET two\n")
print(dct_2)
print(1000*'-')
vari=0
for i in dct_2:
    vari+=1
    if(i==min(dct_2)):
        threshold2= vari                                #finding the threshold for dataset

print("Threshold value is ")
print(threshold2)
kp2= 0
total2 = 1099
kp2 =((threshold2) / total2 )* 100
print("\n% of DATA after applying DCT-I for feature extraction on dataset two")
print(kp2)
plt.plot(dct_2,'ro')                                        #plotting the data
plt.title("DCT on DATASET two prior truncating to threshold")
plt.show()
plt.plot(dct_2[:threshold2], 'ro')                          #Plotting after truncation at threshold
plt.title("DCT on dataset two after truncation")
plt.show()


