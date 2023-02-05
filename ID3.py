#!/usr/bin/env python
# coding: utf-8

# ### Libraries

# In[1]:


#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import math
import copy


# ### Functions & Classes

# In[2]:


#Shuffles the whole data
def shuffle_array(arr):
    np.random.seed(1337)
    l = len(arr)
    for i in range(0,l):
        index = np.random.randint(0, l)
        arr[i] = arr[index]
        
#Splits train and test datasets
def split_train_test(arr, targetCol, k_fold = 1):
    
    # k_fold: k value for k_fold cross validation
    # dropCols: indices of the columns those are going to be dropped
    # targetCol: target column or attribute
    # Return type: list of X_train, y_train, X_test, y_test
    
    #Preparing data
    shuffle_array(arr)
    test_size = int(len(arr) / k_fold)
    k_datasets = list()
    
    #Splitting dataset with respect to the k-fold Cross Validation
    for index in range(k_fold):
        test_array = arr[test_size * index: test_size * (index + 1)]
        train_array = np.delete(arr, slice(test_size * index, test_size * (index + 1)), 0)
        X_train = train_array
        y_train = train_array[:, [targetCol]]
        X_test = test_array
        y_test = test_array[:, [targetCol]]
        k_datasets.append([X_train, y_train, X_test, y_test])
        
    return k_datasets

#Discretisation of continuous attributes
def discretisation(data, continuousCols):
    max_ele, min_ele = data.max(axis = 0), data.min(axis = 0)
    for i in continuousCols:
        mx, mn = max_ele[i], min_ele[i]
        wide = math.ceil((mx - mn) / 5) #5 intervals
        for row in data:
            x = (row[i] - mn) // wide
            ticket = "{} - {}".format((x * wide + mn), ((x+1) * wide + mn - 1))
            row[i] = ticket

#Returns counts of unique values of an attribute                
def count_unique_values(data, col_index):
    attr = data[:, col_index]
    values, counts = np.unique(attr, return_counts = True)
    return values, counts

#Returns entropy of an attribute
def entropy(data, col_index):
    total = 0
    _ , counts = count_unique_values(data, col_index)
    for count in counts:
        proportion = count / len(data)
        total -= (proportion)*np.log2(proportion)
    return total

#Returns information gain of an attribute
def info_gain(data, col_index, target_index):
    gain = entropy(data, target_index)
    values, _ = count_unique_values(data, col_index)
    for i in range(len(values)):
        mask = (data[:, col_index] == values[i])
        value_data = data[mask, :]
        gain -= (len(value_data) / len(data)) * entropy(value_data, target_index)
    return gain

#Node class
class node(object):
    def __init__(self):
        
        self.children = [] #References to child nodes
        self.attribute = None #Attribute name
        self.leafNode = None #Leaf node or not
        self.label = None
        self.value = None
        
    def set_leafNode(self, b):
        self.leafNode = b
        
    def get_leafNode(self):
        return self.leafNode
        
    def set_label(self, tag):
        self.label = tag
        
    def get_label(self):
        return self.label
        
    def get_attribute(self):
        return self.attribute
        
    def set_attribute(self, attribute):
        self.attribute = attribute
    
    def set_children(self, children):
        self.children = children
        
    def get_children(self):
        return self.children
    
    def set_value(self, v):
        self.value = v
        
    def get_value(self):
        return self.value
              
#ID3 Decision Tree Algorithm
def id3(data, target_attr, attr):
    
    root = node()
    values, counts = count_unique_values(data, target_attr)
    
    #If all examples are positive or negative
    if (len(values) == 1):
        root.set_leafNode(True)
        root.set_label(values[0])
    
    #If attributes is empty, return most common most common value of target_attribute in data
    elif (len(attr) == 0):
        root.set_leafNode(True)
        most_common_val_index = [index for index, item in enumerate(counts) if item == max(counts)][0]
        root.set_label(values[most_common_val_index])
    
    #Otherwise
    else:
        #Finding best classifier attribute for the root node
        info_gains = dict()
        for a in attr:
            info_gains[a] = info_gain(data, attr_dict[a], target_attr)
        sorted_info_gains = sorted(info_gains.items(), key=lambda x: x[1], reverse=True)
        best_classifier = sorted_info_gains[0][0]
        best_classifier_index = attr_dict[best_classifier]
        
        #Root node set-up
        root.set_leafNode(False)
        attr.remove(best_classifier)
        root.set_attribute(best_classifier)
        
        #Finding values of the attribute
        best_values, _ = count_unique_values(data, best_classifier_index)
        
        #Initialize children of the root node
        children = [node() for i in range(len(best_values))]
        
        #Set up children
        for val_index in range(len(best_values)):
            #Select the rows with the specific value
            mask = (data[:, best_classifier_index] == best_values[val_index])
            sub_data = data[mask, :]
            
            #If the selected dataset is empty 
            if (len(sub_data) == 0):
                children[val_index].set_leafNode(True)
                most_common_val_index = [index for index, item in enumerate(counts) if item == max(counts)][0]
                children[val_index].set_label(values[most_common_val_index])
            
            #If NOT
            else:
                children[val_index] = id3(sub_data, target_attr, attr)
            
            #Value of the children nodes
            children[val_index].set_value(best_values[val_index])
        
        #Add children to the root node
        root.set_children(children)
        
    #Return root
    return root

#Classify the sample with the ID3 tree
def classify_id3(tree, sample):
    #If the tree is leafNode
    if (tree.get_leafNode() == True):
        return tree.get_label()
    
    #If NOT
    else:
        #Which attribute need to be checked
        attr = tree.get_attribute()
        sample_attr_val = sample[attr_dict[attr]]
        
        #Check the specific attribute's values if that matches with sample's attribute
        for child in tree.get_children():
            if (child.get_value() == sample_attr_val):
                return classify_id3(child, sample)
        
        return None

class ID3:
    
    def __init__(self, train_data, test_data, y_test, classes, target_index, attribute_names):
        
        self.train_data = train_data
        self.test_data = test_data
        self.y_test = y_test
        self.classes = classes
        self.target_index = target_index
        self.attribute_names = attribute_names
        self.confusionMatrix = np.zeros(shape=(len(classes),len(classes)), dtype=int)
        
        #classification part
        tree = id3(self.train_data, self.target_index, self.attribute_names)
        for i in range(len(self.test_data)):
            predicted_class = classify_id3(tree, self.test_data[i])
            if (predicted_class == None): continue
            actual_class = self.y_test[i][0]
            self.confusionMatrix[classes[predicted_class],classes[actual_class]] += 1
            
    def get_confusionMatrix(self):
        return self.confusionMatrix
    
    def get_accuracy(self):
        return np.trace(self.confusionMatrix) / np.sum(self.confusionMatrix) * 100
        
    def print_accuracy(self):
        print("Accuracy: {:.2f}%".format(self.get_accuracy()))
                
    def plot(self):
        df_cm = pd.DataFrame(self.confusionMatrix, index = [i for i in self.classes], columns = [i for i in self.classes])
        plt.figure(figsize = (7,7))
        ax = sn.heatmap(df_cm, annot=True, fmt = "d", cmap="YlGnBu", cbar_kws={'label': 'Scale'})
        ax.set(ylabel="Predicted Label", xlabel="Actual Label")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.title("Confusion Matrix")


# ### Data Preparing Parameters

# In[3]:


#Data preparing parameters
dropColumns = ['EmployeeNumber']
targetColumn = 'Attrition'
continuousColumns = ['Age', 'DailyRate', 'DistanceFromHome',
                     'HourlyRate', 'MonthlyIncome', 'MonthlyRate',
                     'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears',
                     'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
                    'YearsWithCurrManager', 'TrainingTimesLastYear']
attributeColumns = copy.deepcopy(dropColumns)
attributeColumns.append(targetColumn)
csv_file = 'HR-Employee-Attrition.csv'


# ### Data Preparing

# In[4]:


#Data preparing 
df = pd.read_csv(csv_file)
df.drop(columns=dropColumns, inplace = True)
target_index = df.columns.get_loc(targetColumn)
#Preparing attribute dictionary and attribute names
attr_dict = dict()
for index in range(len(df.columns)):
    attr_dict[df.columns[index]] = index
#Dataframe to numpy
arr = df.to_numpy()
#Discretisation
continuous_indices = [df.columns.get_loc(col_name) for col_name in continuousColumns]
discretisation(arr, continuous_indices)
#Classes
classes, _ = count_unique_values(arr, target_index)
classes_dict = {}
for i in range(len(classes)):
    classes_dict[classes[i]] = i


# ### k-Fold Cross Validation

# In[5]:


kFold_list = split_train_test(arr, target_index, k_fold = 5)
_ = 0

for fold in kFold_list:
    train_data = fold[0]
    test_data = fold[2]
    y_test = fold[3]
    
    attribute_names = [key for key in attr_dict.keys() if key not in attributeColumns]
    ID3_obj = ID3(train_data, test_data, y_test, classes_dict, target_index, attribute_names)
    print("k-Fold Cross Validation, Fold Number: {}".format(_), end = " ")
    ID3_obj.print_accuracy()
    _ += 1


# In[ ]:




