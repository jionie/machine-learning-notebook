from scipy import stats
import numpy as np


# This method computes entropy for information gain
def entropy(class_y):
    # Input:            
    #   class_y         : list of class labels 
    
    # TODO: Compute the entropy for a list of classes
    #
    # Example:
    #    entropy([0,0,0,1,1,1,1,1,1]) = 0.9182958340544896
    #    entropy([0,0,0,1,1,1,2,2,2]) = 1.584962500721156
        
    entropy = 0
    ##class_y is empty
    if(len(class_y)==0):
        return entropy

    class_y_set = set(class_y)

    for class_ in class_y_set:
        
        class_compare = class_*np.ones(len(class_y))
        class_prob = sum(np.asarray(class_y)==class_compare)/len(class_y)
        entropy += -1*class_prob*np.log2(class_prob)

    return entropy


def partition_classes(X, y, split_attribute, split_val):
    # Inputs:
    #   X               : data containing all attributes
    #   y               : labels
    #   split_attribute : column index of the attribute to split on
    #   split_val       : either a numerical or categorical value to divide the split_attribute
    #   Numeric Split Attribute:
    #   Split the data X into two lists(X_left and X_right) where the first list has all
    #   the rows where the split attribute is less than or equal to the split value, and the 
    #   second list has all the rows where the split attribute is greater than the split 
    #   value. Also create two lists(y_left and y_right) with the corresponding y labels.
    #
    # Categorical Split Attribute:
    #   Split the data X into two lists(X_left and X_right) where the first list has all 
    #   the rows where the split attribute is equal to the split value, and the second list
    #   has all the rows where the split attribute is not equal to the split value.
    #   Also create two lists(y_left and y_right) with the corresponding y labels.

    '''
    Example:
    
    X = [[3, 'aa', 10],                 y = [1,
         [1, 'bb', 22],                      1,
         [2, 'cc', 28],                      0,
         [5, 'bb', 32],                      0,
         [4, 'cc', 32]]                      1]
    
    Here, columns 0 and 2 represent numeric attributes, while column 1 is a categorical attribute.
    
    Consider the case where we call the function with split_attribute = 0 and split_val = 3 (mean of column 0)
    Then we divide X into two lists - X_left, where column 0 is <= 3  and X_right, where column 0 is > 3.
    
    X_left = [[3, 'aa', 10],                 y_left = [1,
              [1, 'bb', 22],                           1,
              [2, 'cc', 28]]                           0]
              
    X_right = [[5, 'bb', 32],                y_right = [0,
               [4, 'cc', 32]]                           1]

    Consider another case where we call the function with split_attribute = 1 and split_val = 'bb'
    Then we divide X into two lists, one where column 1 is 'bb', and the other where it is not 'bb'.
        
    X_left = [[1, 'bb', 22],                 y_left = [1,
              [5, 'bb', 32]]                           0]
              
    X_right = [[3, 'aa', 10],                y_right = [1,
               [2, 'cc', 28],                           0,
               [4, 'cc', 32]]                           1]
               
    ''' 

    X_left = []
    X_right = []
    
    y_left = []
    y_right = []

    if((isinstance(split_val,int))or(isinstance(split_val,float))):

        for i in range(len(X)):
            if(X[i][split_attribute]<=split_val):
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])

    if(isinstance(split_val,str)):
        
        for i in range(len(X)):
            if(X[i][split_attribute]==split_val):
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])
    
    return (X_left, X_right, y_left, y_right)


def information_gain(previous_y, current_y):
    # Inputs:
    #   previous_y: the distribution of original labels 
    #   current_y:  the distribution of labels after splitting based on a particular
    #               split attribute and split value
    
    
    """
    Example:
    
    previous_y = [0,0,0,1,1,1]
    current_y = [[0,0], [1,1,1,0]]
    
    info_gain = 0.45915
    """
    y_split_0 = current_y[0]
    y_split_1 = current_y[1]

    entropy_split_0 = entropy(y_split_0)
    entropy_split_1 = entropy(y_split_1)

    p_split_0 = len(y_split_0)/len(previous_y)
    p_split_1 = len(y_split_1)/len(previous_y)

    entropy_pre = entropy(previous_y)
    entropy_cur = p_split_0 * entropy_split_0 + p_split_1 * entropy_split_1

    info_gain = entropy_pre - entropy_cur

    return info_gain

def information_gain_ratio(previous_y, current_y):
    # Inputs:
    #   previous_y: the distribution of original labels 
    #   current_y:  the distribution of labels after splitting based on a particular
    #               split attribute and split value

    info_gain = information_gain(previous_y, current_y)
    y_split_0 = current_y[0]
    y_split_1 = current_y[1]
    p_split_0 = len(y_split_0)/len(previous_y)
    p_split_1 = len(y_split_1)/len(previous_y)
    
    if((p_split_0!=0)and(p_split_1!=0)):
        entropy_split = -1*p_split_0*np.log2(p_split_0) + -1*p_split_1*np.log2(p_split_1)
        return info_gain/entropy_split
    else:
        return 0