import numpy as np 
from utils import entropy, partition_classes, information_gain_ratio, information_gain_ratio

class C4_5(object):

    def __init__(self, X, y, max_depth, min_gain):
        # Initializing the tree as an empty dictionary 
        X = np.asarray(X)
        y = np.asarray(y)
        self.tree = {}
        self.tree['max_depth'] = max_depth
        self.tree['min_gain'] = min_gain
        self.tree['features'] = np.arange(0, X.shape[1])
        self.tree['current_features'] = np.arange(0, X.shape[1])
        self.tree['depth'] = 0

    def fit(self, X, y):
        # Train the decision tree (self.tree) using the the sample X and labels y
        # X should be 2D numpy array NxM, N is the number of instances, 
        # M is the number of features.
        # y should be 1D numpy array N, N is the number of instances.
        # max_depth represents the maximum depth of the tree
        # min_gain represents the minimum information gain
        # key "left" and "right" represent the left child and right child

        X = np.asarray(X).astype(float)
        y = np.asarray(y).astype(int)

        #current feature set is empty
        if(self.tree['current_features'].shape[0]==0):
            self.tree['label'] = np.argmax(np.bincount(y))
            return 

        #All instances are same class
        if(len(set(y))==1):
            self.tree['label'] = y[0]
            return

        #Reach max_depth
        if((self.tree['max_depth']>0)and(self.tree['depth']==self.tree['max_depth'])):
            self.tree['label'] = np.argmax(np.bincount(y))
            return

        current_features = self.tree['current_features']
        max_information_gain_ratio_list = []
        max_split_val_list = []

        for split_attribute in current_features:
            X_select = list(set([x[split_attribute] for x in X]))
            max_information_gain_ratio = 0
            max_split_val = X_select[0]

            for split_val in X_select:
                (_, _, y_left, y_right) = partition_classes(X, y, split_attribute, split_val)
                current_information_gain_ratio = information_gain_ratio(y, [y_left, y_right])

                if(current_information_gain_ratio>max_information_gain_ratio):
                    max_information_gain_ratio = current_information_gain_ratio
                    max_split_val = split_val

            max_information_gain_ratio_list.append(max_information_gain_ratio)
            max_split_val_list.append(max_split_val)

        #index of split_attribute in current features
        index = np.argmax(max_information_gain_ratio_list)

        #information gain is less than threshold
        if(max_information_gain_ratio_list[index]<=self.tree['min_gain']):
            self.tree['label'] = np.argmax(np.bincount(y))
            return

        self.tree['split_attribute'] = current_features[index]
        self.tree['split_val'] = max_split_val_list[index]

        #split node
        (X_left, X_right, y_left, y_right) = partition_classes(X, y, \
        self.tree['split_attribute'], self.tree['split_val'])

        left_tree = C4_5(X_left, y_left, self.tree['max_depth'], self.tree['min_gain'])
        right_tree = C4_5(X_right, y_right, self.tree['max_depth'], self.tree['min_gain'])

        current_features = np.delete(current_features, index)
        left_tree.tree['current_features'] = current_features
        right_tree.tree['current_features'] = current_features
        
        left_tree.tree['depth'] = self.tree['depth'] + 1
        right_tree.tree['depth'] = self.tree['depth'] + 1

        left_tree.fit(X_left, y_left)
        right_tree.fit(X_right, y_right)
        
        self.tree['left'] = left_tree
        self.tree['right'] = right_tree

        return


    def predict(self, record):
        # predict the record using self.tree and return the predicted label
        if(('left' in self.tree)and('right' in self.tree)):
    
            split_attribute = self.tree['split_attribute']

            if(isinstance(self.tree['split_val'],int)):
                if(record[split_attribute]<=self.tree['split_val']):
                    return self.tree['left'].predict(record)
                else:
                    return self.tree['right'].predict(record)

            if(isinstance(self.tree['split_val'],float)):
                if(record[split_attribute]<=self.tree['split_val']):
                    return self.tree['left'].predict(record)
                else:
                    return self.tree['right'].predict(record)

            if(isinstance(self.tree['split_val'],str)):
                if(record[split_attribute]==self.tree['split_val']):
                    return self.tree['left'].predict(record)
                else:
                    return self.tree['right'].predict(record)

        elif(('left' not in self.tree)and('right' in self.tree)):
            
            return self.tree['right'].predict(record)

        elif(('left' in self.tree)and('right' not in self.tree)): 

            return self.tree['left'].predict(record)

        else:
            
            return self.tree['label']

    def score(self, X_test, y_test):
        worng_count = 0
        for i in range(X_test.shape[0]):
            if(self.predict(X_test[i])!=y_test[i]):
                worng_count += 1
        
        return 1-worng_count/X_test.shape[0]