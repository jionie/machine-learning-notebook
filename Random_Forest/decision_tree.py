from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        #self.tree = {}
        self.tree = {}

    def learn(self, X, y, max_depth):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)

        if(self.tree['current_features'].shape[0]==0):

            self.tree['label'] = int(round(sum(y)/len(y)))
            #print("no feature")
            return
        
        if(len(X)<=1):

            self.tree['label'] = int(round(sum(y)/len(y)))
            #print("one data")
            return

        if((max_depth>0)and(self.tree['depth']==max_depth)):

            self.tree['label'] = int(round(sum(y)/len(y)))
            #print("max depth")
            return

        current_features = self.tree['current_features']
        information_gain_list = []
        max_split_val_list = []

        for split_attribute in current_features:
            X_select = [x[split_attribute] for x in X]

            max_split_val = 0
            max_information_gain = 0

            for split_val in X_select:
    
                (_, _, y_left, y_right) = partition_classes(X, y, split_attribute, split_val)
                current_y = [y_left, y_right]
                current_information_gain = information_gain(y, current_y)

                if(current_information_gain>max_information_gain):
                    max_information_gain = current_information_gain
                    max_split_val = split_val
            
            information_gain_list.append(max_information_gain)
            max_split_val_list.append(max_split_val)

        index = np.argmax(information_gain_list)
        if(information_gain_list[index]<=self.tree['min_gain']):

            self.tree['label'] = int(round(sum(y)/len(y)))
            #print("little gain")
            return

####if information gain > min_gain, split
        self.tree['split_attribute'] = current_features[index]
        self.tree['split_val'] = max_split_val_list[index]

        (X_left, X_right, y_left, y_right) = partition_classes(X, y, \
        self.tree['split_attribute'], self.tree['split_val'])

        left_tree = DecisionTree()
        right_tree = DecisionTree()

        self.tree['left'] = left_tree
        self.tree['right'] = right_tree

        current_features = np.delete(current_features, index) #delete selected feature
        left_tree.tree['current_features'] = current_features
        right_tree.tree['current_features'] = current_features

        left_tree.tree['features'] = self.tree['features']
        right_tree.tree['features'] = self.tree['features']

        left_tree.tree['depth'] = self.tree['depth']+1
        right_tree.tree['depth'] = self.tree['depth']+1

        left_tree.tree['min_gain'] = self.tree['min_gain']
        right_tree.tree['min_gain'] = self.tree['min_gain']
        
        #print("left tree depth: " + str(self.tree['depth']+1))
        left_tree.learn(X_left, y_left, max_depth)
        #print("right tree depth: " + str(self.tree['depth']+1))
        right_tree.learn(X_right, y_right, max_depth)

        

    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        if(('left' in self.tree)and('right' in self.tree)):

            split_attribute = self.tree['split_attribute']

            if(isinstance(self.tree['split_val'],int)):
                if(record[split_attribute]<=self.tree['split_val']):
                    return self.tree['left'].classify(record)
                else:
                    return self.tree['right'].classify(record)

            if(isinstance(self.tree['split_val'],float)):
                if(record[split_attribute]<=self.tree['split_val']):
                    return self.tree['left'].classify(record)
                else:
                    return self.tree['right'].classify(record)

            if(isinstance(self.tree['split_val'],str)):
                if(record[split_attribute]==self.tree['split_val']):
                    return self.tree['left'].classify(record)
                else:
                    return self.tree['right'].classify(record)

        elif(('left' not in self.tree)and('right' in self.tree)):
            
            return self.tree['left'].classify(record)

        elif(('left' in self.tree)and('right' not in self.tree)): 

            return self.tree['right'].classify(record)

        else:
            
            return self.tree['label']
