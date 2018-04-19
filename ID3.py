from node import Node
import math

valueCountSet = {}
def handleMissingAttributes(examples):
    for key in examples[0]:
        values = []
        for i in range(len(examples)):
            if key not in values and examples[i][key] != '?':
                values.append(examples[i][key])
        valueCountSet[key] = values
        for i in range(len(examples)):
            if examples[i][key] == '?':
                examples[i][key] = valueCountSet[key][0]

def classValueCounter(examples):
    valueCounts = {}
    for example in examples:
        currentValue = example["Class"]
        if currentValue in valueCounts:
            valueCounts[currentValue] = valueCounts[currentValue] + 1
        else:
            valueCounts[currentValue] = 1
    return valueCounts


def entropy(examples):
    valueCounts = classValueCounter(examples)
    h = 0
    for key in valueCounts:
        p = float(valueCounts[key]) / float(len(examples))
        h = h + -p * math.log(p, 2)
    return h


def chooseAttribute(examples):
    infoGains = {}  # dictionary of attribute : infoGain
    hPrior = entropy(examples)
    for key in examples[0]:  # populating the keys with attributes that aren't class
        if key != 'Class':
            infoGains[key] = 0
    for key in infoGains:
        valueCountSet = {}  # a dictionary of possibleValue : numAppearances
        for i in range(len(examples)):  # populating valueCountSet
            if examples[i][key] not in valueCountSet:
                valueCountSet[examples[i][key]] = 1
            else:
                valueCountSet[examples[i][key]] = valueCountSet[examples[i][key]] + 1
        summation = 0  # the term to be subtracted from hPrior
        for value in valueCountSet:
            newDataSet = []  # subset of examples for x = v
            for i in range(len(examples)):  # populating newDataSet
                if examples[i][key] == value:
                    newDataSet.append(examples[i])
            p = float(valueCountSet[value]) / float(len(examples))
            hNew = entropy(newDataSet)
            summation = summation + p * hNew
        infoGains[key] = hPrior - summation

    bestAttribute = ""
    for attribute in infoGains:  # find best attribute
        if bestAttribute == "":
            bestAttribute = attribute
        elif infoGains[attribute] > infoGains[bestAttribute]:
            bestAttribute = attribute
    return bestAttribute

def mode(examples):
    valueCount = classValueCounter(examples)
    maxKey = ""
    for key in valueCount:
        if maxKey == "":
            maxKey = key
        elif valueCount[key] > valueCount[maxKey]:
            maxKey = key
    return maxKey

def ID3(examples, default):
    '''
    Takes in an array of examples, and returns a tree (an instance of Node)
    trained on the examples.  Each example is a dictionary of attribute:value pairs,
    and the target class variable is a special attribute with the name "Class".
    Any missing attributes are denoted with a value of "?"
    '''
    handleMissingAttributes(examples)
    notExistsNontrivial = True
    for key in examples[0]:  # columns of the data set
        if key != 'Class':
            for i in range(len(examples) - 1):  # rows of data set
                if examples[i][key] != examples[i + 1][key]:
                    notExistsNontrivial = False
    sameClassification = True
    for i in range(len(examples) - 1):
        if examples[i]["Class"] != examples[i + 1]["Class"]:
            sameClassification = False

    if len(examples) == 0:
        node = Node()
        node.label = default
        return default
    elif sameClassification or notExistsNontrivial:
        best = Node()
        best.label = mode(examples)
        return best
    else:
        best = Node()
        best.attribute = chooseAttribute(examples)
        #######################
        valueCountSet = {}  # a dictionary of possibleValue : numAppearances
        for i in range(len(examples)):  # populating valueCountSet
            if examples[i][best.attribute] not in valueCountSet:
                valueCountSet[examples[i][best.attribute]] = 0
        #######################
        for value in valueCountSet:
            newDataSet = []  # subset of examples for x = v
            for i in range(len(examples)):  # populating newDataSet
                if examples[i][best.attribute] == value:
                    newDataSet.append(examples[i])

            best.mostCommonClass = mode(newDataSet)
            subtree = ID3(newDataSet, mode(examples))
            subtree.parent = best
            best.children[value] = subtree
        return best

def isBottomAttributeNode(node):
    for child in node.children:
        if node.children[child].attribute is None:
            continue
        else:
            return False
    return True

def pruneHelp(origTree, currNode, examples):
    if currNode.attribute is not None:
        for child in currNode.children:
            pruneHelp(origTree, currNode.children[child], examples)
        else:
            return 0
    for child in currNode.children:
        if currNode.children[child].attriute is not None:
            parentPostPrune = origTree
            newNode = Node()
            newNode.label = currNode.mostCommonClass
            parentPostPrune.children[child] = newNode
            newNode.parent = parentPostPrune
            newTree = newNode
            while newTree.parent is not None:
                newTree = newTree.parent
            if test(newTree, examples) > test(origTree, examples):
                origTree = newTree
    newTree = origTree
    newNode = Node()


def prune(node, examples):
    '''
    Takes in a trained tree and a validation set of examples.  Prunes nodes in order
    to improve accuracy on the validation data; the precise pruning strategy is up to you.
    '''
    pruneHelp(node, node, examples)
    # if node.attribute is not None:
    #     for child in node.children:
    #         prune(node.children[child], examples)
    # else:
    #     return 0
    #
    # for child in node.children:
    #     if node.children[child].attribute is not None:
    #         parentPostPrune = node
    #         newNode = Node()
    #         newNode.label = node.mostCommonClass
    #         parentPostPrune.children[child] = newNode
    #         newNode.parent = parentPostPrune
    #         if test(parentPostPrune, examples) > test(node, examples):
    #             node = parentPostPrune



def test(node, examples):
    '''
    Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
    of examples the tree classifies correctly).
    '''
    correct = 0
    for i in range(len(examples)):
        if evaluate(node, examples[i]) == examples[i]["Class"]:
            correct = correct + 1
    return float(correct) / float(len(examples))


def evaluate(node, example):
    '''
    Takes in a tree and one example.  Returns the Class value that the tree
    assigns to the example.
    '''
    for key in example:
        if example[key] == '?':
            example[key] = valueCountSet[key][0]

    if len(node.children) != 0:
        return evaluate(node.children[example[node.attribute]], example)
    else:
        return node.label
