from node import Node
import math


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
        if key is not "Class":
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


def ID3(examples, default):
    '''
    Takes in an array of examples, and returns a tree (an instance of Node)
    trained on the examples.  Each example is a dictionary of attribute:value pairs,
    and the target class variable is a special attribute with the name "Class".
    Any missing attributes are denoted with a value of "?"
    '''
    notExistsNontrivial = True
    for key in examples[0]:  # columns of the data set
        if key is not "Class":
            for i in range(len(examples) - 1):  # rows of data set
                if examples[i][key] == examples[i + 1][key]:
                    continue
                else:
                    notExistsNontrivial = False
                    break
    sameClassification = True
    for i in range(len(examples) - 1):
        if examples[i]["Class"] == examples[i + 1]["Class"]:
            continue
        else:
            sameClassification = False
            break
    if examples is None:
        return default
    elif sameClassification or notExistsNontrivial:
        valueCount = classValueCounter(examples)
        node = Node()
        maxKey = ""
        for key in valueCount:
            if maxKey is "":
                maxKey = key
            elif valueCount[key] > valueCount[maxKey]:
                maxKey = key
        node.label = maxKey
        return node
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
            subtree = ID3(newDataSet, default)
            best.children[value] = subtree
        return best


# else:
# best = chooseAttribute(examples)
# t = Node()
# t.label = best


def prune(node, examples):
    '''
    Takes in a trained tree and a validation set of examples.  Prunes nodes in order
    to improve accuracy on the validation data; the precise pruning strategy is up to you.
    '''


def test(node, examples):
    '''
    Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
    of examples the tree classifies correctly).
    '''


def evaluate(node, example):
    '''
    Takes in a tree and one example.  Returns the Class value that the tree
    assigns to the example.
    '''
    if len(node.children) is not 0:
        return evaluate(node.children[example[node.attribute]], example)
    else:
        return node.label
