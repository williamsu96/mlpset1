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

    # valueCountSet = {}  # a dictionary of possibleValue : numAppearances
    # for i in range(len(examples)):  # populating valueCountSet
    #     for key in examples[i]:
    #         if examples[i][key] not in valueCountSet and examples[i][key] != '?':
    #             valueCountSet[examples[i][key]] = 0
    #
    # for i in range(len(examples)):
    #     for key in examples[i]:
    #         if examples[i][key] == '?':
    #             examples[i][key] = valueCountSet.keys()[0]


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
    handleMissingAttributes(examples)
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
        # print("leaf!")
        valueCount = classValueCounter(examples)
        best = Node()
        maxKey = ""
        for key in valueCount:
            if maxKey is "":
                maxKey = key
            elif valueCount[key] > valueCount[maxKey]:
                maxKey = key
        best.label = maxKey
        return best
    else:
        # print("branch!")
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

            newClassValues = classValueCounter(newDataSet)
            maxKey = ""
            for key in newClassValues:
                if maxKey is "":
                    maxKey = key
                elif newClassValues[key] > newClassValues[maxKey]:
                    maxKey = key

            best.mostCommonClass = maxKey
            subtree = ID3(newDataSet, default)
            subtree.parent = best
            best.children[value] = subtree
        return best


def findLeafNode(node):
    if len(node.children) is not 0:
        for child in node.children:
            return findLeafNode(node.children[child])
    else:
        return node


# def pruneCheck(currNode, origTree, examples):
#     pruneTree = node
#     for child in node.children:
#         if node.children[child].attribute is not None:
#             newNode = Node()
#             newNode.parent = pruneTree
#             newNode.

def isBottomAttributeNode(node):
    for child in node.children:
        if node.children[child].attribute is None:
            continue
        else:
            return False
    return True


def prune(node, examples):
    '''
    Takes in a trained tree and a validation set of examples.  Prunes nodes in order
    to improve accuracy on the validation data; the precise pruning strategy is up to you.
    '''
    currNode = node

    if node.attribute is not None:
        for child in node.children:
            prune(node.children[child], examples)
    else:
        # print("prune")
        return 0

    for child in node.children:
        if node.children[child].attribute is not None:
            parentPostPrune = node
            newNode = Node()
            newNode.label = node.mostCommonClass
            parentPostPrune.children[child] = newNode
            newNode.parent = parentPostPrune
            # print("ay" + str(test(parentpostPrune, examples)))
            if test(parentPostPrune, examples) > test(node, examples):
                # print("what's poppin")
                node = parentPostPrune

    # if node.attribute is not None:
    #     for child in node.children:
    #         if isBottomAttributeNode(node.children[child]):
    #             parentPostPrune = node
    #             newNode = Node()
    #             newNode.label = node.children[child].mostCommonClass
    #             newNode.parent = node
    #             parentPostPrune.children[child] = newNode
    #         elif hasAttributeNode(node.children[child]):
    #             prune(node.children[child])
    #         else:
    #             return node
    # else:
    #     return node

    # if node.attribute is not None and not isBottomAttributeNode(node):
    #     for child in node.children:
    #         prune(node.children[child], examples)
    # else:

    # leafNode = findLeafNode(node)
    # currNode = leafNode
    # prunedTree = node
    # if currNode.attribute is None and currNode.parent is not None:
    #     currNode = currNode.parent
    # else:
    #     print("hello!")
    #     for child in currNode.children:
    #         if prunedTree.children[child].attribute is not None:
    #             newNode = Node()
    #             newNode.parent = currNode
    #             newNode.label = prunedTree.children[child].mostCommonClass
    #             prunedTree.children[child] = newNode
    #             print(test(prunedTree,examples))


def test(node, examples):
    '''
    Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
    of examples the tree classifies correctly).
    '''
    correct = 0
    for i in range(len(examples)):
        # print(evaluate(node, examples[i]) + " " + examples[i]["Class"])
        if evaluate(node, examples[i]) == examples[i]["Class"]:
            correct = correct + 1
    return float(correct) / float(len(examples))


def evaluate(node, example):
    '''
    Takes in a tree and one example.  Returns the Class value that the tree
    assigns to the example.
    '''
    # handleMissingAttributes([example])
    if len(node.children) != 0:
        return evaluate(node.children[example[node.attribute]], example)
    else:
        return node.label
