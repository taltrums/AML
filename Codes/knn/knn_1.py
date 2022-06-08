import random
import csv
import math
import operator

def load_dataset(filename, split, train_set=[], test_set=[]):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
                if(random.random()<split):
                    train_set.append(dataset[x])
                else:
                    test_set.append(dataset[x])

def euclidean_dist(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x]-instance2[x]),2)
    return math.sqrt(distance)

def normalEuclidean_dist(instance1, instance2, length):
    distance = 0
    for x in range(length):
        a = float(pow((instance1[x]-instance2[x]),2))
        distance += pow(((instance1[x]-instance2[x])/a),2)
    return math.sqrt(distance)

def cosineSimilarity(data_1, data_2):
    dot = np.dot(data_1, data_2[:-1])
    norm_data_1 = np.linalg.norm(data_1)
    norm_data_2 = np.linalg.norm(data_2[:-1])
    cos = dot / (norm_data_1 * norm_data_2)
    return (1-cos)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        #dist = cosineSimilarity(testInstance, trainingSet[x])
        #dist = NeuclideanDistance(testInstance, trainingSet[x], length)
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedvotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedvotes[0][0]

def getAccuracy(test_set, prediction):
    correct = 0
    for x in range(len(test_set)):
        if prediction[x] == test_set[x][-1]:
            correct += 1
    return float(correct/len(test_set))*100.0


def main():
    # prepare data
    trainingSet=[]
    testSet=[]
    split = 0.67
    load_dataset('iris.csv', split, trainingSet, testSet)
    print ('\n Number of Training data: ' + (repr(len(trainingSet))))
    print (' Number of Test Data: ' + (repr(len(testSet))))
    # generate predictions
    predictions=[]
    k = 3
    print('\n The predictions are: ')
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print(' predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('\n The Accuracy is: ' + repr(accuracy) + '%')
    
main()





