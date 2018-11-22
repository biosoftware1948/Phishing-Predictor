from processData import PhishingData

FILE_NAME = "PhishingData.arff"

def label_occurences(data):
    label_dict = {}
    for val in data:
        if val not in label_dict:
            label_dict[val] = 1
        else:
            label_dict[val]+=1
    
    for key, val in label_dict.items():
        print("value " + str(key) + " occurs " + str(val/len(data)) + " percent in dataset")

def main():
    dataSet = PhishingData(FILE_NAME)
    dataSet.load()

    label_occurences(dataSet.y_test)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(dataSet.x_train, dataSet.y_train)
    y_prediction = clf.predict(dataSet.x_test)
    from sklearn.metrics import accuracy_score
    print ("Accuracy of classifier: " + str((accuracy_score(dataSet.y_test, y_prediction))))

if __name__ == "__main__":
    main()