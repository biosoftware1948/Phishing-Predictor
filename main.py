import pandas as pd

from processData import PhishingData

FILE_NAME = "PhishingData.arff"

def label_occurences(data):
    label_dict = {}
    for val in data:
        if val not in label_dict:
            label_dict[val] = 1
        else:
            label_dict[val] += 1

    for key, val in label_dict.items():
        print("value " + str(key) + " occurs " + str(100.0*val/len(data)) + " percent in dataset")

def main():
    dataSet = PhishingData(FILE_NAME)
    dataSet.load()

    label_occurences(dataSet.y_test)

    classifier = int(input("\nChoose a classifier:\n1. Random Forest\n2. Linear SVC\n3. K Neighbors\n4. K Means\n"));
    if classifier == 1:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    elif classifier == 2:
        from sklearn.svm import LinearSVC
        clf = LinearSVC(random_state=0, tol=1e-6)
    elif classifier == 3:
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=12)
    elif classifier == 4:
        kmeans(dataSet)
        return
    else:
        return

    clf.fit(dataSet.x_train, dataSet.y_train)
    y_prediction = clf.predict(dataSet.x_test)

    if(classifier == 1):
        featureImportance(clf)

    from sklearn.metrics import accuracy_score
    print ("Accuracy of classifier: " + str((accuracy_score(dataSet.y_test, y_prediction))))

def featureImportance(classifier):
    labels = ("SFH", "popUpWindow", "SSLfinal_State", "Request_URL", "URL_of_Anchor", "Web_traffic", "URL_Length", "age", "has_IP")
    importances = classifier.feature_importances_
    print("Printing feature importance below")
    print (importances)
    plot(labels, importances)

def plot(labels, importances):
    import matplotlib.pyplot as plt; plt.rcdefaults()
    import numpy as np
    import matplotlib.pyplot as plt
 
    y_pos = np.arange(len(labels))
 
    plt.bar(y_pos, importances, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.ylabel('Importance')
    plt.xlabel("Feature")
    plt.title('Feature Importance when predicting phishing links')

 
    plt.show()

def kmeans(dataSet):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # Retrieve clustered values for train and test data
    train_clusters, train_values, test_clusters, test_values = kmeans_cluster_data(dataSet)
    overall_accuracy_generic, overall_accuracy_clustered = 0, 0
    win, lose, count = 0, 0, 0

    # Create generic ML model to fit all data points
    clf_generic = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf_generic.fit(dataSet.x_train, dataSet.y_train)

    # Predict value for test data in every cluster
    for label in train_clusters:
        if label not in test_clusters:
            continue

        # Create ML model to fit specific cluster
        clf_clustered = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        clf_clustered.fit(train_clusters[label], pd.Series(train_values[label]))

        # Make predictions for test data's value in cluster
        y_prediction_generic = clf_generic.predict(test_clusters[label])
        y_prediction_clustered = clf_clustered.predict(test_clusters[label])

        # Retrieve accuracy
        accuracy_generic = accuracy_score(pd.Series(test_values[label]), y_prediction_generic)
        accuracy_clustered = accuracy_score(pd.Series(test_values[label]), y_prediction_clustered)

        # Print cluster results
        print ("Generic Accuracy for cluster " + str(label) + ": " + str(accuracy_generic))
        print ("Clustered Accuracy for cluster " + str(label) + ": " + str(accuracy_clustered))
        print ("Sample size of : " + str(len(train_values[label])) + " + " + str(len(test_values[label])))
        print ("\n")

        overall_accuracy_generic += accuracy_generic * len(test_values[label])
        overall_accuracy_clustered += accuracy_clustered * len(test_values[label])

        if accuracy_generic > accuracy_clustered:
            lose += 1
        if accuracy_clustered > accuracy_generic:
            win += 1
        count += 1

    # Print overall results
    print ("\nOverall accuracy:")
    print ("Generic: " + str(overall_accuracy_generic/dataSet.y_test.size))
    print ("Clustered: " + str(overall_accuracy_clustered/dataSet.y_test.size))
    print ("Sample size of : " + str(dataSet.y_test.size))

    print ("\n")
    print ("Clustered wins: \t" + str(win))
    print ("Clustered losses: \t" + str(lose))
    print ("Count: \t" + str(count))

def kmeans_cluster_data(dataSet):
    from sklearn.cluster import KMeans

    train_clusters, train_values, test_clusters, test_values = {}, {}, {}, {}

    # Create individual DataFrames for each cluster
    kmeans = KMeans(n_clusters=46, random_state=0).fit(dataSet.x_train)
    for index, label in enumerate(kmeans.labels_):
        if label not in train_clusters:
            train_clusters[label] = pd.DataFrame(columns=dataSet.x_train.columns)
            train_values[label] = []

        train_clusters[label] = train_clusters[label].append(dataSet.x_train.iloc[index], ignore_index=True)
        train_values[label].append(dataSet.y_train.iloc[index])

    # Find associated cluster for all test data
    predictions = kmeans.predict(dataSet.x_test)
    for index, label in enumerate(predictions):
        if label not in test_clusters:
            test_clusters[label] = pd.DataFrame(columns=dataSet.x_test.columns)
            test_values[label] = []

        test_clusters[label] = test_clusters[label].append(dataSet.x_test.iloc[index], ignore_index=True)
        test_values[label].append(dataSet.y_test.iloc[index])

    return train_clusters, train_values, test_clusters, test_values

if __name__ == "__main__":
    main()
