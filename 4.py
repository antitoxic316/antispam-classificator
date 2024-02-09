import requests
import tarfile
import os
import sklearn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

class LoadData:
    def __init__(self, urls):
        self.urls = urls
        self.tar_file_paths = []
        self.data_dir_path = os.path.join(os.getcwd(), "data")

        self.extracted_dir_names = ["easy_ham_2", "hard_ham", "spam_2"]

        self.extracted_data_dirs = {x: os.path.join(self.data_dir_path, x) for x in self.extracted_dir_names}

        self.raw_data = {x: [] for x in self.extracted_dir_names}

        self.text_data = {x: [] for x in self.extracted_dir_names}

        self.uncategorized_text_data = []

        self.X = []
        self.y = []

    
    def fetch_and_save_data(self):
        if not os.path.isdir(self.data_dir_path):    
            os.mkdir(self.data_dir_path)

        for url in self.urls:
            r = requests.get(self.urls[url], stream=True)
            if r.status_code == 200:
                with open(os.path.join(self.data_dir_path, url), "wb") as f:
                    f.write(r.raw.read())
                self.tar_file_paths.append(os.path.join(self.data_dir_path, url))
                print(f"saved {url}")
            else:
                print(f"error {r.status_code}")

    def extract_data(self):
        for file_path in self.tar_file_paths:
            tar = tarfile.open(file_path, "r:bz2")  
            tar.extractall(self.data_dir_path)
            tar.close()

    def get_raw_data(self):
        for dir in self.extracted_data_dirs:
            dir_name = dir
            dir_path = self.extracted_data_dirs[dir]

            for file in os.listdir(dir_path)[:-1]:
                with open(os.path.join(dir_path, file), mode="rb") as f:
                    try:
                        self.raw_data[dir_name].append(f.read().decode())
                    except UnicodeDecodeError:
                        f.close()

    def get_text_from_data(self):
        for data_header in self.extracted_dir_names:
            for i in range(0, len(self.raw_data[data_header])):
                message = self.raw_data[data_header][i]
                message = message[message.find("Message-Id"):]
                message = message[message.find("\n\n"):]

                self.raw_data[data_header][i] = message

                self.uncategorized_text_data.append(message)
                self.text_data[data_header].append(message)

    def get_vectorized_data(self, vectorized: CountVectorizer):
        for category_name in self.text_data:
            for message in self.text_data[category_name]:

                self.X.append(vectorizer.transform([message]).toarray()[0])
                if category_name == "spam_2":
                    self.y.append(1) #is scam
                else:
                    self.y.append(0) #not a scam

urls = [
    "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2"
]

urls_dict = {x.split("/")[-1] : x for x in urls}

data_loader = LoadData(urls_dict)
data_loader.fetch_and_save_data()
data_loader.get_raw_data()
data_loader.get_text_from_data()

vectorizer = CountVectorizer(max_features=250)
vectorizer.fit(data_loader.uncategorized_text_data)

data_loader.get_vectorized_data(vectorizer)

X, y = shuffle(data_loader.X, data_loader.y)

train_X, train_y = X[:int(len(X)*0.70)-1], y[:int(len(y)*0.70)-1]
test_X, test_y = X[int(len(X)*0.70):], y[int(len(y)*0.70):]

regression_clasifier = LogisticRegression()
regression_clasifier.fit(train_X, train_y)
print(regression_clasifier.score(test_X, test_y))

knn = KNeighborsClassifier(10)
knn.fit(train_X, train_y)
print(knn.score(test_X, test_y))