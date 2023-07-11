from metaflow import FlowSpec, step
# conda_base
# @conda_base(libraries={'numpy':'1.23.5', 'scikit-learn':'1.2.2', 'dill':'0.3.2', 'pandas':'1.4.2'}, python='3.9.16')

class CreditClassifierTrain(FlowSpec):
    '''
    Model training flow
    Load training data, split into train and test, train model. Note, I am not doing any hyperparameter tuning here.
    Persist final model, train and test data
    '''
    @step
    def start(self):
        print("Flow is starting")
        self.next(self.load_data)

    @step
    def load_data(self):
        import pandas as pd
        ''' 
        This is where you would load the actual model development data set.
        '''
    
        self.df = pd.read_csv('../../data/credit_card_historical_data.csv')
        self.next(self.split_data)

    @step
    def split_data(self):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        x = self.df[['type','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']]
        le = LabelEncoder()
        x['type_encoded'] = le.fit_transform(x['type'])
        x = x.drop(['type'],axis=1)
        y = self.df["isFraud"].values
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, stratify=y, random_state=42, test_size=0.1, shuffle=True
            )
        print("Data is split")
        self.next(self.create_features)

    @step
    def create_features(self):
        
        from sklearn.preprocessing import StandardScaler

        #standardizing data
        scaler = StandardScaler()

        self.x_train_clean = scaler.fit_transform(self.x_train)
        self.x_test_clean = scaler.transform(self.x_test)
        
        self.next(self.train_model)

    @step
    def train_model(self):
        from sklearn.linear_model import LogisticRegression
        # Train logistic regression classifier
        self.lr = LogisticRegression(C=0.1, solver="sag")
        self.lr.fit(self.x_train_clean, self.y_train)
        print("Model has been trained")
        self.next(self.get_scores)

    @step
    def get_scores(self):
        self.train_scores = self.lr.predict_proba(self.x_train_clean)
        self.test_scores = self.lr.predict_proba(self.x_test_clean)
        print("Test set scores saved for evaluation")
        print("accuracy scores train and test", self.lr.score(self.x_train_clean,self.y_train), self.lr.score(self.x_test_clean,self.y_test))
        self.next(self.end)

    @step
    def end(self):
        print("Training is done")

if __name__ == '__main__':
    CreditClassifierTrain()
