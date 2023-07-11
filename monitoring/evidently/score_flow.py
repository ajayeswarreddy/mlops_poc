from metaflow import FlowSpec, Flow, step, conda_base, Parameter
# @conda_base(libraries={'numpy':'1.23.5', 'scikit-learn':'1.2.2', 'pandas':'1.4.2'}, python='3.9.16')

class CreditClassifierScore(FlowSpec):
    '''
    Model scoring flow
    Load production data
    Load fitted model from the most recent run of RedditClassifierTrain
    Score data
    Persist production data (both raw and tfidf features)
    Persist predictions as a dataFrame
    '''

    file_name = Parameter('file_name', default = '../../data/credit_card_production_data.csv')

    @step
    def start(self):
        print("Flow is starting")
        self.next(self.load_data)

    @step
    def load_data(self):
        import pandas as pd
        import numpy
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        flow = Flow('CreditClassifierTrain').latest_run

        

        print("Data is loading")

        ''' 
        At this point you would load the new production data from wherever it lives. Our "production" data
        is in a csv file called 20230430_reddit.csv
        '''
        x_prod = pd.read_csv(self.file_name)
        
        print("Data is loaded")
        
        print("Creating features")
        y_prod = x_prod['isFraud'].values
        x_prod = x_prod[['type','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']]
        
        le = LabelEncoder()
        x_prod['type_encoded'] = le.fit_transform(x_prod['type'])
        x_prod = x_prod.drop(['type'],axis=1)
        
        scaler = StandardScaler()
        x_prod = scaler.fit_transform(x_prod)

        self.x_prod_features = x_prod
        self.y_prod_labels = y_prod
        print("Features created")
        self.next(self.load_model)

        
    @step
    def load_model(self):

        print('Loading model')
        flow = Flow('CreditClassifierTrain').latest_run
        self._lr_model = flow.data.lr
        print('Model loaded')
        self.next(self.predict_prob)

    @step
    def predict_prob(self):
        print("Making predictions")
        self.prod_scores = self._lr_model.predict_proba(self.x_prod_features)
        print("Predictions made", self._lr_model.score(self.x_prod_features,self.y_prod_labels))
        self.next(self.end)
        
    
    @step
    def end(self):
        print("Flow is ending")

if __name__ == '__main__':
    CreditClassifierScore()
