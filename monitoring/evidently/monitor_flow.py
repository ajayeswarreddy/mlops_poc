from metaflow import FlowSpec, step, Flow, conda_base, card
# @conda_base(libraries={'numpy':'1.23.5', 'pandas':'1.4.2', 'evidently':'0.3.1', 'nltk':'3.8.1'}, python='3.9.16')

class CreditClassifierMonitor(FlowSpec):
    '''
    Model monitoring flow
    From most recent run of RedditClassifierTrain, load the reference data (both raw and features)
    From most recent run of RedditClassifierScore, load the current data (both raw and features)
    We will ignore labels for now
    Run two types of monitoring, one for text data, and one for embeddings
    Note: evidently works with pandas dataFrames only
    '''

    @step
    def start(self):
        print("Start flow")
        self.next(self.load_data)

    @step
    def load_data(self):
        import pandas as pd
        print("Loading reference and test data")
        flow_train = Flow('CreditClassifierTrain').latest_run
        flow_score = Flow('CreditClassifierScore').latest_run

        self.ref_set = pd.DataFrame(flow_train.data.x_train_clean, columns= ['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','type'])
        self.curr_set = pd.DataFrame(flow_score.data.x_prod_features, columns= ['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','type'])
        self.ref_scores = pd.DataFrame(flow_train.data.train_scores[:,1], columns= ['prediction'])
        self.curr_scores = pd.DataFrame(flow_score.data.prod_scores[:,1], columns= ['prediction'])
        self.ref_set_labels = pd.DataFrame(flow_train.data.y_train, columns=['target'])
        self.curr_set_labels = pd.DataFrame(flow_score.data.y_prod_labels, columns=['target'])

        print("All ref and curr data loaded")

        # For monitoring labels we will likely have a separate data store for labels
        self.next(self.process)

    @step
    def process(self):
        import pandas as pd

        print("Creating final ref and curr sets for monitoring")
        self.ref_set = pd.concat([self.ref_set, self.ref_scores, self.ref_set_labels], axis='columns')
        self.curr_set = pd.concat([self.curr_set, self.curr_scores, self.curr_set_labels], axis='columns')

        self.next(self.monitor)

    @card(type='html')
    @step
    def monitor(self):
        from evidently.test_preset import DataStabilityTestPreset
        from evidently.test_suite import TestSuite
        from evidently.report import Report
        from evidently.metric_preset import TextOverviewPreset,ClassificationPreset, DataDriftPreset, TargetDriftPreset
        from evidently.metrics import DatasetDriftMetric, ClassificationQualityMetric
        from evidently import ColumnMapping
        from evidently.descriptors import NonLetterCharacterPercentage, TextLength
        
        from helper_functions import get_evidently_html
        from datetime import datetime
        
        td = datetime.today().strftime('%Y-%m-%d')

        print("Monitoring: data quality tests")
        
        #ClassificationPreset - has lot of clasification results included as a suite
        metrics_report = Report(metrics=[
            ClassificationPreset()
            ]
        )
        
        metrics_report1 = Report(metrics=[
            TargetDriftPreset()
            ]
        )
        
        metrics_report2 = Report(metrics=[
            DataDriftPreset()
            ]
        )


        # metrics_report.run(reference_data=self.ref_set, current_data=self.curr_set)
        # self.html = get_evidently_html(metrics_report)
        # metrics_report.save_html(f"{td}_classification_preset_metrics_results.html")
        
        # metrics_report1.run(reference_data=self.ref_set, current_data=self.curr_set)
        # self.html1 = get_evidently_html(metrics_report1)
        # metrics_report1.save_html(f"{td}_target_drift_metrics_results.html")
        
        # metrics_report2.run(reference_data=self.ref_set, current_data=self.curr_set)
        # self.html2 = get_evidently_html(metrics_report2)
        # metrics_report2.save_html(f"{td}_datadrift_metrics_results.html")
        
        #test suites
        data_stability = TestSuite(tests=[
            DataStabilityTestPreset(),
            ])

        data_stability.run(reference_data=self.ref_set, current_data=self.curr_set)
        self.html3 = get_evidently_html(data_stability)
        data_stability.save_html(f"{td}_data_stability_test_suite_results.html")

        self.next(self.end)
        

    @step
    def end(self):
        print("Flow completed")

if __name__ == '__main__':
    CreditClassifierMonitor()
