import logging
from io import StringIO
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(r'logs/final_model_trainer.log', mode='w')
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        log_format = '%(name)s:%(levelname)s %(asctime)s -  %(message)s'
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        pd.options.display.max_columns = 100
        pd.options.display.max_rows = 500
        pd.options.display.width = 0
        pd.options.display.float_format = '{:,.2f}'.format
        pd.set_option('mode.chained_assignment', None)
        self.train = pd.read_csv('.csv files/train_processed.csv', index_col=0)
        self.test = pd.read_csv('.csv files/test_processed.csv', index_col=0)
        self.test_p = pd.read_csv('.csv files/test.csv')
        self.prediction = self.test_p[['Id', 'idhogar']]
        scorer = make_scorer(f1_score, average='macro', eval_metric='mlogloss')
        self.gb = xgb.XGBClassifier(eval_metric=scorer, silent=True, random_state=42)
        self.buf1 = StringIO()
        self.buf2 = StringIO()

    def start(self):
        self.logger.debug('Starting Class')
        self.df_current_state(self.train, self.buf1, 'train')
        self.df_current_state(self.test, self.buf2, 'test')
        self.create_x_and_y_dfs()
        self.make_train_test_split()
        self.train_model()
        self.make_prediction()
        self.create_prediction_csv()
        self.create_prediction_heatmap()
        self.start_shap()
        self.summary_plots()
        self.logger.debug('Closing Class')

    def df_current_state(self, df, buf, name):
        self.logger.debug(f"Current {name}.head()\n{df.head()}")
        df.info(buf=buf)
        self.logger.debug(f"Current {name}.info()\n{buf.getvalue()}")
        self.logger.debug(f"Current {name}.describe()\n{df.describe(include='all')}")

    def create_x_and_y_dfs(self):
        self.logger.debug('Creating X and y dataframes')
        self.y = self.train['Target']
        self.X = self.train.drop('Target', axis=1)

    def make_train_test_split(self):
        self.logger.debug('Creating train test split')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                stratify=self.y, random_state=42)
        self.y_test = self.y_test.to_numpy()

    def train_model(self):
        self.logger.debug('Training model')
        self.gb = xgb.XGBClassifier(colsample_bytree=0.3453022274731309, learning_rate=0.033555500186925075,
                                    max_depth=72, n_estimators=125, reg_alpha=19.675438226237613,
                                    subsample=0.5228779006843063, eval_metric='mlogloss',
                                    verbosity=0, random_state=42)
        self.gb.fit(self.X_train, self.y_train)

    def make_prediction(self):
        self.logger.debug('Predicting test df')
        self.y_pred = self.gb.predict(self.test)

    def create_prediction_csv(self):
        for i in range(0, 18):
            self.y_pred = np.append(self.y_pred, 4)
        self.prediction['Target'] = 0
        n_idho = self.prediction['idhogar'].unique()
        vi = 0
        for i in n_idho:
            self.prediction.loc[self.prediction['idhogar'] == i, 'Target'] = self.y_pred[vi]
            vi += 1
        self.prediction = self.prediction.drop('idhogar', axis=1)
        self.prediction.to_csv(".csv files/test_prediction.csv", index=False)

    def create_prediction_heatmap(self):
        y_pred_conf = self.gb.predict(self.X_test)
        labels_cm = ['ExtermePoverty', 'ModeratePoverty', 'Vulnerable', 'NonVulnerable']
        cm = confusion_matrix(self.y_test, y_pred_conf)
        df_cm = pd.DataFrame(cm, index=[i for i in labels_cm], columns=[i for i in labels_cm])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap='YlGnBu', ax=ax)
        plt.savefig('plots/prediction_heatmap.png')
        plt.figure().clear()

    def start_shap(self):
        shap.initjs()
        explainer = shap.TreeExplainer(self.gb)
        self.shap_values = explainer.shap_values(self.X_train)

    def summary_plots(self):
        plt.title('ExtermePoverty')
        shap.summary_plot(self.shap_values[0], self.X_train, show=False, plot_size=(16, 6))
        plt.savefig('plots/extpov_summary.png')
        plt.clf()
        plt.title('ModeratePoverty')
        shap.summary_plot(self.shap_values[1], self.X_train, show=False, plot_size=(16, 6))
        plt.savefig('plots/modpov_summary.png')
        plt.clf()
        plt.title('Vulnerable')
        shap.summary_plot(self.shap_values[2], self.X_train, show=False, plot_size=(16, 6))
        plt.savefig('plots/vulnerable_summary.png')
        plt.clf()
        plt.title('NonVulnerable')
        shap.summary_plot(self.shap_values[3], self.X_train, show=False, plot_size=(16, 6))
        plt.savefig('plots/nonvul_summary.png')
        plt.clf()
