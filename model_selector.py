import logging
from io import StringIO
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, make_scorer
import warnings
warnings.filterwarnings('ignore')


class ModelSelector:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(r'logs/model_selector.log', mode='w')
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        log_format = '%(name)s:%(levelname)s %(asctime)s -  %(message)s'
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        pd.options.display.max_columns = 100
        pd.options.display.max_rows = 500
        pd.options.display.width = 0
        pd.options.display.float_format = '{:,.2f}'.format
        self.train = pd.read_csv('.csv files/train_processed.csv', index_col=0)
        self.test = pd.read_csv('.csv files/test_processed.csv', index_col=0)
        scorer = make_scorer(f1_score, average='macro', eval_metric='mlogloss')
        self.scaler = MinMaxScaler()
        self.kneigh = KNeighborsClassifier()
        self.dectree = DecisionTreeClassifier(random_state=42)
        self.forest = RandomForestClassifier(random_state=42)
        self.adab = AdaBoostClassifier(random_state=42)
        self.gb = xgb.XGBClassifier(eval_metric=scorer, verbosity=0, random_state=42)
        self.buf1 = StringIO()
        self.buf2 = StringIO()

    def start(self):
        self.logger.debug('Starting Class')
        self.df_current_state(self.train, self.buf1, 'train')
        self.df_current_state(self.test, self.buf2, 'test')
        self.create_x_and_y_dfs()
        self.make_train_test_split()
        self.test_kneighbor()
        self.test_dectree()
        self.test_random_forest()
        self.test_adab()
        self.test_xgb()
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

    def bayes_search(self, model, param_grid):
        n_iter = 5
        cv = StratifiedKFold(n_splits=n_iter, shuffle=True, random_state=42)
        bsearch = BayesSearchCV(model, param_grid, n_iter=n_iter, scoring='neg_log_loss', cv=cv).fit(
            self.X, self.y)
        self.logger.debug(f"{model}'s best score: {bsearch.best_score_}")
        self.logger.debug(f"{model}'s best parameters: {bsearch.best_params_}")

    def bs_kneighbor(self):
        param_grid = {'n_neighbors': Integer(2, 20),
                      'weights': Categorical(['uniform', 'distance']),
                      'leaf_size': Integer(10, 100)}
        self.bayes_search(self.kneigh, param_grid)

    def test_kneighbor(self):
        self.bs_kneighbor()
        kneigh = KNeighborsClassifier(leaf_size=71, n_neighbors=10, weights='distance')
        kneigh.fit(self.X_train, self.y_train)
        y_pred = kneigh.predict(self.X_test)
        result = f1_score(y_pred, self.y_test, average='macro')
        self.logger.debug(f"Kneighbor's result: {result}")

    def bs_dectree(self):
        param_grid = {'criterion': Categorical(['gini', 'entropy']),
                      'splitter': Categorical(['best', 'random']),
                      'max_depth': Integer(10, 200),
                      'min_samples_split': Integer(5, 50),
                      'max_leaf_nodes': Integer(10, 200),
                      }
        self.bayes_search(self.dectree, param_grid)

    def test_dectree(self):
        self.bs_dectree()
        dectree = DecisionTreeClassifier(max_depth=66, min_samples_split=19, random_state=42)
        dectree.fit(self.X_train, self.y_train)
        y_pred = dectree.predict(self.X_test)
        result = f1_score(y_pred, self.y_test, average='macro')
        self.logger.debug(f"Decision Tree's result: {result}")

    def bs_random_forest(self):
        param_grid = {'n_estimators': Integer(100, 2000),
                      'criterion': Categorical(['gini', 'entropy']),
                      'max_leaf_nodes': Integer(20, 500),
                      'min_samples_split': Integer(5, 50),
                      }
        self.bayes_search(self.forest, param_grid)

    def test_random_forest(self):
        self.bs_random_forest()
        forest = RandomForestClassifier(criterion='entropy', max_leaf_nodes=258, min_samples_split=28,
                                        n_estimators=559, random_state=42)
        forest.fit(self.X_train, self.y_train)
        y_pred = forest.predict(self.X_test)
        result = f1_score(y_pred, self.y_test, average='macro')
        self.logger.debug(f"Random Forest's result: {result}")

    def bs_adab(self):
        param_grid = {'n_estimators': Integer(50, 1000),
                      'learning_rate': Real(0.01, 1, prior='log-uniform')
                      }
        self.bayes_search(self.adab, param_grid)

    def test_adab(self):
        self.bs_adab()
        adab = AdaBoostClassifier(learning_rate=0.06260414650403581, n_estimators=403, random_state=42)
        adab.fit(self.X_train, self.y_train)
        y_pred = adab.predict(self.X_test)
        result = f1_score(y_pred, self.y_test, average='macro')
        self.logger.debug(f"AdaBoost's result: {result}")

    def bs_xgb(self):
        param_grid = {'max_depth': Integer(1, 90),
                      'learning_rate': Real(0.01, 1, prior='log-uniform'),
                      'reg_alpha': Real(0.01, 100),
                      'colsample_bytree': Real(0.2e0, 0.8e0),
                      'subsample': Real(0.2e0, 0.8e0),
                      'n_estimators': Integer(50, 200)}
        self.bayes_search(self.gb, param_grid)

    def test_xgb(self):
        self.bs_xgb()
        gb = xgb.XGBClassifier(colsample_bytree=0.3453022274731309, learning_rate=0.033555500186925075, max_depth=72,
                               n_estimators=125, reg_alpha=19.675438226237613, subsample=0.5228779006843063,
                               eval_metric='mlogloss', random_state=42)
        gb.fit(self.X_train, self.y_train)
        y_pred = gb.predict(self.X_test)
        result = f1_score(y_pred, self.y_test, average='macro')
        self.logger.debug(f"XGBoost's result: {result}")
        self.logger.debug("The best model was XGBoost")
