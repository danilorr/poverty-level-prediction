import logging
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# noinspection PyUnresolvedReferences
class DataAnalyser:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(r'logs/exploratory_data_analyser.log', mode='w')
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        log_format = '%(name)s:%(levelname)s %(asctime)s -  %(message)s'
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        pd.options.display.max_columns = 100
        pd.options.display.max_rows = 500
        pd.options.display.width = 0
        pd.options.display.float_format = '{:,.2f}'.format
        self.train = pd.read_csv('.csv files/train.csv')
        self.buf1 = StringIO()
        self.buf2 = StringIO()

    def start(self):
        self.logger.debug('Starting Class')
        self.df_current_state(self.buf1)
        self.have_refrig_plot()
        self.have_tablets_plot()
        self.number_tablets_plot()
        self.have_computer_plot()
        self.have_television_plot()
        self.have_mobile_phone_plot()
        self.number_mobile_phones_plot()
        self.sumelectronics_plot()
        self.overcrowd_heatmap()
        self.logger.debug('Closing Class')

    def df_current_state(self, buf):
        self.logger.debug(f"Current train.head()\n{self.train.head()}")
        self.train.info(buf=buf)
        self.logger.debug(f"Current train.info()\n{buf.getvalue()}")
        self.logger.debug(f"Current train.describe()\n{self.train.describe(include='all')}")

    def plot_frac(self, col_plot, x_vals, x_label, gtype, name):
        # Creates the dataframe that will be filled with values to be plotted
        df_plot = pd.DataFrame()
        # Defines range used in the for loop
        loop_range = x_vals
        # Loop that fills the df_plot
        for i in loop_range:
            temp_train = self.train[self.train[col_plot] == i]
            temp_val = temp_train.groupby(['Target']).count()[col_plot] / temp_train[col_plot].count()
            df_plot = df_plot.append(temp_val)
        df_plot.set_index(np.array(range(len(loop_range))), inplace=True)
        df_plot.fillna(0, inplace=True)
        df_plot = df_plot.rename(
            columns={1: 'ExtermePoverty', 2: 'ModeratePoverty', 3: 'Vulnerable', 4: 'NonVulnerable'})
        # Creating aliases to reduce the length of code
        ds1d, ds2d, ds3d, ds4d = df_plot['ExtermePoverty'], df_plot['ModeratePoverty'], df_plot['Vulnerable'], df_plot[
            'NonVulnerable']
        fig, ax = plt.subplots()
        # Creating each of the bars, passing the bottom parameter as the sum of the bars under it
        ax.bar(df_plot.index, ds1d, label='ExtermePoverty')
        ax.bar(df_plot.index, ds2d, label='ModeratePoverty', bottom=ds1d)
        ax.bar(df_plot.index, ds3d, label='Vulnerable', bottom=np.array(ds1d) + np.array(ds2d))
        ax.bar(df_plot.index, ds4d, label='NonVulnerable', bottom=np.array(ds1d) + np.array(ds2d) + np.array(ds3d))
        ax.legend()
        # Set the x-axis to the animal gender
        ax.set_xticklabels(x_label)
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(range(50)))
        plt.xticks(rotation=30)
        plt.yticks(np.linspace(0, 1, 11))
        plt.title('Target Fraction based on ' + gtype)
        plt.ylabel('Fraction of Target')
        fig.set_figheight(8)
        fig.set_figwidth(12)
        plt.savefig(f'plots/{name}_plot.png')

    def have_refrig_plot(self):
        self.plot_frac('refrig', [0, 1], ['Not Have', 'Have'], 'Having Refrigerators', 'have_refrig')

    def have_tablets_plot(self):
        self.plot_frac('v18q', [0, 1], ['Not Have', 'Have'], 'Having Tablets', 'have_tablets')

    def number_tablets_plot(self):
        self.plot_frac('v18q1', range(0, 7), range(0, 7), 'Number of Tablets', 'number_tablets')

    def have_computer_plot(self):
        self.plot_frac('computer', [0, 1], ['Not Have', 'Have'], 'Having Computer', 'have_computer')

    def have_television_plot(self):
        self.plot_frac('television', [0, 1], ['Not Have', 'Have'], 'Having Television', 'have_television')

    def have_mobile_phone_plot(self):
        self.plot_frac('mobilephone', [0, 1], ['Not Have', 'Have'], 'Having Mobile Phone', 'have_mobile_phone')

    def number_mobile_phones_plot(self):
        self.plot_frac('qmobilephone', range(0, 11), range(0, 11), 'Number of Mobile Phones', 'number_mobile_phones')

    def sumelectronics_plot(self):
        self.train['sumelectronics'] = self.train[['qmobilephone', 'v18q1', 'computer']].apply(np.sum, axis=1)
        self.plot_frac('sumelectronics', range(0, 15), range(0, 15), 'Number of Electronics', 'sumelectronics')

    def overcrowd_heatmap(self):
        overcrowd_df = self.train[['bedrooms', 'hacdor', 'rooms', 'hacapo',
                                   'tamviv', 'tamhog', 'overcrowding', 'Target']]
        corr_matrix = overcrowd_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, ax=ax)
        plt.savefig('plots/overcrowd_heatmap.png')
