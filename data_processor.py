import logging
from io import StringIO
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# noinspection SpellCheckingInspection
class DataProcessor:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(r'logs/data_processor.log', mode='w')
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
        self.test = pd.read_csv('.csv files/test.csv')
        self.df_list = [self.train, self.test]
        self.buf1 = StringIO()
        self.buf2 = StringIO()

    def start(self):
        self.logger.debug('Starting Class')
        self.df_current_state(self.train, self.buf1, 'train')
        self.df_current_state(self.test, self.buf2, 'test')
        for df in self.df_list:
            self.fill_nas(df)
            self.drop_squared_features(df)
            self.join_columns_as_ordinal(df)
            self.remove_area_redundancy(df)
            self.create_dependencynew_feature(df)
            self.create_headescolari_feature(df)
            self.create_sumelectronics_feature(df)
            self.drop_high_correlated_features(df)
            self.drop_individual_features(df)
            self.drop_low_corr_with_target_features(df)
        self.drop_not_househead_rows()
        self.create_train_df_csv()
        self.create_test_df_csv()
        self.logger.debug('Closing Class')

    def df_current_state(self, df, buf, name):
        self.logger.debug(f"Current {name}.head()\n{df.head()}")
        df.info(buf=buf)
        self.logger.debug(f"Current {name}.info()\n{buf.getvalue()}")
        self.logger.debug(f"Current {name}.describe()\n{df.describe(include='all')}")

    def fill_v2a1_na(self, df):
        df.loc[(df['tipovivi1'] == 1), 'v2a1'] = 0
        df['v2a1-missing'] = df['v2a1'].isnull()
        df['v2a1'].fillna(0, inplace=True)

    def fill_v18q1_na(self, df):
        df['v18q1'] = df['v18q1'].fillna(0)

    def fill_rez_esc_na(self, df):
        df.loc[((df['age'] > 19) | (df['age'] < 7)) & (df['rez_esc'].isnull()), 'rez_esc'] = 0
        df['rez_esc-missing'] = df['rez_esc'].isnull()
        df['rez_esc'].fillna(0, inplace=True)
        df.loc[df['rez_esc'] > 5, 'rez_esc'] = 5

    def fill_meaneduc_na(self, df):
        df['meaneduc'] = df['meaneduc'].fillna(0)

    def fill_sqbmeaned_na(self, df):
        df['SQBmeaned'] = df['SQBmeaned'].fillna(0)

    def fill_nas(self, df):
        self.fill_v2a1_na(df)
        self.fill_v18q1_na(df)
        self.fill_rez_esc_na(df)
        self.fill_meaneduc_na(df)
        self.fill_sqbmeaned_na(df)

    def drop_squared_features(self, df):
        sqbcol = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding',
                  'SQBdependency', 'SQBmeaned', 'agesq']
        df.drop(sqbcol, axis=1, inplace=True)

    def join_columns_as_ordinal_base(self, dic, colname, df):
        for i in dic:
            df.loc[df[i] == 1, colname] = dic[i]
            df.drop(i, axis=1, inplace=True)

    def join_columns_as_ordinal_walltype(self, df):
        df['walltype'] = 0
        wall_dt = {
            'paredother': 1,
            'pareddes': 1,
            'paredfibras': 1,
            'paredzinc': 2,
            'paredzocalo': 3,
            'paredmad': 4,
            'paredpreb': 5,
            'paredblolad': 6
        }
        self.join_columns_as_ordinal_base(wall_dt, 'walltype', df)

    def join_columns_as_ordinal_wallquality(self, df):
        df['wallquality'] = 0
        wallq_dt = {
            'epared1': 1,
            'epared2': 2,
            'epared3': 3,
        }
        self.join_columns_as_ordinal_base(wallq_dt, 'wallquality', df)

    def join_columns_as_ordinal_floortype(self, df):
        df['floortype'] = 0
        floor_dt = {
            'pisonotiene': 1,
            'pisoother': 2,
            'pisonatur': 2,
            'pisocemento': 3,
            'pisomadera': 4,
            'pisomoscer': 5,
        }
        self.join_columns_as_ordinal_base(floor_dt, 'floortype', df)

    def join_columns_as_ordinal_floorqual(self, df):
        df['floorqual'] = 0
        floorq_dt = {
            'eviv1': 1,
            'eviv2': 2,
            'eviv3': 3
        }
        self.join_columns_as_ordinal_base(floorq_dt, 'floorqual', df)

    def join_columns_as_ordinal_rooftype(self, df):
        df['nothaveroof'] = 0
        df.loc[df['cielorazo'] == 0, 'nothaveroof'] = 1
        df.drop('cielorazo', axis=1, inplace=True)
        df['rooftype'] = 0
        roof_dt = {
            'nothaveroof': 1,
            'techootro': 2,
            'techocane': 2,
            'techoentrepiso': 3,
            'techozinc': 4
        }
        self.join_columns_as_ordinal_base(roof_dt, 'rooftype', df)

    def join_columns_as_ordinal_roofqual(self, df):
        df['roofqual'] = 0
        roofq_dt = {
            'etecho1': 1,
            'etecho2': 2,
            'etecho3': 3,
        }
        self.join_columns_as_ordinal_base(roofq_dt, 'roofqual', df)

    def join_columns_as_ordinal_waterprov(self, df):
        df['waterprov'] = 0
        water_dt = {
            'abastaguano': 1,
            'abastaguafuera': 2,
            'abastaguadentro': 3
        }
        self.join_columns_as_ordinal_base(water_dt, 'waterprov', df)

    def join_columns_as_ordinal_elecsource(self, df):
        df['elecsource'] = 0
        elec_dt = {
            'noelec': 1,
            'planpri': 2,
            'coopele': 2,
            'public': 3,
        }
        self.join_columns_as_ordinal_base(elec_dt, 'elecsource', df)

    def join_columns_as_ordinal_toiletdwel(self, df):
        df['toiletdwel'] = 0
        toilet_dt = {
            'sanitario1': 1,
            'sanitario5': 2,
            'sanitario6': 3,
            'sanitario3': 3,
            'sanitario2': 4
        }
        self.join_columns_as_ordinal_base(toilet_dt, 'toiletdwel', df)

    def join_columns_as_ordinal_cookingsource(self, df):
        df['cookingsource'] = 0
        cook_dt = {
            'energcocinar1': 1,
            'energcocinar4': 2,
            'energcocinar3': 3,
            'energcocinar2': 4,
        }
        self.join_columns_as_ordinal_base(cook_dt, 'cookingsource', df)

    def join_columns_as_ordinal_rubbishdisp(self, df):
        df['rubbishdisp'] = 0
        rubbish_dt = {
            'elimbasu6': 1,
            'elimbasu5': 2,
            'elimbasu4': 1,
            'elimbasu3': 3,
            'elimbasu2': 1,
            'elimbasu1': 4
        }
        self.join_columns_as_ordinal_base(rubbish_dt, 'rubbishdisp', df)

    def join_columns_as_ordinal_houseowned(self, df):
        df.rename(columns={'tipovivi4': 'isprecarious'}, inplace=True)
        df['houseowned'] = 0
        houseown_dt = {
            'tipovivi5': 0,
            'tipovivi3': 0,
            'tipovivi2': 1,
            'tipovivi1': 1,
        }
        self.join_columns_as_ordinal_base(houseown_dt, 'houseowned', df)

    def join_columns_as_ordinal_region(self, df):
        df['region'] = 0
        region_dt = {
            'lugar1': 1,
            'lugar2': 2,
            'lugar3': 3,
            'lugar4': 4,
            'lugar5': 5,
            'lugar6': 6
        }
        self.join_columns_as_ordinal_base(region_dt, 'region', df)

    def join_columns_as_ordinal_civilstate(self, df):
        df['civilstate'] = 0
        civil_dt = {
            'estadocivil1': 1,
            'estadocivil2': 3,
            'estadocivil3': 4,
            'estadocivil4': 5,
            'estadocivil5': 5,
            'estadocivil6': 6,
            'estadocivil7': 2
        }
        self.join_columns_as_ordinal_base(civil_dt, 'civilstate', df)

    def join_columns_as_ordinal_education(self, df):
        df['education'] = 0
        educ_dt = {
            'instlevel1': 1,
            'instlevel2': 2,
            'instlevel3': 3,
            'instlevel4': 4,
            'instlevel5': 5,
            'instlevel6': 6,
            'instlevel7': 7,
            'instlevel8': 8,
            'instlevel9': 9
        }
        self.join_columns_as_ordinal_base(educ_dt, 'education', df)

    def join_columns_as_ordinal(self, df):
        self.join_columns_as_ordinal_walltype(df)
        self.join_columns_as_ordinal_wallquality(df)
        self.join_columns_as_ordinal_floortype(df)
        self.join_columns_as_ordinal_floorqual(df)
        self.join_columns_as_ordinal_rooftype(df)
        self.join_columns_as_ordinal_roofqual(df)
        self.join_columns_as_ordinal_waterprov(df)
        self.join_columns_as_ordinal_elecsource(df)
        self.join_columns_as_ordinal_toiletdwel(df)
        self.join_columns_as_ordinal_cookingsource(df)
        self.join_columns_as_ordinal_rubbishdisp(df)
        self.join_columns_as_ordinal_houseowned(df)
        self.join_columns_as_ordinal_region(df)
        self.join_columns_as_ordinal_civilstate(df)
        self.join_columns_as_ordinal_education(df)

    def remove_area_redundancy(self, df):
        df.rename(columns={'area1': 'isurban'}, inplace=True)
        df.drop('area2', axis=1, inplace=True)

    def create_dependencynew_feature(self, df):
        df['n_depend'] = 0
        df['n_indep'] = 0
        n_dep = df[['age', 'idhogar']].loc[(df['age'] < 19) |
                                           (df['age'] > 64)].groupby('idhogar').count()
        n_ind = df[['age', 'idhogar']].loc[(df['age'] >= 19) &
                                           (df['age'] <= 64)].groupby('idhogar').count()
        for i in n_dep.index:
            df.loc[df['idhogar'] == i, 'n_depend'] = int(n_dep.loc[n_dep.index == i, 'age'].values)
        for i in n_ind.index:
            df.loc[df['idhogar'] == i, 'n_indep'] = int(n_ind.loc[n_ind.index == i, 'age'].values)
        df['dependencynew'] = df['n_depend'] / df['n_indep']
        df.loc[df['dependencynew'] == np.inf, 'dependencynew'] = 10
        df.drop(['n_depend', 'n_indep', 'dependency'], axis=1, inplace=True)

    def create_headescolari_feature(self, df):
        df['headescolari'] = 0
        df.loc[df['parentesco1'] == 1, 'headescolari'] = df['escolari']
        df.drop(['edjefe', 'edjefa'], axis=1, inplace=True)

    def create_sumelectronics_feature(self, df):
        df['sumelectronics'] = df[['qmobilephone', 'v18q1', 'computer']].apply(np.sum, axis=1)

    def drop_high_correlated_features(self, df):
        drop_cols = ['r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t2']
        df.drop(drop_cols, axis=1, inplace=True)
        drop_cols = ['refrig', 'v18q', 'v18q1', 'computer', 'television', 'mobilephone', 'qmobilephone']
        df.drop(drop_cols, axis=1, inplace=True)
        drop_cols = ['hhsize', 'hogar_total']
        df.drop(drop_cols, axis=1, inplace=True)
        drop_cols = ['bedrooms', 'hacdor', 'rooms', 'hacapo', 'tamviv', 'tamhog']
        df['overcrowding'] = df['overcrowding'].apply(lambda x: round(x, 1))
        df.drop(drop_cols, axis=1, inplace=True)
        drop_cols = ['hogar_nin', 'hogar_adul', 'hogar_mayor']
        df.drop(drop_cols, axis=1, inplace=True)

    def drop_individual_features(self, df):
        drop_cols = ['parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7',
                     'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12']
        df.rename(columns={'parentesco1': 'ishousehead'}, inplace=True)
        df.drop(drop_cols, axis=1, inplace=True)
        drop_cols = ['escolari', 'rez_esc', 'rez_esc-missing', 'dis', 'male', 'female', 'age', 'civilstate',
                     'education']
        df.drop(drop_cols, axis=1, inplace=True)

    def drop_low_corr_with_target_features(self, df):
        drop_cols = ['cookingsource', 'v2a1', 'toiletdwel', 'rubbishdisp', 'isurban', 'waterprov', 'v14a', 'houseowned',
                     'rooftype', 'elecsource', 'r4t3', 'v2a1-missing', 'Id', 'idhogar']
        df.drop(drop_cols, axis=1, inplace=True)
        df.rename(columns={'r4t1': 'numchilds', 'dependencynew': 'dependency'}, inplace=True)

    def drop_not_househead_rows(self):
        self.train = self.train.loc[self.train['ishousehead'] == 1]
        self.train.drop('ishousehead', axis=1, inplace=True)
        self.test = self.test.loc[self.test['ishousehead'] == 1]
        self.test.drop('ishousehead', axis=1, inplace=True)

    def create_train_df_csv(self):
        self.logger.debug('Creating train.csv')
        self.train.to_csv(r'.csv files/train_processed.csv')

    def create_test_df_csv(self):
        self.logger.debug('Creating test.csv')
        self.test.to_csv(r'.csv files/test_processed.csv')
