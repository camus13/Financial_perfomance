
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np

class Data_format():
    def prepare_data(self, data):
        """ Предобработка данных: заполнение пропущенных дат, замена NaN на последние найденные значения.
            
            Для удобства дальнейших вычислений к дате с курсами валют добавлен курс доллара к доллару и
            timestamp для выделения промежутка времени
        """
        
        data = data.rename(index=str, columns={'Unnamed: 0': 'date'}).reset_index(drop=True)

        list_all_date = []
        for i in pd.date_range(start=data['date'][0], end=list(data['date'])[-1], freq='D'):
            list_all_date.append(i.strftime('%Y-%m-%d'))

        if len(data) != len(list_all_date):
            for ind in range(0, len(list_all_date)):

                if str(data['date'][ind]) != list_all_date[ind]:
                    data.index = list(np.arange(0, ind)) + list(np.arange(ind, len(data)) + 1)
                    data.loc[ind] = [list_all_date[ind]] + [np.nan for i in range(0, len(data.columns) - 1)]
                    data = data.sort_index()
                    
        if 'EUR' in data.columns:
            data['USD'] = [1 for i in range(0, len(data))]
            data['timestamp'] = [int(i.replace('-', '')) for i in data['date']]
        else:
            data['timestamp'] = [int(i.replace('-', '')) for i in data['date']]

        for t in data.index[1:]:                                        # Цикл по строкам
            for i in range(1, len(data.columns) - 1):                   # Цикл по столбцам
                col = data.columns[i]

                if np.isnan(data[col][t - 1]) and t == 1:
                    valid_index = data[col].first_valid_index()      # Индекс отличного от NaN значения в столбце
                    data.loc[t - 1, col] = float(data[col][valid_index])
                if np.isnan(data[col][t]):                              # Проверка значения на NaN
                    valid_index = data[col][:t].last_valid_index()      # Индекс отличного от NaN значения в столбце
                    data.loc[t, col] = float(data[col][valid_index])

        return data


# In[6]:


class Perfomance():
    
    def __init__(self, data_prices, data_currencies, data_exchanges, data_weights):
        self.data_prices = data_prices
        self.data_currencies = data_currencies.rename(index=str, columns={'Unnamed: 0': 'currency_type'}).reset_index(drop=True)
        self.data_exchanges = data_exchanges
        self.data_weights = data_weights
        self.dicts_currencies = dict(zip(self.data_currencies['currency_type'], self.data_currencies['currency']))
    
    @staticmethod
    def calculate_total_t_i(data1, data2=None, dicts_currencies=None):
        
        # Формула 1,2,3 из тестоvого задания
        
        Matrix_t_i = np.zeros((len(data1), len(data1.columns) - 1)) 
        
        if data2 is not None:
            data_all = pd.merge(data1, data2, on=['date'])
        else:
            data_all = data1

        for t in data_all.index[1:]:                                      
            for i in range(1, len(data1.columns) - 1):
                col = data1.columns[i]

                if data2 is not None:
                    c_t_i = data_all[dicts_currencies[col]][t]
                    c_t_1_i = data_all[dicts_currencies[col]][t - 1]
                else:
                    c_t_i = 1
                    c_t_1_i = 1
                
                p_t_i = data_all[col][t]
                p_t_1_i = data_all[col][t - 1]
     
                Matrix_t_i[t][i] = 1/(c_t_1_i * p_t_1_i) * (c_t_i * p_t_i  - c_t_1_i * p_t_1_i)
        
        Matrix_t_i = pd.DataFrame(Matrix_t_i[1:, 1:], columns=data1.columns[1: len(data1.columns) - 1])
        Matrix_t_i['date'] = data_all['date'][1:].reset_index(drop=True)
        
        return Matrix_t_i
    
    @staticmethod
    def calculate_asset_total_t(data_weights, M_t_i):
        
        # Формула 4 и 6
        
        data_all = pd.merge(data_weights, M_t_i, on='date')

        M_t = np.zeros(len(data_all))


        for t in data_all.index[1:]:
            r_t = 0
            for i in range(1, len(data_weights.columns) - 1):
                col_w = data_weights.columns[i] + '_x'
                col_r = data_weights.columns[i] + '_y'

                r_t += data_all[col_w][t] * data_all[col_r][t]

            M_t[t] = r_t
        
        return M_t

    @staticmethod
    def calculate_currency_total_t(data_weights, CR_t_i, dicts_currencies=None):
        
        # Формула 5
        
        data_all = pd.merge(data_weights, CR_t_i, on='date')
    
        CR_t = np.zeros(len(data_all))

        for t in data_all.index[1:]:
            cr_t = 0
            for i in range(1, len(data_prices.columns) - 1):
                col_w = data_prices.columns[i]
                col_cr = dicts_currencies[col_w]
                cr_t += data_all[col_w][t] * data_all[col_cr][t]

            CR_t[t] = cr_t

        return CR_t
    
    @staticmethod
    def calculate_P(R_t):
        
        # Формула 7, 8, 9
        
        P_t = np.ones(len(R_t))

        for t in range(1, len(R_t)):
            P_t[t] = P_t[t - 1] * (1 + R_t[t])
        return pd.Series(P_t)

    def calculate_asset_pefromance(self, start_date, end_date):
        start_date = int(start_date.replace('-', ''))
        end_date = int(end_date.replace('-', ''))
        self.data_prices = self.data_prices[(self.data_prices['timestamp'] > start_date) & (self.data_prices['timestamp'] < end_date)]
        self.data_weights = self.data_weights[(self.data_weights['timestamp'] > start_date) & (self.data_weights['timestamp'] < end_date)]
        self.data_prices.reset_index(inplace=True, drop=True)
        self.data_weights.reset_index(inplace=True, drop=True)
        self.R_t_i = Perfomance.calculate_total_t_i(self.data_prices)
        self.R_t = Perfomance.calculate_asset_total_t(self.data_weights, self.R_t_i)
        self.P_t = Perfomance.calculate_P(self.R_t)
        return self.P_t
    def calculate_currency_pefromance(self, start_date, end_date):
        start_date = int(start_date.replace('-', ''))
        end_date = int(end_date.replace('-', ''))
        self.data_exchanges = self.data_exchanges[(self.data_exchanges['timestamp'] > start_date) & (self.data_exchanges['timestamp'] < end_date)]
        self.data_weights = self.data_weights[(self.data_weights['timestamp'] > start_date) & (self.data_weights['timestamp'] < end_date)]
        self.data_exchanges.reset_index(inplace=True, drop=True)
        self.data_weights.reset_index(inplace=True, drop=True)
        self.CR_t_i = Perfomance.calculate_total_t_i(self.data_exchanges, dicts_currencies=self.dicts_currencies)
        self.CR_t= Perfomance.calculate_currency_total_t(self.data_weights, self.CR_t_i, dicts_currencies=self.dicts_currencies)
        self.CP_t = Perfomance.calculate_P(self.CR_t)
        return self.CP_t
    def calculate_total_pefromance(self, start_date, end_date):
        start_date = int(start_date.replace('-', ''))
        end_date = int(end_date.replace('-', ''))
        self.data_prices = self.data_prices[(self.data_prices['timestamp'] > start_date) & (self.data_prices['timestamp'] < end_date)]
        self.data_exchanges = self.data_exchanges[(self.data_exchanges['timestamp'] > start_date) & (self.data_exchanges['timestamp'] < end_date)]
        self.data_weights = self.data_weights[(self.data_weights['timestamp'] > start_date) & (self.data_weights['timestamp'] < end_date)]
        self.data_prices.reset_index(inplace=True, drop=True)
        self.data_exchanges.reset_index(inplace=True, drop=True)
        self.data_weights.reset_index(inplace=True, drop=True)
        self.TR_t_i = Perfomance.calculate_total_t_i(self.data_prices, self.data_exchanges, dicts_currencies=self.dicts_currencies)
        self.TR_t = Perfomance.calculate_asset_total_t(self.data_weights, self.TR_t_i)
        self.TP_t = Perfomance.calculate_P(self.TR_t)
        return self.TP_t


# In[12]:


if __name__ == '__main__':
    data_prices = pd.read_csv('prices.csv')
    data_prices = Data_format().prepare_data(data_prices)
    data_currencies = pd.read_csv('currencies.csv')
    data_exchanges = pd.read_csv('exchanges.csv')
    data_exchanges = Data_format().prepare_data(data_exchanges)
    data_weights = pd.read_csv('weights.csv')
    data_weights = Data_format().prepare_data(data_weights)
    test = Perfomance(data_prices, data_currencies, data_exchanges, data_weights)
    P_t = test.calculate_asset_pefromance('2014-01-13', '2014-05-19')
    CP_t = test.calculate_currency_pefromance('2014-01-13', '2014-05-19')
    TP_t = test.calculate_total_pefromance('2014-01-13', '2014-05-19')
    

