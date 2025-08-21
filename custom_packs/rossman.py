import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann( object ):
    def __init__( self ):
        self.home_path='' 
        self.rs_competition_distance     = pickle.load(open( self.home_path +  'exports/cicle_products/rs_competition_distance.pkl', 'rb'))
        self.rs_competition_time_month   = pickle.load(open( self.home_path +  'exports/cicle_products/rs_competition_time_month.pkl', 'rb'))
        self.mms_promo_time_week         = pickle.load(open( self.home_path +  'exports/cicle_products/mms_promo_time_week.pkl', 'rb'))
        self.mms_year                    = pickle.load(open( self.home_path +  'exports/cicle_products/mms_year.pkl', 'rb'))
        self.le_store_type               = pickle.load(open( self.home_path +  'exports/cicle_products/le_store_type.pkl', 'rb'))
        self.ohe_state_holiday           = pickle.load(open( self.home_path +  'exports/cicle_products/ohe_state_holiday.pkl', 'rb'))

    def apply_ohe(self, encoder ,encoded_array, variable_name, df):
        '''
        Aplica o One Hot Encoder no dataframe, criando as colunas respectivas às categorias no encoder e elimininado a coluna original.

        Parâmetros:
            encoder (OneHotEncoder): Encoder OneHotEncoder do Scikit-learn pré-configurado.
            encoded_array (ndarray): Array gerado pelo OneHotEncoder.
            variable_name (str): Nome da variável original.
            df (DataFrame): Dataframe a ser alterado.

        Retorna:
            Um novo dataframe com as novas colunas respectivas às categorias no encoder e sem a coluna original.
        '''

        # Criar DataFrame com nomes corretos
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoder.get_feature_names_out([variable_name]),
            index=df.index
        )

        # Concatenar com o DataFrame original, removendo a coluna original
        df_f = pd.concat([df.drop(columns=[variable_name]), encoded_df], axis=1)

        return df_f

    def apply_01 (self, df):
        df_01 = df.copy()

        # Renomeando colunas:
        cols_old = ['Id', 'Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday',
                    'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionDistance',
                    'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
                    'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

        snakecase = lambda x: inflection.underscore( x )

        cols_new = list( map( snakecase, cols_old ) )

        df_01.columns = cols_new


        # Transformando variável 'date' em datetime:
        df_01['date'] = pd.to_datetime( df_01['date'] )


        # Preenchendo dados vazios

        #competition_distance - Transforma os Nan em 200000 (muito maior do que a maior distância máxima no banco de dados):     
        df_01['competition_distance'] = df_01['competition_distance'].apply( lambda x: 200000.0 if math.isnan( x ) else x )

        #competition_open_since_month - Caso seja NA, extrai o mês da coluna 'date':
        df_01['competition_open_since_month'] = df_01.apply( lambda x: x['date'].month if math.isnan( x['competition_open_since_month'] ) else x['competition_open_since_month'], axis=1 )

        #competition_open_since_year - Caso seja NA, extrai o ano da coluna 'date':
        df_01['competition_open_since_year'] = df_01.apply( lambda x: x['date'].year if math.isnan( x['competition_open_since_year'] ) else x['competition_open_since_year'], axis=1 )

        #promo2_since_week - Caso seja NA, extrai a semana da coluna 'date':           
        df_01['promo2_since_week'] = df_01.apply( lambda x: x['date'].week if math.isnan( x['promo2_since_week'] ) else x['promo2_since_week'], axis=1 )

        #promo2_since_year - Caso seja NA, extrai o ano da coluna 'date':           
        df_01['promo2_since_year'] = df_01.apply( lambda x: x['date'].year if math.isnan( x['promo2_since_year'] ) else x['promo2_since_year'], axis=1 )

        #promo_interval - Cria uma coluna 'is_promo' para indicar se está dentro do período de promoção ou não.           
        month_map = {1: 'Jan',  2: 'Fev',  3: 'Mar',  4: 'Apr',  5: 'May',  6: 'Jun',  7: 'Jul',  8: 'Aug',  9: 'Sep',  10: 'Oct', 11: 'Nov', 12: 'Dec'} # Cria o mapa de meses

        df_01['promo_interval'].fillna(0, inplace=True ) # Transforma Nan em 0

        df_01['month_map'] = df_01['date'].dt.month.map( month_map ) # Extrai o mês de 'date' e transforama em letras conforme o mapa

        # Checa se o mês de 'month_map' está contido nos meses de 'promo_interval' e se estiver muda para 1 o valor de 'is_promo' indicando que está no período de promoção
        df_01['is_promo'] = df_01[['promo_interval', 'month_map']].apply(
                                                                        lambda x: 
                                                                        0 if x['promo_interval'] == 0 
                                                                        else 1 if x['month_map'] in x['promo_interval'].split( ',' ) 
                                                                        else 0, axis=1 
                                                                        )


        # Alterando tipagem de dados

        # competiton
        df_01['competition_open_since_month'] = df_01['competition_open_since_month'].astype( int )
        df_01['competition_open_since_year'] = df_01['competition_open_since_year'].astype( int )
            
        # promo2
        df_01['promo2_since_week'] = df_01['promo2_since_week'].astype( int )
        df_01['promo2_since_year'] = df_01['promo2_since_year'].astype( int )

        return df_01

    def apply_02 (self, df):
        df_02 = df.copy()

        # Criando variáveis derivadas

        # year - Nova coluna apenas com o ano da coluna 'date'
        df_02['year'] = df_02['date'].dt.year
        df_02['year'] = np.int64(df_02['year'])

        # month - Nova coluna apenas com o mês da coluna 'date'
        df_02['month'] = df_02['date'].dt.month
        df_02['month'] = np.int64(df_02['month'])

        # day - Nova coluna apenas com o dia da coluna 'date'
        df_02['day'] = df_02['date'].dt.day
        df_02['day'] = np.int64(df_02['day'])

        # week of year - Nova coluna apenas com a semana do ano da coluna 'date'
        df_02['week_of_year'] = df_02['date'].dt.strftime('%W')
        df_02['week_of_year'] = df_02['week_of_year'].astype( int )

        # year week - Nova coluna apenas com semana do ano e o ano da coluna 'date'
        df_02['year_week'] = df_02['date'].dt.strftime( '%Y-%W' )

        # competition since - Converte 'competition_open_since_year' e 'competition_open_since_month' em uma variável com o tempo de existência do concorrente em meses. 
        df_02['competition_since'] = df_02.apply( lambda x: datetime.datetime( year=x['competition_open_since_year'], month=x['competition_open_since_month'],day=1 ), axis=1 )
        df_02['competition_time_month'] = ( ( df_02['date'] - df_02['competition_since'] )/30 ).apply( lambda x: x.days ).astype( int )

        # promo since - Converte 'promo2_since_year' e 'promo2_since_week' em uma variável com o tempo de promoção em semanas. 
        df_02['promo_since'] = df_02['promo2_since_year'].astype( str ) + '-' + df_02['promo2_since_week'].astype( str )
        df_02['promo_since'] = df_02['promo_since'].apply( lambda x: datetime.datetime.strptime( x + '-1', '%Y-%W-%w' ) - datetime.timedelta( days=7 ) )
        df_02['promo_time_week'] = ( ( df_02['date'] - df_02['promo_since'] )/7 ).apply( lambda x: x.days ).astype( int )

        # assortment
        df_02['assortment'] = df_02['assortment'].apply( lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended' )

        # state holiday
        df_02['state_holiday'] = df_02['state_holiday'].apply( lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day' )

        return df_02

    def apply_03 (self, df):
        df_03 = df.copy()

        # Filtrando linhas apenas para dias em que houve vendas:
        df_03 = df_03[(df_03['open'] != 0)]


        # Removendo colunas:
        cols_drop = ['open', 'competition_open_since_year', 'competition_open_since_month', 'promo2_since_year', 'promo2_since_week', 'promo_interval', 'month_map']
        df_03 = df_03.drop( cols_drop, axis=1 )

        return df_03

    def apply_05(self, df):
        df_05 = df.copy()

        # Normalização - Não há dados


        # Rescaling

        # competition_distance
        df_05['competition_distance'] = self.rs_competition_distance.transform( df_05[['competition_distance']].values )

        # competition_time_month
        df_05['competition_time_month'] = self.rs_competition_time_month.transform( df_05[['competition_time_month']].values )

        # promo_time_week
        df_05['promo_time_week'] = self.mms_promo_time_week.transform( df_05[['promo_time_week']].values )

        # year
        df_05['year'] = self.mms_year.transform( df_05[['year']].values )


        # Encoding

        # state_holiday - One Hot Encoding
        encoded_array = self.ohe_state_holiday.transform(df_05[['state_holiday']])
        df_05 = self.apply_ohe(self.ohe_state_holiday, encoded_array, 'state_holiday', df_05)

        # store_type - Label Encoding
        df_05['store_type'] = self.le_store_type.transform( df_05[['store_type']].values )

        # assortment - Ordinal Encoding
        assortment_dict = {'basic': 1,  'extra': 2, 'extended': 3}
        df_05['assortment'] = df_05['assortment'].map( assortment_dict )


        # Transformação de natureza (encoder cíclico)

        # day of week
        df_05['day_of_week_sin'] = df_05['day_of_week'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
        df_05['day_of_week_cos'] = df_05['day_of_week'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )

        # month
        df_05['month_sin'] = df_05['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
        df_05['month_cos'] = df_05['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )

        # day 
        df_05['day_sin'] = df_05['day'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
        df_05['day_cos'] = df_05['day'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )

        # week of year
        df_05['week_of_year_sin'] = df_05['week_of_year'].apply( lambda x: np.sin( x * ( 2. * np.pi/52 ) ) )
        df_05['week_of_year_cos'] = df_05['week_of_year'].apply( lambda x: np.cos( x * ( 2. * np.pi/52 ) ) )


        # Descartando colunas antigas

        cols_drop = ['week_of_year', 'day', 'month', 'day_of_week', 'promo_since', 'competition_since', 'year_week' ]
        df_05 = df_05.drop( cols_drop, axis=1 )

        return df_05

    def apply_06 (self, df):
        df_06 = df.copy()

        cols_selected_boruta = [
                                'id', # Para manter o registro, será descartada antes da aplicação do modelo
                                'store',
                                'promo',
                                'store_type',
                                'assortment',
                                'competition_distance',
                                'promo2',
                                'competition_time_month',
                                'promo_time_week',
                                'day_of_week_sin',
                                'day_of_week_cos',
                                'month_sin',
                                'month_cos',
                                'day_sin',
                                'day_cos'
                                ]

        df_06 = df_06[ cols_selected_boruta ]

        return df_06

    def get_prediction( self, model, original_data, test_data ):

        # Gerando previsões (sem a coluna de id)
        pred = model.predict(test_data.drop(['id'], axis=1))


        # Anexando previsões ao dataframe de teste
        test_data['predictions'] = np.expm1( pred )


        # Anexando previsões ao dataframe original

        # Procurando previsões em test_data e aplicando em original_data com base no id
        pred_map = test_data.set_index('id')['predictions']
        original_data['predictions'] = original_data['Id'].map(pred_map)

        # Preenchendo previsões faltantes com zeros
        original_data['predictions'] = original_data['predictions'].fillna(0)

        
        return original_data.to_json( orient='records', date_format='iso' )
    
