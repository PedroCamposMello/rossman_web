import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann( object ):
    def __init__( self ):
        self.home_path='' 
        self.tool_encoder_store_type                = pickle.load( open( self.home_path + 'exports/cicle_products/tool_encoder_store_type.pkl', 'rb') )
        self.tool_scaler_competition_distance       = pickle.load( open( self.home_path + 'exports/cicle_products/tool_scaler_competition_distance.pkl', 'rb') )
        self.tool_scaler_competition_time_month     = pickle.load( open( self.home_path + 'exports/cicle_products/tool_scaler_competition_time_month.pkl', 'rb') )
        self.tool_scaler_promo_time_week            = pickle.load( open( self.home_path + 'exports/cicle_products/tool_scaler_promo_time_week.pkl', 'rb') )
        self.tool_scaler_year                       = pickle.load( open( self.home_path + 'exports/cicle_products/tool_scaler_year.pkl', 'rb') )

    def data_cleaning( self, df_01 ):
        # 1.1. Renomear colunas

        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 
                    'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

        snakecase = lambda x: inflection.underscore( x )

        cols_new = list( map( snakecase, cols_old ) )

        # Renomeando colunas:
        df_01.columns = cols_new



        # 1.3. Tipagem dos dados - Primeira iteração

        # Transformando variável 'date' em datetime:
        df_01['date'] = pd.to_datetime( df_01['date'] )



        # 1.5. Preenchendo dados vazios

        #competition_distance - Transforma os Nan em 200000 (muito maior do que a maior distância no banco de dados):     
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

        df_01['is_promo'] = df_01[['promo_interval', 'month_map']].apply( lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split( ',' ) else 0, axis=1 ) # Checa se o mês de 'month_map' está contido nos meses de 'promo_interval' e se estiver muda para 1 o valor de 'is_promo' indicando que está no período de promoção



        # 1.6. Tipagem dos dados - Segunda iteração

        # competiton
        df_01['competition_open_since_month'] = df_01['competition_open_since_month'].astype( int )
        df_01['competition_open_since_year'] = df_01['competition_open_since_year'].astype( int )
            
        # promo2
        df_01['promo2_since_week'] = df_01['promo2_since_week'].astype( int )
        df_01['promo2_since_year'] = df_01['promo2_since_year'].astype( int )

        return df_01 

    def feature_engineering( self, df_02 ):

        # 2.0 Feature engineering

        ## 2.4. Criando variáveis derivadas
        
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

        # competition since
        df_02['competition_since'] = df_02.apply( lambda x: datetime.datetime( year=x['competition_open_since_year'], month=x['competition_open_since_month'],day=1 ), axis=1 )
        df_02['competition_time_month'] = ( ( df_02['date'] - df_02['competition_since'] )/30 ).apply( lambda x: x.days ).astype( int )

        # promo since
        df_02['promo_since'] = df_02['promo2_since_year'].astype( str ) + '-' + df_02['promo2_since_week'].astype( str )
        df_02['promo_since'] = df_02['promo_since'].apply( lambda x: datetime.datetime.strptime( x + '-1', '%Y-%W-%w' ) - datetime.timedelta( days=7 ) )
        df_02['promo_time_week'] = ( ( df_02['date'] - df_02['promo_since'] )/7 ).apply( lambda x: x.days ).astype( int )

        # assortment
        df_02['assortment'] = df_02['assortment'].apply( lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended' )

        # state holiday
        df_02['state_holiday'] = df_02['state_holiday'].apply( lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day' )

        return df_02

    def filtering_to_business( self, df_03 ):

        # 3.0 Filtragem de variáveis para o negócio

        ## 3.1. Filtragem de linhas

        # Filtrando linhas apenas para dias em que houve vendas:
        df_03 = df_03[(df_03['open'] != 0)]



        ## 3.2. Seleção de colunas

        # Removendo colunas utilizadas como referência na etapa de feature engeneering:
        cols_drop = ['open', 'promo_interval', 'month_map']
        df_03 = df_03.drop( cols_drop, axis=1 )

        return df_03

    def data_preparation( self, df_05 ):
        # 5.0. Prepararação dos dados

        ## 5.2. Rescaling

        # competition distance
        df_05['competition_distance'] = self.tool_scaler_competition_distance.transform( df_05[['competition_distance']].values ) # Com tratamento de outliers

        # competition time month
        df_05['competition_time_month'] = self.tool_scaler_competition_time_month.transform( df_05[['competition_time_month']].values ) # Com tratamento de outliers

        # promo time week
        df_05['promo_time_week'] = self.tool_scaler_promo_time_week.transform( df_05[['promo_time_week']].values ) # Sem tratamento de outliers

        # year
        df_05['year'] = self.tool_scaler_year.transform( df_05[['year']].values ) # Sem tratamento de outliers



        ## 5.3. Transformação

        ### 5.3.1. Encoding

        # state_holiday - One Hot Encoding
        df_05 = pd.get_dummies( df_05, prefix=['state_holiday'], columns=['state_holiday'] )

        # store_type - Label Encoding
        df_05['store_type'] = self.tool_encoder_store_type.transform( df_05['store_type'] )

        # assortment - Ordinal Encoding
        assortment_dict = {'basic': 1,  'extra': 2, 'extended': 3}
        df_05['assortment'] = df_05['assortment'].map( assortment_dict )



        ### 5.3.2. Transformação da variável resposta

        ### 5.3.3. Transformação de natureza (encoder cíclico)
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



        ## 5.4. Descartando colunas antigas
        cols_drop = ['week_of_year', 'day', 'month', 'day_of_week', 'promo_since', 'competition_since', 'year_week' ]
        df_05 = df_05.drop( cols_drop, axis=1 )

        return df_05

    def feature_selection( self, df_06 ):

        cols_selected = ['store',
                        'promo',
                        'store_type',
                        'assortment',
                        'competition_distance',
                        'competition_open_since_month',
                        'competition_open_since_year',
                        'promo2',
                        'promo2_since_week',
                        'promo2_since_year',
                        'competition_time_month',
                        'promo_time_week',
                        'day_of_week_sin',
                        'day_of_week_cos',
                        'month_sin',
                        'month_cos',
                        'day_sin',
                        'day_cos',
                        'week_of_year_sin',
                        'week_of_year_cos'
                        ]
        
        return df_06[cols_selected]

    def get_prediction( self, model, original_data, test_data ):
        # prediction
        pred = model.predict( test_data )
        
        # join pred into the original data
        original_data['prediction'] = np.expm1( pred )
        
        return original_data.to_json( orient='records', date_format='iso' )
    
