"""
Capstone Project - Interim Report and Code
"""
import time
import datetime as dt

import pandas as pd
import sqlalchemy
import MySQLdb as mysql
from binance.client import Client

from src.config import (
    API_KEY,
    API_SECRET,
    MYSQL_USER,
    MYSQL_PASSWORD)


class Db:
    def __init__(self):
        self.symbol_map = {'TUSDBTC': 1}

    @staticmethod
    @property
    def get_db_engine():
        """hold connection details"""
        return sqlalchemy.create_engine('mysql+pymysql://{0}:{1}@{2}:{3}/{4}'.format(
            MYSQL_USER, MYSQL_PASSWORD, 'localhost', '3306', 'capstone'), echo=False)

    def return_sql(self, sql='SHOW TABLES'):
        """executes any sql and tries to return data"""
        return self.get_db_engine.execute(sql).fetchall()

    def execute_sql(self, sql='SHOW TABLES'):
        """executes any sql and does not try to return data"""
        return self.get_db_engine.execute(sql)

    def write_df(self, df, index, table_name='test_table'):
        """stores the data frame to the database"""
        with self.get_db_engine.engine.connect() as conn:
            df = df.to_sql(name=table_name, con=conn, if_exists='append',
                           index=index)
        return df

    def insert_bid_ask(self, df, target_table):
        conn = mysql.connect(user=MYSQL_USER, passwd=MYSQL_PASSWORD, db='capstone')
        cursor = conn.cursor('')
        cursor.execute('select * from {} where updatedId = {} LIMIT 1'.format(
            target_table, df['updatedId'].iloc[0]))
        already_in = cursor.fetchall()
        if len(already_in) == 0:
            cursor.execute('SHOW COLUMNS FROM {}'.format(target_table))
            rows = cursor.fetchall()
            cols = ', '.join([row[0] for row in rows])
            str_sql = 'insert into {} ({}) values ({})'.format(
                target_table, cols, ', '.join(['{}, ' * len(rows)])[:-2])
            try:
                for row in df.iterrows():
                    cursor.execute(
                        str_sql.format(row[1].iloc[0], row[1].iloc[1], row[1].iloc[2],
                                       "'" + str(row[1].iloc[3]) + "'", self.symbol_map[row[1].iloc[4]])
                    )
                conn.commit()
                print('This data is updated {} for {}'.format(row[1].iloc[2], target_table))
            except mysql.Error as e:
                print(e)
                conn.rollback()
            finally:
                conn.close()
        else:
            print('This data is already updated.')

    def read_sql(self, sql='select * from test'):
        """returns pandas df as the query result"""
        with self.get_db_engine.engine.connect() as conn:
            df = pd.read_sql(sql=sql, con=conn)
        return df

    def read_table(self, sql='select * from table_name', table_name='test'):
        """reads a given table fully in"""
        return self.read_sql(sql.replace('table_name', table_name))

    def clear_some_table(self):
        sql = "TRUNCATE TABLE bid;"
        self.execute_sql(sql)
        sql = "TRUNCATE TABLE ask;"
        self.execute_sql(sql)

    def drop_table(self):
        sql = "DROP TABLE IF EXISTS bid;"
        self.execute_sql(sql)
        sql = "DROP TABLE IF EXISTS ask;"
        self.execute_sql(sql)


class DataEndPoint:
    def __init__(self):
        self.client = Client(API_KEY, API_SECRET)
        self.db = Db()

    def fetch_market_depth(self, symbol='TUSDBTC'):
        """bid and ask prices for given time and symbol"""
        depth = self.client.get_order_book(symbol=symbol)
        db = Db()

        df_bid = pd.DataFrame(depth['bids'], columns=['price', 'amount', 'col3']).drop(labels=['col3'], axis=1)
        df_bid['updatedId'] = depth['lastUpdateId']
        df_bid['myUtc'] = dt.datetime.utcnow()
        df_bid['symbol'] = symbol
        df_bid['price'] = df_bid['price'].astype(float)
        df_bid['amount'] = df_bid['amount'].astype(float)
        self.db.insert_bid_ask(df_bid, 'bid')

        df_ask = pd.DataFrame(depth['asks'], columns=['price', 'amount', 'col3']).drop(labels=['col3'], axis=1)
        df_ask['updatedId'] = depth['lastUpdateId']
        df_ask['myUtc'] = dt.datetime.utcnow()
        df_ask['symbol'] = symbol
        df_ask['price'] = df_ask['price'].astype(float)
        df_ask['amount'] = df_ask['amount'].astype(float)
        self.db.insert_bid_ask(df_ask, 'ask')

    def fetch_all_symbol_prices(self):
        """prices and all symbols in the given time"""
        prices = self.client.get_all_tickers()
        df = pd.DataFrame(prices)
        self.db.write_df(df=df, index=True, table_name='symbols')

    def fetch_exchange_info(self):
        """
        Exchange info includes

        * rateLimits - limits per time intervals - minute, second and day

        * symbols - large set limiting factors for symbols - tick sizes, min prices,
          orders, base and quote assets, quote precision, status, full symbol
          interpretation for a currency pair

        * current server time
        * time zone
        * exchangeFilters - blank field at the moment
        """
        info = self.client.get_exchange_info()
        # info.keys()
        #  [u'rateLimits', u'timezone', u'exchangeFilters', u'serverTime', u'symbols']
        df = pd.DataFrame(info['rateLimits'])
        self.db.write_df(df=df, index=True, table_name='limits')
        df = pd.DataFrame(info['symbols'])
        df1 = df[['baseAsset', 'filters']]
        self.db.write_df(df=df1, index=True, table_name='symbolFilters')
        df.drop(labels='filters', inplace=True, axis=1)
        self.db.write_df(df=df1, index=True, table_name='symbolInfo')
        # unused fields
        # info['serverTime']
        # info['timezone']
        # # UTC
        # info['exchangeFilters']
        self.client.get_all_orders(symbol='TUSDBTC', requests_params={'timeout': 5})

    def fetch_general_end_points(self):
        """
        ping - returns nothing for me
        server time only
        symbol info - limits for a specific symbol
        """
        self.client.ping()
        time_res = self.client.get_server_time()
        status = self.client.get_system_status()
        # this is the same as exchange info
        info = self.client.get_symbol_info('TUSDBTC')
        pd.DataFrame(info)

    def fetch_recent_trades(self):
        """
        500 recent trades for the symbol
        columns:
          id  isBestMatch  isBuyerMaker       price            qty           time
        """
        trades = self.client.get_recent_trades(symbol='TUSDBTC')
        pd.DataFrame(trades)

    def fetch_historical_trades(self):
        """
        seems the same as recent trades
        """
        trades = self.client.get_historical_trades(symbol='TUSDBTC')
        pd.DataFrame(trades)

    def fetch_aggregate_trades(self):
        """not sure what this does, some trade summary but uses letters as column headers"""
        trades = self.client.get_aggregate_trades(symbol='TUSDBTC')
        pd.DataFrame(trades)

    def fetch_aggregate_trade_iterator(self):
        agg_trades = self.client.aggregate_trade_iter(symbol='TUSDBTC', start_str='30 minutes ago UTC')

        # iterate over the trade iterator
        for trade in agg_trades:
            print(trade)
            # do something with the trade data

        # convert the iterator to a list
        # note: generators can only be iterated over once so we need to call it again
        agg_trades = self.client.aggregate_trade_iter(symbol='ETHBTC', start_str='30 minutes ago UTC')
        agg_trade_list = list(agg_trades)

        # example using last_id value - don't run this one - can be very slow
        agg_trades = self.client.aggregate_trade_iter(symbol='ETHBTC', last_id=3263487)
        agg_trade_list = list(agg_trades)

    def fetch_candlesticks(self):
        candles = self.client.get_klines(symbol='BNBBTC', interval=Client.KLINE_INTERVAL_30MINUTE)

        # this works but I am not sure how to use it
        for kline in self.client.get_historical_klines_generator("BNBBTC", Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC"):
            print(kline)
        # do something with the kline

    def fetch_24hr_ticker(self):
        """
        summary of price movements for all symbols
        """
        tickers = self.get_ticker()
        pd.DataFrame(tickers).columns
        # [u'askPrice', u'askQty', u'bidPrice', u'bidQty', u'closeTime', u'count',
        #  u'firstId', u'highPrice', u'lastId', u'lastPrice', u'lastQty',
        #  u'lowPrice', u'openPrice', u'openTime', u'prevClosePrice',
        #  u'priceChange', u'priceChangePercent', u'quoteVolume', u'symbol',
        #  u'volume', u'weightedAvgPrice']

    def fetch_orderbook_tickers(client):
        """
        summary of current orderbook for all markets
        askPrice           askQty       bidPrice           bidQty      symbol
        """
        tickers = client.get_orderbook_tickers()
        pd.DataFrame(tickers)
        pass


def main():

    market_data = DataEndPoint()

    for i in range(3600):
        market_data.fetch_market_depth()
        time.sleep(1)


if __name__ == '__main__':
    main()
