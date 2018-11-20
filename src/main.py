"""
Capstone Project - Interim Report and Code
"""
#ToDo build a good DB structure - start with ID, bid/ask tables


import datetime as dt

import pandas as pd
import sqlalchemy
from binance.client import Client

from src.config import (
    API_KEY,
    API_SECRET,
    MYSQL_USER,
    MYSQL_PASSWORD)


class Db:
    def __init__(self):
        pass

    @staticmethod
    def get_db_engine():
        """hold connection details"""
        return sqlalchemy.create_engine('mysql+pymysql://{0}:{1}@{2}:{3}/{4}'.format(
            MYSQL_USER, MYSQL_PASSWORD, 'localhost', '3306', 'capstone'), echo=False)

    def return_sql(self, sql='SHOW TABLES'):
        """executes any sql and tries to return data"""
        return self.get_db_engine().execute(sql).fetchall()

    def execute_sql(self, sql='SHOW TABLES'):
        """executes any sql and does not try to return data"""
        return self.get_db_engine().execute(sql)

    def write_df(self, df, index, table_name='test_table'):
        """stores the data frame to the database"""
        with self.get_db_engine().engine.connect() as conn:
            df = df.to_sql(name=table_name, con=conn, if_exists='append',
                           index=index)
        return df

    def read_sql(self, sql='select * from test'):
        """returns pandas df as the query result"""
        with self.get_db_engine().engine.connect() as conn:
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


def fetch_market_depth(client, symbol='TUSDBTC'):
    """bid and ask prices for given time and symbol"""
    depth = client.get_order_book(symbol=symbol)
    db = Db()

    df_bid = pd.DataFrame(depth['bids'], columns=['price', 'amount', 'col3']).drop(labels=['col3'], axis=1)
    df_bid['updatedId'] = depth['lastUpdateId']
    df_bid['myUtc'] = dt.datetime.utcnow()
    df_bid['symbol'] = symbol
    df_bid['price'] = df_bid['price'].astype(float)
    df_bid['amount'] = df_bid['amount'].astype(float)
    db.write_df(df=df_bid, index=True, table_name='bid')

    df_ask = pd.DataFrame(depth['asks'], columns=['price', 'amount', 'col3']).drop(labels=['col3'], axis=1)
    df_ask['updatedId'] = depth['lastUpdateId']
    df_ask['myUtc'] = dt.datetime.utcnow()
    df_ask['symbol'] = symbol
    df_ask['price'] = df_ask['price'].astype(float)
    df_ask['amount'] = df_ask['amount'].astype(float)
    db.write_df(df=df_ask, index=True, table_name='ask')


def fetch_all_symbol_prices(client):
    """prices and all symbols in the given time"""
    prices = client.get_all_tickers()
    df = pd.DataFrame(prices)
    db = Db()
    db.write_df(df=df, index=True, table_name='symbols')


def fetch_exchange_info(client):
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
    info = client.get_exchange_info()
    db = Db()
    # info.keys()
    #  [u'rateLimits', u'timezone', u'exchangeFilters', u'serverTime', u'symbols']
    df = pd.DataFrame(info['rateLimits'])
    db.write_df(df=df, index=True, table_name='limits')
    df = pd.DataFrame(info['symbols'])
    df1 = df[['baseAsset', 'filters']]
    df1 = df['filters'][0]
    db.write_df(df=df1, index=True, table_name='symbolFilters')
    df.drop(labels='filters', inplace=True, axis=1)
    db.write_df(df=df1, index=True, table_name='symbolInfo')
    info['serverTime']
    info['timezone']
    # UTC
    info['exchangeFilters']

    client.get_all_orders(symbol='TUSDBTC', requests_params={'timeout': 5})


def fetch_general_end_points(client):
    """
    ping - returns nothing for me
    server time only
    symbol info - limits for a specific symbol
    """
    client.ping()
    time_res = client.get_server_time()
    status = client.get_system_status()
    # returns
    # {
    #     "status": 0,  # 0: normal，1：system maintenance
    #     "msg": "normal"  # normal or System maintenance.
    # }
    # this is the same as exchange info
    info = client.get_symbol_info('TUSDBTC')
    pd.DataFrame(info)


def fetch_recent_trades(client):
    """
    500 recent trades for the symbol
    columns:
      id  isBestMatch  isBuyerMaker       price            qty           time
    """
    trades = client.get_recent_trades(symbol='TUSDBTC')
    pd.DataFrame(trades)


def fetch_historical_trades(client):
    """
    seems the same as recent trades
    """
    trades = client.get_historical_trades(symbol='TUSDBTC')
    pd.DataFrame(trades)


def fetch_aggregate_trades(client):
    """not sure what this does, some trade summary but uses letters as column headers"""
    trades = client.get_aggregate_trades(symbol='TUSDBTC')
    pd.DataFrame(trades)


def fetch_aggregate_trade_iterator(client):
    agg_trades = client.aggregate_trade_iter(symbol='TUSDBTC', start_str='30 minutes ago UTC')

    # iterate over the trade iterator
    for trade in agg_trades:
        print(trade)
        # do something with the trade data

    # convert the iterator to a list
    # note: generators can only be iterated over once so we need to call it again
    agg_trades = client.aggregate_trade_iter(symbol='ETHBTC', start_str='30 minutes ago UTC')
    agg_trade_list = list(agg_trades)

    # example using last_id value - don't run this one - can be very slow
    agg_trades = client.aggregate_trade_iter(symbol='ETHBTC', last_id=3263487)
    agg_trade_list = list(agg_trades)


def fetch_candlesticks(client):
    candles = client.get_klines(symbol='BNBBTC', interval=Client.KLINE_INTERVAL_30MINUTE)

    # this works but I am not sure how to use it
    for kline in client.get_historical_klines_generator("BNBBTC", Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC"):
        print(kline)
    # do something with the kline


def fetch_24hr_ticker(client):
    """
    summary of price movements for all symbols
    """
    tickers = client.get_ticker()
    a = pd.DataFrame(tickers)
    a.columns
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

def main():
    client = Client(API_KEY, API_SECRET)

    db = Db
    df = db.read_sql()
    db.write_df(df=df)
    db.execute_sql(sql='')
    db.read_table(table_name='test_table')


if __name__ == '__main__':
    main()
