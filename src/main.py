"""
Capstone Project - Interim Report and Code
"""

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


def fetch_market_book(client, symbol='TUSDBTC'):
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
    prices = client.get_all_tickers()
    df = pd.DataFrame(prices)
    db = Db()
    db.write_df(df=df, index=True, table_name='symbols')


def main():
    client = Client(API_KEY, API_SECRET)
    fetch_market_book(client)
    fetch_all_symbol_prices(client)

    db = Db
    df = db.read_sql()
    db.write_df(df=df)
    db.execute_sql(sql='')
    db.read_table(table_name='test_table')


if __name__ == '__main__':
    main()
