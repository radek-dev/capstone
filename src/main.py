"""
Capstone Project - Interim Report and Code
"""
from __future__ import division, print_function

import datetime as dt
import time

import MySQLdb as mysql
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy
from binance.client import Client
from scipy.stats import norm

from config import (
    API_KEY,
    API_SECRET,
    MYSQL_USER,
    MYSQL_PASSWORD)

pd.set_option('precision', 10)


class Db:
    """Tools for handling SQL database"""
    def __init__(self):
        self.symbol_map = {'TUSDBTC': 1}

    @staticmethod
    def get_db_engine():
        """hold connection details"""
        return sqlalchemy.create_engine('mysql+pymysql://{0}:{1}@{2}:{3}/{4}'.format(
            MYSQL_USER, MYSQL_PASSWORD, 'localhost', '3306', 'capstone'), echo=False)

    @staticmethod
    def get_mysql_engine():
        return mysql.connect(user=MYSQL_USER, passwd=MYSQL_PASSWORD, db='capstone')

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

    def insert_bid_ask(self, df, target_table):
        conn = self.get_mysql_engine()
        cursor = conn.cursor()
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

    def read_sql(self, sql='select * from symbolRef'):
        """returns pandas df as the query result"""
        with self.get_db_engine().engine.connect() as conn:
            df = pd.read_sql(sql=sql, con=conn)
        return df

    def read_mysql(self, sql='select * from symbolRef'):
        conn = self.get_mysql_engine()
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            headers = [header[0] for header in cursor.description]
            df = pd.DataFrame(list(cursor.fetchall()), columns=headers)
        finally:
            conn.close()
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
    """Tools for connecting to exchange"""
    def __init__(self):
        self.client = Client(API_KEY, API_SECRET)
        self.db = Db()

    def fetch_market_depth(self, symbol='TUSDBTC'):
        """bid and ask prices for given time and symbol"""
        depth = self.client.get_order_book(symbol=symbol)

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


class Kpis():
    def __init__(self, df):
        self.df = df
        self.kpi = {}
        self.trade_summary = None
        self.prepare_trade_summary()
        self.print_kpis()

    def prepare_trade_summary(self):
        df = pd.DataFrame(self.df['position'])
        trades = df.ne(df.shift()).apply(lambda x: x.index[x].tolist())
        data = []
        for i in range(len(trades[0])):
            if trades[0][i] != trades[0][-1]:
                data.append([trades[0][i], trades[0][i+1],
                             self.df.loc[trades[0][i], 'price_mid'],
                             self.df.loc[trades[0][i+1], 'price_mid']])
        df = pd.DataFrame(data=data, columns=['trade_start', 'trade_end', 'start_price', 'end_price'])
        df['return'] = df['end_price'] / df['start_price']
        self.trade_summary = df

    def print_kpis(self):
        self.kpi['mean_annualised_return'] = self.get_mean_annualised_return()
        self.kpi['annualised_std'] = self.get_annualised_std()
        self.kpi['sharp_ratio'] = self.get_sharp_ratio()
        self.kpi['number_of_trades'] = self.get_number_trades()
        self.kpi['win_rate'] = self.get_win_rate()
        self.kpi['avg_trade_return'] = self.get_avg_trade_return()
        self.kpi['avg_win_return'] = self.get_avg_win_return()
        self.kpi['avg_loss_return'] = self.get_avg_loss_return()
        self.kpi['win_loss_ratio'] = self.get_win_loss_ratio()
        self.kpi['max_consecutive_winners'], self.kpi['max_consecutive_losers'] =\
            self.get_max_consecutive_trades()
        self.kpi['max_drawdown'] = self.get_max_drawdown()
        self.kpi['cagr'] = self.get_cagr()
        self.kpi['lake_ratio'] = self.get_lake_ratio()
        print(pd.DataFrame(pd.Series(self.kpi, name='KPIs')))

    def get_mean_annualised_return(self):
        mean_return = self.df[self.df['return'] != 1]['return'].mean()
        days = (len(self.df) / float(3600)) / float(24)
        return mean_return ** (365/days) - 1

    def get_annualised_std(self):
        std_return = self.df[self.df['return'] != 1]['return'].std()
        return std_return * 252

    def get_sharp_ratio(self):
        return self.get_mean_annualised_return() / self.get_annualised_std()

    def get_number_trades(self):
        return len(self.trade_summary)

    def get_win_rate(self):
        return np.where(self.trade_summary['return'] > 1, 1, 0).sum() / float(self.get_number_trades())

    def get_avg_trade_return(self):
        return (self.trade_summary['return']-1).mean()

    def get_avg_win_return(self):
        return (self.trade_summary[self.trade_summary['return'] > 1]['return']-1).mean()

    def get_avg_loss_return(self):
        return (self.trade_summary[self.trade_summary['return'] < 1]['return']-1).mean()

    def get_win_loss_ratio(self):
        return self.get_avg_win_return() / float(self.get_avg_loss_return())

    def get_max_consecutive_trades(self):
        """
        w_max - maximum consecutive winners
        l_max - maximum consecutive losers
        """
        winner_flags, w_max = np.where(self.trade_summary['return'] > 1, 1, 0), 0
        losers_flags, l_max = np.where(self.trade_summary['return'] < 1, 1, 0), 0
        for i in range(1, len(losers_flags)):
            if losers_flags[i] != 0:
                losers_flags[i] += losers_flags[i - 1]
                if losers_flags[i] > l_max:
                    l_max = losers_flags[i]
            if winner_flags[i] != 0:
                winner_flags[i] += winner_flags[i - 1]
                if winner_flags[i] > w_max:
                    w_max = winner_flags[i]
        return w_max, l_max

    def get_max_drawdown(self):
        # noinspection PyBroadException
        try:
            valuation = self.df['cum_ret'].replace(np.nan, 1).values.flatten()
            # end of period
            end_index = np.argmax(np.maximum.accumulate(valuation) - valuation)
            # start of period
            start_index = np.argmax(valuation[:end_index])
            mdd = (valuation[end_index] / valuation[start_index] - 1) * (-1)
        except:
            print('failed to find MDD')
            mdd = np.nan
        return mdd

    def get_cagr(self):
        s_val, e_val = self.df['cum_ret'].replace(np.nan, 1).iloc[0], \
                       self.df['cum_ret'].replace(np.nan, 1).iloc[-1]
        no_years = 1/float(100)
        return (e_val / s_val) ** (1/no_years) - 1

    def get_lake_ratio(self):
        df = pd.DataFrame(self.df['cum_ret'].replace(np.nan, 1).copy())
        df['cum_max'] = df['cum_ret'].cummax()
        df['lakes'] = df['cum_max'] - df['cum_ret']
        return df['lakes'].sum() / df['cum_max'].sum()


def collect_live_data():
    """code snippets to collect data"""
    market_data = DataEndPoint()

    for i in range(3600 * 3):
        market_data.fetch_market_depth()
        time.sleep(1)



def analyse_data():
    """code snippets to used to analyse the data"""
    # collect data
    db = Db()

    str_sql = """
        select distinct bid.updatedId
        from bid
        order by bid.updatedId asc
        limit 60;
        """

    df_updated_id = db.read_sql(str_sql)

    for index, updated_id in df_updated_id.iloc[:, 0].iteritems():
        str_sql = """
            select b.price, sum(b.amount) as amount
            from bid b
              inner join symbolRef sR on b.symbolId = sR.id
            where b.updatedId = {}
            group by b.price
            order by b.price DESC;
            """.format(updated_id)
        df_bid = db.read_sql(str_sql)
        df_bid['cum_amount'] = df_bid['amount'].cumsum()
        str_sql = """
            select a.price, sum(a.amount) as amount
            from ask a
              inner join symbolRef sR on a.symbolId = sR.id
            where a.updatedId = {}
            group by a.price
            order by a.price ASC;
            """.format(updated_id)
        df_ask = db.read_sql(str_sql)
        df_ask['cum_amount'] = df_ask['amount'].cumsum()
        plt.plot(df_bid['price'], df_bid['cum_amount'], 'r', df_ask['price'], df_ask['cum_amount'], 'b')
    plt.title('Bid and Ask over 1 minute')
    plt.xticks(rotation=90)
    plt.show()

    for index, updated_id in df_updated_id.iloc[:, 0].iteritems():
        str_sql = """
            select b.price, sum(b.amount) as amount
            from bid b
              inner join symbolRef sR on b.symbolId = sR.id
            where b.updatedId = {}
                and b.price >= 0.0002245
            group by b.price
            order by b.price DESC;
            """.format(updated_id)
        df_bid = db.read_sql(str_sql)
        df_bid['cum_amount'] = df_bid['amount'].cumsum()
        str_sql = """
            select a.price, sum(a.amount) as amount
            from ask a
              inner join symbolRef sR on a.symbolId = sR.id
            where a.updatedId = {}
                and a.price <= 0.0002265
            group by a.price
            order by a.price ASC;
            """.format(updated_id)
        df_ask = db.read_sql(str_sql)
        df_ask['cum_amount'] = df_ask['amount'].cumsum()
        plt.plot(df_bid['price'], df_bid['cum_amount'], 'r', df_ask['price'], df_ask['cum_amount'], 'b')
    plt.title('Bid and Ask over 1 minute - close to the spread')
    plt.xticks(rotation=90)
    plt.show()

    str_sql = """
    select qryMinAsk.updatedId, qryMaxBid.bidPrice, qryMinAsk.askPrice from
      (select b.updatedId, max(b.price) as bidPrice
      from bid b
      group by b.updatedId) as qryMaxBid
    inner join
      (select a.updatedId, min(a.price) as askPrice
      from ask a
      group by a.updatedId) as qryMinAsk
    on qryMinAsk.updatedId = qryMaxBid.updatedId order by qryMinAsk.updatedId
    """

    df = db.read_sql(str_sql)
    df['spread'] = df['askPrice'] - df['bidPrice']
    df[['bidPrice', 'askPrice']].plot(title='Bid-Ask spread over 1 hour')
    plt.show()
    df['spread'].plot(title='Spread over 1 hour')
    plt.show()


def build_strategy():

    def log_normal_cdf(x=0, mu=0, sigma=0):
        return norm.cdf((np.log(x) - mu) / sigma)

    db = Db()

    str_sql = """
        select distinct bid.updatedId
        from bid
        order by bid.updatedId asc
        """

    df_updated_id = db.read_sql(str_sql)

    data = []
    for index, updated_id in df_updated_id.iloc[:, 0].iteritems():
        str_sql = """
            select * from ask 
            where updatedId = {}
            order by price asc;
        """.format(updated_id)

        df_ask = db.read_sql(str_sql)
        log_values = np.log(df_ask['price'].values)
        mu = (log_values.sum()) / len(df_ask['price'])
        sigma = np.sqrt(((log_values - mu) ** 2).sum() / (len(df_ask['price']) - 1))
        prob = 1 - np.vectorize(log_normal_cdf)(df_ask['price'], mu, sigma)
        prob_weighted_supply = (df_ask['amount'] * prob).sum()
        total_supply = df_ask['amount'].sum()

        str_sql = """
                select * from bid 
                where updatedId = {}
                order by price DESC;
            """.format(updated_id)
        df_bid = db.read_sql(str_sql)
        log_values = np.log(df_bid['price'].values)
        mu = (log_values.sum()) / len(df_bid['price'])
        sigma = np.sqrt(((log_values - mu) ** 2).sum() / (len(df_bid['price']) - 1))
        # need to reverse the order as we want to the mass to start at the highest value
        prob = 1 - np.vectorize(log_normal_cdf)(df_bid['price'], mu, sigma)[::-1]
        prob_weighted_demand = (df_bid['amount'] * prob).sum()
        total_demand = df_ask['amount'].sum()

        best_bid = df_bid['price'].max()
        best_ask = df_ask['price'].min()
        spread_scale = (best_ask / best_bid) - 1
        price_mid = (best_bid + best_ask) / 2
        base = total_supply + total_demand
        mid_point = (total_demand + total_supply) / 2
        ask_price_adjustment = (mid_point - (prob_weighted_demand - prob_weighted_supply))/base * spread_scale
        bid_price_adjustment = (mid_point - (prob_weighted_supply - prob_weighted_demand))/base * spread_scale

        if prob_weighted_demand > prob_weighted_supply:
            price_adj = price_mid / (1 + bid_price_adjustment)
            data.append([best_bid, best_ask, price_mid, price_adj, np.nan])
        else:
            price_adj = price_mid * (1 + ask_price_adjustment)
            data.append([best_bid, best_ask, price_mid, np.nan, price_adj])
        print(index)

    df = pd.DataFrame(data, columns=['max_bid', 'min_ask', 'price_mid', 'price_adj_ask', 'price_adj_bid'])
    df[['max_bid', 'min_ask', 'price_adj_ask', 'price_adj_bid']].plot()
    plt.show()
    df.to_csv('data.csv')


def trade(plotting=False):
    df = pd.read_csv('data.csv', index_col=0)
    df['long_signal'] = np.where(np.isnan(df['price_adj_ask']), 0, 1)
    df['short_signal'] = np.where(np.isnan(df['price_adj_bid']), 0, 1)

    df['position'] = 0
    for n in range(len(df)):
        if n > 60:
            view = df.loc[n-60: n, 'long_signal'].sum()
            if view > 55:
                df.loc[n, 'position'] = 1
            elif view < 3:
                df.loc[n, 'position'] = -1
            else:
                df.loc[n, 'position'] = 0
    # close the trade at the end of the time period
    df.loc[len(df)-1, 'position'] = 0

    df['return'] = (df['min_ask'].pct_change() * df['position']) + 1
    df['cum_ret'] = df['return'].cumprod()

    if plotting:
        df[['max_bid', 'min_ask', 'price_adj_ask', 'price_adj_bid']].plot()
        plt.show()

        df['position'].plot()
        plt.show()

        df[['long_signal']].plot()
        plt.show()

        df[['short_signal']].plot()
        plt.show()

        df['cum_ret'].plot()
        plt.show()

    return df


def main():
    # uncomment if you with to start the collection of live data.
    collect_live_data()
    # analyse_data()
    # build_strategy()
    # df = trade()
    # kpi = Kpis(df)


if __name__ == '__main__':
    main()
