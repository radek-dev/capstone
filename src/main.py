from src.config import (API_KEY, API_SECRET)
from binance.client import Client


def create_db():
    print('hi')


def main():
    client = Client(API_KEY, API_SECRET)
    depth = client.get_order_book(symbol='BTCUSDT')
    print(depth)


if __name__ == '__main__':
    main()
