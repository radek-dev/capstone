from config import (API_KEY, API_SECRET)
from binance.client import Client
client = Client(API_KEY, API_SECRET)
depth = client.get_order_book(symbol='BTCUSDT')


def main():
    client = Client(API_KEY, API_SECRET)
    depth = client.get_order_book(symbol='BTCUSDT')
    print(depth)


if __name__ == '__main__':
    main()
