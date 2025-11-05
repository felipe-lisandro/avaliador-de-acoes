from ib_insync import *
import asyncio
import pandas as pd
import math
from datetime import datetime
import time
import numpy as np

import yfinance as yf
import requests
#import xml.etree.ElementTree as et

import json
import ast
import os
import pickle
import sys

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from threading import Thread

class Config:
    def resource_path(relative_path):
        # Normalize slashes (important on Windows)
        relative_path = relative_path.replace('/', os.sep)
        
        if getattr(sys, 'frozen', False):
            base_path = os.path.dirname(sys.executable)
            # New PyInstaller 6+ layout: check _internal first
            internal_path = os.path.join(base_path, "_internal", relative_path)
            if os.path.exists(internal_path):
                return internal_path
            # Fallback: check next to .exe
            external_path = os.path.join(base_path, relative_path)
            if os.path.exists(external_path):
                return external_path
            # As a last resort, assume inside _internal anyway
            return internal_path
        else:
            # When running from source
            return os.path.join(os.path.abspath("."), relative_path)
    
    PATH_TICKERS_BR = resource_path("tickers/tickers_br.txt")
    PATH_TICKERS_USA = resource_path("tickers/tickers_usa.txt")
    PATH_DEAD_TICKERS = resource_path("tickers/dead_tickers.txt")
    PATH_LAST_FETCH = resource_path("cache/last_fetch.json")
    PATH_GROSS_MARGINS = resource_path("cache/fundamentals.json")
    PATH_FILTERS = resource_path("cache/filters.json")

    TIMEFRAMES = {
        'day': {'duration': '1 Y', 'bar_size': '1 day'},
        'week': {'duration': '5 Y', 'bar_size': '1 W'},
        'month': {'duration': '10 Y', 'bar_size': '1 M'},
        '15min': {'duration': '1 W', 'bar_size': '15 mins'},
        '60min': {'duration': '1 M', 'bar_size': '1 hour'}
    }

    IB_HOST = '127.0.0.1'
    IB_PORT = 4001
    IB_CLIENT_ID = 100

    MAX_CONCURRENT = 40
    REQUEST_DELAY = .01

    MAX_CONCURRENT_YF = 2
    REQUEST_DELAY_YF = 5

    PERCENTILE = 90
    GROSS_MARGIN_FILTER = .1
    MARKET_CAP_FILTER = 200000000000
    GOOD_QUALITY_WEIGHT = 1
    BAD_QUALITY_WEIGHT =.2

    MA_SPAN = {
        'ema9': 9,
        'sma21': 21,
        'sma50': 50,
        'sma200': 200,
        'ema400': 400
    }
    LOOKBACKS = {
        'ema400': 40,
        'sma200': 20,
        'sma50': 13,
        'sma21': 8,
        'ema9': 5
    }
    MA_WEIGHTS = {
        #'ema400': 0.05,
        #'sma200': 0.15,
        #'sma50': 0.35,
        #'sma21': 0.3,
        #'ema9': 0.15
    }
    TIMEFRAME_WEIGHTS = {
        #'day': .2,
        #'week': .45,
        #'month': .3,
        #'15min': 0,
        #'60min': .05
    }
    UPTREND_WEIGHTS = {
        'price_above_ema9': 0.2, 
        'price_above_sma21': 0.5, 
        'price_above_sma50': 5, 
        'price_above_sma200': 4.5,
        'price_above_ema400': 4.0, 
        'ema9_rising': 0.15,
        'sma21_rising': 0.3, 
        'sma50_rising': 2.8, 
        'sma200_rising': 3.0,
        'ema400_rising': 2.5,
        'stacked_mas': 8.8
    }

class Cache:
    def load_list_br():
        if not os.path.exists(Config.PATH_TICKERS_BR):
            return []
        try:
            with open(Config.PATH_TICKERS_BR, 'r') as f:
                return ast.literal_eval(f.read())
            
        except Exception as e:
            print(f"Error loading BR list! {e}")
            return None

    def load_list_usa():
        if not os.path.exists(Config.PATH_TICKERS_USA):
            return []
        try:
            with open(Config.PATH_TICKERS_USA, 'r') as f:
                return ast.literal_eval(f.read())
        except Exception as e:
            print(f"Error loading USA list! {e}")
            return None
    
    def save_list_usa(list):
        try:
            with open(Config.PATH_TICKERS_USA, 'w') as f:
                json.dump(list, f)
                return 1
        except:
            print(f"Error while trying to save USA tickers list!")
            return None

    def load_list_dead():
        if not os.path.exists(Config.PATH_DEAD_TICKERS):
            return []
        try:
            with open(Config.PATH_DEAD_TICKERS, 'r') as f:
                return json.load(f)
        except:
            print(f"Error loading dead tickers list!")
            return []

    def save_list_dead(list: list):
        try:
            with open(Config.PATH_DEAD_TICKERS, 'w') as f:
                json.dump(list, f)
        except:
            print(f"Error while trying to save dead tickers list!")

    def load_last_fetch():
        if not os.path.exists(Config.PATH_LAST_FETCH):
            # print(f"retorna nada!")
            return {}
        try:
            with open(Config.PATH_LAST_FETCH, 'r') as f:
                return json.load(f)
        except:
            return {} 

    def save_last_fetch(cache: dict):
        try:
            with open(Config.PATH_LAST_FETCH, 'w') as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            print(f"Error while trying to save the last fetch cache! {e}")

    def load_ticker_cache(ticker: str, timeframe: str):
        path = f'cache/{timeframe}/{ticker}.pkl'
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading ticker {ticker}: {e}")
        return None

    def save_ticker_cache(ticker:str, timeframe:str, df: pd.DataFrame):
        path = f'cache/{timeframe}/{ticker}.pkl'
        try:
            with open(path, 'wb') as f:
                pickle.dump(df, f, protocol=5)
        except:
            print(f"Error while trying to save the {ticker} {timeframe} bar!")

    def load_gross_margins():
        if not os.path.exists(Config.PATH_GROSS_MARGINS):
            return {}
        try:
            with open(Config.PATH_GROSS_MARGINS, 'r') as f:
                return json.load(f)
        except:
            return {} 
    
    def save_gross_margins(gross_margins):
        #print("salva")
        try:
            with open(Config.PATH_GROSS_MARGINS, 'w') as f:
                json.dump(gross_margins, f, indent=2)
        except Exception as e:
            print(f"Error while trying to save the gross margins cache! {e}")

    def load_filters():
        if os.path.exists(Config.PATH_FILTERS):
            try:
                with open(Config.PATH_FILTERS, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_filters(filters):
        try:
            with open(Config.PATH_FILTERS, "w") as f:
                json.dump(filters, f, indent=2)
        except Exception as e:
            print(f"Error trying to save filters! {e}")

class Helper:
    def get_last_trading_day():
        import datetime
        from datetime import timedelta
        from datetime import datetime
        import pytz
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)
        if now_et.weekday() == 5: # saturday
            return (now_et - timedelta(days=1)).replace(tzinfo=None)
        if now_et.weekday() == 6: # sunday
            return (now_et - timedelta(days=2)).replace(tzinfo=None)
        market_open_hour = 9
        market_open_minute = 30
        current_time_minutes = now_et.hour * 60 + now_et.minute
        market_open_minutes = market_open_hour * 60 + market_open_minute
        if current_time_minutes < market_open_minutes: # before opening
            return (now_et - timedelta(days=1)).replace(tzinfo=None)
        return now_et.replace(tzinfo=None) # open / closed

class TickerManager:
    def __init__(self):
        self.ib = None
        self.active_requests = 0
        self.max_concurrent = Config.MAX_CONCURRENT
        self.semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT)
        self.list_br = Cache.load_list_br()
        self.list_usa = Cache.load_list_usa()

    async def connect(self):
        self.ib = IB()
        self.ib.errorEvent += TickerManager.on_ib_error
        print(f"aqui")
        await self.ib.connectAsync(Config.IB_HOST, Config.IB_PORT, clientId=Config.IB_CLIENT_ID)

    async def disconnect(self):
        if self.ib is not None:
            self.ib.disconnect()

    def get_info_fetch(self, dates, ticker: str, timeframe: str):
        """
        Returns [ticker, durationStr, barSize]
        """
        if ticker not in dates or timeframe not in dates[ticker]: # case where it has to fetch everything
            # print(f"aquiiii {ticker}")
            ret = [ticker, Config.TIMEFRAMES[timeframe]['duration'], Config.TIMEFRAMES[timeframe]['bar_size'], timeframe]
        # case where something is already cached
        else:
            last_date = datetime.strptime(dates[ticker][timeframe], "%Y-%m-%d").date()
            last_trading_day = Helper.get_last_trading_day().date()
            delta_days = (last_trading_day - last_date).days
            if timeframe == 'day':
                ret = [ticker, f"{delta_days} D", Config.TIMEFRAMES[timeframe]['bar_size'], timeframe]
            elif timeframe == 'week':
                weeks = max(1, math.ceil(delta_days / 7))
                ret = [ticker, f"{weeks} W", Config.TIMEFRAMES[timeframe]['bar_size'], timeframe]
            elif timeframe == 'month':
                months = max(1, math.ceil(delta_days / 30))
                ret = [ticker, f"{months} M", Config.TIMEFRAMES[timeframe]['bar_size'], timeframe]
            elif timeframe == '15min':
                ret = [ticker, f"2 D", Config.TIMEFRAMES[timeframe]['bar_size'], timeframe]
            elif timeframe == '60min':
                ret = [ticker, f"2 D", Config.TIMEFRAMES[timeframe]['bar_size'], timeframe]
        # print(ret)
        return ret
    
    def batchify(self, lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]

    @staticmethod
    def on_ib_error(reqId, errorCode, errorString, contract):
        try:
            print(f"[IB ERROR] reqId={reqId}, code={errorCode}, msg={errorString}")
            if errorCode in [200, 321, 366, 420, 162]:
                # Handle 'dead' tickers or similar cases
                list_dead = Cache.load_list_dead()
                if contract.symbol in list_dead:
                    return
                list_dead.append(contract.symbol)
                Cache.save_list_dead(list_dead)
        except Exception as e:
            import traceback
            traceback.print_exc()

    def needs_fetch(self, ticker, timeframe, dates):
        if ticker not in dates or timeframe not in dates[ticker]:
            return True
        last_fetch_date = datetime.fromisoformat(dates[ticker][timeframe]).date()
        last_trading_day = Helper.get_last_trading_day().date()
        if timeframe == 'day':
            return last_fetch_date < last_trading_day
        elif timeframe == 'week':
            y1, w1, _ = last_fetch_date.isocalendar()
            y2, w2, _ = last_trading_day.isocalendar()
            return (y1, w1) < (y2, w2)
        elif timeframe == 'month':
            return (last_fetch_date.year, last_fetch_date.month) < (last_trading_day.year, last_trading_day.month)
        elif timeframe == '60min' or timeframe == '15min':
            return True

    async def fetch_all_ticker_bars(self):
        await self.connect()
        list_ticker = self.list_br + self.list_usa
        list_dead = Cache.load_list_dead()
        actual_list = [ticker for ticker in list_ticker if ticker not in list_dead]
        dates = Cache.load_last_fetch()
        task_list = []
        # print(Helper.get_last_trading_day().date().isoformat())
        # ================= local function
        async def fetch_one(self, ticker: str, duration: str, bar_size: str, timeframe: str):
            exchange = 'SMART' if ticker in self.list_usa else 'BOVESPA'
            currency = 'USD' if ticker in self.list_usa else 'BRL'
            contract = Stock(ticker, exchange, currency)
            async with self.semaphore:
                bars = await self.ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime='',
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow='TRADES',
                    useRTH=True
                )
                await asyncio.sleep(Config.REQUEST_DELAY)
                print(f"Data succesfully fetched for {ticker}: {timeframe}")
                return (ticker, timeframe, bars, duration)
        # ================ local function
        # gets all ticker - timeframe relations that needs update
        print(f"Checking tickers in cache...")
        for ticker in actual_list:
            for timeframe in Config.TIMEFRAMES.keys():
                if not self.needs_fetch(ticker, timeframe, dates):
                    continue
                element = self.get_info_fetch(dates, ticker, timeframe)
                task_list.append(fetch_one(self, element[0], element[1], element[2], element[3]))
        # with the relation, uses it to fetch the updates. I have, on each, the ticker, the durationStr and the barSize
        results = await asyncio.gather(*task_list, return_exceptions=True)
        # now I need to cache every data I got
        list_to_be_cached = []
        list_dead = Cache.load_list_dead()
        for element in results:
            if element[2] is []:
                print("Nothing was fetched!")
                continue
            if element[0] in list_dead:
                # print(f"Ticker {element[0]} is dead!")
                # if os.path.exists(f"cache/{element[1]}/{element[1]}.pkl"):
                    # os.remove(f"cache/{element[1]}/{element[1]}.pkl")
                continue
            list_to_be_cached.append(element)
        # now I have every bar that needs to be saved
        for element in list_to_be_cached:
            df = util.df(element[2])
            try:
                if element[3] == Config.TIMEFRAMES[element[1]]['duration']: # case fetched all data
                    Cache.save_ticker_cache(element[0], element[1], df)
                    if element[0] not in dates.keys():
                        dates[element[0]] = {}
                    dates[element[0]][element[1]] = Helper.get_last_trading_day().date().isoformat()
                    # print(df.head(3))
                    # print(dates)
                else: # case append new data
                    old_df = Cache.load_ticker_cache(element[0], element[1])
                    if old_df['date'].dtype != 'datetime64[ns]':
                        old_df['date'] = pd.to_datetime(old_df['date'])
                    if df['date'].dtype != 'datetime64[ns]':
                        df['date'] = pd.to_datetime(df['date'])
                    last_old_date = old_df['date'].iloc[-1]
                    new_bars = df[df['date'] > last_old_date]
                    if len(new_bars) == 0:
                        merged = old_df
                    else:
                        merged = pd.concat([old_df, new_bars], ignore_index=True, copy=False)
                    Cache.save_ticker_cache(element[0], element[1], merged)
                    if element[0] not in dates.keys():
                        dates[element[0]] = {}
                    dates[element[0]][element[1]] = Helper.get_last_trading_day().date().isoformat()
            except Exception as e:
                print(f"Error trying to save data from {element[0]} for {element[1]} timeframe! {e}")
                continue
        Cache.save_last_fetch(dates)
        
    async def start_batch(self):
        await self.fetch_all_ticker_bars()

class TickerAnalyzer:
    def __init__(self):
        self.list_br = Cache.load_list_br()
        self.list_usa = Cache.load_list_usa()
        self.gross_margins = Cache.load_gross_margins()


    def get_ma(self, df, simple: bool, span: int, look_back: int):
        if df is None or len(df) == 0:
            return None
        len_bars = len(df)
        if len_bars < span + look_back:
            return None
        if simple:
            return float(df['close'].rolling(window=span).mean().iloc[- (1 + look_back)])
        return float(df['close'].ewm(span=span, adjust=False).mean().iloc[- (1 + look_back)])
    
    def get_mas(self, df):
        result = {}
        for ma in Config.MA_SPAN.keys():
            simple = False
            if ma in ['sma200', 'sma50', 'sma21']:
                simple = True 
            temp_ma = self.get_ma(df, simple=simple, span=Config.MA_SPAN[ma], look_back=0)
            if temp_ma is None:
                continue
            result[ma] = temp_ma
        if len(result.keys()) > 0:
            result['close'] = float(df['close'].iloc[-1])
        return result

    def get_slope(self, df, simple: bool, span: int, look_back):
        if df is None or len(df) == 0:
            return None
        len_bars = len(df)
        if len_bars < span + look_back:
            #look_back = len_bars - 1 # will do it this way for now
            return 0
        prev_ma = self.get_ma(df, simple, span, look_back)
        actual_ma = self.get_ma(df, simple, span, look_back=0)
        if prev_ma is None or actual_ma is None:
            return 0
        return (actual_ma - prev_ma) / abs(prev_ma)
    
    def get_slopes(self, df):
        result = {}
        for ma in Config.MA_SPAN.keys():
            simple = False
            if ma in ['sma200', 'sma50', 'sma21']:
                simple = True 
            temp_slope = self.get_slope(df, simple=simple, span=Config.MA_SPAN[ma], look_back=Config.LOOKBACKS[ma])
            if temp_slope is None:
                continue
            result[ma] = temp_slope
        return result
    
    def get_uptrend(self, mas, slopes):
        if mas is None or slopes is None:
            return None
        signals = []
        for ma in mas:
            if ma == 'close':
                continue
            if mas['close'] > mas[ma]:
                signals.append(f"price_above_{ma}")
        for ma, slope in slopes.items():
            if slope > 0:
                signals.append(f"{ma}_rising")
        ordered_mas = ['ema9', 'sma21', 'sma50', 'sma200', 'ema400']
        last_val = float('inf')
        stacked = True
        for ma in ordered_mas:
            if ma in mas and mas[ma] > last_val:
                stacked = False
                break
            if ma in mas:
                last_val = mas[ma]
        if stacked:
            signals.append("stacked_mas")
        return signals

    def analyze_one(self, ticker, timeframe):
        print(f"Analyzing {ticker}: {timeframe}")
        df = Cache.load_ticker_cache(ticker, timeframe)
        result = {}
        mas = self.get_mas(df)
        result['good'] = Config.GOOD_QUALITY_WEIGHT
        if len(mas) == 0:
            result['good'] = Config.BAD_QUALITY_WEIGHT
            result['points'] = 0
            return result
        slopes = self.get_slopes(df)
        if slopes == {}:
            result['good'] = Config.BAD_QUALITY_WEIGHT
            result['points'] = 0
            return result
        uptrends = self.get_uptrend(mas, slopes)
        if uptrends is None:
            result['good'] = Config.BAD_QUALITY_WEIGHT
            result['points'] = 0
            return result
        score = sum([Config.UPTREND_WEIGHTS[s] for s in uptrends])
        result['uptrends'] = uptrends
        #if ticker in ['CLS', 'DCI', 'ELA', 'HSSI', 'HURN', 'CSGS', 'CX']:
            #print(f"{ticker}|{timeframe} -> {uptrends}: {score}")
            #time.sleep(3)
        #if ticker == 'ANET':
            #print(f"{ticker} | {timeframe}: {uptrends} -> {score}")
            #time.sleep(5)
        if score < 16:
            result['good'] = Config.BAD_QUALITY_WEIGHT
        result['points'] = score
        return result

    def analyze_all_tickers(self):
        """
        ticker: timeframe: {slopes: [], index: }
        """
        timeframe_order = ['week','month', 'day', '60min', '15min']
        list_all = self.list_br + self.list_usa
        list_all = [ticker for ticker in list_all if ticker not in Cache.load_list_dead()]
        result = {}
        for ticker in list_all:
            result[ticker] = {}
            for timeframe in timeframe_order:
                result[ticker][timeframe] = self.analyze_one(ticker, timeframe)
                if timeframe == 'week' and result[ticker][timeframe]['good'] == Config.BAD_QUALITY_WEIGHT:
                    break
            #result[ticker]['score'] = sum([Config.TIMEFRAME_WEIGHTS[s] * result[ticker][s]['points'] * result[ticker][s]['good'] for s in Config.TIMEFRAME_WEIGHTS.keys()])
            weighted_sum = 0
            for timeframe in timeframe_order:
                if timeframe not in result[ticker]:
                    break
                #print(Config.TIMEFRAME_WEIGHTS)
                weighted_sum += Config.TIMEFRAME_WEIGHTS[timeframe] * result[ticker][timeframe]['points'] * result[ticker][timeframe]['good']
            result[ticker]['score'] = weighted_sum
            #if ticker == 'ANET':
                #print(f"->> {result[ticker]['score']}")
                #time.sleep(3)
        return result
    
    def analysis(self, uptrend_filter: bool, gross_margin_filter: float=0, market_cap_filter: float=0):
        results = self.analyze_all_tickers()
        if not uptrend_filter:
            return results
        tickers_to_delete = []
        if gross_margin_filter > 0:
            for ticker in results.keys():
                if ticker not in self.gross_margins:
                    continue
                if self.gross_margins[ticker]['gross_margin'] is None or self.gross_margins[ticker]['gross_margin'] == 0:
                    continue
                if self.gross_margins[ticker]['gross_margin'] < gross_margin_filter and ticker not in tickers_to_delete:
                    tickers_to_delete.append(ticker)
        # quality analysis
        scores = [results[key]['score'] for key in results.keys() if 'score' in results[key]]
        threshold = np.percentile(scores, Config.PERCENTILE)
        for ticker in results.keys():
            if 'score' not in results[ticker] or results[ticker]['score'] < threshold:
                tickers_to_delete.append(ticker)
        # gross margin filter
        if gross_margin_filter > 0:
            for ticker in results.keys():
                if ticker not in self.gross_margins:
                    continue
                if self.gross_margins[ticker]['gross_margin'] is None or self.gross_margins[ticker]['gross_margin'] == 0 or self.gross_margins[ticker]['gross_margin'] == 1:
                    tickers_to_delete.append(ticker)
                    continue
                if self.gross_margins[ticker]['gross_margin'] < gross_margin_filter and ticker not in tickers_to_delete:
                    
                    tickers_to_delete.append(ticker)
        if market_cap_filter > 0:
            for ticker in results.keys():
                if ticker not in self.gross_margins:
                    continue
                if self.gross_margins[ticker]['market_cap'] is None or self.gross_margins[ticker]['market_cap'] == 0:
                    tickers_to_delete.append(ticker)
                    continue
                if self.gross_margins[ticker]['market_cap'] > market_cap_filter and ticker not in tickers_to_delete:
                    tickers_to_delete.append(ticker)
        if len(tickers_to_delete) == 0:
            return results
        tickers_to_delete = set(tickers_to_delete)
        for ticker in tickers_to_delete:
            del results[ticker]
        # results = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
        for ticker, info in sorted(results.items(), key=lambda x: x[1]["score"], reverse=True):
            #print(f"{ticker} -> {results[ticker]['score']}")
            print(f"{ticker}")
        return results

class MarginFetcher:
    def __init__(self):
        self.list_usa = Cache.load_list_usa()
        self.list_br = Cache.load_list_br()
        self.list_dead = Cache.load_list_dead()

    def needs_fetch(self, ticker, gross_margins_cache):
        if ticker not in gross_margins_cache: # case where there is nothing cached
            return True
        if gross_margins_cache[ticker]['gross_margin'] == 0 or gross_margins_cache[ticker]['gross_margin'] == 1: # probably those cases are wrong. I need a failsafe for fetching new data when these are fetched
            return True
        last_fetch_date = datetime.fromisoformat(gross_margins_cache[ticker]['date']).date()
        last_trading_day = Helper.get_last_trading_day().date()
        return last_fetch_date.month < last_trading_day.month and last_fetch_date.day - last_trading_day.day >= 29

    async def fetch_all_margins(self): # se por algum motivo isso der erro, nao da pra mostrar na UI ainda
        list_all_tickers = [ticker for ticker in self.list_br + self.list_usa if ticker not in self.list_dead]
        to_be_fetched = []
        margins = Cache.load_gross_margins()
        for ticker in list_all_tickers:
            if self.needs_fetch(ticker, margins):
                to_be_fetched.append(ticker)
        """
        # ============== local function
        async def fetch_ibkr_async(ticker_symbol):
            ib = IB()
            await ib.connectAsync('127.0.0.1', 4001, clientId=Config.IB_CLIENT_ID)
            exchange = 'SMART' if ticker_symbol in Cache.load_list_usa() else 'BOVESPA'
            currency = 'USD' if ticker_symbol in Cache.load_list_usa() else 'BRL'
            contract = Stock(ticker_symbol, exchange, currency)
            try:
                data = await ib.reqFundamentalDataAsync(contract, reportType='ReportSnapshot')
                root = et.fromstring(data)
                market_cap = root.find('.//MarketCap').text
                gross_margin = root.find('.//GrossMargin').text  # adjust based on XML
                return gross_margin, market_cap
            finally:
                ib.disconnect()
        # ============== local function
        # ============== local function
        async def fetch_using_ibkr(ticker):
            try:
                return await fetch_ibkr_async(ticker)
            except Exception as e:
                print(f"Error fetching IBKR data for {ticker}: {e}")
                return None, None
        # ============== local function
        """
        # ============== local function
        def fetch_one(ticker):
            try:
                yf_ticker = yf.Ticker(ticker)
                info = yf_ticker.info
                gross_margin = info.get("grossMargins")
                #if gross_margin == 0 or gross_margin == 1 or gross_margin is None:
                    #ibkr_gross_margin = await fetch_using_ibkr(ticker)
                    #if ibkr_gross_margin is not None:
                        #gross_margin = ibkr_gross_margin
                        #pass
                market_cap = info.get("marketCap")
                print(f"{ticker}: {gross_margin} - {market_cap}")
                time.sleep(Config.REQUEST_DELAY_YF)
                return (ticker, gross_margin, market_cap)
            except Exception as e:
                print(f"Error while trying to fetch fundamentals for {ticker}! {e}")
                return None
        # ============== local function
        async def fetch_all_asynchronous(self, ticker):
            return await asyncio.to_thread(fetch_one, ticker)
        # ============== local function
        tasks = [fetch_all_asynchronous(self, ticker) for ticker in to_be_fetched]
        results = await asyncio.gather(*tasks)
        for element in results:
            if element is None:
                continue
            if element not in margins:
                margins[element[0]] = {}
            if element[1] not in (0, 1, None):
                margins[element[0]]['gross_margin'] = element[1]
            if 'gross_margin' not in margins[element[0]]:
                margins[element[0]]['gross_margin'] = 0
            margins[element[0]]['market_cap'] = element[2]
            margins[element[0]]['date'] = Helper.get_last_trading_day().date().isoformat()
        # print(margins)
        Cache.save_gross_margins(margins)
        return 1

    def fetch_bulk(self):
        url = f"https://financialmodelingprep.com/api/v4/income-statement/AAPL?apikey={Config.API_KEY}&datatype=csv"
        #params = {"apikey":Config.API_KEY,"datatype":"csv"}
        r = requests.get(url)
        print(r)
        df = pd.DataFrame(r)
        print(df.head())

class StockEvaluatorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Avaliador de Ações")
        self.root.geometry("1100x700")
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        # filters loading
        self.filters = Cache.load_filters()
        if len(self.filters.keys()) == 0:
            self.current_filter = None
        else:
            first = list(self.filters.keys())[0]
            self.current_filter = self.filters[first]
        # create window
        self.create_data_buttons()
        self.create_filter_section()
        self.create_stock_buttons()
        self.create_table()
        # create helper atributes
        self.ticker_manager = TickerManager()
        self.ticker_analyzer = TickerAnalyzer()
        self.fundamental_fetcher = MarginFetcher()
        
    def create_data_buttons(self):
        data_frame = tk.Frame(self.root, padx=10, pady=10)
        data_frame.grid(row=0, column=0, sticky="ew")
        btn_market = tk.Button(
            data_frame, 
            text="Buscar Dados de Mercado",
            command=self.fetch_market_data,
            width=30,
            bg="#7FBBEC",
            fg="white",
            font=("Arial", 10, "bold"),
            relief="flat",
            borderwidth=0
        )
        btn_market.pack(side="left", padx=0)
        btn_fundamentals = tk.Button(
            data_frame,
            text="Buscar Dados Fundamentais",
            command=self.fetch_fundamentals,
            width=30,
            bg="#7FBBEC",
            fg="white",
            font=("Arial", 10, "bold"),
            relief="flat",
            borderwidth=0
        )
        btn_fundamentals.pack(side="left", padx=5)
        
    def create_filter_section(self):
        filter_frame = tk.Frame(self.root, padx=10, pady=5)
        filter_frame.grid(row=1, column=0, sticky="ew")
        label = tk.Label(
            filter_frame,
            text="Filtro:",
            font=("Arial", 11, "bold")
        )
        label.pack(side="left", padx=0)    
        # Filter selector (combobox)
        self.filter_var = tk.StringVar()
        self.filter_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.filter_var,
            values=list(self.filters.keys()),
            state="readonly",
            width=25,
            font=("Arial", 10)
        )
        self.filter_combo.pack(side="left", padx=5)
        self.filter_combo.bind("<<ComboboxSelected>>", self.on_filter_selected)
        btn_create_filter = tk.Button(
            filter_frame,
            text="Criar Filtro",
            command=self.create_filter,
            width=15,
            bg="#66BB6A",
            fg="white",
            font=("Arial", 10, "bold"),
            relief="flat",
            borderwidth=0
        )
        if self.filters:
            self.filter_combo['values'] = list(self.filters.keys())
            first_key = next(iter(self.filters))
            self.filter_var.set(first_key)
            self.on_filter_selected(event="<<ComboboxSelected>>")
        btn_create_filter.pack(side="left", padx=5)
        btn_delete_filter = tk.Button(
            filter_frame,
            text="Deletar Filtro",
            command=self.delete_filter,
            width=15,
            bg="#EF5350",
            fg="white",
            font=("Arial", 10, "bold"),
            relief="flat",
            borderwidth=0
        )
        btn_delete_filter.pack(side="left", padx=5)
        
    def create_stock_buttons(self):
        stock_frame = tk.Frame(self.root, padx=10, pady=0)
        stock_frame.grid(row=2, column=0, sticky="ew")
        btn_add = tk.Button(
            stock_frame,
            text="Adicionar Ação",
            command=self.add_stock,
            width=15,
            bg="#7FBBEC",
            fg="white",
            font=("Arial", 10, "bold"),
            relief="flat",
            borderwidth=0
        )
        btn_add.pack(side="left", padx=0)
        btn_remove = tk.Button(
            stock_frame,
            text="Remover Ação",
            command=self.remove_stock,
            width=15,
            bg="#7FBBEC",
            fg="white",
            font=("Arial", 10, "bold"),
            relief="flat",
            borderwidth=0
        )
        btn_remove.pack(side="left", padx=5)
        
    def create_table(self):
        # Frame for table and analysis button
        table_frame = tk.Frame(self.root, padx=10, pady=10)
        table_frame.grid(row=3, column=0, sticky="nsew")
        table_frame.grid_rowconfigure(1, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        btn_analyze = tk.Button(
            table_frame,
            text="Análise",
            command=self.run_analysis_background,
            width=15,
            bg="#7FBBEC",
            fg="white",
            font=("Arial", 11, "bold"),
            height=2,
            relief="flat",
            borderwidth=0
        )
        btn_analyze.grid(row=0, column=0, pady=(0, 10), sticky="w")
        # Create Treeview with scrollbars
        tree_frame = tk.Frame(table_frame)
        tree_frame.grid(row=1, column=0, sticky="nsew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        hsb.grid(row=1, column=0, sticky="ew")
        # Treeview (table)
        self.tree = ttk.Treeview(
            tree_frame,
            columns=("simbolo", "nota", "tendencias"),
            show="headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set
        )
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)
        # Define columns
        self.tree.heading("simbolo", text="Símbolo")
        self.tree.heading("nota", text="Nota")
        self.tree.heading("tendencias", text="N° Tendências de Alta / Timeframe")
        # Column widths
        self.tree.column("simbolo", width=130, anchor="center")
        self.tree.column("nota", width=80, anchor="center")
        self.tree.column("tendencias", width=640, anchor="center")
        # Style configuration
        style = ttk.Style()
        style.configure("Treeview", rowheight=25, font=("Arial", 9))
        style.configure("Treeview.Heading", font=("Arial", 11, "bold"))

    def on_filter_selected(self, event):
        """Handle filter selection"""
        filter_name = self.filter_var.get()
        if filter_name in self.filters:
            self.current_filter = self.filters[filter_name]
            Config.TIMEFRAME_WEIGHTS = self.current_filter['timeframe_weights']
            Config.MA_WEIGHTS = self.current_filter['ma_weights']
            Config.MARKET_CAP_FILTER = self.current_filter['market_cap']
            Config.GROSS_MARGIN_FILTER = self.current_filter['gross_margin']
            Config.PERCENTILE = self.current_filter['percentile']
            #print(Config.TIMEFRAME_WEIGHTS)
            #print(Config.MA_WEIGHTS)
            #print(Config.MARKET_CAP_FILTER)
            #print(Config.GROSS_MARGIN_FILTER)
            #print(Config.PERCENTILE)

    def create_filter(self):
        dialog = FilterCreatorDialog(self.root)
        result = dialog.result
        if result:
            filter_name = result['name']
            if filter_name in self.filters:
                overwrite = messagebox.askyesno(
                    "Filtro Existente",
                    f"Filtro '{filter_name}' já existe. Deseja sobrescrever?"
                )
                if not overwrite:
                    return
            #print(type(result))
            self.filters[filter_name] = result
            #print(filters.keys())
            #print(self.filters)
            Cache.save_filters(self.filters)
            # Update combobox
            self.filter_combo['values'] = list(self.filters.keys())
            self.filter_var.set(filter_name)
            self.current_filter = self.filters[filter_name]
            self.on_filter_selected(event="<<ComboboxSelected>>")
            messagebox.showinfo("Sucesso", f"Filtro '{filter_name}' criado!")
    
    def delete_filter(self):
        """Delete the selected filter"""
        filter_name = self.filter_var.get()
        if not filter_name:
            messagebox.showwarning("Aviso", "Selecione um filtro para deletar!")
            return
        confirm = messagebox.askyesno("Confirmar", f"Tem certeza que deseja deletar o filtro '{filter_name}'?")
        if confirm:
            del self.filters[filter_name]
            Cache.save_filters(self.filters)
            # Update combobox
            self.filter_combo['values'] = list(self.filters.keys())
            self.filter_var.set("")
            self.current_filter = None
            messagebox.showinfo("Sucesso", f"Filtro '{filter_name}' deletado!")
    
    # Table methods
    def add_to_table(self, ticker, score, uptrends):
        """Add a row to the table"""
        self.tree.insert("", "end", values=(ticker, f"{score:.2f}", uptrends))
        
    def clear_table(self):
        """Clear all rows from the table"""
        for item in self.tree.get_children():
            self.tree.delete(item)
            
    def get_selected_ticker(self):
        """Get the currently selected ticker from the table"""
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            return item['values'][0]
        return None
    
    # Action methods
    def fetch_market_data(self):
        """Fetch market data - integrate with TickerManager"""
        # Create a small modal dialog
        loading = tk.Toplevel(self.root)
        loading.title("Aguarde...")
        loading.geometry("250x100")
        loading.transient(self.root)
        loading.grab_set()  # make it modal
        loading.resizable(False, False)
        # Center it relative to main window
        self.root.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (250 // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (100 // 2)
        loading.geometry(f"250x100+{x}+{y}")
        tk.Label(loading, text="Carregando dados de mercado...", font=("Arial", 10, "bold")).pack(pady=20)
        progress = ttk.Progressbar(loading, mode="indeterminate")
        progress.pack(fill="x", padx=20)
        progress.start()
        # ====== local function
        def task():
            try:
                # heavy operation
                asyncio.run(self.ticker_manager.start_batch())
            finally:
                # close dialog safely from main thread
                self.root.after(0, lambda: loading.destroy())
        # ====== local function
        # Run task in background thread
        Thread(target=task, daemon=True).start()
        print("Fetching market data...")
        
    def fetch_fundamentals(self):
        """Fetch fundamental data - integrate with MarginFetcher"""
        loading = tk.Toplevel(self.root)
        loading.title("Aguarde...")
        loading.geometry("250x100")
        loading.transient(self.root)
        loading.grab_set()  # make it modal
        loading.resizable(False, False)
        # Center it relative to main window
        self.root.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (250 // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (100 // 2)
        loading.geometry(f"250x100+{x}+{y}")
        tk.Label(loading, text="Carregando dados fundamentais...", font=("Arial", 10, "bold")).pack(pady=20)
        progress = ttk.Progressbar(loading, mode="indeterminate")
        progress.pack(fill="x", padx=20)
        progress.start()
        # ====== local function
        def task():
            try:
                # heavy operation
                asyncio.run(self.fundamental_fetcher.fetch_all_margins())
            finally:
                # close dialog safely from main thread
                self.root.after(0, lambda: loading.destroy())
        # ====== local function
        # Run task in background thread
        Thread(target=task, daemon=True).start()
        print("Fetching fundamentals...")

    def run_analysis_background(self):
        """Runs analysis in a separate thread."""
        loading = tk.Toplevel(self.root)
        loading.title("Aguarde...")
        loading.geometry("250x100")
        loading.transient(self.root)
        loading.grab_set()  # make it modal
        loading.resizable(False, False)
        # Center it relative to main window
        self.root.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (250 // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (100 // 2)
        loading.geometry(f"250x100+{x}+{y}")
        tk.Label(loading, text="Realizando análise...", font=("Arial", 10, "bold")).pack(pady=20)
        progress = ttk.Progressbar(loading, mode="indeterminate")
        progress.pack(fill="x", padx=20)
        progress.start()
        # ====== local function
        def task():
            try:
                # heavy computation here
                results = self.ticker_analyzer.analysis(
                    True, Config.GROSS_MARGIN_FILTER, Config.MARKET_CAP_FILTER
                )
                # schedule GUI update in main thread
                self.root.after(0, lambda: self.display_results(results))
            except Exception as e:
                # capture e safely
                self.root.after(0, lambda e=e: messagebox.showerror("Erro", f"lalala {str(e)}"))
            finally:
                # destroy loading window once task is done (success or fail)
                self.root.after(0, loading.destroy)
        # ======= local function
        Thread(target=task, daemon=True).start()

    def _run_analysis_task(self):
        """Heavy work here, safely separated."""
        try:
            results = self.ticker_analyzer.analysis(True, Config.GROSS_MARGIN_FILTER, Config.MARKET_CAP_FILTER)
            # Once done, update UI from main thread
            self.root.after(0, lambda: self.display_results(results))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Erro", str(e)))
        
    def display_results(self, results):
        """Run analysis on all tickers - integrate with TickerAnalyzer"""
        #messagebox.showinfo("Info", "Executando análise...")
        self.clear_table()   
        sorted_results = dict(sorted(results.items(), key=lambda item: item[1].get('score', 0), reverse=True))
        for ticker in sorted_results.keys():
            all_uptrends = ""
            for timeframe in sorted_results[ticker].keys():
                if timeframe == 'score':
                    continue
                match timeframe:
                    case 'day':
                        display = 'Diário'
                    case 'week':
                        display = 'Semanal'
                    case 'month':
                        display = 'Mensal'
                    case '60min':
                        display = '60 Minutos'
                    case '15min':
                        display = '15 Minutos'
                #print(type(results[ticker][timeframe]), results[ticker][timeframe])
                num = 0
                for uptrend in sorted_results[ticker][timeframe]['uptrends']:
                    num += 1
                all_uptrends += f"{display}: {num}; "
            self.add_to_table(ticker, sorted_results[ticker]['score'], all_uptrends)
        
    def add_stock(self):
        """Add a new stock ticker"""
        dialog = AddTickerDialog(self.root, "Adicionar Ação")
        ticker = dialog.result
        if ticker:
            if self.validate_ticker(ticker):
                messagebox.showinfo("Sucesso", f"Ação {ticker} adicionada com sucesso!")
            else:
                messagebox.showerror("Erro", f"Ação {ticker} inválida, já presente na lista ou algum erro ocorreu. Tente novamente.")
    
    def remove_stock(self):
        """Remove a stock ticker"""
        selected = self.get_selected_ticker()
        if selected:
            ticker = selected
        else:
            dialog = AddTickerDialog(self.root, "Remover Ação")
            ticker = dialog.result
        if ticker:
            confirm = messagebox.askyesno("Confirmar", f"Tem certeza que deseja remover {ticker}?")
            if confirm:
                list_usa = Cache.load_list_usa()
                if ticker in list_usa:
                    list_usa.remove(ticker)
                else:
                    messagebox.showinfo("Falha", f"Ação {ticker} não está presente na lista de ações.")
                    return
                confirm = Cache.save_list_usa(list_usa)
                if confirm is None:
                    messagebox.showinfo("Falha", f"Ação {ticker} não pode ser removida. Tente novamente")
                else:
                    messagebox.showinfo("Sucesso", f"Ação {ticker} removida!")
    
    def validate_ticker(self, ticker):
        """Validate if ticker exists"""
        ticker = ticker.upper().strip()
        if len(ticker) < 2 or len(ticker) > 10:
            return False
        list_all = Cache.load_list_br() + Cache.load_list_usa()
        list_valid_tickers = [ticker for ticker in list_all if ticker not in Cache.load_list_dead()]
        if ticker in list_valid_tickers:
            return False
        list_usa = Cache.load_list_usa()
        list_usa.append(ticker)
        confirm = Cache.save_list_usa(list_usa)
        if not confirm:
            return False
        return True

class FilterCreatorDialog:
    """Dialog window for creating custom filters"""
    def __init__(self, parent):
        self.result = None
        
        # Default values from Config
        self.default_timeframe = {
            'day': 0.2,
            'week': 0.45,
            'month': 0.3,
            '15min': 0,
            '60min': 0.05
        }
        self.default_ma = {
            'ema400': 0.05,
            'sma200': 0.15,
            'sma50': 0.35,
            'sma21': 0.3,
            'ema9': 0.15
        }
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Criar Filtro")
        #self.dialog.geometry("500x650")
        self.dialog.transient(parent)
        self.dialog.grab_set()   
        # Center the dialog
        self.dialog.update_idletasks()
        #x = (self.dialog.winfo_screenwidth() // 2) - (250)
        #y = (self.dialog.winfo_screenheight() // 2) - (325)
        #self.dialog.geometry(f"500x650+{x}+{y}")
        # Create scrollable frame
        main_frame = tk.Frame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        # Filter name
        name_frame = tk.Frame(main_frame)
        name_frame.pack(fill="x", pady=(0, 15))
        tk.Label(name_frame, text="Nome do Filtro:", font=("Arial", 11, "bold")).pack(anchor="w")
        self.name_entry = tk.Entry(name_frame, font=("Arial", 11), width=40)
        self.name_entry.pack(fill="x", pady=5)
        # Timeframe weights section
        tf_label = tk.Label(main_frame, text="Pesos dos Timeframes:", font=("Arial", 11, "bold"))
        tf_label.pack(anchor="w", pady=(10, 5))
        tf_frame = tk.LabelFrame(main_frame, text="Timeframes", padx=10, pady=10)
        tf_frame.pack(fill="x", pady=5)
        self.tf_entries = {}
        for i, (key, value) in enumerate(self.default_timeframe.items()):
            row_frame = tk.Frame(tf_frame)
            row_frame.pack(fill="x", pady=3)
            match key:
                case 'day':
                    display = 'Diário'
                case 'week':
                    display = 'Semanal'
                case 'month':
                    display = 'Mensal'
                case '60min':
                    display = '60 Minutos'
                case '15min':
                    display = '15 Minutos'
            tk.Label(row_frame, text=f"{display}:", width=10, anchor="w").pack(side="left")
            entry = tk.Entry(row_frame, width=15)
            entry.insert(0, str(value))
            entry.pack(side="left", padx=5)
            self.tf_entries[key] = entry
        # MA weights section
        ma_label = tk.Label(main_frame, text="Pesos das Médias Móveis:", font=("Arial", 11, "bold"))
        ma_label.pack(anchor="w", pady=(15, 5))
        ma_frame = tk.LabelFrame(main_frame, text="Médias Móveis", padx=10, pady=10)
        ma_frame.pack(fill="x", pady=5)
        self.ma_entries = {}
        for key, value in self.default_ma.items():
            row_frame = tk.Frame(ma_frame)
            row_frame.pack(fill="x", pady=3)
            match key:
                case 'ema400':
                    display = 'mme400'
                case 'sma200':
                    display = 'mms200'
                case 'sma50':
                    display = 'mms50'
                case 'sma21':
                    display = 'mms21'
                case 'ema9':
                    display = 'mme9'
            tk.Label(row_frame, text=f"{display}:", width=10, anchor="w").pack(side="left")
            entry = tk.Entry(row_frame, width=15)
            entry.insert(0, str(value))
            entry.pack(side="left", padx=5)
            self.ma_entries[key] = entry
        self.default_gross_margin = 0.1
        self.default_market_cap = 200
        self.default_percentile = .95
        # Gross Margin Filter section (slider)
        gm_label = tk.Label(main_frame, text="Filtro de Qualidade:", font=("Arial", 11, "bold"))
        gm_label.pack(anchor="w", pady=(15, 5))
        gm_frame = tk.LabelFrame(main_frame, text="Margem Bruta (%)", padx=10, pady=10)
        gm_frame.pack(fill="x", pady=5)
        # Slider value label
        self.gm_value_label = tk.Label(gm_frame, text=f"{self.default_gross_margin * 100:.1f}%", font=("Arial", 10, "bold"))
        self.gm_value_label.pack()
        # Slider
        self.gross_margin_slider = tk.Scale(
            gm_frame,
            from_=0,
            to=100,
            orient="horizontal",
            resolution=1,
            length=400,
            showvalue=False,
            command=self.update_gm_label
        )
        self.gross_margin_slider.set(self.default_gross_margin * 100)
        self.gross_margin_slider.pack(pady=5)
        # Percentil melhores ações
        perc_frame = tk.LabelFrame(main_frame, text="Melhores Ações (%)", padx=10, pady=10)
        perc_frame.pack(fill="x", pady=5)
        # Slider value label
        self.perc_value_label = tk.Label(perc_frame, text=f"{self.default_gross_margin * 100:.1f}%", font=("Arial", 10, "bold"))
        self.perc_value_label.pack()
        # Slider
        self.percentile_slider = tk.Scale(
            perc_frame,
            from_=0,
            to=100,
            orient="horizontal",
            resolution=1,
            length=400,
            showvalue=False,
            command=self.update_perc_label
        )
        self.percentile_slider.set(self.default_percentile * 100)
        self.percentile_slider.pack(pady=5)
        # Market Cap Filter section (input)
        mc_label = tk.Label(main_frame, text="Filtro de Valor de Mercado:", font=("Arial", 11, "bold"))
        mc_label.pack(anchor="w", pady=(15, 5))
        mc_frame = tk.LabelFrame(main_frame, text=" (USD)", padx=10, pady=10)
        mc_frame.pack(fill="x", pady=5)
        mc_row = tk.Frame(mc_frame)
        mc_row.pack(fill="x", pady=3)
        tk.Label(mc_row, text="Valor Máximo:", width=15, anchor="w").pack(side="left")
        self.market_cap_entry = tk.Entry(mc_row, width=20)
        self.market_cap_entry.insert(0, str(self.default_market_cap))
        self.market_cap_entry.pack(side="left", padx=5)
        tk.Label(mc_row, text="(Valor em Bilhões de Dólares)", font=("Arial", 8)).pack(side="left", padx=5)
        # Buttons
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(pady=20) 
        btn_save = tk.Button(
            btn_frame,
            text="Salvar",
            command=self.save_clicked,
            width=12,
            bg="#66BB6A",
            fg="white",
            font=("Arial", 10, "bold"),
            relief="flat",
            borderwidth=0
        )
        btn_save.pack(side="left", padx=5)
        btn_cancel = tk.Button(
            btn_frame,
            text="Cancelar",
            command=self.cancel_clicked,
            width=12,
            bg="#EF5350",
            fg="white",
            font=("Arial", 10, "bold"),
            relief="flat",
            borderwidth=0
        )
        btn_cancel.pack(side="left", padx=5)
        self.dialog.wait_window()
    
    def update_gm_label(self, value):
        """Update the gross margin percentage label"""
        self.gm_value_label.config(text=f"{float(value):.1f}%")

    def update_perc_label(self, value):
        self.perc_value_label.config(text=f"{float(value):.1f}%")
    
    def save_clicked(self):
        """Save the filter configuration"""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Erro", "Digite um nome para o filtro!")
            return
        try:
            timeframe_np_array = []
            # Get timeframe weights
            timeframe_weights = {}
            for key, entry in self.tf_entries.items():
                timeframe_np_array.append(float(entry.get()))
            timeframe_np_array = np.array(timeframe_np_array)
            ma_np_array = []
            # Get MA weights
            ma_weights = {}
            for key, entry in self.ma_entries.items():
                print()
                ma_np_array.append(float(entry.get()))
            # normalize
            ma_np_array = np.array(ma_np_array)
            # print(f" - {timeframe_np_array.max()}. -- {timeframe_np_array.min()} / - {ma_np_array.max()}. -- {ma_np_array.min()}")
            timeframe_np_array = timeframe_np_array / timeframe_np_array.sum()
            i = 0
            for key in self.tf_entries.keys():
                timeframe_weights[key] = timeframe_np_array[i]
                i += 1
            ma_np_array = ma_np_array / ma_np_array.sum()
            i = 0
            for key in self.ma_entries.keys():
                ma_weights[key] = ma_np_array[i]
                i += 1
            gm_value = self.gross_margin_slider.get() / 100 
            mc_value = float(self.market_cap_entry.get()) * 1000000000
            perc_value = self.percentile_slider.get()
            self.result = {
                'name': name,
                'timeframe_weights': timeframe_weights,
                'ma_weights': ma_weights,
                'gross_margin': gm_value,
                'market_cap': mc_value,
                'percentile': perc_value
            }
            self.dialog.destroy()
            
        except ValueError:
            messagebox.showerror("Erro", "Valores inválidos! Use números decimais (ex: 0.5)")
    
    def cancel_clicked(self):
        """Cancel filter creation"""
        self.result = None
        self.dialog.destroy()


class AddTickerDialog:
    """Dialog window for adding/removing tickers"""
    def __init__(self, parent, title):
        self.result = None
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("300x150")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (150)
        y = (self.dialog.winfo_screenheight() // 2) - (75)
        self.dialog.geometry(f"300x150+{x}+{y}")
        
        # Label
        label = tk.Label(
            self.dialog,
            text="Digite o símbolo da ação:",
            font=("Arial", 11)
        )
        label.pack(pady=20)
        
        # Entry
        self.entry = tk.Entry(self.dialog, font=("Arial", 12), width=20)
        self.entry.pack(pady=10)
        self.entry.focus()
        
        # Buttons frame
        btn_frame = tk.Frame(self.dialog)
        btn_frame.pack(pady=10)
        
        btn_ok = tk.Button(
            btn_frame,
            text="OK",
            command=self.ok_clicked,
            width=10,
            bg="#7FBBEC",
            fg="white",
            relief="flat",
            borderwidth=0
        )
        btn_ok.pack(side="left", padx=5)
        
        btn_cancel = tk.Button(
            btn_frame,
            text="Cancelar",
            command=self.cancel_clicked,
            width=10,
            bg="#7FBBEC",
            fg="white",
            relief="flat",
            borderwidth=0
        )
        btn_cancel.pack(side="left", padx=5)
        self.entry.bind("<Return>", lambda e: self.ok_clicked())
        self.dialog.wait_window()
    
    def ok_clicked(self):
        self.result = self.entry.get().strip().upper()
        self.dialog.destroy()
    
    def cancel_clicked(self):
        self.result = None
        self.dialog.destroy()

async def main():
    #ticker_manager = TickerManager()
    #analyzer = TickerAnalyzer()
    #await ticker_manager.start_batch()
    #analyzer.analysis(True, Config.GROSS_MARGIN_FILTER, Config.MARKET_CAP_FILTER)
    #await ticker_manager.disconnect()
    #margin_fetcher = MarginFetcher()
    #await margin_fetcher.fetch_all_margins()
    root = tk.Tk()
    app = StockEvaluatorUI(root)
    root.mainloop()

if __name__ == "__main__":
    asyncio.run(main())