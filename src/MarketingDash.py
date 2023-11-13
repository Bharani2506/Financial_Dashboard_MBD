###############################################################################
# FINANCIAL DASHBOARD EXAMPLE - v3
###############################################################################

#==============================================================================
# Initiating
#==============================================================================

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
import pandas_datareader.data as web
from numerize import numerize
from plotly.subplots import make_subplots




import requests
import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret
#==============================================================================
# Header
#==============================================================================

def render_header():
    """
    This function render the header of the dashboard with the following items:
        - Title
        - Dashboard description
        - 3 selection boxes to select: Ticker, Start Date, End Date
    """
    # Add dashboard title and description
    #st.title("Marketing Insights💡")
    st.markdown("<h1 style='text-align: center; color: grey;'>Marketing Insights💡</h1>", unsafe_allow_html=True)
    # Add the ticker selection on the sidebar
    # Get the list of stock tickers from S&P500
   
#==============================================================================
# Tab 1
#==============================================================================
def render_tab1():
    """
    This function render the Tab 1 - Company Profile of the dashboard.
    """
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    get = st.button("Refresh", key="get")
    # Add the selection boxes
    col1, col2, col3 = st.columns(3)  # Create 3 columns
    # Ticker name
    global ticker  # Set this variable as global, so the functions in all of the tabs can read it
    ticker = col1.selectbox("Ticker", ticker_list)
    # Begin and end dates
    global start_date, end_date  # Set this variable as global, so all functions can read it
    start_date = col2.date_input("Start date", datetime.today().date() - timedelta(days=30))
    end_date = col3.date_input("End date", datetime.today().date())
    # Show to stock image
    col1, col2, col3 = st.columns([1, 3, 1])
    col2.image('./img/stock.jpg', use_column_width=True,
               caption='Company Stock Information')
    # Get the company information
    @st.cache_data
    def GetCompanyInfo(ticker):
        """
        This function get the company information from Yahoo Finance.
        """
        return YFinance(ticker).info
   # Get the stock price data for the selected duration
    @st.cache_data
    def GetStockData(ticker, duration):
        end_date = datetime.today().date()
        if duration == '1M':
            start_date = end_date - timedelta(days=30)
        elif duration == '3M':
            start_date = end_date - timedelta(days=3 * 30)
        elif duration == '6M':
            start_date = end_date - timedelta(days=6 * 30)
        elif duration == 'YTD':
            start_date = datetime(end_date.year, 1, 1).date()
        elif duration == '1Y':
            start_date = end_date - timedelta(days=365)
        elif duration == '3Y':
            start_date = end_date - timedelta(days=3 * 365)
        elif duration == '5Y':
            start_date = end_date - timedelta(days=5 * 365)
        elif duration == 'MAX':
            start_date = None  # Fetch data from the beginning
        else:
            st.error("Invalid duration selected.")
            return None
        stock_df = yf.Ticker(ticker).history(start=start_date, end=end_date)
        stock_df.reset_index(inplace=True)  # Drop the indexes
        stock_df['Date'] = stock_df['Date'].dt.date  # Convert date-time to date
        return stock_df
    # If the ticker is already selected
    if ticker != '':
        # Get the company information in list format
        info = GetCompanyInfo(ticker)
        # Show the company description using markdown + HTML
        st.write('**1. Business Summary:**')
        st.markdown('<div style="text-align: justify;">' + \
                    info['longBusinessSummary'] + \
                    '</div><br>',
                    unsafe_allow_html=True)
        # Show some statistics as a DataFrame
        st.write('**2. Key Statistics:**')
        info_keys = {'previousClose':'Previous Close',
                     'open'         :'Open',
                     'bid'          :'Bid',
                     'ask'          :'Ask',
                     'marketCap'    :'Market Cap',
                     'volume'       :'Volume'}
        company_stats = {}  # Dictionary
        for key in info_keys:
            company_stats.update({info_keys[key]:info[key]})
        company_stats = pd.DataFrame({'Value':pd.Series(company_stats)})  # Convert to DataFrame
        st.dataframe(company_stats)
        # Dropdown for selecting different durations
        st.write('**3.Stock Price Line Graph**')
        duration_options = ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'MAX']
        selected_duration = st.selectbox('Select Duration:', duration_options)
        # Get stock price data for the selected duration
        stock_price = GetStockData(ticker, selected_duration)
        # Plot the line graph of stock prices
        fig, ax = plt.subplots()
        ax.plot(stock_price['Date'], stock_price['Close'], label='Closing Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price')
        ax.set_title(f'Stock Price Over Time ({selected_duration})')
        ax.legend()
        st.pyplot(fig)
    @st.cache
    def GetMajorHolders(ticker):
        for tick in ticker:
            holders = yf.Ticker(tick).major_holders
            holders = holders.rename(columns={0:"Value", 1:"Breakdown"})
            holders = holders.set_index('Breakdown')
            holders.loc[['Number of Institutions Holding Shares']].style.format({'Number of Institutions Holding Shares': '{:0,.0f}'.format})
        return holders
    
   
    holders = GetMajorHolders([ticker])
    st.write('**4.Major Holders**')
    st.table(holders)
    
#==============================================================================
# Tab 2
#==============================================================================
def fetch_close_price(ticker):
    stock_name = yf.Ticker(ticker)
    stock_price = stock_name.history(period='1mo')
    return stock_price['Close']

# Function to run Monte Carlo simulation
def monte_carlo_simulation(close_price, n_simulations, time_period):
    daily_return = close_price.pct_change()
    daily_volatility = np.std(daily_return)
    
    simulation_df = pd.DataFrame()
    
    for i in range(n_simulations):
        next_price = []
        last_price = close_price[-1]
        
        for _ in range(time_period):
            future_return = np.random.normal(0, daily_volatility)
            future_price = last_price * (1 + future_return)
            next_price.append(future_price)
            last_price = future_price
        
        simulation_df[i] = next_price
    
    return simulation_df

# Function to plot Monte Carlo simulation
def plot_monte_carlo_simulation(mc_sim, close_price, time_period):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(mc_sim)
    plt.title(f'Monte Carlo simulation for stock in next {time_period} days')
    plt.xlabel('Day')
    plt.ylabel('Price')
    
    plt.axhline(y=close_price[-1], color='red')
    plt.legend([f'Current stock price is: {np.round(close_price[-1], 2)}'])
    ax.get_legend().legendHandles[0].set_color('red')
    
    st.pyplot(fig)

# Function to calculate and plot Value at Risk (VaR)
def calculate_and_plot_var(mc_sim, close_price):
    ending_price = mc_sim.iloc[-1, :].values
    fig1, ax = plt.subplots(figsize=(15, 10))
    ax.hist(ending_price, bins=50)
    
    percentile_5 = np.percentile(ending_price, 5)
    plt.axvline(percentile_5, color='red', linestyle='--', linewidth=1)
    plt.legend([f'5th Percentile of the Future Price: {np.round(percentile_5, 2)}'])
    
    plt.title('Distribution of the Ending Price')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    
    st.pyplot(fig1)
    
    VaR = close_price[-1] - percentile_5
    st.subheader('Value at Risk (VaR)')
    st.write(f'VaR at 95% confidence interval is: {np.round(VaR, 2)} USD')

# Function to render Tab 4
def render_tab4():
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    ticker = st.selectbox("Select a ticker", ticker_list, index=45, key="unique")
    
    if ticker != '-':
        close_price = fetch_close_price(ticker)
        n_simulations = st.selectbox("Number of Simulations", [200, 500, 1000])
        time_period = st.selectbox("Time Horizon", [30, 60, 90])
        
        mc_sim = monte_carlo_simulation(close_price, n_simulations, time_period)
        plot_monte_carlo_simulation(mc_sim, close_price, time_period)
        calculate_and_plot_var(mc_sim, close_price)
#==============================================================================
# Tab 3
#==============================================================================
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    stock_df = yf.Ticker(ticker).history(start=start_date, end=end_date)
    stock_df.reset_index(inplace=True)
    stock_df['Date'] = stock_df['Date'].dt.date
    return stock_df

# Function to render the line chart for stock prices
def render_stock_prices_line_chart(ticker):
    start_date_stock = st.date_input("Select start date for stock prices:", pd.to_datetime('today') - pd.DateOffset(days=365))
    end_date_stock = st.date_input("Select end date for stock prices:", pd.to_datetime('today'))
    stock_data = get_stock_data(ticker, start_date_stock, end_date_stock)
    st.line_chart(stock_data.set_index('Date')['Close'], use_container_width=True)

# Function to render financial statements tab
def render_financial_statements_tab(ticker, statement_type):
    tabA, tabB = st.tabs(["Annual", "Quarter"])
    with tabA:
        financial_statement = getattr(yf.Ticker(ticker), statement_type)
        financial_statement = financial_statement.rename(lambda t: t.strftime('%Y-%m-%d'), axis='columns')
        financial_statement = financial_statement.astype(float).apply(np.floor)
        st.dataframe(financial_statement.style.format(formatter='{:,.0f}'))
    with tabB:
        quarterly_statement = getattr(yf.Ticker(ticker), f"quarterly_{statement_type}")
        quarterly_statement = quarterly_statement.rename(lambda t: t.strftime('%Y-%m-%d'), axis='columns')
        quarterly_statement = quarterly_statement.astype(float).apply(np.floor)
        st.dataframe(quarterly_statement.style.format(formatter='{:,.0f}'))

# Function to render financial statements and stock prices
def render_financial_statements_and_stock_prices():
    st.title("Financial Statements and Stock Prices")

    # Ticker selection
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    ticker = st.selectbox("Select a ticker", ticker_list, index=45)

    # Render stock prices line chart
    render_stock_prices_line_chart(ticker)

    # Tabs for financial statements
    tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])

    # Income Statement
    with tab1:
        render_financial_statements_tab(ticker, "financials")

    # Balance Sheet
    with tab2:
        render_financial_statements_tab(ticker, "balance_sheet")

    # Cash Flow
    with tab3:
        render_financial_statements_tab(ticker, "cashflow")
#==============================================================================
# Tab 4
#==============================================================================
def render_tab2():
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    ticker_selectbox_key = "unique_ticker_selectbox"
    ticker = col1.selectbox("Select a ticker", ticker_list, index=45, key=ticker_selectbox_key)
    start_date = col2.date_input("Select Start Date", datetime.today().date() - timedelta(days=30))
    end_date = col3.date_input("Select End Date", datetime.today().date())
    duration = col4.selectbox("Select Duration", ['Date Range', '1mo', '3mo', '6mo', 'ytd','1y', '3y','5y', 'max'])          
    interval = col5.selectbox("Select Interval", ['1d', '1mo', '1y'])
    plot = col6.selectbox("Select Plot", ['Candle', 'Line'])
    
    @st.cache_data
    def GetStockData(ticker, start_date, end_date):
        stock_df = yf.Ticker(ticker).history(start=start_date, end=end_date)
        stock_df.reset_index(inplace=True)
        stock_df['Date'] = stock_df['Date'].dt.date
        return stock_df
        
        if duration != 'Date Range':        
            stock_price = yf.download(ticker, period = duration, interval = interval)
            stock_price = stock_price.reset_index()
            return stock_price
        else:
            stock_price = yf.download(ticker, start_date, end_date, interval = interval)
            stock_price = stock_price.reset_index()
            return stock_price
        
    stock_price = GetStockData(ticker, start_date, end_date) 
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    if plot == 'Candle':
        fig.add_trace(go.Candlestick(x=stock_price['Date'], open=stock_price['Open'], high=stock_price['High'], low=stock_price['Low'], close=stock_price['Close'],increasing_fillcolor= 'red', increasing_line_color='rgba(0, 0, 0, 0)',decreasing_fillcolor= 'cyan',decreasing_line_color='rgba(0, 0, 0, 0)'))
        fig.add_trace(go.Bar(x = stock_price['Date'], y = stock_price['Volume'], name = 'Volume', marker_color = 'white', marker_line_color='rgba(0, 0, 0, 0)'), secondary_y = True)
        fig.add_trace(go.Scatter(x=stock_price['Date'], y=stock_price['Close'].rolling(50).mean(), opacity=0.7, line=dict(color='yellow', width=2), name='50-Day MA'))
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", showline=False, spikedash="dot", spikecolor="white", spikethickness=0.1)
        fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor", showline=False, spikedash="dot", spikecolor="white", spikethickness=0.1)
        fig.update_yaxes(range=[0, stock_price['Volume'].max()*5], autorange=False, showticklabels=False, secondary_y=True, showgrid=False)
        fig.update_layout(margin=go.layout.Margin(l=20, r=20, b=20, t=20))
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                           'paper_bgcolor': 'rgba(0, 0, 0, 0)',}, 
                           autosize=True, width=750, height=330,
                           xaxis=dict(showgrid=False),
                           yaxis=dict(showgrid=False),
                           xaxis_rangeslider_visible=False,
                           showlegend = False)
        
    elif plot == 'Line':
        fig.add_trace(go.Scatter(x=stock_price['Date'], y=stock_price['Close'], mode='lines', name = 'Close', marker_color = 'cyan'), secondary_y = False)
        fig.add_trace(go.Bar(x = stock_price['Date'], y = stock_price['Volume'], name = 'Volume', marker_color = 'white', marker_line_color='rgba(0, 0, 0, 0)'), secondary_y = True)
        fig.add_trace(go.Scatter(x=stock_price['Date'], y=stock_price['Close'].rolling(50).mean(), opacity=0.7, line=dict(color='yellow', width=2), name='50-Day MA'))
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", showline=False, spikedash="dot", spikecolor="white", spikethickness=0.1)
        fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor", showline=False, spikedash="dot", spikecolor="white", spikethickness=0.1)
        fig.update_yaxes(range=[0, stock_price['Volume'].max()*5], autorange=False, showticklabels=False, secondary_y=True, showgrid=False)
        fig.update_layout(margin=go.layout.Margin(l=20, r=20, b=20, t=20))
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                           'paper_bgcolor': 'rgba(0, 0, 0, 0)',}, 
                           autosize=True, width=750, height=330,
                           xaxis=dict(showgrid=False),
                           yaxis=dict(showgrid=False),
                           xaxis_rangeslider_visible=False,
                           showlegend = False)
    
    st.plotly_chart(fig)
   
    # Add options to view historical data and export to excel
    col1, col2 = st.columns([4,1])
    show_data = col1.checkbox("Show Historical Data")
    export_data = col2.button("Export to Excel")

    if show_data:
        stock_price = stock_price.sort_values('Date', ascending=False)
        stock_price = stock_price.set_index('Date')
        stock_price = stock_price.style.format({'Open': '{:,.2f}'.format,
                                   'High': '{:,.2f}'.format,
                                   'Low': '{:,.2f}'.format,
                                   'Close': '{:,.2f}'.format,
                                   'Adj Close': '{:,.2f}'.format,
                                   'MA': '{:,.2f}'.format,
                                   'Volume': '{:0,.0f}'.format})
        st.table(stock_price)
    if export_data:
       stock_price.to_excel(r'C:/Users/bmurugesan1/Documents/Streamlit/Stock_data.xlsx', index=False)
#==============================================================================
# Tab 5
#==============================================================================
def render_tab5():
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    ticker_selectbox_key = "blah"
    ticker = st.selectbox("Select a ticker", ticker_list, index=45, key=ticker_selectbox_key)
    tab1, tab2 = st.tabs(["Top Institutional Holders", "Top Mutual Fund Holders"])
    with tab1:
        @st.cache
        def GetInstHolders(ticker):
            for tick in ticker:
                inst_holders = yf.Ticker(tick).institutional_holders
                inst_holders['Shares'] = [numerize.numerize(y) for y in  inst_holders['Shares']]
                inst_holders['Value'] = [numerize.numerize(y) for y in  inst_holders['Value']]
                #inst_holders = inst_holders.style.format({'% Out': '{:0,.0f}'.format})
                inst_holders = inst_holders.set_index('Holder')
                return inst_holders
    
   
        inst_holders = GetInstHolders([ticker])
        st.write('**Top Institutional Holders**')
        st.table(inst_holders)
    with tab2:
        @st.cache
        def GetFundHolders(ticker):
            for tick in ticker:
                fund_holders = yf.Ticker(tick).fund_holders
                fund_holders['Shares'] = [numerize.numerize(y) for y in  fund_holders['Shares']]
                return fund_holders
    
    
        fund_holders = GetInstHolders([ticker])
        st.write('**Top Mutual Fund Holders**')
        st.table(fund_holders)
#==============================================================================
# Main body
#==============================================================================
# Render the header
render_header()

# Render the tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Chart", "Financials", "MonteCarlo", "Holders"])
with tab1:
    render_tab1()
with tab2:
    render_tab2()
with tab3:
    render_financial_statements_and_stock_prices()
with tab4:
    render_tab4()
with tab5:
    render_tab5()    
#Customize the dashboard with CSS
st.markdown(
     """
     <style>
         .stApp {
             background: #F0F8FF;
         }
     </style>
     """,
     unsafe_allow_html=True,
)
    
###############################################################################
# END
###############################################################################
