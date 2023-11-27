import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from xbbg import blp
from datetime import date
import itertools
import math
from joblib import Parallel, delayed
import streamlit as st 
from babel.numbers import format_currency

def target_price_df(ticker):
    tp_df = pd.DataFrame(columns = ['Best TP', 'Value'])
    
    hi_val = blp.bdp(ticker, 'BEST_TARGET_HI').loc[ticker][0]
    mean_val = blp.bdp(ticker, 'BEST_TARGET_MEDIAN').loc[ticker][0]
    low_val = blp.bdp(ticker, 'BEST_TARGET_LO').loc[ticker][0]
    upside_pct = (-(blp.bdp(ticker,'PX_LAST').loc[ticker][0] - mean_val)/mean_val)*100
    
    vals = [hi_val, mean_val, low_val, upside_pct]
    labels = ['Hi', 'Mean', 'Lo', 'Upside_%']
    tp_df['Best TP'] = labels 
    tp_df['Value'] = vals
    tp_df = tp_df.round(2)
    return tp_df

def get_bdp_data(ticker,field,year):
    df = blp.bdp(ticker,field,EQY_FUND_YEAR = year)
    value= 0 
    
    if df.empty:
        value= 1
    
    else:
        value =blp.bdp(ticker,field,EQY_FUND_YEAR = year)
        value = value[field.lower()].iloc[0]
        value = round(value,2)
        
    return value

def best_bdp(ticker,field,period):
    df = blp.bdp(ticker,field, BEST_FPERIOD_OVERRIDE=period)
    value = 0
    if df.empty:
        value= 1
    
    else:
        value =blp.bdp(ticker,field,BEST_FPERIOD_OVERRIDE = period)
        value = value[field.lower()].iloc[0]
    return value

def calc_margin(other_sales,total_sales):
    return (other_sales/total_sales)*100
    
def calc_return(initial_value, final_value):
    val = ((final_value-initial_value)/initial_value) * 100
    val = round(val,2)
    return val

def RunParallelGetBdp(stock,field):
    years = ['-3FY',"-2FY", "-1FY","0FY"]
    return Parallel(n_jobs=-1)(delayed(get_bdp_data)(stock,field,year) for year in years) 

def financial_statement_df(ticker):
    labels = [str(i) for i in range(2019,2023)]
    index_labels = ["Sales","Ebit","NI","OPM","NPM", "Sales g",'Ebit g',
                    "NI g", "ROA","ROE"]
    fs_df = pd.DataFrame(index = index_labels,columns = labels)
    
    sales = RunParallelGetBdp(ticker,"SALES_REV_TURN")
    
    ebit =  RunParallelGetBdp(ticker,"EBIT")
    
    ebitda =  RunParallelGetBdp(ticker,"EBITDA")
    
    net_income = RunParallelGetBdp(ticker,"NET_INCOME")
    
    operating_margin = RunParallelGetBdp(ticker,"OPER_MARGIN")
    
    net_margin = RunParallelGetBdp(ticker,"PROF_MARGIN")
    
    sales_g =  RunParallelGetBdp(ticker,"SALES_GROWTH")
    
    ebit_g = RunParallelGetBdp(ticker,"EBIT_MARGIN")
    
    ni_g = RunParallelGetBdp(ticker,"NET_INC_GROWTH")
    
    roa =  RunParallelGetBdp(ticker,"RETURN_ON_ASSET")
        
    roe = RunParallelGetBdp(ticker,"RETURN_COM_EQY")
    

    
    fs_df.loc['Sales'] = sales
    fs_df.loc['Ebit'] = ebit
    fs_df.loc['EBITDA'] = ebitda
    fs_df.loc['NI'] = net_income
    fs_df.loc["OPM"] = operating_margin
    fs_df.loc["NPM"] = net_margin
    fs_df.loc["Sales g"] = sales_g
    fs_df.loc["Ebit g"] = ebit_g
    fs_df.loc["NI g"] = ni_g
    fs_df.loc["ROA"] = roa
    fs_df.loc["ROE"] = roe
    
    fs_df = fs_df.round(2) 
    return fs_df
    

# get eps data from 2018 to 2022
def get_eps_data(ticker):
    year = [str(i) for i in range(2018,2024)]
    eps_vals = []
    periods = ['-4FY','-3FY','-2FY','-1FY','0FY']
    
    for period in periods: 
        eps_vals.append(blp.bdp(ticker, "IS_EPS",EQY_FUND_YEAR=period).iloc[0][0])
    
    eps_vals.append(blp.bdp(ticker,"TRAIL_12M_EPS").iloc[0][0])
    
    eps_df = pd.DataFrame(columns=['Year','EPS'])
    eps_df['Year'] = year
    eps_df['EPS'] = eps_vals
    eps_df = eps_df.set_index('Year')
    eps_df.index = pd.to_datetime(eps_df.index)


    diff = eps_df.diff().values.tolist()
    diff = list(itertools.chain.from_iterable(diff))
    diff = [round(elem,2) for elem in diff]
    diff = [0 if math.isnan(x) else x for x in diff]

    eps_df = eps_df.resample("D").asfreq()
    eps_df['year'] = eps_df.index.year
    
    for i in range(len(eps_df)):
        if (eps_df.iloc[i,1] == 2018) & ( math.isnan(eps_df.iloc[i,0])):
            eps_df.iloc[i,0] = eps_df.iloc[i-1,0] + (diff[1]/365)
   
        elif (eps_df.iloc[i,1] == 2019) & ( math.isnan(eps_df.iloc[i,0])):
            eps_df.iloc[i,0] = eps_df.iloc[i-1,0] + (diff[2]/365)
        
        elif(eps_df.iloc[i,1] == 2020) & ( math.isnan(eps_df.iloc[i,0])):
            eps_df.iloc[i,0] = eps_df.iloc[i-1,0] + (diff[3]/365)
        
        elif(eps_df.iloc[i,1] == 2021) & ( math.isnan(eps_df.iloc[i,0])):
            eps_df.iloc[i,0] = eps_df.iloc[i-1,0] + (diff[4]/365)
    
        elif(eps_df.iloc[i,1] == 2022) & ( math.isnan(eps_df.iloc[i,0])):
            eps_df.iloc[i,0] = eps_df.iloc[i-1,0] + (diff[5]/365)
        
        else:
            eps_df.iloc[i,0] = eps_df.iloc[i,0]
            
    eps_df = eps_df.drop(['year'],axis=1)
            
    return eps_df

def get_price_data(ticker):
    price = blp.bdh(ticker,'PX_LAST','2018-01-01','2022-12-30', Fill='P', Days='A')
    price.columns = price.columns.droplevel()
    price.columns = ['Close']
    price.index = pd.to_datetime(price.index)
    return price

def get_pe_data(ticker):
    eps = get_eps_data(ticker)
    price = get_price_data(ticker)
    pe_df = pd.concat([price,eps],axis=1).dropna()
    pe_df['PE_RATIO'] = round((pe_df["Close"]/pe_df["EPS"]),2)
    return pe_df

def plot_pe(ticker):
    sns.set_style('darkgrid')
    data = blp.bdh(ticker, 'PE_RATIO','2018-01-01',date.today())
    data.columns = ['PE_RATIO']
    mean = data["PE_RATIO"].mean()
    std = data["PE_RATIO"].std()

    plus_one = mean + std
    plus_two = mean + (2*std)
    minus_one = mean - std
    minus_two = mean - (2*std)
    
    plus_one = mean + std
    plus_two = mean + (2*std)
    minus_one = mean - std
    minus_two = mean - (2*std)
    
    
    fig,ax = plt.subplots(figsize=(20,20))
    ax.plot(data.index, data['PE_RATIO'],color='green',linewidth='2',label='PE Ratio')
    plt.axhline(y = mean, color = 'red', linestyle = '--',label= 'mean')
    plt.axhline(y = plus_one, color = 'blue', linestyle = '--',label= '+1')
    plt.axhline(y = plus_two, color = 'black', linestyle = '--',label= '+2')
    plt.axhline(y = minus_one, color = 'purple', linestyle = '--',label= '-1')
    plt.axhline(y = minus_two, color = 'grey', linestyle = '--',label= '-2')
    plt.title("PE Ratio",fontsize=15)
    ax.set_xlabel('Date',fontsize=15)
    ax.set_ylabel('PE Ratio',fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    
    plt.legend(loc=0,fontsize='large',frameon=True, fancybox=True, shadow=True)
    plt.show()

# get book value data data from 2018 to 2022
def get_bv_data(ticker):
    year = [str(i) for i in range(2018,2024)]
    bv_vals = []
    periods = ['-4FY','-3FY','-2FY','-1FY','0FY']
    
    for period in periods: 
        bv_vals.append(blp.bdp(ticker, "BOOK_VAL_PER_SH",EQY_FUND_YEAR=period).iloc[0][0])
    
    bv_vals.append(blp.bdp(ticker,"BEST_EBPS_CUR_YR_MEAN").iloc[0][0])
    
    bv_df = pd.DataFrame(columns=['Year','BVPS'])
    bv_df['Year'] = year
    bv_df['BVPS'] = bv_vals
    bv_df = bv_df.set_index('Year')
    bv_df.index = pd.to_datetime(bv_df.index)


    diff = bv_df.diff().values.tolist()
    diff = list(itertools.chain.from_iterable(diff))
    diff = [round(elem,2) for elem in diff]
    diff = [0 if math.isnan(x) else x for x in diff]

    bv_df = bv_df.resample("D").asfreq()
    bv_df['year'] = bv_df.index.year
    
    for i in range(len(bv_df)):
        if (bv_df.iloc[i,1] == 2018) & ( math.isnan(bv_df.iloc[i,0])):
            bv_df.iloc[i,0] = bv_df.iloc[i-1,0] + (diff[1]/365)
   
        elif (bv_df.iloc[i,1] == 2019) & ( math.isnan(bv_df.iloc[i,0])):
            bv_df.iloc[i,0] = bv_df.iloc[i-1,0] + (diff[2]/365)
        
        elif(bv_df.iloc[i,1] == 2020) & ( math.isnan(bv_df.iloc[i,0])):
            bv_df.iloc[i,0] = bv_df.iloc[i-1,0] + (diff[3]/365)
        
        elif(bv_df.iloc[i,1] == 2021) & ( math.isnan(bv_df.iloc[i,0])):
            bv_df.iloc[i,0] = bv_df.iloc[i-1,0] + (diff[4]/365)
    
        elif(bv_df.iloc[i,1] == 2022) & ( math.isnan(bv_df.iloc[i,0])):
            bv_df.iloc[i,0] = bv_df.iloc[i-1,0] + (diff[5]/365)
        
        else:
            bv_df.iloc[i,0] = bv_df.iloc[i,0]
            
    bv_df = bv_df.drop(['year'],axis=1)
            
    return bv_df

def get_pb_data(ticker):
    bv = get_bv_data(ticker)
    price = get_price_data(ticker)
    pb_df = pd.concat([price,bv],axis=1).dropna()
    pb_df['PB_RATIO'] = round((pb_df["Close"]/pb_df["BVPS"]),2)
    return pb_df

def plot_pb(ticker):
    sns.set_style('darkgrid')
    data = blp.bdh(ticker,"PX_TO_BOOK_RATIO","2018-01-01",date.today())
    data.columns = ['PB_RATIO']
    mean = data["PB_RATIO"].mean()
    std = data["PB_RATIO"].std()
    
    plus_one = mean + std
    plus_two = mean + (2*std)
    minus_one = mean - std
    minus_two = mean - (2*std)
    
    
    fig,ax = plt.subplots(figsize=(20,20))
    ax.plot(data.index, data['PB_RATIO'],color='green',linewidth='2',label='PB RATIO')
    
    plt.axhline(y = mean, color = 'red', linestyle = '--',label= 'mean')
    plt.axhline(y = plus_one, color = 'blue', linestyle = '--',label= '+1')
    plt.axhline(y = plus_two, color = 'black', linestyle = '--',label= '+2')
    plt.axhline(y = minus_one, color = 'purple', linestyle = '--',label= '-1')
    plt.axhline(y = minus_two, color = 'grey', linestyle = '--',label= '-2')
    
    plt.title("PB Ratio",fontsize=15)
    ax.set_xlabel('Date',fontsize=15)
    ax.set_ylabel('PB Ratio',fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    
    plt.legend(loc=0,fontsize='large',frameon=True, fancybox=True, shadow=True)
    plt.show()

def RunParallelGetSales(stock,field):
    years = ["-4FY","-3FY","-2FY", "-1FY","0FY"]
    return Parallel(n_jobs=-1)(delayed(get_bdp_data)(stock,field,year) for year in years)

def get_quarterly_sales_data(ticker):
    sales_df_list = []
    index = ["Q" + str(i) for i in range(1,5)]
    start_years = [str(i)+"-01-01" for i in range(2018,2023)]
    end_years = [str(i)+"-01-01" for i in range(2019,2024)]

    for start, end in zip(start_years,end_years):
        df = blp.bdh(ticker, "SALES_REV_TURN", start, end)
        df.index = pd.to_datetime(df.index)
        df['Year'] = df.index.year
        df['Quarter'] = index
        df = df.reset_index(drop=True)
        sales_df_list.append(df)

    sales_qrt = pd.concat(sales_df_list)
    sales_qrt.columns = ['Sales', 'Year', 'Quarter']
    sales_qrt['Sales'] = sales_qrt['Sales']/10**6
    return sales_qrt

def get_quarterly_sales_data_usd(ticker):
    sales_df_list = []
    index = ["Q" + str(i) for i in range(1,5)]
    start_years = [str(i)+"-01-01" for i in range(2018,2023)]
    end_years = [str(i)+"-01-01" for i in range(2019,2024)]

    for start, end in zip(start_years,end_years):
        df = blp.bdh(ticker, "SALES_REV_TURN", start, end)
        df.index = pd.to_datetime(df.index)
        df['Year'] = df.index.year
        df['Quarter'] = index
        df = df.reset_index(drop=True)
        sales_df_list.append(df)

    sales_qrt = pd.concat(sales_df_list)
    sales_qrt.columns = ['Sales', 'Year', 'Quarter']
    sales_qrt['Sales'] = sales_qrt['Sales']
    return sales_qrt

def get_quarterly_net_margin_data(ticker):
    year = [i for i in range(2018,2023)]
    net_margin = RunParallelGetSales(ticker,"PROF_MARGIN")
    
    margin_df = pd.DataFrame(columns=["Net_Margin", 'Year'])
    margin_df['Net_Margin'] = net_margin
    margin_df['Year'] = year
    return margin_df

def get_quarterly_oper_margin_data(ticker):
    year = [i for i in range(2018,2023)]
    oper_margin = RunParallelGetSales(ticker,"OPER_MARGIN")
    
    margin_df = pd.DataFrame(columns=["Oper_Margin", 'Year'])
    margin_df['Oper_Margin'] = oper_margin
    margin_df['Year'] = year
    return margin_df

def plot_margin(ticker):
    if (blp.bdp(ticker,"EQY_FUND_CRNCY").iloc[0][0] == 'IDR'):
        sales_data = get_quarterly_sales_data(ticker)
        net_margin_data = get_quarterly_net_margin_data(ticker)
        oper_margin_data = get_quarterly_oper_margin_data(ticker)

        fig, ax = plt.subplots(figsize=(10,8))

        ax.ticklabel_format(useOffset=False,style='plain')

        pivot_sales = pd.pivot_table(data=sales_data,index=['Year'],columns=['Quarter'],values=['Sales'])
        pivot_sales.columns = pivot_sales.columns = ['Q1','Q2','Q3',"Q4"]
        pivot_sales.plot(ax=ax,kind='bar',stacked=True)
        ax.legend(pivot_sales.columns)
        ax.set_ylabel("Revenue In IDR Trillions",fontsize=15)
        ax2 = ax.twinx() 

        ln2 = ax2.plot(ax.get_xticks(),oper_margin_data['Oper_Margin'], label = 'OPM', linewidth=3, linestyle= '--', marker='o',color='green')
        ln3 = ax2.plot(ax.get_xticks(),net_margin_data['Net_Margin'], label = 'NPM',linewidth=3, alpha=0.5,marker='o',color='orange')
        ax2.set_ylabel('Margin %', fontsize = 15)
        ax2.legend(loc=9)

        plt.title("Quarterly Margin and Sales",fontsize=15)
        plt.xlabel("Year", fontsize=15)
        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12)
        plt.show()
    
    else:
        sales_data = get_quarterly_sales_data_usd(ticker)
        net_margin_data = get_quarterly_net_margin_data(ticker)
        oper_margin_data = get_quarterly_oper_margin_data(ticker)

        fig, ax = plt.subplots(figsize=(10,8))

        ax.ticklabel_format(useOffset=False,style='plain')

        pivot_sales = pd.pivot_table(data=sales_data,index=['Year'],columns=['Quarter'],values=['Sales'])
        pivot_sales.columns = pivot_sales.columns = ['Q1','Q2','Q3',"Q4"]
        pivot_sales.plot(ax=ax,kind='bar',stacked=True)
        ax.legend(pivot_sales.columns)
        ax.set_ylabel("Revenue In USD Millions",fontsize=15)
        ax2 = ax.twinx() 

        ln2 = ax2.plot(ax.get_xticks(),oper_margin_data['Oper_Margin'], label = 'OPM', linewidth=3, linestyle= '--', marker='o',color='green')
        ln3 = ax2.plot(ax.get_xticks(),net_margin_data['Net_Margin'], label = 'NPM',linewidth=3, alpha=0.5,marker='o',color='orange')
        ax2.set_ylabel('Margin %', fontsize = 15)
        ax2.legend(loc=9)

        plt.title("Quarterly Margin and Sales",fontsize=15)
        plt.xlabel("Year", fontsize=15)
        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12)
        plt.show()

def plot_seasonality(ticker):
    data = get_quarterly_sales_data(ticker)
    pivot_sales = pd.pivot_table(data=data,index=['Year'],columns=['Quarter'],values=['Sales'])
    pivot_sales.columns = pivot_sales.columns = ['Q1','Q2','Q3',"Q4"]
    pivot_sales['Total_Sales'] = pivot_sales.sum(axis=1)

    pivot_perc = pd.DataFrame()
    pivot_perc['Q1'] = ((pivot_sales['Q1'] / pivot_sales["Total_Sales"]) * 100).round(2)
    pivot_perc['Q2'] = ((pivot_sales['Q2'] / pivot_sales["Total_Sales"]) * 100).round(2)
    pivot_perc['Q3'] = ((pivot_sales['Q3'] / pivot_sales["Total_Sales"]) * 100).round(2)
    pivot_perc['Q4'] = ((pivot_sales['Q4'] / pivot_sales["Total_Sales"]) * 100).round(2)
    pivot_perc['Total_Sales_Percentage'] = pivot_perc.sum(axis=1)
    pivot_desc = pivot_perc.describe().drop(['Total_Sales_Percentage'],axis=1)
    
    # Plot for seasonality
    fig,ax = plt.subplots(figsize=(10,8))
    plt.title("Sales Seasonality")

    lns1 = ax.plot(pivot_desc.columns,pivot_desc.loc['min'], label = 'min', linewidth=5)
    lns2 = ax.plot(pivot_desc.columns,pivot_desc.loc['mean'],label = 'mean', linewidth=5)
    lns3 = ax.plot(pivot_desc.columns,pivot_desc.loc['max'], label = 'max', linewidth=5)

    ax.fill_between(pivot_desc.columns , pivot_desc.loc['max'],pivot_desc.loc['min'], color = '#F5F5F5')
    plt.xlabel("Quarter")
    plt.ylabel("Sales %")
    plt.legend(loc=0)
    plt.show()

def get_analyst_rec(ticker):
    rec = blp.bds(ticker, "BEST_ANALYST_RECS_BULK")
    rec = rec[['firm_name','analyst','recommendation','rating','target_price','date','1_year_return']]
    rec['target_price'] = rec['target_price'].apply(lambda x: format_currency(x, currency="IDR", locale="id_ID"))
    rec['1_year_return'] = rec['1_year_return'].apply(lambda x: (x/100))
    rec['1_year_return'] = rec['1_year_return'].apply(lambda x: "{:.2%}".format(x))
    rec['security_name'] = blp.bdp(rec.index[0],'SECURITY_NAME')
    temp_cols= rec.columns.tolist()
    new_cols=temp_cols[-1:] + temp_cols[:-1]
    rec=rec[new_cols]
    return rec[['firm_name','recommendation','target_price','date']]

def main():
    
    st.header("Stock Cards")
    
    ticker = st.sidebar.text_input('Ticker', 'BBCA IJ Equity')
    
    name = blp.bdp(ticker,"SECURITY_NAME").iloc[0][0]
    idx = blp.bdp(ticker,'INDUSTRY_GROUP_INDEX').iloc[0][0]
    country = blp.bdp(ticker,'COUNTRY_FULL_NAME').iloc[0][0]
    
    with st.sidebar:
        
        st.text_input('Security_Name', name)
        st.text_input('Industry_Group_Index',idx)
        st.text_input("Country",country)
    
    col1 , col2 = st.columns((2),gap='Large')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    with col1:
        st.pyplot(plot_pe(ticker))
        st.pyplot(plot_pb(ticker))
        st.pyplot(plot_seasonality(ticker))

              
    with col2:
        st.write(financial_statement_df(ticker))      
        st.pyplot(plot_margin(ticker))
        st.write(target_price_df(ticker))
    
    st.write(get_analyst_rec(ticker))

        
if __name__ == "__main__":
    main()
    


