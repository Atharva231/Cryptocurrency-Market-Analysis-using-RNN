import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

files=["BTCUSD.csv","DASHUSD.csv","DOGEUSD.csv","ETHUSD.csv","LTCUSD.csv","XMRUSD.csv"];
for i in files:
    df = pd.read_csv(i)
    df = df[['Date','Close']] 
    df['Date'] = pd.to_datetime(df.Date,format='%d/%m/%Y')
    df['Date']=df['Date'].map(dt.datetime.toordinal)
    X = df.iloc[:, 0].values.reshape(-1, 1)  
    Y = df.iloc[:, 1].values.reshape(-1, 1) 
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    linear_regressor = LinearRegression()
    linear_regressor.fit(x_train, y_train)
    y_pred = linear_regressor.predict(x_test)
    plt.title(i)
    plt.scatter(x_test, y_test,color='#808080')
    plt.plot(x_test, y_pred, color='black', linewidth=3)
    plt.xlabel('Date(in Ordinal form)', fontsize=14)
    plt.ylabel('Close Price(in USD)', fontsize=14)
    plt.grid(True)
    print(i)
    print('Mean squared error: %.2f' % mean_squared_error(y_test,y_pred))
    print('Coefficient of determination %.2f\n'% r2_score(y_test, y_pred))
    plt.show()
