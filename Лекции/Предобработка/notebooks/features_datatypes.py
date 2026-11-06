import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Особенности формирования признаков для данных различных типов

    - Большое количество техник обработки признаков ориентировано на числовые и категориальные признаки.
    - А как быть с признаками других видов? Например таких, как дата-время, IP-адрес, URL-адрес, строка (которая не является категориальным признаком).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Загрузка и первичный анализ данных

    Используем набор данных [web log dataset.](https://www.kaggle.com/shawon10/web-log-dataset)
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import datetime
    import ipaddress
    import tqdm
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    sns.set(style="ticks")
    return LabelEncoder, MinMaxScaler, datetime, np, pd


@app.cell
def _(pd):
    data = pd.read_csv('data/weblog.csv', sep=',')
    return (data,)


@app.cell
def _(data):
    data.shape
    return


@app.cell
def _(data):
    data.head()
    return


@app.cell
def _(data):
    # Целевой признак
    data['Status'].value_counts()
    return


@app.cell
def _(LabelEncoder, data):
    # Закодируем целевой признак
    target_le = LabelEncoder()
    data['status_le'] = target_le.fit_transform(data['Status'])
    data['status_le'].unique(), target_le.inverse_transform(data['status_le'].unique())
    return


@app.cell
def _(data):
    data.dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Выделение признаков из URL-адреса
    """)
    return


@app.cell
def _(data):
    # Проверка наличия подстроки в URL
    def substr_in_url(substr):
        lsubstr = substr.lower()
        return data.apply(lambda x: 1 if lsubstr in x['URL'].lower() else 0, axis=1)

    return (substr_in_url,)


@app.cell
def _(data, substr_in_url):
    data['is_get'] = substr_in_url('GET')
    data['is_php'] = substr_in_url('PHP')
    data['is_js'] = substr_in_url('JS')
    data['is_woff'] = substr_in_url('WOFF')
    url_features = ['is_get', 'is_php', 'is_js', 'is_woff']
    return


@app.cell
def _(data):
    data.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Обработка даты и времени
    """)
    return


@app.cell
def _(data, pd):
    # Сконвертируем дату и время в нужный формат
    data['dt'] = data.apply(lambda x: pd.to_datetime(x['Time'], format='[%d/%b/%Y:%H:%M:%S'), axis=1)
    return


@app.cell
def _(data):
    data.head()
    return


@app.cell
def _(data):
    data.dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Для работы с датами, временем и временными интервалами используются [возможности библиотеки NumPy.](https://numpy.org/doc/stable/reference/arrays.datetime.html)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Выделение стандартных признаков даты и времени
    """)
    return


@app.cell
def _(data):
    # День
    data['day'] = data['dt'].dt.day
    # Месяц
    data['month'] = data['dt'].dt.month
    # Год
    data['year'] = data['dt'].dt.year
    # Часы
    data['hour'] = data['dt'].dt.hour
    #Минуты
    data['minute'] = data['dt'].dt.minute
    #Секунды
    data['second'] = data['dt'].dt.second
    #Неделя года
    data['week'] = data['dt'].dt.isocalendar().week
    #Квартал
    data['quarter'] = data['dt'].dt.quarter
    #День недели 
    data['dayofweek'] = data['dt'].dt.dayofweek
    #Выходной день
    data['day_name'] = data['dt'].dt.day_name()
    data['is_holiday'] = data.apply(lambda x: 1 if x['dt'].dayofweek in [5,6] else 0, axis=1)
    return


@app.cell
def _(data):
    #Проверка определения выходных дней
    data[data['day_name']=='Monday'] 
    data[data['day_name']=='Sunday'] 
    data[data['day_name']=='Saturday']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    По аналогии с выходными днями часто формируется календарь потенциальных аномалий (например, праздники по китайскому календарю).
    """)
    return


@app.cell
def _(data):
    data.head()
    return


@app.cell
def _(MinMaxScaler, data):
    # Создадим масштабируемые признаки для дальнейших экспериментов
    dt_features = ['year', 'day', 'month', 'hour', 'minute', 'second', 'week', 'quarter', 'dayofweek']
    dt_features_scaled = []
    for _f in dt_features:
        f_new = str(_f + '_scaled')
        dt_features_scaled.append(f_new)
        data[f_new] = MinMaxScaler().fit_transform(data[[_f]])
    dt_features_scaled
    return (dt_features,)


@app.cell
def _(data):
    data.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Временные интервалы
    """)
    return


@app.cell
def _(data, datetime):
    # Разница между датами 
    data['now'] = datetime.datetime.today()
    data['diff'] = data['now'] - data['dt']
    data.dtypes
    return


@app.cell
def _(data):
    data.head()
    return


@app.cell
def _(data, np):
    # Чтобы получить разницу между датами в заданных единицах 
    # (минутах, днях и т.д.) нужно разделить timedelta64 на длительность
    data['diff_days']=data['diff']/np.timedelta64(1,'D')
    data.dtypes
    return


@app.cell
def _(data):
    data.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Отображение на круг

    Кодирование времени с помощью часов не дает представления о том, что между 23:00 и 01:00 расстояние в 2 часа. Это может помешать эффективной работе метрических алгоритмов.

    Для решения проблемы час $v$ заменяется двумя признаками:

    $$ t_1 = cos \Big( \frac{v \cdot 2\pi}{T} \Big) $$

    $$ t_2 = sin \Big( \frac{v \cdot 2\pi}{T} \Big) $$

    Значение $T$ в знаменателе формулы это период кодируемой величины. В случае кодирования часов $T=24$.

    Аналогичным образом может кодироваться день, месяц и т.д. В этом случае вместо 24 часов в знаменатель формулы подставляется соответствующий период.
    """)
    return


@app.cell
def _(np):
    def round_code(v, T, cos_flag = True):
        x = 2*np.pi*v/T
        if cos_flag:
            return np.cos(x)
        else:
            return np.sin(x)

    return (round_code,)


@app.cell
def _(data, dt_features):
    # Опеределим периоды для признаков
    for _f in dt_features:
        print(_f, data[_f].min(), data[_f].max())
    return


@app.cell
def _():
    # периоды для признаков
    dt_features_periods = [0, 31, 12, 24, 60, 60, 52, 4, 7]
    return (dt_features_periods,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Год можно было закодировать с помощью one-hot encoding, но года всего два, поэтому закодируем год флагом да/нет для 2017 года.
    """)
    return


@app.cell
def _(data):
    data['year_round'] = data.apply(lambda x: 1 if x['year']==2017 else 0, axis=1)
    dt_features_round = ['year_round']
    return (dt_features_round,)


@app.cell
def _(data):
    data.head()
    return


@app.cell
def _(data, dt_features, dt_features_periods, dt_features_round, round_code):
    # Построим отображение признаков на круг, год пропускаем
    for _f, p in zip(dt_features[1:], dt_features_periods[1:]):
        f_cos = str(_f + '_cos')
        f_sin = str(_f + '_sin')
        data[f_cos] = data.apply(lambda x: round_code(x[_f], p), axis=1)
        data[f_sin] = data.apply(lambda x: round_code(x[_f], p, False), axis=1)
        dt_features_round.append(f_cos)
        dt_features_round.append(f_sin)
    dt_features_round
    return


@app.cell
def _(data):
    data.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [IP-адрес](https://ru.wikipedia.org/wiki/IP-%D0%B0%D0%B4%D1%80%D0%B5%D1%81)

    IP-адрес может быть закодирован непосредственно или может являться источником для извлечения дополнительных признаков.

    В случае непосредственного кодирования:
    1. Каждый бит может быть закодирован в виде отдельного признака (аналог one-hot кодирования). В случае IPv4 это 4 байта * 8 битов = 32 признака. В случае IPv6 это 6 байтов * 8 битов = 48 признаков.
    1. Для того, чтобы получить информацию об иерархической вложенности адресов, можно кодировать адреса последовательными N-граммами. Например, для адреса "10.130.2.1" будут созданы четыре признака:
        - 10
        - 10_130
        - 10_130_2
        - 10_130_2_1
        - Полученные признаки могут рассматривать как категориальные и далее кодироваться по правилам для категориальных признаков.

    Также IP-адрес может рассматриваться как источник для извлечения дополнительных признаков, в частности, на основе геолокации. Для Python существуют развитые библиотеки геолокации, в частности [geoip2.](https://pypi.org/project/geoip2/) Геолокационная информация (регион, страна, город, долгота, широта) могут далее кодироваться как категориальные или числовые признаки.
    """)
    return


@app.function
def ip_code(ip, level):
    arr = ip.split('.')
    if level > 4:
        level=4
    if level == 1:
        return arr[0]
    else:
        return '_'.join(arr[:level])


@app.cell
def _(data):
    ip_features = []
    for _i in range(4):
        _f = str('ip_' + str(_i + 1))
        data[_f] = data.apply(lambda x: ip_code(x['IP'], _i + 1), axis=1)
        ip_features.append(_f)
    ip_features
    return (ip_features,)


@app.cell
def _(data):
    data.head()
    return


@app.cell
def _(data, ip_features, pd):
    df_ip = pd.get_dummies(data[ip_features],drop_first=True)
    df_ip.shape
    return (df_ip,)


@app.cell
def _(df_ip):
    df_ip.head()
    return


@app.cell
def _(df_ip):
    for _col in df_ip:
        print(_col, df_ip[_col].unique(), len(df_ip[_col].unique()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Дополнительные источники
    - [Открытый курс машинного обучения. Тема 6. Построение и отбор признаков](https://habr.com/ru/company/ods/blog/325422/)
    - [Feature Engineering, о чём молчат online-курсы](https://habr.com/ru/company/mailru/blog/346942/)
    - [Discover Feature Engineering, How to Engineer Features and How to Get Good at It](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)
    - https://heartbeat.fritz.ai/hands-on-with-feature-engineering-techniques-variables-types-b2120e534680
    - https://www.kaggle.com/kashnitsky/topic-6-feature-engineering-and-feature-selection
    - https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering-p2
    - https://hammer.figshare.com/articles/thesis/Encoding_IP_Address_as_a_Feature_for_Network_Intrusion_Detection/11307287
    - https://hammer.figshare.com/articles/thesis/Encoding_IP_Address_as_a_Feature_for_Network_Intrusion_Detection/11307287
    """)
    return


if __name__ == "__main__":
    app.run()
