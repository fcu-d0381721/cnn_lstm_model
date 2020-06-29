from selenium import webdriver  #從library中引入webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time
# browser = webdriver.Chrome()    #開啟chrome browser
Monitoring_station = ['C0AH10']
ms = ["01", "02", "03", "04", "05", "06"]
ds = [31, 28, 31, 30, 31, 30]
for n in Monitoring_station:
    for m in range(len(ms)):
        browser = webdriver.Chrome()
        all = []
        for d in range(1, ds[m]+1):
            print(ms[m], " ", d)

            if d < 10:
                dd = "0" + str(d)
                browser.get('https://e-service.cwb.gov.tw/HistoryDataQuery/DayDataController.do?command=viewMain&'
                            'station=' + n + '&stname=%25E5%258D%2597%25E5%25B1%25AF&datepicker=2018-'
                            + str(ms[m]) + '-' + dd)
            else:
                browser.get(
                    'https://e-service.cwb.gov.tw/HistoryDataQuery/DayDataController.do?command=viewMain&station='
                    + n + '&stname=%25E5%258D%2597%25E5%25B1%25AF&datepicker=2018-' + str(
                        ms[m]) + '-' + str(d))
            time.sleep(3)
            count = 0
            data = []

            soup = BeautifulSoup(browser.page_source)
            for ele in soup.select('table#MyTable tbody tr td'):

                if count == 16:
                    data.append(ele.text.split()[0])
                    data.append(str(ms[m]) + '-' + str(d))
                    all.append(data)
                    count = 0
                    data = []
                else:
                    try:
                        print(ele.text)
                        data.append(ele.text.split()[0])
                        count += 1
                    except:
                        data.append(0)
                        count += 1
        browser.close()
        print(all)
        a = pd.DataFrame(all, columns=['觀測時間', '測站氣壓', '海平面氣壓', '氣溫', '露點溫度', '相對溼度', '風速', '風向',
                                       '最大陣風', '最大陣風風向', '降水量', '降水時數', '日照時數', '全天空日射量', '能見度',
                                       '紫外線指數', '總雲量', '日期'])
        a.to_csv('./永和/2018-' + str(ms[m]) + '-' + str(d) + '.csv')