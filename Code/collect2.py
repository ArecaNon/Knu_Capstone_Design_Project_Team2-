from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import  By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
from urllib.parse import quote_plus

from urllib.request import (urlopen, urlparse, urlunparse, urlretrieve)
import os
import pandas as pd

def selenium_scroll_option():
  SCROLL_PAUSE_SEC = 3
  
  # 스크롤 높이 가져옴
  last_height = driver.execute_script("return document.body.scrollHeight")
  
  while True:
    # 끝까지 스크롤 다운
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # 1초 대기
    time.sleep(SCROLL_PAUSE_SEC)

    # 스크롤 다운 후 스크롤 높이 다시 가져옴
    new_height = driver.execute_script("return document.body.scrollHeight")
  
    if new_height == last_height:
        break
    last_height = new_height

#url 제작
def make_url(search_term):
    # 네이버 이미지 검색
    base_url = 'https://search.naver.com/search.naver?where=image&section=image&query='
    # CCL 상업적 이용 가능 옵션
    end_url = '&res_fr=0&res_to=0&sm=tab_opt&color=&ccl=2' \
              '&nso=so%3Ar%2Ca%3Aall%2Cp%3Aall&recent=0&datetype=0&startdate=0&enddate=0&gif=0&optStr=&nso_open=1'
    return base_url + quote_plus(search_term) + end_url

# 키워드 검색하기

a=input("검색할 키워드를 입력 : ")
image_name = input("저장할 이미지 이름 : ")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get(make_url(a))
# 클래스를 찾고 해당 클래스의 src 리스트 제작

selenium_scroll_option() # 스크롤하여 이미지를 많이 확보

'''이미지 src요소를 리스트업해서 이미지 url 저장'''

images = driver.find_elements(By.CLASS_NAME, "_image") #  클래스 네임에서 공백은 .을 찍어줌
images_url = []
for i in images: 
   
   if i.get_attribute('src')!= None :
        images_url.append(i.get_attribute('src'))
   else :
       images_url.append(i.get_attribute('data-src'))
driver.close()



# 겹치는 이미지 url 제거

print("전체 다운로드한 이미지 개수: {}\n동일한 이미지를 제거한 이미지 개수: {}".format(len(images_url), len(pd.DataFrame(images_url)[0].unique())))
images_url=pd.DataFrame(images_url)[0].unique()


        
#'''해당하는 파일에 이미지 다운로드'''        
os.makedirs('./' + a); #검색어로 폴더 생성
for t, url in enumerate(images_url, 0):        
    urlretrieve(url, './' + a + '/' + image_name + '_' + str(t) + '.jpg')


