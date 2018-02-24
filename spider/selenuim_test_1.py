from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os

chrome_driver = "C:/Users/Administrator/AppData/Local/Google/Chrome/Application/chromedriver.exe"
os.environ["webdriver.chrome.driver"] = chrome_driver

ff_options = Options()
ff_options.add_argument("--headless")       # define headless

# driver = webdriver.Firefox(firefox_options=ff_options)
driver = webdriver.Chrome(executable_path=chrome_driver, chrome_options=ff_options)

# 将刚刚复制的帖在这
driver.get("https://morvanzhou.github.io/")
driver.find_element_by_xpath(u"//img[@alt='强化学习 (Reinforcement Learning)']").click()
driver.find_element_by_link_text("About").click()
driver.find_element_by_link_text(u"赞助").click()
driver.find_element_by_link_text(u"教程 ▾").click()
driver.find_element_by_link_text(u"数据处理 ▾").click()
driver.find_element_by_link_text(u"网页爬虫").click()

# 得到网页 html, 还能截图
html = driver.page_source  # get html
print(html)
driver.get_screenshot_as_file("./img/sreenshot1.png")
driver.close()