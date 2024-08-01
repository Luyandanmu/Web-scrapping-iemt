#!/usr/bin/env python

import requests
from bs4 import BeautifulSoup

def main():
    url = 'https://www.example.com/local-weather'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    weather = soup.find('div', class_='weather-forecast').text
    print(f'Local Weather: {weather}')
if __name__ == '__main__':
    main()
