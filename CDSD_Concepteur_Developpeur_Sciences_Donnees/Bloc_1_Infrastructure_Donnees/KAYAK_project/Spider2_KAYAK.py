import os
import logging
import scrapy
from scrapy.crawler import CrawlerProcess
import pandas as pd
import pickle

# Load the variable from the saved file
with open('list_of_urls.pkl', 'rb') as f:
    list_of_urls = pickle.load(f)

class Spider2_KAYAK(scrapy.Spider):
    name = "Spider2_KAYAK"

    def start_requests(self):
        for hotel_url in list_of_urls:
            url = f'{hotel_url}'
            yield scrapy.Request(url=url, callback=self.parse)

    # Callback function that will be called when the spider starts
    def parse(self, response):

        # Extract hotel details using XPath

        hotel_gps = response.xpath("//a[contains(@data-atlas-latlng, '')]/@data-atlas-latlng").get()
        hotel_desc = response.xpath("/html/body/div[4]/div/div/div[1]/div[1]/div[2]/div/div[1]/div[1]/div[1]/div/div/p[1]/text()").getall()
        hotel_mark = response.xpath("/html/body/div[4]/div/div/div[1]/div[1]/div[1]/div[1]/div[4]/div/div[1]/div[1]/div/div[1]/a/div/div/div/div[1]/text()").get()

        yield {
            'hotel_url': response.url,
            'hotel_gps': hotel_gps,
            'hotel_desc': hotel_desc,
            'hotel_mark' : hotel_mark
        }

filename = "hotels_details.json"

# Remove the file if it already exists
if os.path.exists(f'kayak_results/{filename}'):
    os.remove(f'kayak_results/{filename}')

# Initialize the CrawlerProcess with specific settings
process = CrawlerProcess(settings={
    'USER_AGENT': 'Chrome/97.0',
    'LOG_LEVEL': logging.INFO,
    "FEEDS": {
        'kayak_results/' + filename: {"format": "json"},
    },
})

# Start crawling with the defined Spider
process.crawl(Spider2_KAYAK)
process.start()
