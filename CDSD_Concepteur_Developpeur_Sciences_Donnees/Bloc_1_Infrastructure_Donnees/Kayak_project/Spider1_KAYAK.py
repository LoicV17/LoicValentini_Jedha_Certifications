import os
import logging
import scrapy
from scrapy.crawler import CrawlerProcess
import numpy as np
import pickle

# Load the variable from the saved file
with open('TOP5_list_of_cities.pkl', 'rb') as f:
    TOP5_list_of_cities = pickle.load(f)


class Spider1_KAYAK(scrapy.Spider):
    name = "Spider1_KAYAK"

    def start_requests(self):

        # Generate URL for each city
        for city in TOP5_list_of_cities:
            url = f'https://www.booking.com/searchresults.fr.html?ss={city}'
            yield scrapy.Request(url=url, callback=self.parse, meta={'city': city})

    # Callback function that will be called when the spider starts
    def parse(self, response):
        
        # Retrieve the city name from meta
        city = response.meta['city']

        # Extract hotel names, URLs using XPath
 
        hotel_names = response.xpath("/html/body/div[4]/div/div/div/div[2]/div[3]/div[2]/div[2]/div[3]/div/div[1]/div[2]/div/div/div[1]/div/div[1]/div/h3/a/div[1]/text()").getall()
        hotel_urls = response.xpath("/html/body/div[4]/div/div/div/div[2]/div[3]/div[2]/div[2]/div[3]/div/div[1]/div[2]/div/div[1]/div[1]/div/div[1]/div/h3/a/@href").getall()

        return {
            'city': city,
            'hotel_names': hotel_names,
            'hotel_urls': hotel_urls,
        }

filename = "List_of_URLs_per_TOP5_cities.json"

    
# Remove the file if it already exists
if os.path.exists(f'kayak_results/{filename}'):
    os.remove(f'kayak_results/{filename}')

# Initialize the CrawlerProcess with specific settings
process = CrawlerProcess(settings = {
    'USER_AGENT': 'Chrome/97.0',  # Set the user agent to Chrome
    'LOG_LEVEL': logging.INFO,  # Set the log level to INFO
    "FEEDS": {
                'kayak_results/' + filename : {"format": "json"},  # Define the output file format as JSON
    }
})

# Start crawling with the defined Spider
process.crawl(Spider1_KAYAK)
process.start()
