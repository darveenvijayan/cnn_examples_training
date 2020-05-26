from bs4 import BeautifulSoup
import numpy as np
import requests
import cv2
import PIL.Image
import urllib

classes_id = {
	'king_penguin': 'n02056570', # 2022
	'giant_panda': 'n02510455', # 1832
	'red_panda': 'n02509815', # 1686
	'wombat': 'n01877812', # 1222
	'megalonychidae': 'n02457586', # I don't know
	'llama': 'n02437616', # 1304
	'hippo': 'n02437616', # 1391
	'alaskan_malamute': 'n02110063', # 1634
	'baboon': 'n02486410', # 1635
	'otter': 'n02444819', # 1547
}

# 1. Ger the lists of URLs for the images of the synset
page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04194289")#ship synset
print(page.content)
# BeautifulSoup is an HTML parsing library
soup = BeautifulSoup(page.content, 'html.parser') #puts the content of the website into the soup variable, each url on a different line