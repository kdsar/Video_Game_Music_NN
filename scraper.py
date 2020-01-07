import os
import requests
from requests.compat import urljoin
from bs4 import BeautifulSoup

URLS = {'nes': 'https://www.vgmusic.com/music/console/nintendo/nes/',
        'gameboy': 'https://www.vgmusic.com/music/console/nintendo/gameboy/',
        'snes': 'https://www.vgmusic.com/music/console/nintendo/snes/',
        'tg16': 'https://www.vgmusic.com/music/console/nec/tg16/',
        'genesis': 'https://www.vgmusic.com/music/console/sega/genesis/'}

MIDI_FOLDER = "./data/"


def scrap_urls():
    print('Beginning scrap')
    errors = 0
    for console in URLS:
        directory = os.path.join(MIDI_FOLDER, console)
        if not os.path.exists(directory):
            os.mkdir(directory)
        print(f'Beginning {console}')
        response = requests.get(URLS[console])
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "lxml")
            for anchor_tag in soup.find_all('a', href=True):
                if anchor_tag['href'].endswith('.mid'):
                    game = anchor_tag.parent.parent.find_previous_sibling(
                        'tr', class_='header').find_all('a')[0].get_text()
                    game_directory = os.path.join(directory, game)
                    if not os.path.exists(game_directory):
                        os.mkdir(game_directory)
                    try:
                        midi_file = requests.get(
                            urljoin(URLS[console], anchor_tag['href']))
                        open(os.path.join(game_directory, anchor_tag['href']), 'wb').write(
                            midi_file.content)
                    except requests.exceptions.Timeout:
                        errors += 1
    print(f'{errors} errors occured while scraping')


scrap_urls()
