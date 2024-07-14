import os
import requests
from bs4 import BeautifulSoup
import urllib.parse

# Set up the base URLs and credentials
base_url = 'https://aux.ecmwf.int/ecpds/data/list/RMMS/ecmwf/reforecasts/'
username = 's2sidx'
password = 's2sidx'

# Set up the local directory path
data_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dir_path = os.path.join(data_dir, 'data', 'ecmwf_reforecasts')

def is_valid_path(href, base_url):
    full_url = urllib.parse.urljoin(base_url, href)
    base_file_url = base_url.replace('list', 'file')
    return (full_url.startswith(base_url) or full_url.startswith(base_file_url)) and full_url != base_url

def get_relative_path(url, base_url):
    return url[len(base_url):].lstrip('/')

def download_recursive(url, local_base_path):
    session = requests.Session()
    session.auth = (username, password)

    response = session.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a')

    for link in links:
        href = link.get('href')
        if href and not href.startswith('?') and not href.startswith('javascript:') and href != '../':
            full_url = urllib.parse.urljoin(url, href)
            
            if is_valid_path(href, url):
                relative_path = get_relative_path(full_url, base_url)
                full_local_path = os.path.join(local_base_path, relative_path)

                if href.endswith('/'):
                    # It's a directory
                    if not os.path.exists(full_local_path):
                        os.makedirs(full_local_path, exist_ok=True)
                        print(f"Created directory: {full_local_path}")
                        download_recursive(full_url, local_base_path)
                    else:
                        print(f"Directory already exists: {full_local_path}")
                elif href.endswith('.txt'):
                    # It's a .txt file
                    print(f"Downloading: {full_url}")
                    try:
                        file_response = session.get(full_url)
                        file_response.raise_for_status()
                        os.makedirs(os.path.dirname(full_local_path), exist_ok=True)
                        with open(full_local_path, 'wb') as f:
                            f.write(file_response.content)
                    except requests.exceptions.RequestException as e:
                        print(f"Error downloading {full_url}: {e}")

def main():
    # Create the main directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    # Start the recursive download
    download_recursive(base_url, dir_path)

    print("Download completed!")

if __name__ == "__main__":
    main()