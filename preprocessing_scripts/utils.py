import bs4 as bs


def clean_body(body):
    soup = bs.BeautifulSoup(body, "lxml")
    for math in soup.find_all('span', {'class':"math-container"}):
        math.replace_with(f'${math.text}$')
    return soup.text
