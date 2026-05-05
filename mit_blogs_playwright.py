from playwright.sync_api import sync_playwright
import pandas as pd
import time

def crawl_mit_blogs():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Open the blog list page
        print("Opening blog list page...")
        page.goto('https://mitadmissions.org/blogs/', timeout=30000)
        page.wait_for_load_state('networkidle', timeout=15000)

        # 2. Collect all article URLs (filter out #disqus_thread links)
        links = page.query_selector_all('a[href*="/blogs/entry/"]')
        urls = []
        for link in links:
            href = link.get_attribute('href')
            if href and '#' not in href and href not in urls:
                urls.append(href)
        print(f"Found {len(urls)} articles")

        # 3. Visit each article and extract data
        data = []
        for i, url in enumerate(urls):
            try:
                print(f"[{i+1}/{len(urls)}] Crawling: {url}")
                page.goto(url, timeout=30000)
                page.wait_for_load_state('networkidle', timeout=15000)

                # Title
                title_el = page.query_selector('h1.page-topper__title')
                title = title_el.inner_text().strip() if title_el else ''

                # Author
                author_el = page.query_selector('span.page-topper__title__name')
                author = author_el.inner_text().strip() if author_el else ''

                # Comment Count
                comment_el = page.query_selector('.comment-count')
                comments = comment_el.inner_text().strip() if comment_el else '0'

                # Time
                time_el = page.query_selector('p.page-topper__date-mod')
                pub_time = time_el.inner_text().strip() if time_el else ''

                # Article Content
                content_el = page.query_selector('div.article__body')
                content = content_el.inner_text().strip() if content_el else ''

                # Images In Article
                images = ''
                if content_el:
                    imgs = content_el.query_selector_all('img')
                    img_urls = []
                    for img in imgs:
                        src = img.get_attribute('src')
                        if src:
                            img_urls.append(src)
                    images = '; '.join(img_urls)

                data.append([title, author, comments, pub_time, content, images])
                print(f"  Done: {title[:50]}")

            except Exception as e:
                print(f"  Error: {url}, reason: {e}")

        browser.close()

    # 4. Save to CSV
    columns = ['Title', 'Author', 'Comment Count', 'Time',
               'Article Content', 'Images In Article']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('E:/PythonProject_1/mit_blogs_playwright.csv', index=False, encoding='utf-8-sig')
    print(f"\nDone! Crawled {len(data)} articles, saved to mit_blogs_playwright.csv")

if __name__ == '__main__':
    crawl_mit_blogs()
