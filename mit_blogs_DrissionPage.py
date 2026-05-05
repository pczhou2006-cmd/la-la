from DrissionPage import ChromiumPage
import pandas as pd
import time

page = ChromiumPage() #自动打开浏览器
page.get('https://mitadmissions.org/blogs/') #打开网页
time.sleep(5)

#获取列表页所有文章链接
links = page.eles('css:a[href="/blogs/entry/"]')
urls = [       'https://mitadmissions.org/blogs/entry/east-campus-part-2/',       'https://mitadmissions.org/blogs/entry/cpw-2026-camera-roll-dump/',       'https://mitadmissions.org/blogs/entry/kindness-friday-live-blog/',       'https://mitadmissions.org/blogs/entry/gay-beaver-marriage-is-real/',       'https://mitadmissions.org/blogs/entry/check-in-on-my-cross-stitch-livestream-during-cpw-link-inside/',       'https://mitadmissions.org/blogs/entry/the-double-doubled-b1ner-cribs/',       'https://mitadmissions.org/blogs/entry/some-cpw-events-ill-be-at/',       'https://mitadmissions.org/blogs/entry/spring-break-qa-part-2/',       'https://mitadmissions.org/blogs/entry/all-the-classes-i-didnt-take/',       'https://mitadmissions.org/blogs/entry/guest-post-a-circle-of-shared-breaths/',       'https://mitadmissions.org/blogs/entry/a-lonely-but-exciting-road/',       'https://mitadmissions.org/blogs/entry/i-made-the-mit-admissions-blog-tcg/',       'https://mitadmissions.org/blogs/entry/behind-the-scenes-of-thirty-pies/',       'https://mitadmissions.org/blogs/entry/the-life-i-couldve-lived/',       'https://mitadmissions.org/blogs/entry/everyone-knows-each-other/',       'https://mitadmissions.org/blogs/entry/cpw-will-not-be-perfect/',   ]
for i in links:
    url = i.attr(href="/blogs/author/anika/")
    if url and url not in urls:
        urls.append(url)
        print(f'共{len(urls)}篇文章')
#遍历每篇文章
data = []
for url in urls:
  try:
    page.get(url)
    time.sleep(5)
        #标题
    title = page.ele('css:h1.page-topper__title').text
        #作者
    try:
     author = page.ele('css:span.page-topper__title__name').text
    except:
            author = "null"
        #评论数
    try:
            comments =page.ele('css:.comment-count').text
    except:
            comments = '0'
        #时间
    try:
            sleep_time = page.ele('css:p.page-topper__date-mod').text
    except:
             sleep_time = 'null'
        #正文
    try:
         content_el = page.ele('css:div.article__body')  
         content = content_el.text
         #正文中的图片
         imgs = content_el.eles('css:img')
         img_urls = [img.attr('src') for img in imgs if img.attr('src')]
         images = '; '.join(img_urls)
    except:
         content = ''
         images = ''
    data.append([title, author, comments, sleep_time, content, images])
    print(f'已爬取: {title}')

  except Exception as e:
          print(f'出错: {url}, 原因: {e}')

page.quit()

    #保存CSV
columns = ['Title', 'Author', 'Comment Count', 'Time',
             'Article Content', 'Images In Article']
df = pd.DataFrame(data, columns=columns)
df.to_csv('mit_blogs_drissionpage.csv', index=False, encoding='utf-8-sig')
print('DrissionPage爬取完成！')
