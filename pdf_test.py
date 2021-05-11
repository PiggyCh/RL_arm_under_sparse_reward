# !/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    实现目标：爬某一博主的所有博客
    1.作者url+headers
    2.看作者所在的url是否是静态网页
    3.解析网页，获取作者的每个作品的url，及作者名字
    4.根据每个作品url继续访问，然后数据分析
    5.提取html文本，标题
    6.创建多级文件夹
    7.保存html文本
    8.转换pdf文本
'''
import requests,parsel,os,pdfkit
from lxml import etree
from pprint import pprint
def main():
    #1.author_url+headers
    author_url=input('请输入csdn博主的url:')
    headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/87.0.4280.88 Safari/537.36'}
    response = requests.get(author_url,headers=headers).text
    # 2.作者所在的url是静态网页,xpath解析每个文章url
    html_xpath = etree.HTML(response)

    try:
        author_name = html_xpath.xpath(r'//*[@class="user-profile-head-name"]/div/text()')[0]
        # print(author_name)
        author_book_urls = html_xpath.xpath(r'//*[@class="blog-list-box"]/a/@href')
        # print(author_book_urls)
    except Exception as e:
        author_name = html_xpath.xpath(r'//*[@id="uid"]/span/text()')[0]
        author_book_urls = html_xpath.xpath(r'//*[@class="article-list"]/div/h4/a/@href')

    # print(author_name,author_book_urls,sep='\n')

    #3.遍历循环每个作品网址，请求网页
    for author_book_url in author_book_urls:
        book_res = requests.get(author_book_url,headers = headers).text
        #4.将响应分别用xpath，css选择器解析
        html_book_xpath = etree.HTML(book_res)
        html_book_css = parsel.Selector(book_res)
        book_title = html_book_xpath.xpath(r'//*[@id="articleContentId"]/text()')[0]
        html_book_content = html_book_css.css('#mainBox > main > div.blog-content-box').get()

        #5.拼接构造网页框架，加入文章html内容
        html =\
            '''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>Title</title>
            </head>
            <body>
                {}
            </body>
            </html>
        '''.format(html_book_content)

        #6.创建博主文件夹
        if not os.path.exists(r'./{}'.format(author_name)):
            os.mkdir(r'./{}'.format(author_name))

        #7.保存html文本
        try:
            with open(r'./{}/{}.html'.format(author_name,book_title),'w',encoding='utf-8') as f:
                f.write(html)
            print('***{}.html文件下载成功****'.format(book_title))
        except Exception as e:
            continue

        #8.转换pdf文本,导转换包
        try:
            config = pdfkit.configuration(
                wkhtmltopdf=r'D:\programs\wkhtmltopdf\bin\wkhtmltopdf.exe'
            )
            pdfkit.from_file(
                r'./{}/{}.html'.format(author_name,book_title),
                './{}/{}.pdf'.format(author_name,book_title),
                configuration=config
            )
            print(r'******{}.pdf文件保存成功******'.format(book_title))
        except Exception as e:
            continue



if __name__ == '__main__':
    main()