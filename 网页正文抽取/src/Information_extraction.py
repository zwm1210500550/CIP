from lxml import etree
import re 

def get_content(infile_name, outfile_name, decode):
    infile = open(infile_name, 'r', encoding=decode)
    outfile = open(outfile_name, 'w', encoding='utf-8')
    content = infile.read()
    root = etree.HTML(content)
    title = root.xpath('/html/head/title')
    outfile.write('title:\n')
    outfile.write(title[0].text + '\n')
    outfile.write('\n')
    
    body = root.xpath('/html/body')
    global body_content
    body_content = ''
    get_body(body[0])
    body_content = re.sub(r'(\n+\s*)+', '\n', body_content.strip())
    outfile.write('body:\n')
    outfile.write(body_content + '\n')
    outfile.write('\n')

    hrefs = root.xpath('//a')
    outfile.write('link:\n')
    for href in hrefs:
        outfile.write(href.text + '\t' + href.get('href') + '\n')

def get_body(elemtree):
    global body_content
    if elemtree.tag != 'script':
        if elemtree.text != None:
            body_content += elemtree.text
        for i in range(len(elemtree)):
            get_body(elemtree[i])
        if elemtree.tail != None:
            body_content += elemtree.tail
        

if __name__ == '__main__':
    get_content('../data/1.html', '../result/out_1.txt', 'GBK')     #两个网页的编码方式不同
    get_content('../data/2.html', '../result/out_2.txt', 'UTF-8')
