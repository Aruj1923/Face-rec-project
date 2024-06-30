from django.shortcuts import render 
from django.http import HttpResponse
# Create your views here.
import requests 
import urllib.request 
from bs4 import BeautifulSoup 
import re




def about(request):


  return render(request , 'newabout.html')



def home_view(request):
    data  = request.POST.get('your_name')
    print(data)



    if data != None:
      search = data                   # input('Enter topic that you want to get data')
      url = 'https://www.google.com/search'

      headers = {
	'Accept' : '*/*',
	'Accept-Language': 'en-US,en;q=0.5',
	'User-Agent': 'Mozilla/6.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.82',
      }
      parameters = {'q': search}

      content = requests.get(url, headers = headers, params = parameters).text
      soup = BeautifulSoup(content, 'html.parser')

      search = soup.find(id = 'search')
      first_link = search.find_all('a')
      for fl in first_link:
       
    
# URL = first_link['href']
        if fl['href'] != '#':

          URL = fl['href']
          a = URL.replace('/',' ')
          b = a.replace('.',' ')
          b = b.replace('-',' ')
        # b=  b.replace('',' ')
          b = b.split(' ')
          lists = ['translate','support','youtube','dictionary']

          k = 0
          for i in b:
       
           for j in lists:
            if i != '':
              if re.search(i,j) != None:
                
                
                print('i :',i,'j :',j)
                k = 23
                break
              else:
                
                pass
            else:
              pass
          if k != 23:
         
        #  print(URL)
           try:
             html = urllib.request.urlopen(URL) 
             break
           except:
           
             pass
          else :
           pass
     
    # print('html :',html)
    # html = urllib.request.urlopen(URL)

  
    # URL = "http://www.values.com/inspirational-quotes" 
    # URL = 'https://www.geeksforgeeks.org/caching-page-tables/'
    # r = requests.get(URL) 


  
    # parsing the html file 
      htmlParse = BeautifulSoup(html, 'html.parser') 
  
    # getting all the paragraphs 
      para_list = []
      for para in htmlParse.find_all("p"): 
    
       para_list.append(para.get_text())

    ## For Image link extraction
    
    # htmldata = urlopen('https://www.geeksforgeeks.org/') 
    # soup = BeautifulSoup(htmldata, 'html.parser') 
      images = htmlParse.find_all('img') 

      imges = []
  
      for item in images: 
       imges.append(item['src'])
       
      working_img = []

      for i in imges:
      
       try:
        html = urllib.request.urlopen(URL) 
        working_img.append(i)
       except:   
        pass
      print(working_img)
      
 
     # # logic of view will be implemented here
      return render(request, "example.html" , context= {'para_list':para_list , 'Topic_name':data , 'imges': working_img})
    
    else :
      data = 'Face Recognition'
      para_list = [None]
      working_img = [None]
      data1 = 'Images'
      return render(request, "example.html" , context= {'para_list':para_list , 'Topic_name':data ,'imges': working_img })

    



    # print(request.GET)
    # print(names)
    # people = [
    #     {'name':'Aruj', 'age':23} ,
    #     {  'name': 'Apooerva', 'age' : 24},
    #     {  'name':  'Ankit', 'age': 28},
    #     {  'name':  'khushi', 'age': 16},
    #     { 'name':   'akansha', 'age':12},
    #     { 'name':   'ruchi', 'age':16}
    # ]
    

