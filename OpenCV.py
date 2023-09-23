#!/usr/bin/env python
# coding: utf-8

# In[18]:


get_ipython().system('pip install opencv-python')


# In[23]:


import cv2
import numpy as np
import os


# In[24]:


files=(
    'subway.jpg',
    'breakfast.jpg',
    'dinner.jpg',
    'building.jpg',
)

f=os.path.join('images', files[0])


# In[25]:


def view_image(i):
    cv2.imshow('view', i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[26]:


i=cv2.imread(f)
view_image(i)


# In[29]:


print(i.shape)
print(i[0,0,:])


# Task 2

# In[34]:


i_gray=cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
print(i_gray.shape)
print(i_gray[0,0])
view_image(i_gray)


# Gradient Image

# In[40]:


sobelx=cv2.Sobel(i_gray, cv2.CV_64F,1,0)
abs_sobelx=np.absolute(sobelx)
view_image(abs_sobelx / np.max(abs_sobelx))


# In[41]:


sobely=cv2.Sobel(i_gray, cv2.CV_64F,0,1)
abs_sobely=np.absolute(sobely)
view_image(abs_sobely / np.max(abs_sobely))


# In[44]:


magnitude=np.sqrt(sobelx**2+ sobely**2)
view_image(magnitude / np.max(magnitude))


# TASK 3

# edges detection

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import cv2
import os
import matplotlib.pyplot as plt


# In[ ]:


folder_path = "images"


# In[ ]:


image_files = ['subway.jpg', 'breakfast.jpg', 'dinner.jpg', 'building.jpg']


# In[ ]:


images = {}


# In[ ]:


or image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)
    
    # Convert BGR to RGB (OpenCV loads images in BGR format by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    images[image_file] = image

# Display the images in Jupyter notebook using matplotlib
for title, img in images.items():
    plt.figure(figsize=(5,5))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




