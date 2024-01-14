# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:30:08 2023

@author: dbda
"""
import numpy as np

prices=np.array([12.33,45.5,67.4,12.4])
prices
print(prices)

quantities=np.array([12,34,52,44])
quantities
print(quantities)

revenue_per_product=prices*quantities
revenue_per_product
print(revenue_per_product)

total_revenue=np.sum(revenue_per_product)
total_revenue
print(np.round(total_revenue))

print("--------------------------------------------------------------------")


views=np.array([1000,500,400,350,670])
print(views)

max_views=np.max(views)
print(max_views)

min_views=np.min(views)
print(min_views)

average_views=np.round(np.mean(views),2)
print(average_views)

total_views=np.sum(views)
print(total_views)



print("-------------------splitting order into batches--------------------------------")

order_ids=np.array([1001,1002,1003,1004,1005,1006,1007,1008])
print(order_ids)

batches=np.split(order_ids,4)
print(batches)

for batch in batches:
    print(batch)
    
print("--------------------categorizing product rating-------------------------------")

ratings=np.array([4.5,5.0,3.4,2.4,1.2,4.8,4.1,0.3])
print(ratings)

positive_ratings=ratings[ratings>=3]
print(positive_ratings)

negative_ratings=ratings[ratings<=3]
print(negative_ratings)

print("--------------calculate total and average quantities sold--------------")

order_quantities=np.array([[5,6,4,3],[1,2,7,8]])
print(order_quantities)

total_quantities_sold=np.sum(order_quantities,axis=0)
print(total_quantities_sold)

total_products_per_order=np.sum(order_quantities,axis=1)
print(total_products_per_order)

average_quantities_sold=np.mean(order_quantities,axis=0)
print(average_quantities_sold)

print("-----------calcualte average product rating and max rating per category---------------")

product_reviews=np.array([[4.5,3.2,2.2,4.1],[4,3.8,2.2,1.1],[2.0,4.3,3.5,1.6]])
print(product_reviews)

average_rating=np.mean(product_reviews,axis=1)
print(average_rating)

max_rating_per_category=np.max(product_reviews,axis=0)
print(max_rating_per_category)

#max_rating_per_category=np.max(product_reviews,axis=1)
#print(max_rating_per_category)



















