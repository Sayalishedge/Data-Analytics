import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#define possiblle genders
genders=np.array(["male","female"])

#generate random indices for genders
gender_indices=np.random.choice(range(len(genders)),size=1000)

#generate random ages between 18 and 65
ages=np.random.randint(18,35,size=1000)

#create a dataframe from the array
data={"gender": genders[gender_indices],"Age": ages}
print(data)

#create a dataframe from the dictionary
df=pd.DataFrame(data)
#A=Gender=female ,B=Age>30
#p(A and B)
p_a_and_b=len(df[(df['gender']=='female')&(df['Age']>30)])/len(df)
print(p_a_and_b)
#p_a_and_p_b=len(df.query("Gender=='female' and Age>30"))/len(df)

#p(B)
p_b=len(df[df['Age']>30])/len(df)
print(p_b)

#p(A)
p_a = len(df[df['gender']=='female'])/len(df)
print(p_a)

#p(A|B)=p(Gender ='F' given that age>30)
p_a_given_b=p_a_and_b /p_b
print(p_a_given_b)

#p(B|A)=p(Age>30 given that gender=F)
p_b_given_a = p_a_and_b/p_a
print(p_b_given_a)


#Grouping by itself does not change the data in the dataframe. It creates a separate Groupby object.
#If we want to see the effect of grouping, we need to perform an aggregation or filtering operation on the grouped Dataframe.


#grouping by gender
grouped_df= df.groupby('gender')
print(type(grouped_df))
print(df.head())
print(grouped_df.head())
print(grouped_df['Age'].mean())

#No of people over 30 yrs
over30_count = grouped_df['Age'].apply(lambda x : (x>30).sum())
print(type(over30_count))
print(over30_count)

#Calculate total number of people in each group
total_count = grouped_df['Age'].count()
print(total_count)

#Conditional probability for each group
conditional_probabilities = over30_count / total_count
print(conditional_probabilities)

#barchart
plt.bar(conditional_probabilities.index , conditional_probabilities)
plt.xlabel('Gender')
plt.ylabel('Probability')
plt.title('Probability of being over 30')
plt.show()


































