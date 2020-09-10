#Multiple Feature Linear Regression Model
#predicting the outcome for a tennis player
#based on their playing habits by analyzing
#and modeling the Association of Tennis Professionals data.
#=========================================================================================
#Libraries
import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#=========================================================================================
#A function to take all the features of a player and make a data frame for training/testing.
def all_features(df,*exceptions):
  x = []
  for column in df.columns:
    if column not in exceptions:
      x.append(str(column))
  return x
#=========================================================================================
#Creating exploratory analysis in order to find relationships between features and outcomes.
data = pd.read_csv('tennis_stats.csv')
df = pd.DataFrame(data)
print(data) 
#x = data['BreakPointsOpportunities']
#y = data['Ranking']
#plt.scatter(x,y)
#plt.show()
#=========================================================================================
#Uncomment the labeled section to see the relationship
#between features and outcome.
#----------------------------------------------
#Single feature linear regression model on data:

#Max value of an outcome will help plot a line of
#best fit between predictions and actual values.
outcome_variable = 'Winnings'
max_range = max(df[outcome_variable])
#----------------------------------------------

#Breaking Points Opportunities vs Winnings:
#x = df[['BreakPointsOpportunities']]
#y = df[['Winnings']]
#Thoughts : Strong Correlation
#Train Score: 0.8192831169997737
#Test Score: 0.7738275747278562

#----------------------------------------------

#Return Games Won vs Winnings
#df = pd.DataFrame(data)
#x = df[['ReturnGamesWon']]
#y = df[['Winnings']]
#Thoughts: Poor Correlation
#Train Score: 0.07459072184459647
#Test Score: 0.08906877188977358

#----------------------------------------------

#Break Points Opportunities vs Ranking
#x = df[['BreakPointsOpportunities']]
#y = df[['Ranking']]
#Thoughts: Poor Correlation
#Train Score: 0.1098804399338642
#Test Score: 0.119886542172287

#----------------------------------------------

#Return Games Won vs Ranking
#df = pd.DataFrame(data)
#x = df[['ReturnGamesWon']]
#y = df[['Ranking']]
#Thoughts: Poor Correlation
#Train Score: 0.03343398688116317
#Test Score: 0.045670619333290086

#----------------------------------------------
#Multiple feature linear regression model on data:

#All features (except: Ranking,Player) vs Ranking
#df = pd.DataFrame(data)
#x = df[all_features(df,'Ranking','Player')]
#y = df[['Ranking']]
#Thoughts: Poor Correlation
#Train Score: 0.16856352491417892
#Test Score: 0.15751394849416767

#----------------------------------------------

#All features (except: Winnings,Player) vs Winnings
#df = pd.DataFrame(data)
#x = df[all_features(df,'Winnings','Player')]
#y = df[['Winnings']]
#Thoughts: Strong Correlation
#Train Score: 0.887804899085904
#Test Score: 0.9120986001823826

#----------------------------------------------

#All features (except: Winnings,Player) vs Winnings
#df = pd.DataFrame(data)
x = df[all_features(df,'Winnings','Player')]
y = df[['Winnings']]
#Thoughts: Strong Correlation
#Train Score: 0.887804899085904
#Test Score: 0.9120986001823826

#----------------------------------------------
#Split the data set into train/test sets
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, test_size = 0.2,)

lm = LinearRegression()
model = lm.fit(x_train,y_train)

y_predict = lm.predict(x_test)
plt.plot(y_test,y_predict,'o')
plt.plot(range(max_range),range(max_range))
plt.xlabel("Independed Variable")
plt.ylabel("Depended Variable")
plt.title("ATP Data Modeling")
plt.show()








