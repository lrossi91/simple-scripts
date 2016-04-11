import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

#Counting Vitae Take the plaintext version of your resume (or linkedin profile)
#and create a bar chart of character frequency.
#(Bonus: programmatically strip out punctuation and whitespace.)

f = open('LORENZO ROSSI Resume.txt', 'r')
resume=f.read()

punctuation =[' ',',','.',';',':','(',')','-','0','1','2','3','4','5','6','7','8','9','/','\n','@','*']

#function that takes the string you want to analyze, and the punctuation you want to remove
def clean_text(text_string, special_characters):
    cleaned_string = text_string
    for string in special_characters:
        cleaned_string = cleaned_string.replace(string,'')
    cleaned_string = cleaned_string.lower()  
    return(cleaned_string)

resume = clean_text(resume, punctuation)

'''
#the following code counts how many times a letter appears in the string

from collections import Counter
str = "Mary had a little lamb"
counter = Counter(str)
print counter['a']
'''
#you can check the frequency for each letter with the code below
#example with letter a
counter = Counter(resume)
a = counter['a']

#splitting the resume by each character
frequency = list(resume)

letter_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
letter_count = []

#The following function takes the text from which we want to count the letters
#and the specific letters, and it returns a list of numbers
def add_letters_frequency(text, letters):
    counter = Counter(text)
    for letter in letters:
        value = counter[letter]
        letter_count.append(value)
    return (letter_count)

#now let's apply the function to letter_count
letter_count = add_letters_frequency(frequency, letter_list)


final_dataframe = pd.DataFrame({'Letter': letter_list,
                                'Count' : letter_count
                                })


y_pos = np.arange(len(letter_list))

print(final_dataframe)

plt.bar(y_pos, letter_count,  align ='center', alpha=0.5)
plt.xticks(y_pos, letter_list)
plt.title('Bar Chart for Letter Frequency in my Resume')
plt.ylabel('Frequency')
plt.show()






   
    






