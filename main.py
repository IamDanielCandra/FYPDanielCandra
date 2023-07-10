import streamlit as st
import pickle
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import re
nltk.download('averaged_perceptron_tagger')
nltk.download("wordnet")
nltk.download('vader_lexicon')

pickle_in = open("xgbmodel.pkl","rb")
classifier = pickle.load(pickle_in)
scaler = pickle.load(open('scaler.pkl', 'rb'))

def prediction(listofinput):
    prediction = classifier.predict(listofinput)
    print(prediction)
    return prediction

def reviewCleaning(reviewList): #Remove everything, only keep alphabets and numbers
    #Removing Symbols, etc (Non Characters, Numbers, and "_")
    if reviewList:
        rev = []
        for review in reviewList:
            tokens = re.findall(r'[A-Za-z0-9_]+', review)
            review = ' '.join(tokens)
            rev.append(review.lower())
        return rev


def sentimentScoring(reviewList):
    if reviewList and len(reviewList)>=1:
        scores = []
        sia = SentimentIntensityAnalyzer()
        for review in reviewList:
            score = sia.polarity_scores(review)['compound']
            scores.append(score)
        scores = np.array(scores)
        return scores.mean()


#Get the wordnet part of speech
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


lemmatizer = WordNetLemmatizer()
def reviewLemmatizer(reviewList):
    #Performing Lemmatization with WordNetLemmatizer
    if reviewList:
        revList = []
        for review in reviewList:
            tokens = review.split()
            eachReview = []
            words_and_tags = nltk.pos_tag(tokens)
            for word,tag in words_and_tags:
                lemma = lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
                eachReview.append(lemma)
            lemmatized = ' '.join(eachReview)
            revList.append(lemmatized)
        return revList

def main():
    st.title("Restaurant Price Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Restaurant Price Prediction App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    rating = st.selectbox("Select your restaurant rating:", [5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5, 0])
    totalrating = st.number_input("Number/Total reviews:", 0, 100000)
    location = st.selectbox("Select your restaurant location:", ['Kuala Lumpur', 'Penang', 'Petaling Jaya', 'George Town'])
    reviews = []
    counter = 1
    inputs = {
        'Rating': 0, 'Total Rating': 0, 'Lemmatized Score': 0, 'Dinner': 0,
        'Breakfast': 0, 'Drinks': 0, 'After-hours': 0, 'Vegetarian Friendly': 0, 'Vegan Options': 0,
        'Halal': 0, 'Gluten Free Options': 0, 'Asian': 0, 'Malaysian': 0, 'Chinese': 0, 'Cafe': 0,
        'Japanese': 0, 'Indian': 0, 'Bar': 0, 'European': 0, 'Seafood': 0, 'Italian': 0, 'Pub': 0,
        'International': 0, 'Sushi': 0, 'Pizza': 0, 'Barbecue': 0, 'Korean': 0, 'Southwestern': 0,
        'Healthy': 0, 'Steakhouse': 0, 'Wine Bar': 0, 'French': 0, 'Fast food': 0, 'Grill': 0,
        'Contemporary': 0, 'Arabic': 0, 'Spanish': 0, 'Street Food': 0, 'Diner': 0,
        'Indonesian': 0, 'Vietnamese': 0, 'Japanese Fusion': 0, 'Gastropub': 0, 'Turkish': 0,
        'Central-Italian': 0, 'Southern-Italian': 0, 'Argentinian': 0, 'Kaiseki': 0,
        'Cambodian': 0, 'Russian': 0, 'Mongolian': 0, 'Sicilian': 0, 'Sardinian': 0,
        'Hokkaido cuisine': 0, 'Austrian': 0, 'Lazio': 0, 'Northern-Italian': 0, 'Romana': 0,
        'Kappo': 0, 'Reservations': 0, 'Table Service': 0, 'Takeout': 0,
        'Wheelchair Accessible': 0, 'Serves Alcohol': 0, 'Free Wifi': 0,
        'Parking Available': 0, 'Highchairs Available': 0, 'Outdoor Seating': 0,
        'Wine and Beer': 0, 'Delivery': 0, 'Full Bar': 0, 'Private Dining': 0,
        'Digital Payments': 0, 'Television': 0, 'Family style': 0, 'Buffet': 0,
        'Valet Parking': 0, 'Non-smoking restaurants': 0, 'Validated Parking': 0,
        'Live Music': 0, 'Gift Cards Available': 0, 'Sports bars': 0,
        'Location_George Town': 0, 'Location_Kuala Lumpur': 0, 'Location_Penang': 0,
        'Location_Petaling Jaya': 0, 'Accept Cards': 0
    }
    while counter <= 15:
        review = st.text_input(f"Enter your most recent review up to 15 (stop filling after done). No {counter}:")
        if review:
            reviews.append(review)
            st.write("Review added:", review)
            st.write(reviews)
            counter += 1
        else:
            break

    meals = st.multiselect("Serves (can select multiple):", ['Dinner', 'Breakfast', 'Drinks', 'After-hours'])
    diets = st.multiselect("Special Diets (can select multiple):", ['Vegetarian Friendly', 'Vegan Options', 'Halal', 'Gluten Free Options'])
    cuisines = st.multiselect("Cuisine type (can select multiple):", ['Asian', 'Malaysian', 'Chinese', 'Cafe', 'Japanese', 'Indian', 'Bar',
                                                'European', 'Seafood', 'Italian', 'Pub', 'International', 'Sushi',
                                                'Pizza', 'Barbecue', 'Korean', 'Southwestern', 'Healthy', 'Steakhouse',
                                                'Wine Bar', 'French', 'Fast food', 'Grill','Contemporary', 'Arabic',
                                                'Spanish', 'Street Food', 'Diner','Indonesian', 'Vietnamese',
                                                'Japanese Fusion', 'Gastropub', 'Turkish', 'Central-Italian',
                                                'Southern-Italian', 'Argentinian', 'Kaiseki', 'Cambodian', 'Russian',
                                                'Mongolian', 'Sicilian', 'Sardinian','Hokkaido cuisine', 'Austrian',
                                                'Lazio', 'Northern-Italian', 'Romana', 'Kappo'])
    features = st.multiselect("Restaurant features (can select multiple):", ['Reservations', 'Table Service', 'Takeout',
                                                       'Wheelchair Accessible', 'Serves Alcohol', 'Free Wifi',
                                                       'Parking Available', 'Highchairs Available', 'Outdoor Seating',
                                                       'Wine and Beer', 'Delivery', 'Full Bar', 'Private Dining',
                                                       'Digital Payments', 'Television', 'Family style', 'Buffet',
                                                       'Valet Parking', 'Non-smoking restaurants', 'Validated Parking',
                                                       'Live Music', 'Gift Cards Available', 'Sports bars',
                                                       'Accept Cards'])

    result=""
    price = ""
    prange = ""
    if st.button("Predict"):
        inputs['Rating'] = rating
        inputs['Total Rating'] = totalrating
        inputs['Lemmatized Score'] = sentimentScoring(reviewLemmatizer(reviewCleaning(reviews)))
        for meal in meals:
            inputs[meal] = 1
        for diet in diets:
            inputs[diet] = 1
        for cuisine in cuisines:
            inputs[cuisine] = 1
        for feature in features:
            inputs[feature] = 1
        loc = "Location_" + location
        inputs[loc] = 1

        listOfInput = list(inputs.values())
        listOfInput = np.array(listOfInput).reshape(1, -1)
        scaled = scaler.transform(listOfInput)
        print(scaled)
        result = prediction(scaled)


        if result == 0:
            price = "Low"
            prange = "RM 10 - RM 35"
        elif result == 1:
            price = "Medium"
            prange = "RM 27 - 139"
        else:
            price = "High"
            prange = "RM 101 - RM 1139"
    st.success('The output is: {} Price Range: {}'.format(price, prange))
    if st.button("About"):
        st.text("FYP - RPPS")
        st.text("Built by Daniel Candra TP060288")


if __name__=='__main__':
    main()
    #use streamlit run main.py in terminal
