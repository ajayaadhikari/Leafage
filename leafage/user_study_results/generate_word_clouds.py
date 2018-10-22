import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


types_of_explanations = ["No Explanation", "Feature Importance", "Example-based", "Feature and Example-based"]
indices = [288, 285, 286, 287]
df = pd.read_csv("../data/user_study_results/Housing_market_114.csv", header=None)
df = df.iloc[2:, :]
df = df.fillna("")

stopwords = set(STOPWORDS)
stopwords.update(["explanation", "high", "low", "prediction", "information", "value", "know", "predict",
                  "type", "understand", "given", "made", "go", "provides", "quality", "make", "one", "much", "predictions",
                  "decision", "rated", "really", "give", "factors", "see", "dislike", "like", "gives", "dislikes", "likes",
                  "house"])

for explanation_type, index in zip(types_of_explanations, indices):
    text = " ".join(review for review in df[index].values.tolist())

    plt.figure()
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", stopwords=stopwords).generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    #plt.title(explanation_type)

plt.show()
