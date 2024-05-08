
# Sentiment Analysis and Sentiment Modeling for Amazon Reviews

import pandas as pd
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import Word, TextBlob
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from warnings import filterwarnings

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 200)




# TEXT PRE-PROCESSING



# amazon.xlsx datasını okutunuz.
df = pd.read_excel("amazon.xlsx")
df.head()
df.info()

'''  
  Veri seti şu değişkenlerden oluşmaktadır:
 -Review: Ürüne yapılan yorum
 -Title: Yorum içeriğine verilen başlık, kısa yorum
 -HelpFul: Yorumu faydalı bulan kişi sayısı
 -Star: Ürüne verilen yıldız sayısı
 
'''


# 1-) Normalizing Case Folding
# "Review" değişkeni üzerinde tüm harfleri küçük harfe çeviriniz.
df['Review'] = df['Review'].str.lower()


# 2-) Punctuations
# "Review" değişkeni üzerinde noktalama işaretlerini çıkarınız
df['Review'] = df['Review'].str.replace('[^\w\s]', '')


# 3-) Numbers
# "Review" değişkeni üzerinde yorumlarda bulunan sayısal ifadeleri çıkarınız.
df['Review'] = df['Review'].str.replace('\d', '')

# 4-) Stopwords
nltk.download('stopwords')
sw = stopwords.words('english')
# "Review" değişkeni üzerinde bilgi içermeyen kelimeleri (stopwords) veriden çıkarınız.
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))


# 5-) Rarewords / Custom Words
# "Review" değişkeni üzerinde 1000'den az geçen kelimeleri veriden çıkarınız
sil = pd.Series(' '.join(df['Review']).split()).value_counts()[-1000:]
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))



# 6-) Lemmatization

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
# "Review" değişkeni üzerinde Lemmatization işlemini uygulayınız
df['Review'] = df['Review'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
df['Review'].head(10)





# TEXT VIRTUALIZATION



# Barplot (Barplot görselleştirme işlemi)


# 1-) "Review" değişkeninin içerdiği kelimeleri frekanslarını hesaplayınız, tf olarak kaydediniz
tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
# 2-) tf dataframe'inin sütunlarını yeniden adlandırınız: "words", "tf" şeklinde.
tf.columns = ["words", "tf"]
# 3-) "tf" değişkeninin değeri 500'den çok olanlara göre filtreleme işlemi yaparak barplot ile görselleştirme işlemini tamamlayınız.
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()



# Wordcloud (WordCloud görselleştirme işlemi)

# 1-) "Review" değişkeninin içerdiği tüm kelimeleri "text" isminde string olarak kaydediniz
text = " ".join(i for i in df.Review)

# 2-) WordCloud kullanarak şablon şeklinizi belirleyip kaydediniz
# 3-) Kaydettiğiniz wordcloud'u ilk adımda oluşturduğunuz string ile generate ediniz.
wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)

# 4-) Görselleştirme adımlarını tamamlayınız. (figure, imshow, axis, show)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()




# SENTIMENT ANALYSIS



# ADIM 1: Python içerisindeki NLTK paketinde tanımlanmış olan SentimentIntensityAnalyzer nesnesini oluşturunuz
sia = SentimentIntensityAnalyzer()

# ADIM 2: SentimentIntensityAnalyzer nesnesi ile polarite puanlarının incelenmesi

# 1-) "Review" değişkeninin ilk 10 gözlemi için polarity_scores() hesaplayınız
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))

# 2-) İncelenen ilk 10 gözlem için compund skorlarına göre filtrelenerek tekrar gözlemleyiniz
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

# 3-) 10 gözlem için compound skorları 0'dan büyükse "pos" değilse "neg" şeklinde güncelleyiniz
df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

# 4-) "Review" değişkenindeki tüm gözlemler için pos-neg atamasını yaparak yeni bir değişken olarak dataframe'e ekleyiniz
df["Sentiment_Label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df.groupby("Sentiment_Label")["Star"].mean()

# NOT:SentimentIntensityAnalyzer ile yorumları etiketleyerek, yorum sınıflandırma makine öğrenmesi modeli için bağımlı değişken oluşturulmuş oldu.



# PREPARATION FOR MACHINE LEARNING!



# ADIM 1: Bağımlı ve bağımsız değişkenlerimizi belirleyerek datayı train test olara ayırınız.

# Test-Train
train_x, test_x, train_y, test_y = train_test_split(df["Review"],
                                                    df["Sentiment_Label"],
                                                    random_state=42)
# ADIM 2: Makine öğrenmesi modeline verileri verebilmemiz için temsil şekillerini sayısala çevirmemiz gerekmekte.
# TF-IDF Word Level

# 1-) TfidfVectorizer kullanarak bir nesne oluşturunuz.
# 2-) Daha önce ayırmış olduğumuz train datamızı kullanarak oluşturduğumuz nesneye fit ediniz
tf_idf_word_vectorizer = TfidfVectorizer().fit(train_x)

# 3-) Oluşturmuş olduğumuz vektörü train ve test datalarına transform işlemini uygulayıp kaydediniz
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)






# MODELLING (LOGISTIC REGRESYON)



# ADIM 1: Lojistik regresyon modelini kurarak train dataları ile fit ediniz.
log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)



# ADIM 2: Kurmuş olduğunuz model ile tahmin işlemleri gerçekleştiriniz.

# 1-) Predict fonksiyonu ile test datasını tahmin ederek kaydediniz.
y_pred = log_model.predict(x_test_tf_idf_word)

# 2-) classification_report ile tahmin sonuçlarınızı raporlayıp gözlemleyiniz.
print(classification_report(y_pred, test_y))

# 3-) cross validation fonksiyonunu kullanarak ortalama accuracy değerini hesaplayınız
cross_val_score(log_model, x_test_tf_idf_word, test_y, cv=5).mean()



# ADIM 3: Veride bulunan yorumlardan ratgele seçerek modele sorulması.


# 1-) sample fonksiyonu ile "Review" değişkeni içerisinden örneklem seçierek yeni bir değere atayınız
random_review = pd.Series(df["Review"].sample(1).values)

# 2-) Elde ettiğiniz örneklemi modelin tahmin edebilmesi için CountVectorizer ile vektörleştiriniz.
# 3-) Vektörleştirdiğiniz örneklemi fit ve transform işlemlerini yaparak kaydediniz.
yeni_yorum = CountVectorizer().fit(train_x).transform(random_review)

# 4-) Kurmuş olduğunuz modele örneklemi vererek tahmin sonucunu kaydediniz.
pred = log_model.predict(yeni_yorum)

# 5-) Örneklemi ve tahmin sonucunu ekrana yazdırınız.
print(f'Review:  {random_review[0]} \n Prediction: {pred}')




# MODELLING (RANDOM FOREST)




# ADIM 1: Random Forest modeli ile tahmin sonuçlarının gözlenmesi;

# 1-) RandomForestClassifier modelini kurup fit ediniz.
rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)

# 2-) Cross validation fonksiyonunu kullanarak ortalama accuracy değerini hesaplayınız.
cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=5, n_jobs=-1).mean()















































