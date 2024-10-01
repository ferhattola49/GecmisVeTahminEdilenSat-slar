# csv dosyalarını okumak için
import pandas as pd
# 2 boyutlu grafik oluşturmak için
import matplotlib.pyplot as plt

# csv dosyamızı okuduk.
data = pd.read_csv('C:/Users/Administrator/Desktop/satislar.csv')

# Aylar isimli kolonu bir değişkene atadık
Aylar = data[['Aylar']]

# Satislar isimli kolonu bir değişkene atadık
Satislar = data[['Satislar']]

# sklearn kütüphanesini kullanarak verileri test ve eğitim olarak böleceğimiz fonksiyonu import ettik.
from sklearn.model_selection import train_test_split

# Veri kümemizi test ve train şeklinde bölüyoruz
x_train, x_test, y_train, y_test = train_test_split(Aylar, Satislar, test_size=0.33, random_state=0)

# sklearn kütüphanesini kullanarak LinearRegression sınıfını dahil ediyoruz.
from sklearn.linear_model import LinearRegression
# Sınıftan bir nesne oluşturuyoruz.
lr = LinearRegression()

# Train veri kümelerini vererek makineyi eğitiyoruz.
lr.fit(x_train, y_train)

# Aylar'ın test kümesini vererek Satislar'ı tahmin etmesini sağlıyoruz. Üst satırda makinemizi eğitmiştik.
tahmin = lr.predict(x_test)

# Verileri grafikte düzenli göstermek için index numaralarına göre sıralıyoruz.
x_train = x_train.sort_index()
y_train = y_train.sort_index()

# Grafik şeklinde ekrana basıyoruz.
plt.figure(figsize=(10,6))  # Grafiğin boyutunu ayarlıyoruz.
plt.plot(x_train, y_train, color='blue', label='Geçmiş Veriler')  # Geçmiş verileri mavi ile gösteriyoruz.
plt.plot(x_test, tahmin, color='red', label='Tahmin Edilen Veriler')  # Tahmin edilen verileri kırmızı ile gösteriyoruz.

# Başlık ve eksen isimleri ekliyoruz.
plt.title('Geçmiş ve Tahmin Edilen Satışlar')
plt.xlabel('Aylar')
plt.ylabel('Satışlar')

# Grafikteki açıklamayı (legend) ekliyoruz.
plt.legend()

# Grafiği gösteriyoruz.
plt.show()
