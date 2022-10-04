import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)  # Show all raws
pd.set_option('display.float_format', lambda x: '%.3f' %
              x)
pd.set_option('display.width', 500)

df = pd.read_csv(
    "dataset\heart_failure_clinical_records_dataset.csv")

df.columns = [col.upper() for col in df.columns]


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

# Değişkenlerin bir biri ile olan kolerasyonlarını incelememize olanak sağlıyor.
# Kolerasyon, değişkenlerin bir biri ile olan bağlılık, benzerliği ifade ediyor.
# Kolerasyonu yüksek iki değişken esasında aynı şeyi söylüyor anlamına gelir.
# Kolerasyonun yüksek olmasını istemeyiz.
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), vmin=-1, cmap='coolwarm', annot=True)
plt.show()


# TODO : Fonksiyon şeklinde tanımlama yaptığım zaman .head() üzerinden herhangi bir dönüş alamıyorum.
# def load():
#     data = pd.read_csv(
#         "D:\CODING_AREA\PROJECTS\Heart_Failure_Prediction\dataset\heart_failure_clinical_records_dataset.csv")
#     return data
# df = load()
# df.head()
# Verilen veri seti ve kolunun üst ve alt limit değerlerini döndürür. Aykırı değer hesabı için kullanacağız.

# Kategorik değişkenlerimizi özetleme fonksiyonumuz.
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

# Sayısal değişkenlerimizi özetleme fonksiyonumuz.


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40,
                 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

# Bağımlı değişkeni sayısal bir değişken ile özetlemek istediğimizde.


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

# Bağımlı değişkeni kategorik bir değişken ile özetlemek istediğimizde.


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(
        categorical_col)[target].mean()}), end="\n\n\n")

# Sayısal değişkenlerin bir biri arasındaki kolerasyonu hesaplamaya yarar.


def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={
                      'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

# Kategorik, sayıasl ve kategorik gibi gözüken kardinal değişkenleri verir.
# 3 adet return değeri mevcuttur.


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


# Değişken türlerinin ayrıştırılması
# cat_th=5 ifadesi ile 5 değerinde az ise değişkenlik kategorik olarak al demek.
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)


# Kategorik değişkenlerin incelenmesi
for col in cat_cols:
    cat_summary(df, col)

# Sayısal değişkenlerin incelenmesi
# Max ve min değerlerine bakarak anormel bir durum var mı gözlemlenir.
df[num_cols].describe().T


# Sayısal değişkenlerin grafik ile ifade edilmesi.
# for col in num_cols:
#     num_summary(df, col, plot=True)

# Sayısal değişkenkerin birbirleri ile korelasyonu
# Veri değişkenlerinin bir birini etkileme durumuna göz attık.
# Yüksek kolerasyon istediğimiz bir şey değil.
correlation_matrix(df, num_cols)


# Target ile sayısal değişkenlerin incelemesi
# Yani bağımlı değişkenmizi hangi değişkenlerin ortalamasında hangi değeri aldığını görmemiz mümkün.
# Genel bir bilgi edindik.
for col in num_cols:
    target_summary_with_num(df, "DEATH_EVENT", col)


# Eşik değeri matematiksel yolla hesaplar.
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# Aykırı değeri sınır değer ile değiştirebiliriz.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Aykırı değer var mı yok mu mesajını bize döner.


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Kategorik değişkenleri sayısal değişkenlere dönüştürmemize yarayacak.


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(
        dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

# TODO: Yeni değişken türetme işlemini araştır. Şimdilik bu proje için bu konuyu atlıyoruz.


# Standartlaştırma
X_scaled = StandardScaler().fit_transform(df[num_cols])
# Standartlaştırılmış değerlerimizi isimleri ile beraber verimize gönderiyoruz.
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

# Bağımlı bağımsız değişken ataması yaptık.
y = df["DEATH_EVENT"]
X = df.drop(["DEATH_EVENT"], axis=1)

# Genel kontrolümüzü yapıyoruz.
check_df(X)


# XGBoost modelinde hata çıktı onları kapatmak için ekleme yaptık.
# Tüm modellerimizi deniyerek nasıl bir performans verdiklerini gözlemliyoruz.
def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(
                       use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False)) # Çok uzun sürdüğü için comentledik
                   ]
    # Tüm modelleri deniyoruz.
    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(
            f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


# Tüm modellerimizin başarılarına bakıyoruz.
# Modeller şuan ham durumda tabi.
base_models(X, y, scoring="accuracy")
