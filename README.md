
# Binance TR Akıllı Coin Tarayıcı v2

## Yeni özellik
- Binance TR'nin aktif TRY paritelerini API'den otomatik alır.
- Tüm TRY coinlerini tek tuşla tarar.
- GPS/TRY mevcutsa otomatik olarak taramaya dahil eder.
- En güçlü AL adaylarını puana göre sıralar.
- En zayıf / SAT adaylarını ayrıca gösterir.
- Coin adına göre tarama sonucu arama.
- CSV dışa aktarma.
- Tek coin için ayrıntılı mum grafiği ve teknik göstergeler.
- Risk / stop-loss / take-profit hesaplama.

## Teknik göstergeler
EMA20, EMA50, EMA200, RSI(14), MACD, Bollinger Bands, ATR, hacim ortalaması,
kısa dönem momentum, destek/direnç.

## Kurulum
Windows:
1. Python 3.11 veya üzerini kurun.
2. ZIP dosyasını açın.
3. `baslat.bat` dosyasına çift tıklayın.
4. Tarayıcıda uygulama açılır.

Manuel:
    pip install -r requirements.txt
    streamlit run app.py

## Güvenlik
Bu sürüm yalnızca halka açık piyasa verisini okur. API key istemez ve gerçek AL/SAT emri göndermez.
