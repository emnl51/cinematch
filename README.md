# CineMatch - Akıllı Film Öneri Sistemi

CineMatch, yapay zeka destekli bir film ve dizi öneri sistemidir. Kullanıcıların beğenilerini öğrenerek kişiselleştirilmiş öneriler sunar.

## 🚀 Özellikler

### 🎯 Akıllı Öğrenme Sistemi
- **4 Aşamalı Öğrenme**: Başlangıç → Profil Geliştirme → Test → Optimizasyon
- **Hibrit Öneri Algoritması**: Content-based + Collaborative filtering
- **Adaptif Strateji**: Kullanıcı davranışına göre öneri stratejisi değişimi
- **Soğuk Başlangıç Desteği**: Yeni kullanıcılar için popüler içerik önerileri

### 🎬 Filmografi Özellikleri
- **Gelişmiş Filmografi Görüntüleyici**: Oyuncu/yönetmen filmografilerini görüntüleme
- **Akıllı Önbellekleme**: LRU cache ile performans optimizasyonu
- **Hata Yönetimi**: Kapsamlı hata yakalama ve kullanıcı dostu mesajlar
- **Throttling**: API isteklerini optimize etme

### 🔧 Production-Ready Özellikler
- **Merkezi Logging**: Development/production ortamları için ayrı log seviyeleri
- **Cache Yönetimi**: Çoklu cache katmanları ve istatistikler
- **Error Boundaries**: Hata yakalama ve graceful degradation
- **Test Coverage**: Kapsamlı unit ve integration testleri

## 🛠️ Teknoloji Stack

- **Frontend**: React 18 + TypeScript + Vite
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **Charts**: Recharts
- **API**: TMDb (The Movie Database)
- **Testing**: Vitest + React Testing Library
- **Linting**: ESLint + TypeScript ESLint

## 📋 Gereksinimler

- Node.js 18+ 
- npm veya yarn
- TMDb API anahtarı

## 🔧 Kurulum

1. **Projeyi klonlayın**:
```bash
git clone <repository-url>
cd cinematch
```

2. **Bağımlılıkları yükleyin**:
```bash
npm install
```

3. **Ortam değişkenlerini ayarlayın**:
```bash
cp .env.example .env
```

`.env` dosyasını düzenleyin ve TMDb API kimlik bilgilerinizi ekleyin (API anahtarı veya Read Access Token):
```env
VITE_TMDB_API_KEY=your_tmdb_api_key_here
VITE_TMDB_READ_ACCESS_TOKEN=your_tmdb_read_access_token_here
```

4. **Geliştirme sunucusunu başlatın**:
```bash
npm run dev
```

Uygulama `http://localhost:3000` adresinde çalışacaktır.

## 🧪 Test Komutları

```bash
# Tüm testleri çalıştır
npm run test

# Test UI'ı ile çalıştır
npm run test:ui

# Coverage raporu ile çalıştır
npm run test:coverage

# Watch modunda çalıştır
npm run test -- --watch
```

## 🏗️ Build ve Deploy

**Production build oluşturun**:
```bash
npm run build
```

**Build'i önizleyin**:
```bash
npm run preview
```

## 🧪 Geliştirme Komutları

```bash
# Geliştirme sunucusu
npm run dev

# TypeScript tip kontrolü
npm run type-check

# ESLint kontrolü
npm run lint

# ESLint otomatik düzeltme
npm run lint:fix

# Production build
npm run build

# Build önizleme
npm run preview
```

## 📁 Proje Yapısı

```
src/
├── components/          # React bileşenleri
│   ├── Header.tsx
│   ├── MovieCard.tsx
│   ├── ProfileSection.tsx
│   ├── FilmList.tsx
│   ├── PersonFilmographyExample.tsx
│   ├── ContentFilters.tsx
│   └── ...
├── hooks/              # Custom React hooks
│   ├── useMovieData.ts
│   ├── usePersonFilmography.ts
│   └── useRecommendation.ts
├── services/           # API ve iş mantığı servisleri
│   ├── tmdb.ts
│   ├── profileService.ts
│   ├── recommendationService.ts
│   └── ...
├── utils/              # Yardımcı fonksiyonlar
│   ├── logger.ts
│   ├── cache.ts
│   └── recommendation/
│       ├── collaborative.ts
│       ├── contentBased.ts
│       └── mixStrategy.ts
├── types/              # TypeScript tip tanımları
│   └── index.ts
├── test/               # Test konfigürasyonu
│   └── setup.ts
├── App.tsx             # Ana uygulama bileşeni
└── main.tsx           # Uygulama giriş noktası
```

## 🎬 usePersonFilmography Hook Kullanımı

Yeni eklenen `usePersonFilmography` hook'u ile oyuncu/yönetmen filmografilerini görüntüleyebilirsiniz:

```tsx
import { usePersonFilmography } from './hooks/usePersonFilmography';
import { FilmList } from './components/FilmList';

function ActorProfile({ actorId }: { actorId: number }) {
  const { 
    movieCredits, 
    tvCredits, 
    loading, 
    error,
    allCredits,
    castCredits,
    crewCredits,
    refetch
  } = usePersonFilmography(actorId);

  if (loading) return <div>Yükleniyor...</div>;
  if (error) return <div>Hata: {error}</div>;

  return (
    <div>
      <h2>Toplam {allCredits.length} içerik</h2>
      <button onClick={refetch}>Yenile</button>
      <FilmList credits={movieCredits} title="Filmler" showFilters={true} />
      <FilmList credits={tvCredits} title="Diziler" showFilters={true} />
    </div>
  );
}
```

## 🤖 Öneri Algoritması Kullanımı

Hibrit öneri sistemi ile gelişmiş öneriler alın:

```tsx
import { useRecommendation } from './hooks/useRecommendation';

function RecommendationSection({ profile, ratings, genres }) {
  const { 
    recommendations, 
    loading, 
    error,
    generateRecommendations,
    refreshRecommendations
  } = useRecommendation(profile, ratings, genres);

  // Özel strateji ile öneri oluştur
  const customStrategy = {
    profileWeight: 0.8,    // %80 profil bazlı
    surpriseWeight: 0.15,  // %15 sürpriz
    diversityWeight: 0.05  // %5 çeşitlilik
  };

  return (
    <div>
      <button onClick={() => generateRecommendations(customStrategy)}>
        Özel Öneriler Oluştur
      </button>
      <button onClick={refreshRecommendations}>
        Önerileri Yenile
      </button>
      {/* Öneriler listesi */}
    </div>
  );
}
```

## 🔑 TMDb API Anahtarı Alma

1. [TMDb](https://www.themoviedb.org/) sitesine kaydolun
2. Hesap ayarlarından API bölümüne gidin
3. API anahtarınızı kopyalayın
4. `.env` dosyasına ekleyin

## 🎯 Kullanım

1. **İlk Kullanım**: Farklı türlerden 5+ film/dizi puanlayın
2. **Profil Geliştirme**: 50 içeriğe kadar puanlama yaparak profilinizi detaylandırın
3. **Test Aşaması**: Sistem önerilerini test eder ve doğruluk oranını hesaplar
4. **Optimizasyon**: Sürekli öğrenme ile kişiselleştirilmiş öneriler alın
5. **Filmografi**: Favori oyuncularınızın tüm filmografisini keşfedin

## 🔧 Yapılandırma

### Cache Ayarları
```typescript
// Cache TTL ve boyut ayarları
const cacheConfig = {
  personFilmography: { ttl: 60000, maxSize: 50 },
  movieDetails: { ttl: 300000, maxSize: 100 },
  search: { ttl: 30000, maxSize: 20 }
};
```

### Logging Konfigürasyonu
```typescript
// Development'ta tüm loglar, production'da sadece warn/error
import { logger } from './utils/logger';

logger.debug('Debug mesajı'); // Sadece dev'de görünür
logger.info('Bilgi mesajı');
logger.warn('Uyarı mesajı');
logger.error('Hata mesajı');
```

### Öneri Algoritması Ayarları
```typescript
// Adaptif strateji ayarları
const strategies = {
  newUser: { profileWeight: 0.5, surpriseWeight: 0.3, diversityWeight: 0.2 },
  experienced: { profileWeight: 0.8, surpriseWeight: 0.15, diversityWeight: 0.05 }
};
```

## 🐛 Sorun Giderme

### API Hataları
- **401 Unauthorized**: TMDb API anahtarınızı kontrol edin
- **404 Not Found**: Kişi/içerik ID'sinin geçerli olduğundan emin olun
- **429 Too Many Requests**: API rate limit'e takıldınız, biraz bekleyin

### Build Hataları
- `npm run type-check` ile tip hatalarını kontrol edin
- `npm run lint` ile kod kalitesi sorunlarını kontrol edin

### Test Hataları
- `npm run test -- --reporter=verbose` ile detaylı test çıktısı alın
- Mock'ların doğru çalıştığından emin olun

### Cache Sorunları
```typescript
// Cache'i temizlemek için
tmdbService.clearCache();

// Cache istatistiklerini görmek için
console.log(tmdbService.getCacheStats());
```

## 📊 Performans Metrikleri

- **Cache Hit Rate**: %85+ hedeflenir
- **API Response Time**: <500ms ortalama
- **Bundle Size**: <2MB gzipped
- **Test Coverage**: >80%

## 🔒 Güvenlik

- API anahtarları environment variables'da saklanır
- Client-side'da hassas bilgi saklanmaz
- CORS politikaları uygulanır
- Input validation yapılır

## 📄 Lisans

Copyright (c) 2025 MoleQ

Bu yazılım yalnızca kişisel kullanım içindir. Ticari kullanım, yeniden satış, dağıtım veya bu yazılımın herhangi bir şekilde gelir elde etmek amacıyla kullanılması yasaktır.

Yazılım, olduğu gibi (AS IS) sağlanmaktadır, hiçbir garanti verilmez. Yazılımı kullanarak, bu koşulları kabul etmiş olursunuz.

İzin verilenler:
- Kişisel projelerde kullanım
- Kaynak kodun incelenmesi, değiştirilmesi

Yasaklar:
- Ticari amaçlı kullanım
- Üçüncü taraflara satış, dağıtım veya lisanslama
- Ticari hizmetlerde entegre edilmesi




## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Testlerinizi yazın ve çalıştırın (`npm run test`)
4. Commit yapın (`git commit -m 'Add amazing feature'`)
5. Push yapın (`git push origin feature/amazing-feature`)
6. Pull Request oluşturun

## 📞 İletişim

Sorularınız için issue açabilir veya pull request gönderebilirsiniz.

---

**CineMatch** - Yapay zeka ile film keşfetmenin yeni yolu! 🎬✨
