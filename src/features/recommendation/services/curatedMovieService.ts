import type { Movie, TVShow, Genre, UserRating } from '../../content/types';
import type { CuratedContentFilters } from '../components/CuratedContentFilters';
import { tmdbService } from '../../content/services/tmdb';

export class CuratedMovieService {
  // Temel türler ve kombinasyonlar (hem film hem dizi için)
  private static readonly GENRE_GROUPS = [
    { name: 'Aksiyon', genres: [28] },
    { name: 'Komedi', genres: [35] },
    { name: 'Dram', genres: [18] },
    { name: 'Bilim Kurgu', genres: [878] },
    { name: 'Korku', genres: [27] },
    { name: 'Romantik', genres: [10749] },
    { name: 'Gerilim', genres: [53] },
    { name: 'Animasyon', genres: [16] },
    { name: 'Suç', genres: [80] },
    { name: 'Macera', genres: [12] },
    { name: 'Fantastik', genres: [14] },
    { name: 'Müzikal', genres: [10402] },
    { name: 'Belgesel', genres: [99] },
    { name: 'Aile', genres: [10751] },
    // Kombinasyonlar
    { name: 'Aksiyon + Bilim Kurgu', genres: [28, 878] },
    { name: 'Komedi + Dram', genres: [35, 18] },
    { name: 'Korku + Gerilim', genres: [27, 53] },
    { name: 'Romantik + Komedi', genres: [10749, 35] },
    { name: 'Aksiyon + Macera', genres: [28, 12] },
    { name: 'Dram + Suç', genres: [18, 80] },
    { name: 'Fantastik + Macera', genres: [14, 12] },
    { name: 'Bilim Kurgu + Gerilim', genres: [878, 53] },
    { name: 'Animasyon + Aile', genres: [16, 10751] },
    { name: 'Dram + Romantik', genres: [18, 10749] }
  ];

  static async getCuratedInitialContent(
    existingRatings: UserRating[] = [],
    progressCallback?: (progress: { current: number; total: number; message: string }) => void,
    watchlistItems: { id: number }[] = [],
    settings?: {
      discoveryContentCount?: number;
      showKidsContent?: boolean;
    }
  ): Promise<(Movie | TVShow)[]> {
    const curatedContent: (Movie | TVShow)[] = [];
    const usedContentIds = new Set<number>();
    
    // Daha önce puanlanan içerikleri ve watchlist'teki içerikleri set'e ekle
    const excludedContentIds = new Set([
      ...existingRatings.map(r => r.movieId),
      ...watchlistItems.map(item => item.id)
    ]);

    const targetContentCount = settings?.discoveryContentCount || 20;

    const allowKidsContent = settings?.showKidsContent ?? false;
    const filterKidsContent = <T extends { genre_ids?: number[] }>(items: T[]) => (
      allowKidsContent
        ? items
        : items.filter(content => !((content.genre_ids || []).includes(16) || (content.genre_ids || []).includes(10751)))
    );

    // Filtreleme yoksa yüksek kaliteli popüler içerikler getir
    if (existingRatings.length === 0) {
      progressCallback?.({ 
        current: 0, 
        total: 4, 
        message: 'En popüler ve yüksek puanlı içerikler aranıyor...' 
      });

      try {
        // En popüler filmler
        const popularMoviesResponse = await tmdbService.discoverMovies({
          'vote_average.gte': 8.0,
          'vote_count.gte': 200,
          sort_by: 'popularity.desc',
          page: 1
        });

        progressCallback?.({ 
          current: 1, 
          total: 4, 
          message: 'En yüksek puanlı filmler aranıyor...' 
        });

        // En yüksek puanlı filmler
        const topRatedMoviesResponse = await tmdbService.discoverMovies({
          'vote_average.gte': 8.0,
          'vote_count.gte': 200,
          sort_by: 'vote_average.desc',
          page: 1
        });

        progressCallback?.({ 
          current: 2, 
          total: 4, 
          message: 'En popüler diziler aranıyor...' 
        });

        // En popüler diziler
        const popularTVResponse = await tmdbService.discoverTVShows({
          'vote_average.gte': 8.0,
          'vote_count.gte': 200,
          sort_by: 'popularity.desc',
          page: 1
        });

        progressCallback?.({ 
          current: 3, 
          total: 4, 
          message: 'En yüksek puanlı diziler aranıyor...' 
        });

        // En yüksek puanlı diziler
        const topRatedTVResponse = await tmdbService.discoverTVShows({
          'vote_average.gte': 8.0,
          'vote_count.gte': 200,
          sort_by: 'vote_average.desc',
          page: 1
        });

        // Tüm içerikleri birleştir ve karıştır
        const allContent = filterKidsContent([
          ...popularMoviesResponse.results.slice(0, 8).map(movie => ({ ...movie, media_type: 'movie' as const })),
          ...topRatedMoviesResponse.results.slice(0, 6).map(movie => ({ ...movie, media_type: 'movie' as const })),
          ...popularTVResponse.results.slice(0, 6).map(show => ({ ...show, media_type: 'tv' as const })),
          ...topRatedTVResponse.results.slice(0, 4).map(show => ({ ...show, media_type: 'tv' as const }))
        ]);

        // İçerikleri karıştır ve tekrarları kaldır
        const shuffledContent = allContent
          .filter(content => !usedContentIds.has(content.id) && !excludedContentIds.has(content.id))
          .sort(() => Math.random() - 0.5);

        // Hedef sayıda içerik seç
        const selectedContent = shuffledContent.slice(0, targetContentCount);
        
        for (const content of selectedContent) {
          curatedContent.push(content);
          usedContentIds.add(content.id);
        }

        progressCallback?.({ 
          current: 4, 
          total: 4, 
          message: 'Yüksek kaliteli içerikler hazırlandı!' 
        });

        return curatedContent;

      } catch (error) {
        console.error('Error fetching high-quality content:', error);
        // Hata durumunda normal akışa devam et
      }
    }

    // Normal akış - tür bazlı içerik seçimi
    progressCallback?.({ 
      current: 0, 
      total: 15, 
      message: 'Tür bazlı içerikler aranıyor...' 
    });

    // Rastgele 15 grup seç (hem film hem dizi için)
    const shuffledGroups = [...this.GENRE_GROUPS].sort(() => Math.random() - 0.5);
    const selectedGroups = shuffledGroups.slice(0, 15);

    // Her grup için hem film hem dizi seç
    for (let i = 0; i < selectedGroups.length; i++) {
      const group = selectedGroups[i];
      
      progressCallback?.({ 
        current: i + 1, 
        total: selectedGroups.length, 
        message: `${group.name} türünden içerikler aranıyor...` 
      });
      
      try {
        // Film seç
        const movie = await this.getRandomMovieFromGroup(group.genres, usedContentIds, excludedContentIds, allowKidsContent);
        if (movie) {
          curatedContent.push(movie);
          usedContentIds.add(movie.id);
        }

        // Dizi seç (50% şans)
        if (Math.random() > 0.5) {
          const tvShow = await this.getRandomTVShowFromGroup(group.genres, usedContentIds, excludedContentIds, allowKidsContent);
          if (tvShow) {
            curatedContent.push(tvShow);
            usedContentIds.add(tvShow.id);
          }
        }
      } catch (error) {
        console.error(`Error fetching content for group ${group.name}:`, error);
        // Continue with next group instead of failing completely
      }
    }

    // Eğer yeterli içerik yoksa, popüler içeriklerle tamamla
    if (curatedContent.length < targetContentCount) {
      progressCallback?.({ 
        current: selectedGroups.length, 
        total: selectedGroups.length, 
        message: 'Ek popüler içerikler aranıyor...' 
      });
      
      try {
        const additionalContent = await this.getAdditionalPopularContent(usedContentIds, excludedContentIds, targetContentCount - curatedContent.length, undefined, allowKidsContent);
        curatedContent.push(...additionalContent);
      } catch (error) {
        console.error('Error fetching additional popular content:', error);
      }
    }

    progressCallback?.({ 
      current: selectedGroups.length, 
      total: selectedGroups.length, 
      message: 'İçerikler hazırlandı!' 
    });

    return curatedContent.slice(0, targetContentCount);
  }

  private static async getRandomMovieFromGroup(
    genreIds: number[],
    usedContentIds: Set<number>,
    excludedContentIds: Set<number>,
    allowKidsContent: boolean
  ): Promise<Movie | null> {
    // Birden fazla sayfa dene
    for (let attempt = 0; attempt < 8; attempt++) {
      const randomPage = Math.floor(Math.random() * 5) + 1; // İlk 5 sayfa daha kaliteli
      
      const discoverParams = {
        with_genres: genreIds.join(','),
        'vote_average.gte': 8.0, // Daha yüksek minimum puan
        'vote_count.gte': 200,   // Minimum oy sayısı
        sort_by: attempt % 2 === 0 ? 'vote_average.desc' : 'popularity.desc',
        page: randomPage
      };

      try {
        const response = await tmdbService.discoverMovies(discoverParams);
        
        // Puanlanmamış, kullanılmamış ve watchlist'te olmayan filmleri filtrele
        const availableMovies = response.results
          .filter(movie => !usedContentIds.has(movie.id) && !excludedContentIds.has(movie.id))
          .filter(movie => allowKidsContent || !((movie.genre_ids || []).includes(16) || (movie.genre_ids || []).includes(10751)))
          .slice(0, 8);
        
        if (availableMovies.length > 0) {
          const randomIndex = Math.floor(Math.random() * availableMovies.length);
          const selectedMovie = availableMovies[randomIndex];
          selectedMovie.media_type = 'movie';
          return selectedMovie;
        }
      } catch (error) {
        console.error('Error in getRandomMovieFromGroup:', error);
        // Continue to next attempt instead of throwing
      }
    }

    return null;
  }

  private static async getRandomTVShowFromGroup(
    genreIds: number[],
    usedContentIds: Set<number>,
    excludedContentIds: Set<number>,
    allowKidsContent: boolean
  ): Promise<TVShow | null> {
    // Birden fazla sayfa dene
    for (let attempt = 0; attempt < 8; attempt++) {
      const randomPage = Math.floor(Math.random() * 5) + 1; // İlk 5 sayfa daha kaliteli
      
      const discoverParams = {
        with_genres: genreIds.join(','),
        'vote_average.gte': 8.0, // Daha yüksek minimum puan
        'vote_count.gte': 200,   // Minimum oy sayısı
        sort_by: attempt % 2 === 0 ? 'vote_average.desc' : 'popularity.desc',
        page: randomPage
      };

      try {
        const response = await tmdbService.discoverTVShows(discoverParams);
        
        // Puanlanmamış, kullanılmamış ve watchlist'te olmayan dizileri filtrele
        const availableShows = response.results
          .filter(show => !usedContentIds.has(show.id) && !excludedContentIds.has(show.id))
          .filter(show => allowKidsContent || !((show.genre_ids || []).includes(16) || (show.genre_ids || []).includes(10751)))
          .slice(0, 8);
        
        if (availableShows.length > 0) {
          const randomIndex = Math.floor(Math.random() * availableShows.length);
          const selectedShow = availableShows[randomIndex];
          selectedShow.media_type = 'tv';
          return selectedShow;
        }
      } catch (error) {
        console.error('Error in getRandomTVShowFromGroup:', error);
        // Continue to next attempt instead of throwing
      }
    }

    return null;
  }

  private static async getAdditionalPopularContent(
    usedContentIds: Set<number>,
    excludedContentIds: Set<number>,
    count: number,
    filters?: CuratedContentFilters,
    allowKidsContent = false
  ): Promise<(Movie | TVShow)[]> {
    const additionalContent: (Movie | TVShow)[] = [];
    
    // Hem film hem dizi için yüksek kaliteli popüler içerik al
    const sortOptions = ['vote_average.desc', 'popularity.desc', 'vote_count.desc'];
    
    for (let attempt = 0; attempt < 15 && additionalContent.length < count; attempt++) {
      try {
        const randomSort = sortOptions[Math.floor(Math.random() * sortOptions.length)];
        const randomPage = Math.floor(Math.random() * 5) + 1; // İlk 5 sayfa daha kaliteli
        
        // Film veya dizi seç (50-50 şans)
        if (Math.random() > 0.5) {
          // Film - daha yüksek kalite kriterleri
          const response = await tmdbService.discoverMovies({
            'vote_average.gte': 8.0, // Daha yüksek minimum puan
            'vote_count.gte': 200,   // Minimum oy sayısı
            sort_by: randomSort,
            page: randomPage,
            ...(filters && filters.languages && filters.languages.length > 0 ? { with_original_language: filters.languages.join(',') } : {})
          });

          const availableMovies = response.results.filter(
            movie => !usedContentIds.has(movie.id) && !excludedContentIds.has(movie.id)
          ).filter(movie => allowKidsContent || !((movie.genre_ids || []).includes(16) || (movie.genre_ids || []).includes(10751)));
          
          for (const movie of availableMovies) {
            if (additionalContent.length >= count) break;
            movie.media_type = 'movie';
            additionalContent.push(movie);
            usedContentIds.add(movie.id);
          }
        } else {
          // Dizi - daha yüksek kalite kriterleri
          const response = await tmdbService.discoverTVShows({
            'vote_average.gte': 8.0, // Daha yüksek minimum puan
            'vote_count.gte': 200,   // Minimum oy sayısı
            sort_by: randomSort,
            page: randomPage,
            ...(filters && filters.languages && filters.languages.length > 0 ? { with_original_language: filters.languages.join(',') } : {})
          });

          const availableShows = response.results.filter(
            show => !usedContentIds.has(show.id) && !excludedContentIds.has(show.id)
          ).filter(show => allowKidsContent || !((show.genre_ids || []).includes(16) || (show.genre_ids || []).includes(10751)));
          
          for (const show of availableShows) {
            if (additionalContent.length >= count) break;
            show.media_type = 'tv';
            additionalContent.push(show);
            usedContentIds.add(show.id);
          }
        }
      } catch (error) {
        console.error('Error fetching additional popular content:', error);
        // Continue to next attempt
      }
    }

    return additionalContent;
  }

  static getGenreGroupName(content: Movie | TVShow, genres: Genre[]): string {
    const contentGenreIds = content.genre_ids || [];
    
    // Önce kombinasyonları kontrol et
    for (const group of this.GENRE_GROUPS) {
      if (group.genres.length > 1) {
        const hasAllGenres = group.genres.every(genreId => contentGenreIds.includes(genreId));
        if (hasAllGenres) {
          return group.name;
        }
      }
    }
    
    // Sonra tekil türleri kontrol et
    for (const group of this.GENRE_GROUPS) {
      if (group.genres.length === 1) {
        if (contentGenreIds.includes(group.genres[0])) {
          return group.name;
        }
      }
    }
    
    // Hiçbiri uymazsa genel tür isimlerini döndür
    return contentGenreIds
      .map(id => genres.find(g => g.id === id)?.name)
      .filter(Boolean)
      .slice(0, 2)
      .join(' + ') || 'Çeşitli';
  }

  static async getInitialRatingContent(
    progressCallback?: (progress: { current: number; total: number; message: string }) => void
  ): Promise<(Movie | TVShow)[]> {
    const initialContent: (Movie | TVShow)[] = [];
    const usedContentIds = new Set<number>();
    
    progressCallback?.({ 
      current: 0, 
      total: 4, 
      message: 'Yüksek kaliteli içerikler aranıyor...' 
    });

    try {
      // En az 7 puan ve 200 oy alan popüler filmler
      const popularMoviesResponse = await tmdbService.discoverMovies({
        'vote_average.gte': 8.0,
        'vote_count.gte': 200,
        sort_by: 'popularity.desc',
        page: 1
      });

      progressCallback?.({ 
        current: 1, 
        total: 4, 
        message: 'Yüksek kaliteli diziler aranıyor...' 
      });

      // En az 7 puan ve 200 oy alan popüler diziler
      const popularTVResponse = await tmdbService.discoverTVShows({
        'vote_average.gte': 8.0,
        'vote_count.gte': 200,
        sort_by: 'popularity.desc',
        page: 1
      });

      progressCallback?.({ 
        current: 2, 
        total: 4, 
        message: 'İçerikler karıştırılıyor...' 
      });

      // Tüm içerikleri birleştir
      const allContent = [
        ...popularMoviesResponse.results.slice(0, 50).map(movie => ({ ...movie, media_type: 'movie' as const })),
        ...popularTVResponse.results.slice(0, 50).map(show => ({ ...show, media_type: 'tv' as const }))
      ].filter(content => !((content.genre_ids || []).includes(16) || (content.genre_ids || []).includes(10751)));

      // İçerikleri karıştır ve tekrarları kaldır
      const shuffledContent = allContent
        .filter(content => !usedContentIds.has(content.id))
        .sort(() => Math.random() - 0.5);

      progressCallback?.({ 
        current: 3, 
        total: 4, 
        message: 'AI öğrenme için 10 içerik seçiliyor...' 
      });

      // İlk 100 içeriği seç
      const selectedContent = shuffledContent.slice(0, 100);
      
      for (const content of selectedContent) {
        initialContent.push(content);
        usedContentIds.add(content.id);
      }

      // Eğer içerik 100'den azsa, ek popüler içeriklerle tamamla
      if (initialContent.length < 100) {
        try {
          const additionalContent = await this.getAdditionalPopularContent(usedContentIds, new Set(), 100 - initialContent.length);
          initialContent.push(...additionalContent);
        } catch (error) {
          console.error('Error fetching additional onboarding content:', error);
        }
      }

      progressCallback?.({ 
        current: 4, 
        total: 4, 
        message: 'AI öğrenme içerikleri hazırlandı!' 
      });

      return initialContent;

    } catch (error) {
      console.error('Error fetching initial rating content:', error);
      
      // Hata durumunda yine de yüksek kaliteli içerikler döndürmeye çalış
      try {
        const fallbackResponse = await tmdbService.discoverMovies({
          'vote_average.gte': 8.0,
          'vote_count.gte': 200,
          sort_by: 'popularity.desc',
          page: 1
        });

        return fallbackResponse.results
          .slice(0, 10)
          .map(movie => ({ ...movie, media_type: 'movie' as const }))
          .filter(movie => !((movie.genre_ids || []).includes(16) || (movie.genre_ids || []).includes(10751)));
      } catch (fallbackError) {
        console.error('Fallback content loading also failed:', fallbackError);
        return [];
      }
    }
  }

  static async getCuratedContentWithFilters(
    existingRatings: UserRating[] = [],
    filters: CuratedContentFilters,
    watchlistItems: { id: number }[] = [],
    settings?: {
      discoveryContentCount?: number;
      showKidsContent?: boolean;
    }
  ): Promise<(Movie | TVShow)[]> {
    const curatedContent: (Movie | TVShow)[] = [];
    const usedContentIds = new Set<number>();
    const excludedContentIds = new Set([
      ...existingRatings.map(r => r.movieId),
      ...watchlistItems.map(item => item.id)
    ]);

    const allowKidsContent = settings?.showKidsContent ?? false;
    const filterKidsContent = <T extends { genre_ids?: number[] }>(items: T[]) => (
      allowKidsContent
        ? items
        : items.filter(content => !((content.genre_ids || []).includes(16) || (content.genre_ids || []).includes(10751)))
    );

    try {
      // Filtrelere göre içerik arama parametreleri
      const baseParams: any = {
        'vote_average.gte': filters.minRating,
        'vote_average.lte': filters.maxRating,
        'vote_count.gte': filters.minVoteCount,
        sort_by: filters.sortBy === 'rating' ? 'vote_average.desc' : 
                filters.sortBy === 'year' ? 'release_date.desc' : 
                filters.sortBy === 'title' ? 'title.asc' : 'popularity.desc'
      };

      // Yıl filtresi
      if (filters.minYear > 1900) {
        baseParams['primary_release_date.gte'] = `${filters.minYear}-01-01`;
      }
      if (filters.maxYear < new Date().getFullYear()) {
        baseParams['primary_release_date.lte'] = `${filters.maxYear}-12-31`;
      }

      // Tür filtresi
      if (filters.genres.length > 0) {
        baseParams['with_genres'] = filters.genres.join(',');
      }

      // Dil filtresi
      if (filters.languages && filters.languages.length > 0) {
        baseParams['with_original_language'] = filters.languages.join(',');
      }

      // Media type filtresi
      if (filters.mediaType === 'movie') {
        // Sadece film ara
        const movieResponse = await tmdbService.discoverMovies({
          ...baseParams,
          page: 1
        });
        
        const contentCount = settings?.discoveryContentCount || 20;
        const availableMovies = filterKidsContent(movieResponse.results)
          .filter(movie => !usedContentIds.has(movie.id) && !excludedContentIds.has(movie.id))
          .slice(0, contentCount);
        
        for (const movie of availableMovies) {
          movie.media_type = 'movie';
          curatedContent.push(movie);
          usedContentIds.add(movie.id);
        }
      } else if (filters.mediaType === 'tv') {
        // Sadece dizi ara
        const tvResponse = await tmdbService.discoverTVShows({
          ...baseParams,
          page: 1
        });
        
        const minContentCount = settings?.discoveryContentCount || 20;
        const availableShows = filterKidsContent(tvResponse.results)
          .filter(show => !usedContentIds.has(show.id) && !excludedContentIds.has(show.id))
          .slice(0, minContentCount);
        
        for (const show of availableShows) {
          show.media_type = 'tv';
          curatedContent.push(show);
          usedContentIds.add(show.id);
        }
      } else {
        // Hem film hem dizi ara
        const [movieResponse, tvResponse] = await Promise.all([
          tmdbService.discoverMovies({
            ...baseParams,
            page: 1
          }),
          tmdbService.discoverTVShows({
            ...baseParams,
            page: 1
          })
        ]);

        const halfCount = Math.ceil((settings?.discoveryContentCount || 20) / 2);
        const availableMovies = filterKidsContent(movieResponse.results)
          .filter(movie => !usedContentIds.has(movie.id) && !excludedContentIds.has(movie.id))
          .slice(0, halfCount);
        
        const availableShows = filterKidsContent(tvResponse.results)
          .filter(show => !usedContentIds.has(show.id) && !excludedContentIds.has(show.id))
          .slice(0, halfCount);

        for (const movie of availableMovies) {
          movie.media_type = 'movie';
          curatedContent.push(movie);
          usedContentIds.add(movie.id);
        }

        for (const show of availableShows) {
          show.media_type = 'tv';
          curatedContent.push(show);
          usedContentIds.add(show.id);
        }
      }

      // Eğer yeterli içerik yoksa, ek içerik ara
      const minContentCount = Math.ceil((settings?.discoveryContentCount || 20) * 0.75);
      if (curatedContent.length < minContentCount) {
        const additionalContent = await this.getAdditionalPopularContent(
          usedContentIds,
          excludedContentIds,
          minContentCount - curatedContent.length,
          filters,
          allowKidsContent
        );
        curatedContent.push(...additionalContent);
      }

      // Sıralama yönüne göre düzenle
      if (filters.sortOrder === 'asc') {
        curatedContent.reverse();
      }

      const finalContentCount = settings?.discoveryContentCount || 20;
      return curatedContent.slice(0, finalContentCount);

    } catch (error) {
      console.error('Error in getCuratedContentWithFilters:', error);
      // Hata durumunda varsayılan içerik döndür
      return this.getCuratedInitialContent(existingRatings, undefined, watchlistItems, settings);
    }
  }
}
