import type { Movie, TVShow, UserProfile, Recommendation, Genre, UserRating } from '../../content/types';
import { tmdbService } from '../../content/services/tmdb';
import { TMDbRecommendationService } from './tmdbRecommendationService';
import { ContentBasedFiltering } from '../../../shared/utils/recommendation/contentBased';
import { NeuralRecommendationService } from './neuralRecommendationService';
import { logger } from '../../../shared/utils/logger';

export class RecommendationService {
  static async generateRecommendations(
    profile: UserProfile | null, 
    movieGenres: Genre[], 
    tvGenres: Genre[],
    existingRatings: UserRating[] = [],
    filters?: {
      genres: number[];
      minYear: number;
      maxYear: number;
      minRating: number;
      maxRating: number;
      mediaType: 'all' | 'movie' | 'tv';
      sortBy: 'match_score' | 'rating' | 'year' | 'title';
      minMatchScore: number;
      languages?: string[]; // Yeni eklenen dil filtresi
      showKidsContent: boolean;
      showAnimationContent: boolean;
      showAnimeContent: boolean;
    },
    settings?: {
      recommendationCount?: number;
      minTmdbScore?: number;
    },
    watchlistIds?: number[]
  ): Promise<Recommendation[]> {
    // Null profile kontrolü
    if (!profile) {
      logger.warn('Profile is null, returning empty recommendations');
      return [];
    }

    // Profile'ın required fields kontrolü
    if (!profile.totalRatings || typeof profile.totalRatings !== 'number' || profile.totalRatings < 0) {
      logger.warn('Profile has invalid totalRatings, using default');
      profile.totalRatings = 0;
    }

    if (!profile.genreDistribution || typeof profile.genreDistribution !== 'object') {
      logger.warn('Profile has invalid genreDistribution, using default');
      profile.genreDistribution = {};
    }

    const recommendations: Recommendation[] = [];
    const allGenres = [...movieGenres, ...tvGenres];

    // PUANLANAN İÇERİKLERİ FİLTRELE
    const ratedContentIds = new Set(existingRatings.map(r => r.movieId));
    const watchlistContentIds = new Set(watchlistIds || []);
    logger.info(`Filtering out rated content: ${ratedContentIds.size} items, watchlist: ${watchlistContentIds.size} items`);

    // Initialize filteredTMDbRecommendations outside try block
    let filteredTMDbRecommendations: Recommendation[] = [];

    try {
      // 1. TMDb API tabanlı öneriler (Yeni!)
      logger.info('Generating TMDb-based recommendations...');
      const tmdbRecommendations = await TMDbRecommendationService.generateTMDbBasedRecommendations(
        profile,
        existingRatings,
        allGenres,
        filters
      );
      
      // TMDb önerilerini ekle (puanlanmış ve watchlistteki içerikleri filtrele)
      filteredTMDbRecommendations = tmdbRecommendations.filter(
        rec => !ratedContentIds.has(rec.movie.id) && !watchlistContentIds.has(rec.movie.id)
      );
      recommendations.push(...filteredTMDbRecommendations);
      
      logger.info(`Added ${filteredTMDbRecommendations.length} TMDb-based recommendations`);

      // 2. Neural Network tabanlı öneriler (Yeni!)
      if (profile.totalRatings >= 20) { // Minimum veri gereksinimi
        logger.info('Generating neural network-based recommendations...');
        try {
          const candidateContent = await this.getCandidateContentForNeural();
          const neuralRecommendations = await NeuralRecommendationService.generateNeuralRecommendations(
            candidateContent,
            profile,
            15,
            filters?.languages ? { languages: filters.languages } : undefined
          );
          
          const filteredNeuralRecommendations = neuralRecommendations.filter(
            rec => !ratedContentIds.has(rec.movie.id) && !watchlistContentIds.has(rec.movie.id)
          );
          recommendations.push(...filteredNeuralRecommendations);
          
          logger.info(`Added ${filteredNeuralRecommendations.length} neural network-based recommendations`);
        } catch (error) {
          logger.warn('Neural network recommendations failed, continuing with other methods:', error);
        }
      }

      // 3. Content-based filtering önerileri (Mevcut sistem)
      logger.info('Generating content-based recommendations...');
      await this.generateContentBasedRecommendations(
        recommendations, profile, allGenres, ratedContentIds
      );

      // 4. Tür-spesifik öneriler
      await this.generateGenreSpecificRecommendations(
        recommendations, profile, allGenres, ratedContentIds
      );

      // 5. Oyuncu bazlı öneriler
      await this.generateActorBasedRecommendations(
        recommendations, profile, ratedContentIds
      );

      // 6. Çeşitlilik önerileri
      await this.generateDiversityRecommendations(
        recommendations, profile, allGenres, ratedContentIds
      );

    } catch (error) {
      logger.error('Error in recommendation generation:', error);
    }

    // Final filtreleme ve sıralama
    let filteredRecommendations = recommendations
      .filter(rec => !ratedContentIds.has(rec.movie.id) && !watchlistContentIds.has(rec.movie.id))
      // Çocuk içeriklerini her zaman filtrele
      .filter(rec => {
        if (!rec?.movie?.genre_ids) return true;
        // 16: Animasyon, 10751: Aile
        const isKidsGenre = rec.movie.genre_ids.includes(16) || rec.movie.genre_ids.includes(10751);
        return !isKidsGenre;
      });

    // Filtreleri uygula
    if (filters) {
      // Minimum eşleşme oranı filtresi
      if (filters.minMatchScore > 0) {
        filteredRecommendations = filteredRecommendations.filter(rec => 
          rec && typeof rec.matchScore === 'number' && 
          rec.matchScore >= filters.minMatchScore
        );
      }

      // Genre filter
      if (Array.isArray(filters.genres) && filters.genres.length > 0) {
        filteredRecommendations = filteredRecommendations.filter(rec => {
          if (!rec?.movie?.genre_ids || !Array.isArray(rec.movie.genre_ids)) return false;
          return filters.genres.some(genreId => 
            rec.movie.genre_ids.includes(genreId)
          );
        });
      }

      // Media type filter
      if (filters.mediaType !== 'all') {
        filteredRecommendations = filteredRecommendations.filter(rec => {
          if (!rec?.movie) return false;
          if (filters.mediaType === 'movie') {
            return rec.movie.media_type === 'movie' || 'title' in rec.movie;
          } else {
            return rec.movie.media_type === 'tv' || 'name' in rec.movie;
          }
        });
      }

      // Year filter
      filteredRecommendations = filteredRecommendations.filter(rec => {
        if (!rec?.movie) return false;
        let year: number;
        if ('release_date' in rec.movie && rec.movie.release_date) {
          year = new Date(rec.movie.release_date).getFullYear();
        } else if ('first_air_date' in rec.movie && rec.movie.first_air_date) {
          year = new Date(rec.movie.first_air_date).getFullYear();
        } else {
          return true;
        }
        
        return year >= filters.minYear && year <= filters.maxYear;
      });

      // Rating filter
      const minRating = Math.max(filters.minRating, settings?.minTmdbScore ?? 0);
      filteredRecommendations = filteredRecommendations.filter(rec => {
        if (!rec?.movie || typeof rec.movie.vote_average !== 'number') return false;
        return rec.movie.vote_average >= minRating && 
               rec.movie.vote_average <= filters.maxRating;
      });

      // Language filter
      if (filters.languages && filters.languages.length > 0) {
        filteredRecommendations = filteredRecommendations.filter(rec => {
          if (!rec?.movie?.original_language) return false;
          return filters.languages!.includes(rec.movie.original_language);
        });
      }

      // Çizgi film (animasyon) filtresi
      if (filters.showAnimationContent === false) {
        filteredRecommendations = filteredRecommendations.filter(rec => {
          if (!rec?.movie?.genre_ids) return true;
          // 16: Animasyon
          return !rec.movie.genre_ids.includes(16);
        });
      }
      // Anime filtresi
      if (filters.showAnimeContent === false) {
        filteredRecommendations = filteredRecommendations.filter(rec => {
          if (!rec?.movie?.genre_ids) return true;
          // 16: Animasyon ve orijinal dil Japonca ise anime kabul etme
          const isAnime = rec.movie.genre_ids.includes(16) && rec.movie.original_language === 'ja';
          return !isAnime;
        });
      }

      // Sort with filters
      filteredRecommendations.sort((a, b) => {
        try {
          switch (filters.sortBy) {
            case 'match_score':
              return (b.matchScore || 0) - (a.matchScore || 0);
            case 'rating':
              return (b.movie?.vote_average || 0) - (a.movie?.vote_average || 0);
            case 'year':
              const yearA = 'release_date' in (a.movie || {}) ? 
                new Date((a.movie as any).release_date || '').getFullYear() : 
                new Date((a.movie as any).first_air_date || '').getFullYear();
              const yearB = 'release_date' in (b.movie || {}) ? 
                new Date((b.movie as any).release_date || '').getFullYear() : 
                new Date((b.movie as any).first_air_date || '').getFullYear();
              return (yearB || 0) - (yearA || 0);
            case 'title':
              const titleA = 'title' in (a.movie || {}) ? (a.movie as any).title : (a.movie as any).name;
              const titleB = 'title' in (b.movie || {}) ? (b.movie as any).title : (b.movie as any).name;
              return (titleA || '').localeCompare(titleB || '', 'tr');
            default:
              return 0;
          }
        } catch (error) {
          console.warn('Sort error:', error);
          return 0;
        }
      });
    } else {
      // Varsayılan sıralama
      filteredRecommendations.sort((a, b) => b.matchScore - a.matchScore);
    }

    // Use settings recommendation count or default to 25
    const recommendationCount = settings?.recommendationCount || 25;
    
    // Eğer filtreleme sonrası yeterli içerik yoksa, ek içerik yükle
    if (filteredRecommendations.length < recommendationCount) {
      logger.info(`Only ${filteredRecommendations.length} recommendations after filtering, need ${recommendationCount}. Loading additional content...`);
      
      try {
        // Ek içerik yükleme - daha geniş kriterlerle
        const additionalRecommendations = await this.generateAdditionalRecommendations(
          profile,
          allGenres,
          ratedContentIds,
          watchlistContentIds,
          recommendationCount - filteredRecommendations.length
        );
        
        // Yeni içerikleri mevcut listeye ekle (duplicate kontrolü ile)
        const existingIds = new Set(filteredRecommendations.map(r => r.movie.id));
        for (const rec of additionalRecommendations) {
          if (!existingIds.has(rec.movie.id) && !watchlistContentIds.has(rec.movie.id)) {
            filteredRecommendations.push(rec);
            existingIds.add(rec.movie.id);
          }
        }
        
        logger.info(`Added ${additionalRecommendations.length} additional recommendations`);
      } catch (error) {
        logger.warn('Error loading additional recommendations:', error);
      }
    }
    
    filteredRecommendations = filteredRecommendations.slice(0, recommendationCount);

    // Çeşitlilik için karıştır (top 10'u koru, geri kalanını karıştır)
    const topRecommendations = filteredRecommendations.slice(0, 10);
    const remainingRecommendations = filteredRecommendations.slice(10);
    const shuffledRemaining = remainingRecommendations.sort(() => Math.random() - 0.5);

    const finalRecommendations = [...topRecommendations, ...shuffledRemaining];

    logger.info('Final recommendations after filtering:', {
      total: recommendations.length,
      afterFiltering: finalRecommendations.length,
      tmdbBased: filteredTMDbRecommendations.length,
      contentBased: finalRecommendations.length - filteredTMDbRecommendations.length
    });

    return finalRecommendations;
  }

  // Content-based filtering önerileri
  private static async generateContentBasedRecommendations(
    recommendations: Recommendation[],
    profile: UserProfile,
    allGenres: Genre[],
    ratedContentIds: Set<number>
  ) {
    // Null kontrolü
    if (!profile || !profile.genreDistribution) {
      logger.warn('Profile or genreDistribution is null in generateContentBasedRecommendations');
      return;
    }

    try {
      // Kullanıcının tür kombinasyonu tercihlerini analiz et
      const userGenreCombinationsObj = ContentBasedFiltering.analyzeUserGenreCombinations(profile);
      const userGenreCombinations = Array.isArray(userGenreCombinationsObj) ? userGenreCombinationsObj : Object.values(userGenreCombinationsObj);

      // Get top genres
      const topGenres = Object.entries(profile.genreDistribution)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 8)
        .map(([genreId]) => genreId);

      // Film önerileri
      await this.generateEnhancedMovieRecommendations(
        recommendations, profile, topGenres, userGenreCombinations, 
        allGenres, ratedContentIds
      );
      
      // Dizi önerileri
      await this.generateEnhancedTVRecommendations(
        recommendations, profile, topGenres, userGenreCombinations, 
        allGenres, ratedContentIds
      );
    } catch (error) {
      logger.error('Error in generateContentBasedRecommendations:', error);
    }
  }

  // Tür-spesifik öneriler
  private static async generateGenreSpecificRecommendations(
    recommendations: Recommendation[],
    profile: UserProfile,
    allGenres: Genre[],
    ratedContentIds: Set<number>
  ) {
    // Null kontrolü
    if (!profile || !profile.genreDistribution) {
      logger.warn('Profile or genreDistribution is null in generateGenreSpecificRecommendations');
      return;
    }

    try {
      const topUserGenres = Object.entries(profile.genreDistribution)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5)
        .map(([genreId, score]) => ({ id: parseInt(genreId), score }));

      for (const userGenre of topUserGenres) {
        if (userGenre.score > 10) {
          await this.generateSpecificGenreRecommendations(
            recommendations, profile, userGenre.id, allGenres, ratedContentIds
          );
        }
      }
    } catch (error) {
      logger.error('Error in generateGenreSpecificRecommendations:', error);
    }
  }

  private static async generateSpecificGenreRecommendations(
    recommendations: Recommendation[],
    profile: UserProfile,
    genreId: number,
    allGenres: Genre[],
    ratedContentIds: Set<number>
  ) {
    const genre = allGenres.find(g => g.id === genreId);
    const genreName = genre?.name || 'Bilinmeyen';

    try {
      // Film önerileri
      const movieConfigs = [
        {
          with_genres: genreId.toString(),
          'vote_average.gte': 6.0,
          'vote_count.gte': 100,
          sort_by: 'vote_average.desc',
          page: Math.floor(Math.random() * 3) + 1,
          type: `premium_${genreName.toLowerCase()}`
        },
        {
          with_genres: genreId.toString(),
          'vote_average.gte': 6.0,
          'vote_count.gte': 100,
          sort_by: 'popularity.desc',
          page: Math.floor(Math.random() * 5) + 1,
          type: `popular_${genreName.toLowerCase()}`
        }
      ];

      // Dizi önerileri
      const tvConfigs = [
        {
          with_genres: genreId.toString(),
          'vote_average.gte': 6.0,
          'vote_count.gte': 100,
          sort_by: 'vote_average.desc',
          page: Math.floor(Math.random() * 3) + 1,
          type: `premium_tv_${genreName.toLowerCase()}`
        },
        {
          with_genres: genreId.toString(),
          'vote_average.gte': 6.0,
          'vote_count.gte': 100,
          sort_by: 'popularity.desc',
          page: Math.floor(Math.random() * 5) + 1,
          type: `popular_tv_${genreName.toLowerCase()}`
        }
      ];

      // Film önerilerini işle
      for (const config of movieConfigs) {
        try {
          const response = await tmdbService.discoverMovies(config);
          for (const movie of response.results.slice(0, 4)) {
            // PUANLANAN İÇERİKLERİ ATLA
            if (ratedContentIds.has(movie.id)) continue;

            const matchScore = this.calculateGenreSpecificScore(movie, profile, genreId, config.type);
            
            if (matchScore >= 50 && !recommendations.find(r => r.movie.id === movie.id)) {
              movie.media_type = 'movie';
              recommendations.push({
                movie,
                matchScore,
                reasons: this.generateGenreSpecificReasons(movie, profile, genreName, config.type),
                confidence: matchScore / 100,
                novelty: this.calculateNovelty(movie, profile),
                diversity: this.calculateDiversity(movie, profile),
                explanation: {
                  primaryFactors: [`${genreName} türünde yüksek puan vermişsin`],
                  secondaryFactors: [`Kalite: ${movie.vote_average.toFixed(1)}/10`],
                  riskFactors: []
                },
                recommendationType: matchScore >= 80 ? 'safe' : 'exploratory'
              });
            }
          }
        } catch (error) {
          logger.warn(`Error fetching ${config.type} movies:`, error);
        }
      }

      // Dizi önerilerini işle
      for (const config of tvConfigs) {
        try {
          const response = await tmdbService.discoverTVShows(config);
          for (const tvShow of response.results.slice(0, 4)) {
            // PUANLANAN İÇERİKLERİ ATLA
            if (ratedContentIds.has(tvShow.id)) continue;

            const matchScore = this.calculateGenreSpecificScore(tvShow, profile, genreId, config.type);
            
            if (matchScore >= 50 && !recommendations.find(r => r.movie.id === tvShow.id)) {
              tvShow.media_type = 'tv';
              recommendations.push({
                movie: tvShow,
                matchScore,
                reasons: this.generateGenreSpecificReasons(tvShow, profile, genreName, config.type),
                confidence: matchScore / 100,
                novelty: this.calculateNovelty(tvShow, profile),
                diversity: this.calculateDiversity(tvShow, profile),
                explanation: {
                  primaryFactors: [`${genreName} türünde yüksek puan vermişsin`],
                  secondaryFactors: [`Kalite: ${tvShow.vote_average.toFixed(1)}/10`],
                  riskFactors: []
                },
                recommendationType: matchScore >= 80 ? 'safe' : 'exploratory'
              });
            }
          }
        } catch (error) {
          logger.warn(`Error fetching ${config.type} TV shows:`, error);
        }
      }

    } catch (error) {
      logger.error(`Error generating ${genreName} specific recommendations:`, error);
    }
  }

  // Çeşitlilik önerileri
  private static async generateDiversityRecommendations(
    recommendations: Recommendation[],
    profile: UserProfile,
    allGenres: Genre[],
    ratedContentIds: Set<number>
  ) {
    // Null kontrolü
    if (!profile || !profile.genreDistribution) {
      logger.warn('Profile or genreDistribution is null in generateDiversityRecommendations');
      return;
    }

    try {
      const allGenreIds = allGenres.map(g => g.id);
      const userGenreIds = Object.keys(profile.genreDistribution).map(id => parseInt(id));
      const unexploredGenres = allGenreIds.filter(id => !userGenreIds.includes(id));

    const popularGenres = [28, 35, 18, 878, 27, 10749, 53, 16];
    const genresToExplore = unexploredGenres.filter(id => popularGenres.includes(id)).slice(0, 3);

    for (const genreId of genresToExplore) {
      try {
        const movieResponse = await tmdbService.discoverMovies({
          with_genres: genreId.toString(),
          'vote_average.gte': 6.0,
          'vote_count.gte': 100,
          sort_by: 'vote_average.desc',
          page: 1
        });

        for (const movie of movieResponse.results.slice(0, 2)) {
          // PUANLANAN İÇERİKLERİ ATLA
          if (ratedContentIds.has(movie.id)) continue;

          if (!recommendations.find(r => r.movie.id === movie.id)) {
            movie.media_type = 'movie';
            const genre = allGenres.find(g => g.id === genreId);
            recommendations.push({
              movie,
              matchScore: 65,
              reasons: [`Yeni tür keşfi: ${genre?.name}`, 'Kaliteli yapım', 'Çeşitlilik önerisi'],
              confidence: 0.65,
              novelty: 0.8,
              diversity: 0.9,
              explanation: {
                primaryFactors: [`Yeni tür keşfi: ${genre?.name}`],
                secondaryFactors: ['Kaliteli yapım'],
                riskFactors: ['Alışık olmadığın bir tür olabilir']
              },
              recommendationType: 'serendipitous'
            });
          }
        }

        const tvResponse = await tmdbService.discoverTVShows({
          with_genres: genreId.toString(),
          'vote_average.gte': 6.0,
          'vote_count.gte': 100,
          sort_by: 'vote_average.desc',
          page: 1
        });

        for (const tvShow of tvResponse.results.slice(0, 2)) {
          // PUANLANAN İÇERİKLERİ ATLA
          if (ratedContentIds.has(tvShow.id)) continue;

          if (!recommendations.find(r => r.movie.id === tvShow.id)) {
            tvShow.media_type = 'tv';
            const genre = allGenres.find(g => g.id === genreId);
            recommendations.push({
              movie: tvShow,
              matchScore: 65,
              reasons: [`Yeni tür keşfi: ${genre?.name}`, 'Kaliteli dizi', 'Çeşitlilik önerisi'],
              confidence: 0.65,
              novelty: 0.8,
              diversity: 0.9,
              explanation: {
                primaryFactors: [`Yeni tür keşfi: ${genre?.name}`],
                secondaryFactors: ['Kaliteli dizi'],
                riskFactors: ['Alışık olmadığın bir tür olabilir']
              },
              recommendationType: 'serendipitous'
            });
          }
        }
      } catch (error) {
        logger.warn(`Error generating diversity recommendations for genre ${genreId}:`, error);
      }
    }
    } catch (error) {
      logger.error('Error in generateDiversityRecommendations:', error);
    }
  }

  private static calculateGenreSpecificScore(
    content: Movie | TVShow,
    profile: UserProfile,
    targetGenreId: number,
    type: string
  ): number {
    let score = 0;

    const contentGenres = content.genre_ids || [];
    if (contentGenres.includes(targetGenreId)) {
      const userPreference = profile.genreDistribution[targetGenreId] || 0;
      score += (userPreference / 100) * 50;
    }

    const qualityScore = Math.min(100, (content.vote_average / 10) * 100);
    score += qualityScore * 0.3;

    const popularityScore = Math.min(100, (content.vote_count / 1000) * 10);
    score += popularityScore * 0.15;

    if (contentGenres.length >= 3) {
      score += 5;
    }

    if (type.includes('premium')) score *= 1.2;
    else if (type.includes('recent')) score *= 1.1;

    if (content.vote_average >= 8.5) score *= 1.15;

    return Math.min(100, Math.max(0, score));
  }

  private static generateGenreSpecificReasons(
    content: Movie | TVShow,
    _profile: UserProfile,
    genreName: string,
    type: string
  ): string[] {
    const reasons: string[] = [];

    reasons.push(`${genreName} türünde yüksek puan vermişsin`);

    if (content.vote_average >= 8.0) {
      reasons.push(`Üstün kalite (${content.vote_average.toFixed(1)}/10)`);
    } else if (content.vote_average >= 7.0) {
      reasons.push(`Yüksek kalite (${content.vote_average.toFixed(1)}/10)`);
    }

    if (type.includes('premium')) {
      reasons.push('Premium kalite seçimi');
    } else if (type.includes('popular')) {
      reasons.push('Çok beğenilen yapım');
    }

    if (content.vote_count > 5000) {
      reasons.push('Geniş kitle tarafından izlendi');
    }

    return reasons.slice(0, 3);
  }

  private static calculateNovelty(content: Movie | TVShow, profile: UserProfile): number {
    let novelty = 0.5;

    // Yıl faktörü
    const releaseDate = 'release_date' in content ? content.release_date : content.first_air_date;
    if (releaseDate) {
      const year = new Date(releaseDate).getFullYear();
      const currentYear = new Date().getFullYear();
      const yearDiff = currentYear - year;
      
      if (yearDiff <= 2) novelty += 0.3;
      else if (yearDiff <= 5) novelty += 0.2;
      else if (yearDiff >= 20) novelty += 0.1;
    }

    // Tür yeniliği
    const contentGenres = content.genre_ids || [];
    const userGenres = Object.keys(profile.genreDistribution).map(id => parseInt(id));
    const newGenres = contentGenres.filter(id => !userGenres.includes(id));
    
    if (newGenres.length > 0) {
      novelty += 0.2;
    }

    return Math.max(0, Math.min(1, novelty));
  }

  private static calculateDiversity(content: Movie | TVShow, profile: UserProfile): number {
    let diversity = 0.5;

    // Genre diversity
    const contentGenres = content.genre_ids || [];
    const userTopGenres = Object.entries(profile.genreDistribution)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3)
      .map(([genreId]) => parseInt(genreId));

    const diverseGenres = contentGenres.filter(id => !userTopGenres.includes(id));
    if (diverseGenres.length > 0) {
      diversity += 0.3;
    }

    // Rating diversity
    const expectedRating = profile.averageScore * 2;
    const ratingDiff = Math.abs(content.vote_average - expectedRating);
    if (ratingDiff > 1) {
      diversity += 0.2;
    }

    return Math.max(0, Math.min(1, diversity));
  }

  private static async generateActorBasedRecommendations(
    recommendations: Recommendation[],
    profile: UserProfile,
    ratedContentIds: Set<number>
  ) {
    const tmdbApiKey = import.meta.env.VITE_TMDB_API_KEY;
    const tmdbReadAccessToken = import.meta.env.VITE_TMDB_READ_ACCESS_TOKEN;
    const hasValidApiKey = Boolean(tmdbApiKey && tmdbApiKey !== 'your_tmdb_api_key_here');
    const tmdbHeaders: HeadersInit = tmdbReadAccessToken
      ? {
          'Accept': 'application/json',
          'Authorization': `Bearer ${tmdbReadAccessToken}`
        }
      : {
          'Accept': 'application/json'
        };
    const topActors = Object.values(profile.favoriteActors)
      .sort((a, b) => b.count - a.count)
      .slice(0, 3)
      .map(actor => actor.name);

    for (const actorName of topActors) {
      try {
        const searchUrl = new URL('https://api.themoviedb.org/3/search/person');
        searchUrl.searchParams.set('language', 'tr-TR');
        searchUrl.searchParams.set('query', actorName);
        if (hasValidApiKey) {
          searchUrl.searchParams.set('api_key', tmdbApiKey);
        }

        const searchResponse = await fetch(searchUrl.toString(), {
          headers: tmdbHeaders
        });
        const searchData = await searchResponse.json();
        
        if (searchData.results && searchData.results.length > 0) {
          const actor = searchData.results[0];
          
          const creditsUrl = new URL(`https://api.themoviedb.org/3/person/${actor.id}/movie_credits`);
          creditsUrl.searchParams.set('language', 'tr-TR');
          if (hasValidApiKey) {
            creditsUrl.searchParams.set('api_key', tmdbApiKey);
          }

          const creditsResponse = await fetch(creditsUrl.toString(), {
            headers: tmdbHeaders
          });
          const creditsData = await creditsResponse.json();
          
          if (creditsData.cast) {
            const topMovies = creditsData.cast
              .filter((movie: any) => movie.vote_average >= 6.0 && movie.vote_count >= 100)
              .sort((a: any, b: any) => b.vote_average - a.vote_average)
              .slice(0, 4);
            
            for (const movie of topMovies) {
              // PUANLANAN İÇERİKLERİ ATLA
              if (ratedContentIds.has(movie.id)) continue;

              if (!recommendations.find(r => r.movie.id === movie.id)) {
                const matchScore = this.calculateGenreSpecificScore(movie, profile, 0, 'actor_based') + 15;
                
                if (matchScore >= 55) {
                  movie.media_type = 'movie';
                  recommendations.push({
                    movie,
                    matchScore,
                    reasons: [`Favori oyuncun ${actorName} bu filmde oynuyor`, `Kalite: ${movie.vote_average.toFixed(1)}/10`],
                    confidence: matchScore / 100,
                    novelty: this.calculateNovelty(movie, profile),
                    diversity: this.calculateDiversity(movie, profile),
                    explanation: {
                      primaryFactors: [`Favori oyuncun ${actorName} bu filmde oynuyor`],
                      secondaryFactors: [`Kalite: ${movie.vote_average.toFixed(1)}/10`],
                      riskFactors: []
                    },
                    recommendationType: 'safe'
                  });
                }
              }
            }
          }
        }
      } catch (error) {
        logger.warn(`Error getting recommendations for actor ${actorName}:`, error);
      }
    }
  }

  private static async generateEnhancedMovieRecommendations(
    recommendations: Recommendation[], 
    profile: UserProfile, 
    topGenres: string[], 
    userGenreCombinations: any[],
    allGenres: Genre[],
    ratedContentIds: Set<number>
  ) {
    const discoveryConfigs = [
      {
        with_genres: topGenres.slice(0, 3).join(','),
        'vote_average.gte': Math.max(4.5, profile.averageScore - 2),
        sort_by: 'vote_average.desc',
        page: Math.floor(Math.random() * 3) + 1,
        type: 'single_genre'
      },
      ...userGenreCombinations.slice(0, 4).filter(combo => combo.genres.length === 2).map(combo => ({
        with_genres: combo.genres.join(','),
        'vote_average.gte': Math.max(4.0, profile.averageScore - 2.5),
        sort_by: 'popularity.desc',
        page: Math.floor(Math.random() * 5) + 1,
        type: 'dual_combination',
        combinationName: combo.name
      })),
      ...userGenreCombinations.slice(0, 3).filter(combo => combo.genres.length === 3).map(combo => ({
        with_genres: combo.genres.join(','),
        'vote_average.gte': Math.max(5.5, profile.averageScore - 1.5),
        sort_by: 'vote_average.desc',
        page: Math.floor(Math.random() * 3) + 1,
        type: 'triple_combination',
        combinationName: combo.name
      }))
    ];

    for (const config of discoveryConfigs) {
      try {
        const response = await tmdbService.discoverMovies(config);
        
        for (const movie of response.results.slice(0, 8)) {
          // PUANLANAN İÇERİKLERİ ATLA
          if (ratedContentIds.has(movie.id)) continue;

          const matchScore = this.calculateEnhancedMovieMatchScore(movie, profile, config.type);
          
          if (matchScore >= 45 && !recommendations.find(r => r.movie.id === movie.id)) {
            const reasons = this.generateEnhancedMovieReasons(movie, profile, allGenres);
            movie.media_type = 'movie';
            recommendations.push({
              movie,
              matchScore,
              reasons,
              confidence: matchScore / 100,
              novelty: this.calculateNovelty(movie, profile),
              diversity: this.calculateDiversity(movie, profile),
              explanation: {
                primaryFactors: reasons.slice(0, 2),
                secondaryFactors: reasons.slice(2),
                riskFactors: []
              },
              recommendationType: matchScore >= 80 ? 'safe' : 'exploratory'
            });
          }
        }
      } catch (error) {
        logger.warn('Error generating enhanced movie recommendations:', error);
      }
    }
  }

  private static async generateEnhancedTVRecommendations(
    recommendations: Recommendation[], 
    profile: UserProfile, 
    topGenres: string[], 
    userGenreCombinations: any[],
    allGenres: Genre[],
    ratedContentIds: Set<number>
  ) {
    const discoveryConfigs = [
      {
        with_genres: topGenres.slice(0, 3).join(','),
        'vote_average.gte': Math.max(4.0, profile.averageScore - 2.5),
        sort_by: 'vote_average.desc',
        page: Math.floor(Math.random() * 5) + 1,
        type: 'single_genre'
      },
      ...userGenreCombinations.slice(0, 5).filter(combo => combo.genres.length === 2).map(combo => ({
        with_genres: combo.genres.join(','),
        'vote_average.gte': Math.max(3.5, profile.averageScore - 3),
        sort_by: 'popularity.desc',
        page: Math.floor(Math.random() * 8) + 1,
        type: 'dual_combination',
        combinationName: combo.name
      })),
      ...topGenres.slice(0, 5).map(genreId => ({
        with_genres: genreId,
        'vote_average.gte': Math.max(4.0, profile.averageScore - 2.5),
        sort_by: Math.random() > 0.5 ? 'vote_average.desc' : 'popularity.desc',
        page: Math.floor(Math.random() * 6) + 1,
        type: 'pure_genre'
      }))
    ];

    for (const config of discoveryConfigs) {
      try {
        const response = await tmdbService.discoverTVShows(config);
        
        for (const tvShow of response.results.slice(0, 12)) {
          // PUANLANAN İÇERİKLERİ ATLA
          if (ratedContentIds.has(tvShow.id)) continue;

          const matchScore = this.calculateEnhancedTVMatchScore(tvShow, profile, config.type);
          
          if (matchScore >= 40 && !recommendations.find(r => r.movie.id === tvShow.id)) {
            const reasons = this.generateEnhancedTVReasons(tvShow, profile, allGenres);
            tvShow.media_type = 'tv';
            recommendations.push({
              movie: tvShow,
              matchScore,
              reasons,
              confidence: matchScore / 100,
              novelty: this.calculateNovelty(tvShow, profile),
              diversity: this.calculateDiversity(tvShow, profile),
              explanation: {
                primaryFactors: reasons.slice(0, 2),
                secondaryFactors: reasons.slice(2),
                riskFactors: []
              },
              recommendationType: matchScore >= 80 ? 'safe' : 'exploratory'
            });
          }
        }
      } catch (error) {
        logger.warn('Error generating enhanced TV recommendations:', error);
      }
    }
  }

  private static calculateEnhancedMovieMatchScore(movie: Movie, profile: UserProfile, type: string): number {
    let score = 0;
    const safeProfile = profile || {};
    const movieGenres = movie.genre_ids || [];
    const { score: combinationScore } = ContentBasedFiltering.calculateGenreCombinationScore(safeProfile, movieGenres);
    score += combinationScore * 35;
    let genreScore = 0;
    const safeGenreDistribution = (safeProfile.genreDistribution && typeof safeProfile.genreDistribution === 'object') ? safeProfile.genreDistribution : {};
    movieGenres.forEach(genreId => {
      if (safeGenreDistribution[genreId]) {
        genreScore += safeGenreDistribution[genreId];
      }
    });
    score += (genreScore / Math.max(1, movieGenres.length)) * 0.3;

    const qualityScore = Math.min(100, (movie.vote_average / 10) * 100);
    score += qualityScore * 0.2;

    const popularityScore = Math.min(100, (movie.vote_count / 1000) * 10);
    score += popularityScore * 0.1;

    if (movie.release_date) {
      const movieYear = new Date(movie.release_date).getFullYear();
      const movieDecade = `${Math.floor(movieYear / 10) * 10}s`;
      const periodScore = (safeProfile.periodPreference && typeof safeProfile.periodPreference === 'object' && safeProfile.periodPreference[movieDecade]) ? safeProfile.periodPreference[movieDecade] : 0;
      score += periodScore * 0.05;
    }

    if (type === 'triple_combination') score *= 1.25;
    else if (type === 'dual_combination') score *= 1.15;
    else if (type === 'pure_genre') score *= 1.1;

    // Demografik faktörleri uygula
    score = this.applyDemographicFactors(score, movie, safeProfile);

    return Math.min(100, Math.max(0, score));
  }

  private static calculateEnhancedTVMatchScore(tvShow: TVShow, profile: UserProfile, type: string): number {
    let score = 0;
    const safeProfile = profile || {};
    const tvGenres = tvShow.genre_ids || [];
    const { score: combinationScore } = ContentBasedFiltering.calculateGenreCombinationScore(safeProfile, tvGenres);
    score += combinationScore * 35;
    let genreScore = 0;
    const safeGenreDistribution = (safeProfile.genreDistribution && typeof safeProfile.genreDistribution === 'object') ? safeProfile.genreDistribution : {};
    tvGenres.forEach(genreId => {
      if (safeGenreDistribution[genreId]) {
        genreScore += safeGenreDistribution[genreId];
      }
    });
    score += (genreScore / Math.max(1, tvGenres.length)) * 0.3;

    const qualityScore = Math.min(100, (tvShow.vote_average / 10) * 100);
    score += qualityScore * 0.2;

    const popularityScore = Math.min(100, (tvShow.vote_count / 500) * 10);
    score += popularityScore * 0.1;

    if (tvShow.first_air_date) {
      const tvYear = new Date(tvShow.first_air_date).getFullYear();
      const tvDecade = `${Math.floor(tvYear / 10) * 10}s`;
      const periodScore = (safeProfile.periodPreference && typeof safeProfile.periodPreference === 'object' && safeProfile.periodPreference[tvDecade]) ? safeProfile.periodPreference[tvDecade] : 0;
      score += periodScore * 0.05;
    }

    if (tvGenres.length >= 3) {
      score += 10;
    }

    if (type === 'triple_combination') score *= 1.25;
    else if (type === 'dual_combination') score *= 1.15;
    else if (type === 'pure_genre') score *= 1.1;

    if (tvShow.vote_average >= 8.5) score *= 1.2;

    // Demografik faktörleri uygula
    score = this.applyDemographicFactors(score, tvShow, safeProfile);

    return Math.min(100, Math.max(0, score));
  }

  // Get candidate content for neural network training
  private static async getCandidateContentForNeural(): Promise<(Movie | TVShow)[]> {
    try {
      // Fetch popular movies and TV shows for neural network recommendations
      const [movieResponse, tvResponse] = await Promise.all([
        tmdbService.getPopularMovies(1),
        tmdbService.getPopularTVShows(1)
      ]);

      const movies = movieResponse.results.map(movie => ({ ...movie, media_type: 'movie' }));
      const tvShows = tvResponse.results.map(tv => ({ ...tv, media_type: 'tv' }));

      return [...movies, ...tvShows] as (Movie | TVShow)[];
    } catch (error) {
      logger.warn('Error fetching candidate content for neural network:', error);
      return [];
    }
  }

  private static generateEnhancedMovieReasons(movie: Movie, profile: UserProfile, genres: Genre[]): string[] {
    const reasons: string[] = [];

    // Null kontrolü
    if (!profile || !profile.genreDistribution) {
      reasons.push('Kaliteli yapım');
      return reasons;
    }

    const movieGenres = movie.genre_ids || [];
    
    try {
      const { matchedCombinations } = ContentBasedFiltering.calculateGenreCombinationScore(profile, movieGenres);
      
      if (matchedCombinations && matchedCombinations.length > 0) {
        reasons.push(`Sevdiğin kombinasyonlar: ${matchedCombinations[0]}`);
      }

      if (!matchedCombinations || matchedCombinations.length === 0) {
        const matchingGenres = movieGenres.filter(genreId => 
          profile.genreDistribution && profile.genreDistribution[genreId] && profile.genreDistribution[genreId] > 8
        );
        if (matchingGenres.length > 0) {
          const genreNames = matchingGenres.map(genreId => {
            const genre = genres.find(g => g.id === genreId);
            return genre?.name || 'Bilinmeyen';
          });
          reasons.push(`${genreNames.join(' ve ')} türlerinde yüksek puan vermişsin`);
        }
      }
    } catch (error) {
      logger.warn('Error in generateEnhancedMovieReasons:', error);
      reasons.push('Kaliteli yapım');
    }

    if (movie.vote_average >= 8.0) {
      reasons.push(`Üstün kalite (${movie.vote_average.toFixed(1)}/10)`);
    } else if (movie.vote_average >= 7.0) {
      reasons.push(`Yüksek kalite (${movie.vote_average.toFixed(1)}/10)`);
    }

    if (movie.vote_count > 5000) {
      reasons.push('Çok izlenen ve beğenilen yapım');
    }

    // Demografik faktörlere dayalı nedenler
    if (profile.demographics) {
      const demographics = profile.demographics;
      
      // Yaş bazlı nedenler
      if (demographics.age && movie.release_date) {
        const movieYear = new Date(movie.release_date).getFullYear();
        const movieAge = new Date().getFullYear() - movieYear;
        
        if (demographics.age < 25 && movieAge <= 5) {
          reasons.push('Yaş grubuna uygun güncel yapım');
        } else if (demographics.age >= 40 && movieAge >= 20) {
          reasons.push('Nostaljik klasik, yaş grubuna uygun');
        }
      }
      
      // Cinsiyet bazlı nedenler
      if (demographics.gender) {
        if (demographics.gender === 'female' && movieGenres.includes(10749)) {
          reasons.push('Cinsiyet tercihlerine uygun romantik içerik');
        } else if (demographics.gender === 'male' && movieGenres.includes(28)) {
          reasons.push('Cinsiyet tercihlerine uygun aksiyon içerik');
        }
      }
      
      // Eğitim seviyesi bazlı nedenler
      if (demographics.education && movie.vote_average >= 7.5) {
        if (demographics.education === 'university' || demographics.education === 'graduate') {
          reasons.push('Eğitim seviyene uygun kaliteli yapım');
        }
      }
      
      // İlişki durumu bazlı nedenler
      if (demographics.relationshipStatus) {
        if ((demographics.relationshipStatus === 'in_relationship' || demographics.relationshipStatus === 'married') && movieGenres.includes(10751)) {
          reasons.push('İlişki durumuna uygun aile dostu içerik');
        } else if (demographics.relationshipStatus === 'single' && movieGenres.includes(10749)) {
          reasons.push('Bekar yaşam tarzına uygun romantik içerik');
        }
      }
      
      // Çocuk durumu bazlı nedenler
      if (demographics.hasChildren && movieGenres.includes(10751)) {
        reasons.push('Çocuklu aileye uygun içerik');
      }
    }

    return reasons.slice(0, 3);
  }

  private static generateEnhancedTVReasons(tvShow: TVShow, profile: UserProfile, genres: Genre[]): string[] {
    const reasons: string[] = [];

    // Null kontrolü
    if (!profile || !profile.genreDistribution) {
      reasons.push('Kaliteli dizi');
      return reasons;
    }

    const tvGenres = tvShow.genre_ids || [];
    
    try {
      const { matchedCombinations } = ContentBasedFiltering.calculateGenreCombinationScore(profile, tvGenres);
      
      if (matchedCombinations && matchedCombinations.length > 0) {
        reasons.push(`Sevdiğin kombinasyonlar: ${matchedCombinations[0]}`);
      }

      if (!matchedCombinations || matchedCombinations.length === 0) {
        const matchingGenres = tvGenres.filter(genreId => 
          profile.genreDistribution && profile.genreDistribution[genreId] && profile.genreDistribution[genreId] > 8
        );
        if (matchingGenres.length > 0) {
          const genreNames = matchingGenres.map(genreId => {
            const genre = genres.find(g => g.id === genreId);
            return genre?.name || 'Bilinmeyen';
          });
          reasons.push(`${genreNames.join(' ve ')} türlerinde yüksek puan vermişsin`);
        }
      }
    } catch (error) {
      logger.warn('Error in generateEnhancedTVReasons:', error);
      reasons.push('Kaliteli dizi');
    }

    if (tvShow.vote_average >= 8.0) {
      reasons.push(`Üstün kalite (${tvShow.vote_average.toFixed(1)}/10)`);
    } else if (tvShow.vote_average >= 7.0) {
      reasons.push(`Yüksek kalite (${tvShow.vote_average.toFixed(1)}/10)`);
    }

    if (tvGenres.length >= 3) {
      const genreNames = tvGenres.map(genreId => {
        const genre = genres.find(g => g.id === genreId);
        return genre?.name;
      }).filter(Boolean);
      
      if (genreNames.length >= 3) {
        reasons.push(`Zengin tür karışımı: ${genreNames.slice(0, 3).join(', ')}`);
      }
    }

    if (tvShow.number_of_seasons) {
      if (tvShow.number_of_seasons === 1) {
        reasons.push('Tek sezonluk tamamlanmış hikaye');
      } else if (tvShow.number_of_seasons <= 3) {
        reasons.push('Kısa ve yoğun hikaye anlatımı');
      } else {
        reasons.push('Uzun soluklu epik hikaye');
      }
    }

    // Demografik faktörlere dayalı nedenler
    if (profile.demographics) {
      const demographics = profile.demographics;
      
      // Yaş bazlı nedenler
      if (demographics.age && tvShow.first_air_date) {
        const tvYear = new Date(tvShow.first_air_date).getFullYear();
        const tvAge = new Date().getFullYear() - tvYear;
        
        if (demographics.age < 25 && tvAge <= 5) {
          reasons.push('Yaş grubuna uygun güncel dizi');
        } else if (demographics.age >= 40 && tvAge >= 15) {
          reasons.push('Nostaljik klasik dizi, yaş grubuna uygun');
        }
      }
      
      // Cinsiyet bazlı nedenler
      if (demographics.gender) {
        if (demographics.gender === 'female' && tvGenres.includes(10749)) {
          reasons.push('Cinsiyet tercihlerine uygun romantik dizi');
        } else if (demographics.gender === 'male' && tvGenres.includes(28)) {
          reasons.push('Cinsiyet tercihlerine uygun aksiyon dizi');
        }
      }
      
      // Eğitim seviyesi bazlı nedenler
      if (demographics.education && tvShow.vote_average >= 7.5) {
        if (demographics.education === 'university' || demographics.education === 'graduate') {
          reasons.push('Eğitim seviyene uygun kaliteli dizi');
        }
      }
      
      // İlişki durumu bazlı nedenler
      if (demographics.relationshipStatus) {
        if ((demographics.relationshipStatus === 'in_relationship' || demographics.relationshipStatus === 'married') && tvGenres.includes(10751)) {
          reasons.push('İlişki durumuna uygun aile dostu dizi');
        } else if (demographics.relationshipStatus === 'single' && tvGenres.includes(10749)) {
          reasons.push('Bekar yaşam tarzına uygun romantik dizi');
        }
      }
      
      // Çocuk durumu bazlı nedenler
      if (demographics.hasChildren && tvGenres.includes(10751)) {
        reasons.push('Çocuklu aileye uygun dizi');
      }
    }

    return reasons.slice(0, 3);
  }

  // Demografik faktörleri hesaplama fonksiyonları
  private static calculateDemographicScore(content: Movie | TVShow, profile: UserProfile): number {
    if (!profile.demographics) return 0;

    let score = 0;
    const demographics = profile.demographics;

    // Yaş bazlı skorlama
    if (demographics.age) {
      score += this.calculateAgeBasedScore(content, demographics.age);
    }

    // Cinsiyet bazlı skorlama
    if (demographics.gender) {
      score += this.calculateGenderBasedScore(content, demographics.gender);
    }

    // Ülke bazlı skorlama
    if (demographics.country) {
      score += this.calculateCountryBasedScore(content, demographics.country);
    }

    // Dil bazlı skorlama
    if (demographics.language) {
      score += this.calculateLanguageBasedScore(content, demographics.language);
    }

    // Eğitim seviyesi bazlı skorlama
    if (demographics.education) {
      score += this.calculateEducationBasedScore(content, demographics.education);
    }

    // İlişki durumu bazlı skorlama
    if (demographics.relationshipStatus) {
      score += this.calculateRelationshipBasedScore(content, demographics.relationshipStatus);
    }

    // Çocuk durumu bazlı skorlama
    if (demographics.hasChildren !== undefined) {
      score += this.calculateChildrenBasedScore(content, demographics.hasChildren, demographics.childrenAge);
    }

    return Math.min(100, Math.max(0, score));
  }

  private static calculateAgeBasedScore(content: Movie | TVShow, userAge: number): number {
    let score = 0;
    
    // İçeriğin yılını al
    let contentYear = 0;
    if ('release_date' in content && content.release_date) {
      contentYear = new Date(content.release_date).getFullYear();
    } else if ('first_air_date' in content && content.first_air_date) {
      contentYear = new Date(content.first_air_date).getFullYear();
    }

    if (contentYear > 0) {
      const contentAge = new Date().getFullYear() - contentYear;
      
      // Yaş gruplarına göre tercihler
      if (userAge < 25) {
        // Genç kullanıcılar daha yeni içerikleri tercih eder
        if (contentAge <= 5) score += 15;
        else if (contentAge <= 10) score += 10;
        else if (contentAge <= 20) score += 5;
      } else if (userAge < 40) {
        // Orta yaş kullanıcılar dengeli tercih
        if (contentAge <= 10) score += 12;
        else if (contentAge <= 20) score += 10;
        else if (contentAge <= 30) score += 8;
      } else if (userAge < 60) {
        // Yetişkin kullanıcılar klasikleri de sever
        if (contentAge <= 15) score += 10;
        else if (contentAge <= 25) score += 12;
        else if (contentAge <= 35) score += 8;
      } else {
        // Yaşlı kullanıcılar nostaljik içerikleri tercih eder
        if (contentAge >= 20) score += 15;
        else if (contentAge >= 10) score += 10;
        else score += 5;
      }
    }

    return score;
  }

  private static calculateGenderBasedScore(content: Movie | TVShow, gender: string): number {
    let score = 0;
    
    // Cinsiyet bazlı tür tercihleri (genel istatistiklere dayalı)
    const contentGenres = content.genre_ids || [];
    
    if (gender === 'female') {
      // Kadın kullanıcılar genellikle romantik, dram, komedi türlerini tercih eder
      if (contentGenres.includes(10749)) score += 12; // Romantik
      if (contentGenres.includes(18)) score += 10;    // Dram
      if (contentGenres.includes(35)) score += 8;     // Komedi
      if (contentGenres.includes(10751)) score += 6;  // Aile
    } else if (gender === 'male') {
      // Erkek kullanıcılar genellikle aksiyon, bilim kurgu, gerilim türlerini tercih eder
      if (contentGenres.includes(28)) score += 12;    // Aksiyon
      if (contentGenres.includes(878)) score += 10;   // Bilim Kurgu
      if (contentGenres.includes(53)) score += 8;     // Gerilim
      if (contentGenres.includes(12)) score += 6;     // Macera
    }

    return score;
  }

  private static calculateCountryBasedScore(_content: Movie | TVShow, userCountry: string): number {
    let score = 0;
    
    // Ülke bazlı içerik tercihleri
    // Bu kısım TMDb'den gelen original_language veya production_countries bilgisi ile geliştirilebilir
    
    // Şimdilik genel tercihler
    if (userCountry === 'TR') {
      // Türk kullanıcılar için yerel içerik tercihi
      // TMDb'den Türkiye yapımı içerikler için bonus
      score += 5;
    } else if (['US', 'GB', 'CA', 'AU'].includes(userCountry)) {
      // İngilizce konuşan ülkeler için Hollywood içerikleri
      score += 8;
    } else if (['DE', 'AT', 'CH'].includes(userCountry)) {
      // Almanca konuşan ülkeler için Alman içerikleri
      score += 6;
    } else if (['FR', 'BE', 'CH'].includes(userCountry)) {
      // Fransızca konuşan ülkeler için Fransız içerikleri
      score += 6;
    }

    return score;
  }

  private static calculateLanguageBasedScore(_content: Movie | TVShow, userLanguage: string): number {
    let score = 0;
    
    // Dil bazlı tercihler
    // Bu kısım TMDb'den gelen original_language bilgisi ile geliştirilebilir
    
    // Şimdilik genel tercihler
    if (userLanguage === 'tr') {
      // Türkçe konuşan kullanıcılar için Türkçe içerik tercihi
      score += 10;
    } else if (userLanguage === 'en') {
      // İngilizce konuşan kullanıcılar için İngilizce içerik tercihi
      score += 8;
    } else if (['de', 'fr', 'es', 'it'].includes(userLanguage)) {
      // Diğer Avrupa dilleri için kendi dillerindeki içerikler
      score += 6;
    }

    return score;
  }

  private static calculateEducationBasedScore(content: Movie | TVShow, education: string): number {
    let score = 0;
    
    const contentGenres = content.genre_ids || [];
    const voteAverage = content.vote_average || 0;
    
    if (education === 'university' || education === 'graduate') {
      // Yüksek eğitimli kullanıcılar daha karmaşık ve kaliteli içerikleri tercih eder
      if (voteAverage >= 7.5) score += 10;
      if (contentGenres.includes(18)) score += 8;     // Dram
      if (contentGenres.includes(9648)) score += 6;   // Gizem
      if (contentGenres.includes(878)) score += 6;    // Bilim Kurgu
    } else if (education === 'secondary') {
      // Orta eğitimli kullanıcılar için dengeli tercih
      if (voteAverage >= 7.0) score += 8;
      if (contentGenres.includes(35)) score += 8;     // Komedi
      if (contentGenres.includes(28)) score += 6;     // Aksiyon
      if (contentGenres.includes(18)) score += 6;     // Dram
    } else {
      // Diğer eğitim seviyeleri için genel tercihler
      if (voteAverage >= 6.5) score += 6;
      if (contentGenres.includes(35)) score += 6;     // Komedi
      if (contentGenres.includes(28)) score += 6;     // Aksiyon
    }

    return score;
  }

  private static calculateRelationshipBasedScore(content: Movie | TVShow, relationshipStatus: string): number {
    let score = 0;
    
    const contentGenres = content.genre_ids || [];
    
    if (relationshipStatus === 'single') {
      // Bekar kullanıcılar için
      if (contentGenres.includes(10749)) score += 8;  // Romantik
      if (contentGenres.includes(35)) score += 6;     // Komedi
      if (contentGenres.includes(28)) score += 6;     // Aksiyon
    } else if (relationshipStatus === 'in_relationship' || relationshipStatus === 'married') {
      // İlişkisi olan kullanıcılar için
      if (contentGenres.includes(10751)) score += 10; // Aile
      if (contentGenres.includes(10749)) score += 8;  // Romantik
      if (contentGenres.includes(35)) score += 6;     // Komedi
    } else if (relationshipStatus === 'divorced') {
      // Boşanmış kullanıcılar için
      if (contentGenres.includes(18)) score += 8;     // Dram
      if (contentGenres.includes(35)) score += 6;     // Komedi
      if (contentGenres.includes(28)) score += 6;     // Aksiyon
    }

    return score;
  }

  private static calculateChildrenBasedScore(content: Movie | TVShow, hasChildren: boolean, childrenAge?: number[]): number {
    let score = 0;
    
    const contentGenres = content.genre_ids || [];
    
    if (hasChildren) {
      // Çocuğu olan kullanıcılar için
      if (contentGenres.includes(10751)) score += 12; // Aile
      if (contentGenres.includes(16)) score += 8;     // Animasyon
      if (contentGenres.includes(35)) score += 6;     // Komedi
      
      // Çocuk yaşlarına göre ek tercihler
      if (childrenAge && childrenAge.length > 0) {
        const avgChildAge = childrenAge.reduce((a, b) => a + b, 0) / childrenAge.length;
        
        if (avgChildAge < 10) {
          // Küçük çocuklar için daha uygun içerikler
          if (contentGenres.includes(10751)) score += 5; // Aile
          if (contentGenres.includes(16)) score += 5;    // Animasyon
        } else if (avgChildAge < 16) {
          // Genç çocuklar için
          if (contentGenres.includes(10751)) score += 3; // Aile
          if (contentGenres.includes(35)) score += 3;    // Komedi
        }
      }
    } else {
      // Çocuğu olmayan kullanıcılar için
      if (contentGenres.includes(27)) score += 8;     // Korku
      if (contentGenres.includes(53)) score += 6;     // Gerilim
      if (contentGenres.includes(18)) score += 6;     // Dram
    }

    return score;
  }

  // Demografik faktörleri match score'a entegre etme
  private static applyDemographicFactors(baseScore: number, content: Movie | TVShow, profile: UserProfile): number {
    const demographicScore = this.calculateDemographicScore(content, profile);
    
    // Demografik skoru %20 ağırlıkla ekle
    const adjustedScore = baseScore * 0.8 + demographicScore * 0.2;
    
    return Math.min(100, Math.max(0, adjustedScore));
  }

  private static async generateAdditionalRecommendations(
    profile: UserProfile,
    allGenres: Genre[],
    ratedContentIds: Set<number>,
    watchlistContentIds: Set<number>,
    count: number
  ): Promise<Recommendation[]> {
    const recommendations: Recommendation[] = [];
    
    try {
      // Popüler filmler ve dizilerden ek öneriler al
      const [movieResponse, tvResponse] = await Promise.all([
        tmdbService.getPopularMovies(1),
        tmdbService.getPopularTVShows(1)
      ]);

      const movieResults = movieResponse.results;
      const tvResults = tvResponse.results;

    // Film önerileri
    for (const movie of movieResults) {
      if (ratedContentIds.has(movie.id) || watchlistContentIds.has(movie.id)) continue;
        
        const matchScore = this.calculateEnhancedMovieMatchScore(movie, profile, 'popular');
        recommendations.push({
          movie,
          matchScore,
          reasons: this.generateEnhancedMovieReasons(movie, profile, allGenres),
          confidence: 0.7,
          novelty: 0.3,
          diversity: 0.4,
          explanation: {
            primaryFactors: [`Popüler film: ${movie.title}`],
            secondaryFactors: ['Yüksek izlenme oranı'],
            riskFactors: []
          },
          recommendationType: 'safe'
        });
    }

    // Dizi önerileri
    for (const tvShow of tvResults) {
      if (ratedContentIds.has(tvShow.id) || watchlistContentIds.has(tvShow.id)) continue;
        
        const matchScore = this.calculateEnhancedTVMatchScore(tvShow, profile, 'popular');
        recommendations.push({
          movie: tvShow,
          matchScore,
          reasons: this.generateEnhancedTVReasons(tvShow, profile, allGenres),
          confidence: 0.7,
          novelty: 0.3,
          diversity: 0.4,
          explanation: {
            primaryFactors: [`Popüler dizi: ${tvShow.name}`],
            secondaryFactors: ['Yüksek izlenme oranı'],
            riskFactors: []
          },
          recommendationType: 'safe'
        });
      }
    } catch (error) {
      console.error('Error generating additional recommendations:', error);
    }
    
    return recommendations.slice(0, count);
  }
}
