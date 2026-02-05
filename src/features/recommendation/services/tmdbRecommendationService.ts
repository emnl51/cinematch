import type { Movie, TVShow, UserProfile, Recommendation, Genre, UserRating } from '../../content/types';
import { tmdbService } from '../../content/services/tmdb';
import { logger } from '../../../shared/utils/logger';

interface TMDbRecommendationResponse {
  page: number;
  results: (Movie | TVShow)[];
  total_pages: number;
  total_results: number;
}

interface SimilarContentResponse {
  page: number;
  results: (Movie | TVShow)[];
  total_pages: number;
  total_results: number;
}

export class TMDbRecommendationService {
  // TMDb API'nin kendi öneri sistemini kullan
  static async getMovieRecommendations(movieId: number, page = 1): Promise<TMDbRecommendationResponse> {
    try {
      logger.debug(`Fetching TMDb recommendations for movie ${movieId}`);
      
      const response = await tmdbService.makeRequest<TMDbRecommendationResponse>(
        `movie/${movieId}/recommendations?language=tr-TR&page=${page}`
      );
      
      // Media type ekle ve tip dönüşümü yap
      response.results = response.results.map(movie => ({
        ...movie,
        media_type: 'movie' as const,
        title: 'title' in movie ? movie.title : '',
        release_date: 'release_date' in movie ? movie.release_date : '',
        genre_ids: movie.genre_ids || []
      })) as (Movie | TVShow)[];
      
      logger.debug(`Found ${response.results.length} movie recommendations`);
      return response;
    } catch (error) {
      logger.error(`Error fetching movie recommendations for ${movieId}:`, error);
      return { page: 1, results: [], total_pages: 0, total_results: 0 };
    }
  }

  static async getTVRecommendations(tvId: number, page = 1): Promise<TMDbRecommendationResponse> {
    try {
      logger.debug(`Fetching TMDb recommendations for TV show ${tvId}`);
      
      const response = await tmdbService.makeRequest<TMDbRecommendationResponse>(
        `tv/${tvId}/recommendations?language=tr-TR&page=${page}`
      );
      
      // Media type ekle ve tip dönüşümü yap
      response.results = response.results.map(tvShow => ({
        ...tvShow,
        media_type: 'tv' as const,
        name: 'name' in tvShow ? tvShow.name : '',
        first_air_date: 'first_air_date' in tvShow ? tvShow.first_air_date : '',
        genre_ids: tvShow.genre_ids || []
      })) as (Movie | TVShow)[];
      
      logger.debug(`Found ${response.results.length} TV recommendations`);
      return response;
    } catch (error) {
      logger.error(`Error fetching TV recommendations for ${tvId}:`, error);
      return { page: 1, results: [], total_pages: 0, total_results: 0 };
    }
  }

  // Benzer içerikler
  static async getSimilarMovies(movieId: number, page = 1): Promise<SimilarContentResponse> {
    try {
      logger.debug(`Fetching similar movies for ${movieId}`);
      
      const response = await tmdbService.makeRequest<SimilarContentResponse>(
        `movie/${movieId}/similar?language=tr-TR&page=${page}`
      );
      
      // Media type ekle ve tip dönüşümü yap
      response.results = response.results.map(movie => ({
        ...movie,
        media_type: 'movie' as const,
        title: 'title' in movie ? movie.title : '',
        release_date: 'release_date' in movie ? movie.release_date : '',
        genre_ids: movie.genre_ids || []
      })) as (Movie | TVShow)[];
      
      logger.debug(`Found ${response.results.length} similar movies`);
      return response;
    } catch (error) {
      logger.error(`Error fetching similar movies for ${movieId}:`, error);
      return { page: 1, results: [], total_pages: 0, total_results: 0 };
    }
  }

  static async getSimilarTVShows(tvId: number, page = 1): Promise<SimilarContentResponse> {
    try {
      logger.debug(`Fetching similar TV shows for ${tvId}`);
      
      const response = await tmdbService.makeRequest<SimilarContentResponse>(
        `tv/${tvId}/similar?language=tr-TR&page=${page}`
      );
      
      // Media type ekle ve tip dönüşümü yap
      response.results = response.results.map(tvShow => ({
        ...tvShow,
        media_type: 'tv' as const,
        name: 'name' in tvShow ? tvShow.name : '',
        first_air_date: 'first_air_date' in tvShow ? tvShow.first_air_date : '',
        genre_ids: tvShow.genre_ids || []
      })) as (Movie | TVShow)[];
      
      logger.debug(`Found ${response.results.length} similar TV shows`);
      return response;
    } catch (error) {
      logger.error(`Error fetching similar TV shows for ${tvId}:`, error);
      return { page: 1, results: [], total_pages: 0, total_results: 0 };
    }
  }

  // Kullanıcının puanladığı içeriklere dayalı TMDb önerileri
  static async generateTMDbBasedRecommendations(
    profile: UserProfile,
    existingRatings: UserRating[],
    genres: Genre[],
    filters?: {
      genres: number[];
      minYear: number;
      maxYear: number;
      minRating: number;
      maxRating: number;
      mediaType: 'all' | 'movie' | 'tv';
      sortBy: 'match_score' | 'rating' | 'year' | 'title';
      minMatchScore: number;
      languages?: string[];
    }
  ): Promise<Recommendation[]> {
    const recommendations: Recommendation[] = [];
    const processedContentIds = new Set<number>();
    const ratedContentIds = new Set(existingRatings.map(r => r.movieId));

    try {
      logger.info('Generating TMDb-based recommendations...');

      // Kullanıcının en yüksek puan verdiği içerikleri al (4+ puan)
      const highRatedContent = existingRatings
        .filter(rating => typeof rating.rating === 'number' && rating.rating >= 4)
        .sort((a, b) => (b.rating as number) - (a.rating as number))
        .slice(0, 10); // En iyi 10 içerik

      logger.debug(`Found ${highRatedContent.length} high-rated content for TMDb recommendations`);

      // Her yüksek puanlı içerik için öneriler ve benzer içerikler al
      for (const rating of highRatedContent) {
        const contentId = rating.movieId;
        const mediaType = rating.mediaType || 'movie';

        try {
          // TMDb önerileri al
          const tmdbRecommendations = mediaType === 'tv' 
            ? await this.getTVRecommendations(contentId, 1)
            : await this.getMovieRecommendations(contentId, 1);

          // Benzer içerikler al
          const similarContent = mediaType === 'tv'
            ? await this.getSimilarTVShows(contentId, 1)
            : await this.getSimilarMovies(contentId, 1);

          // Önerileri işle
          await this.processRecommendationResults(
            tmdbRecommendations.results,
            recommendations,
            processedContentIds,
            ratedContentIds,
            profile,
            genres,
            rating.rating as number,
            'tmdb_recommendation',
            filters
          );

          // Benzer içerikleri işle
          await this.processRecommendationResults(
            similarContent.results,
            recommendations,
            processedContentIds,
            ratedContentIds,
            profile,
            genres,
            rating.rating as number,
            'similar_content',
            filters
          );

          // Rate limiting için kısa bekleme
          await new Promise(resolve => setTimeout(resolve, 100));

        } catch (error) {
          logger.warn(`Error processing recommendations for content ${contentId}:`, error);
          continue;
        }
      }

      // Popüler içeriklerden de öneriler al (eğer yeterli öneri yoksa)
      if (recommendations.length < 15) {
        await this.addPopularRecommendations(
          recommendations,
          processedContentIds,
          ratedContentIds,
          profile,
          genres,
          15 - recommendations.length,
          filters
        );
      }

      logger.info(`Generated ${recommendations.length} TMDb-based recommendations`);
      return recommendations.sort((a, b) => b.matchScore - a.matchScore);

    } catch (error) {
      logger.error('Error in TMDb-based recommendation generation:', error);
      return [];
    }
  }

  private static async processRecommendationResults(
    results: (Movie | TVShow)[],
    recommendations: Recommendation[],
    processedContentIds: Set<number>,
    ratedContentIds: Set<number>,
    profile: UserProfile,
    genres: Genre[],
    sourceRating: number,
    recommendationType: 'tmdb_recommendation' | 'similar_content',
    filters?: {
      genres: number[];
      minYear: number;
      maxYear: number;
      minRating: number;
      maxRating: number;
      mediaType: 'all' | 'movie' | 'tv';
      sortBy: 'match_score' | 'rating' | 'year' | 'title';
      minMatchScore: number;
      languages?: string[];
    }
  ): Promise<void> {
    for (const content of results.slice(0, 8)) { // Her kaynak için max 8 öneri
      // Zaten işlenmiş veya puanlanmış içerikleri atla
      if (processedContentIds.has(content.id) || ratedContentIds.has(content.id)) {
        continue;
      }

      // Kalite filtresi
      if (content.vote_average < 6.0 || content.vote_count < 100) {
        continue;
      }

      // Filtreleri uygula
      if (filters) {
        // Genre filter
        if (Array.isArray(filters.genres) && filters.genres.length > 0) {
          const contentGenres = content.genre_ids || [];
          const hasMatchingGenre = filters.genres.some(genreId => 
            contentGenres.includes(genreId)
          );
          if (!hasMatchingGenre) {
            continue;
          }
        }

        // Media type filter
        if (filters.mediaType !== 'all') {
          const isMovie = 'title' in content;
          const isTV = 'name' in content;
          if (filters.mediaType === 'movie' && !isMovie) {
            continue;
          }
          if (filters.mediaType === 'tv' && !isTV) {
            continue;
          }
        }

        // Year filter
        const releaseDate = 'release_date' in content ? content.release_date : content.first_air_date;
        if (releaseDate) {
          const year = new Date(releaseDate).getFullYear();
          if (year < filters.minYear || year > filters.maxYear) {
            continue;
          }
        }

        // Rating filter
        if (content.vote_average < filters.minRating || content.vote_average > filters.maxRating) {
          continue;
        }

        // Language filter
        if (filters.languages && filters.languages.length > 0) {
          const contentLanguage = content.original_language;
          if (!contentLanguage || !filters.languages.includes(contentLanguage)) {
            continue;
          }
        }
      }

      processedContentIds.add(content.id);

      // Match score hesapla
      const matchScore = this.calculateTMDbMatchScore(
        content,
        profile,
        sourceRating,
        recommendationType
      );

      // Minimum match score filtresi
      const minMatchScore = filters?.minMatchScore ?? 60;
      if (matchScore < minMatchScore) {
        continue;
      }

      const reasons = this.generateTMDbReasons(
        content,
        profile,
        genres,
        sourceRating,
        recommendationType
      );

      recommendations.push({
        movie: content,
        matchScore,
        reasons,
        confidence: matchScore / 100,
        novelty: this.calculateNovelty(content, profile),
        diversity: this.calculateDiversity(content, profile),
        explanation: {
          primaryFactors: reasons.slice(0, 2),
          secondaryFactors: reasons.slice(2),
          riskFactors: this.generateRiskFactors(content, profile)
        },
        recommendationType: matchScore >= 85 ? 'safe' : matchScore >= 70 ? 'exploratory' : 'serendipitous'
      });
    }
  }

  private static calculateTMDbMatchScore(
    content: Movie | TVShow,
    profile: UserProfile,
    sourceRating: number,
    recommendationType: 'tmdb_recommendation' | 'similar_content'
  ): number {
    let score = 0;

    // Base score from TMDb recommendation strength
    const baseScore = recommendationType === 'tmdb_recommendation' ? 70 : 65;
    score += baseScore;

    // Source rating influence
    const ratingBonus = (sourceRating - 3) * 5; // 4 puan = +5, 5 puan = +10
    score += ratingBonus;

    // Quality score
    const qualityScore = Math.min(20, (content.vote_average / 10) * 20);
    score += qualityScore;

    // Popularity score (normalized)
    const popularityScore = Math.min(10, (content.vote_count / 1000) * 5);
    score += popularityScore;

    // Genre alignment
    const contentGenres = content.genre_ids || [];
    let genreScore = 0;
    let genreCount = 0;
    for (const genreId of contentGenres) {
      const userPreference = profile.genreDistribution[genreId] || 0;
      if (userPreference > 0) {
        genreScore += userPreference;
        genreCount++;
      }
    }

    if (genreCount > 0) {
      const avgGenreScore = (genreScore / genreCount) / 100;
      score += avgGenreScore * 15; // Max 15 points from genre alignment
    }

    // Period preference
    const releaseDate = 'release_date' in content ? content.release_date : content.first_air_date;
    if (releaseDate) {
      const year = new Date(releaseDate).getFullYear();
      const decade = `${Math.floor(year / 10) * 10}s`;
      const periodPreference = profile.periodPreference[decade] || 0;
      score += (periodPreference / 100) * 5; // Max 5 points
    }

    // Demografik faktörleri uygula
    score = this.applyDemographicFactors(score, content, profile);

    return Math.min(100, Math.max(0, score));
  }

  private static generateTMDbReasons(
    content: Movie | TVShow,
    profile: UserProfile,
    genres: Genre[],
    sourceRating: number,
    recommendationType: 'tmdb_recommendation' | 'similar_content'
  ): string[] {
    const reasons: string[] = [];

    // Primary reason based on recommendation type
    if (recommendationType === 'tmdb_recommendation') {
      reasons.push(`${sourceRating} puan verdiğin içeriğe dayalı TMDb önerisi`);
    } else {
      reasons.push(`${sourceRating} puan verdiğin içeriğe benzer yapım`);
    }

    // Quality reason
    if (content.vote_average >= 8.0) {
      reasons.push(`Üstün kalite (${content.vote_average.toFixed(1)}/10)`);
    } else if (content.vote_average >= 7.0) {
      reasons.push(`Yüksek kalite (${content.vote_average.toFixed(1)}/10)`);
    }

    // Genre alignment
    const contentGenres = content.genre_ids || [];
    const matchingGenres = contentGenres.filter(genreId => 
      profile.genreDistribution[genreId] > 10
    );

    if (matchingGenres.length > 0) {
      const genreNames = matchingGenres.slice(0, 2).map(genreId => {
        const genre = genres.find(g => g.id === genreId);
        return genre?.name || 'Bilinmeyen';
      });
      reasons.push(`Sevdiğin türler: ${genreNames.join(', ')}`);
    }

    // Popularity
    if (content.vote_count > 5000) {
      reasons.push('Çok izlenen ve beğenilen yapım');
    }

    // Recent content
    const releaseDate = 'release_date' in content ? content.release_date : content.first_air_date;
    if (releaseDate) {
      const year = new Date(releaseDate).getFullYear();
      const currentYear = new Date().getFullYear();
      if (currentYear - year <= 3) {
        reasons.push('Güncel yapım');
      }
    }

    // Demografik faktörlere dayalı nedenler
    if (profile.demographics) {
      const demographics = profile.demographics;
      
      // Yaş bazlı nedenler
      if (demographics.age && releaseDate) {
        const contentYear = new Date(releaseDate).getFullYear();
        const contentAge = new Date().getFullYear() - contentYear;
        
        if (demographics.age < 25 && contentAge <= 5) {
          reasons.push('Yaş grubuna uygun güncel içerik');
        } else if (demographics.age >= 40 && contentAge >= 20) {
          reasons.push('Nostaljik klasik, yaş grubuna uygun');
        }
      }
      
      // Cinsiyet bazlı nedenler
      if (demographics.gender) {
        if (demographics.gender === 'female' && contentGenres.includes(10749)) {
          reasons.push('Cinsiyet tercihlerine uygun romantik içerik');
        } else if (demographics.gender === 'male' && contentGenres.includes(28)) {
          reasons.push('Cinsiyet tercihlerine uygun aksiyon içerik');
        }
      }
      
      // Eğitim seviyesi bazlı nedenler
      if (demographics.education && content.vote_average >= 7.5) {
        if (demographics.education === 'university' || demographics.education === 'graduate') {
          reasons.push('Eğitim seviyene uygun kaliteli yapım');
        }
      }
      
      // İlişki durumu bazlı nedenler
      if (demographics.relationshipStatus) {
        if ((demographics.relationshipStatus === 'in_relationship' || demographics.relationshipStatus === 'married') && contentGenres.includes(10751)) {
          reasons.push('İlişki durumuna uygun aile dostu içerik');
        } else if (demographics.relationshipStatus === 'single' && contentGenres.includes(10749)) {
          reasons.push('Bekar yaşam tarzına uygun romantik içerik');
        }
      }
      
      // Çocuk durumu bazlı nedenler
      if (demographics.hasChildren && contentGenres.includes(10751)) {
        reasons.push('Çocuklu aileye uygun içerik');
      }
    }

    return reasons.slice(0, 4);
  }

  private static calculateNovelty(content: Movie | TVShow, profile: UserProfile): number {
    let novelty = 0.5;

    // Genre novelty
    const contentGenres = content.genre_ids || [];
    const userGenres = Object.keys(profile.genreDistribution).map(id => parseInt(id));
    const newGenres = contentGenres.filter(id => !userGenres.includes(id));
    
    if (newGenres.length > 0) {
      novelty += 0.3;
    }

    // Release year novelty
    const releaseDate = 'release_date' in content ? content.release_date : content.first_air_date;
    if (releaseDate) {
      const year = new Date(releaseDate).getFullYear();
      const currentYear = new Date().getFullYear();
      
      if (currentYear - year <= 2) {
        novelty += 0.2;
      }
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
    if (userCountry === 'TR') {
      // Türk kullanıcılar için yerel içerik tercihi
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

  private static generateRiskFactors(content: Movie | TVShow, profile: UserProfile): string[] {
    const risks: string[] = [];

    // Low user genre preference
    const contentGenres = content.genre_ids || [];
    const lowPreferenceGenres = contentGenres.filter(genreId => {
      const preference = profile.genreDistribution[genreId] || 0;
      return preference < 5;
    });

    if (lowPreferenceGenres.length > 0) {
      risks.push('Alışık olmadığın türler içeriyor');
    }

    // Very different from user's average rating
    const expectedRating = profile.averageScore * 2;
    const ratingDiff = Math.abs(content.vote_average - expectedRating);
    if (ratingDiff > 2) {
      risks.push('Ortalama beğeni seviyenden farklı olabilir');
    }

    // Old content for users who prefer recent
    const releaseDate = 'release_date' in content ? content.release_date : content.first_air_date;
    if (releaseDate) {
      const year = new Date(releaseDate).getFullYear();
      const currentYear = new Date().getFullYear();
      
      if (currentYear - year > 20) {
        const recentPreference = Object.entries(profile.periodPreference)
          .filter(([decade]) => parseInt(decade) >= 2000)
          .reduce((sum, [, pref]) => sum + pref, 0);
        
        if (recentPreference > 60) {
          risks.push('Eski bir yapım olabilir');
        }
      }
    }

    return risks;
  }

  private static async addPopularRecommendations(
    recommendations: Recommendation[],
    processedContentIds: Set<number>,
    ratedContentIds: Set<number>,
    profile: UserProfile,
    genres: Genre[],
    count: number,
    filters?: {
      genres: number[];
      minYear: number;
      maxYear: number;
      minRating: number;
      maxRating: number;
      mediaType: 'all' | 'movie' | 'tv';
      sortBy: 'match_score' | 'rating' | 'year' | 'title';
      minMatchScore: number;
      languages?: string[];
    }
  ): Promise<void> {
    try {
      logger.debug(`Adding ${count} popular recommendations`);

      // Güvenli profil ve genreDistribution kontrolü
      const safeProfile = profile || {};
      const safeGenreDistribution = (safeProfile.genreDistribution && typeof safeProfile.genreDistribution === 'object') ? safeProfile.genreDistribution : {};
      const topGenres = Object.entries(safeGenreDistribution)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 3)
        .map(([genreId]) => parseInt(genreId));

      for (const genreId of topGenres) {
        if (recommendations.length >= count) break;
        try {
          // Film önerileri
          const movieResponse = await tmdbService.discoverMovies({
            with_genres: genreId.toString(),
            'vote_average.gte': 7.0,
            'vote_count.gte': 1000,
            sort_by: 'popularity.desc',
            page: 1,
            ...(filters?.languages && filters.languages.length > 0 && {
              with_original_language: filters.languages[0] // TMDb API supports only one language at a time
            })
          });

          await this.processRecommendationResults(
            movieResponse.results.slice(0, 3),
            recommendations,
            processedContentIds,
            ratedContentIds,
            profile,
            genres,
            4, // Orta seviye kaynak rating
            'similar_content',
            filters
          );

          // Dizi önerileri
          const tvResponse = await tmdbService.discoverTVShows({
            with_genres: genreId.toString(),
            'vote_average.gte': 7.0,
            'vote_count.gte': 500,
            sort_by: 'popularity.desc',
            page: 1,
            ...(filters?.languages && filters.languages.length > 0 && {
              with_original_language: filters.languages[0] // TMDb API supports only one language at a time
            })
          });

          await this.processRecommendationResults(
            tvResponse.results.slice(0, 3),
            recommendations,
            processedContentIds,
            ratedContentIds,
            profile,
            genres,
            4, // Orta seviye kaynak rating
            'similar_content',
            filters
          );

        } catch (error) {
          logger.warn(`Error fetching popular content for genre ${genreId}:`, error);
        }
      }

    } catch (error) {
      logger.error('Error adding popular recommendations:', error);
    }
  }
}
