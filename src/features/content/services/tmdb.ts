const TMDB_API_KEY = import.meta.env.VITE_TMDB_API_KEY;
const TMDB_READ_ACCESS_TOKEN = import.meta.env.VITE_TMDB_READ_ACCESS_TOKEN;
const TMDB_BASE_URL = import.meta.env.VITE_TMDB_BASE_URL || 'https://api.themoviedb.org/3';
const TMDB_IMAGE_BASE_URL = import.meta.env.VITE_TMDB_IMAGE_BASE_URL || 'https://image.tmdb.org/t/p';

import type { 
  Movie, 
  Genre, 
  TMDbResponse, 
  TVShow, 
  PersonCreditsResponse, 
  TMDbErrorResponse
} from '../types';
import { TMDbError } from '../types';
import { logger } from '../../../shared/utils/logger';
import { personFilmographyCache, movieDetailsCache, searchCache, userContentCache, generateCacheKey } from '../../../shared/utils/cache';

interface DiscoverMovieParams {
  with_genres?: string;
  with_cast?: string;
  with_crew?: string;
  'release_date.gte'?: string;
  'release_date.lte'?: string;
  'vote_average.gte'?: number;
  'vote_average.lte'?: number;
  'vote_count.gte'?: number;
  sort_by?: string;
  page?: number;
  with_original_language?: string;
}

interface DiscoverTVParams {
  with_genres?: string;
  with_cast?: string;
  with_crew?: string;
  'first_air_date.gte'?: string;
  'first_air_date.lte'?: string;
  'vote_average.gte'?: number;
  'vote_average.lte'?: number;
  'vote_count.gte'?: number;
  sort_by?: string;
  page?: number;
  with_original_language?: string;
}

interface Person {
  id: number;
  name: string;
  profile_path: string | null;
  known_for_department: string;
  known_for: (Movie | TVShow)[];
}

interface SearchResult {
  movies: Movie[];
  tvShows: TVShow[];
  people: Person[];
  totalResults: number;
  searchType: 'content' | 'person' | 'mixed';
}

interface QueuedRequest {
  url: string;
  cacheKey: string | undefined;
  resolve: (value: any) => void;
  reject: (error: any) => void;
  retryCount: number;
}

export class TMDbService {
  private static instance: TMDbService;
  private genres: Genre[] = [];
  private tvGenres: Genre[] = [];
  private requestQueue: Map<string, Promise<any>> = new Map();
  
  // Concurrency control
  private readonly MAX_CONCURRENT_REQUESTS = 6; // Reduced from default to be more conservative
  private currentActiveRequests = 0;
  private pendingRequestQueue: QueuedRequest[] = [];
  private readonly MAX_RETRIES = 3;
  private readonly RETRY_DELAY_BASE = 1000; // Base delay in ms
  private readonly RATE_LIMIT_DELAY = 2000; // Delay for rate limiting

  static getInstance(): TMDbService {
    if (!TMDbService.instance) {
      TMDbService.instance = new TMDbService();
    }
    return TMDbService.instance;
  }

  private hasValidApiKey(): boolean {
    return Boolean(TMDB_API_KEY && TMDB_API_KEY !== 'your_tmdb_api_key_here');
  }

  private hasReadAccessToken(): boolean {
    return Boolean(TMDB_READ_ACCESS_TOKEN);
  }

  private validateCredentials(): void {
    if (!this.hasValidApiKey() && !this.hasReadAccessToken()) {
      const errorMessage = 'TMDb API credentials are not configured. Please follow these steps:\n\n' +
        '1. Get a free API key or Read Access Token from https://www.themoviedb.org/settings/api\n' +
        '2. Copy your API key or Read Access Token\n' +
        '3. Open the .env file in your project root\n' +
        '4. Set VITE_TMDB_API_KEY or VITE_TMDB_READ_ACCESS_TOKEN\n' +
        '5. Restart the development server\n\n' +
        'For detailed instructions, see the README.md file.';
      
      logger.error(errorMessage);
      throw new TMDbError(errorMessage, undefined, undefined, false, true);
    }
  }

  private appendApiKey(url: string): string {
    if (!this.hasValidApiKey()) {
      return url;
    }

    const separator = url.includes('?') ? '&' : '?';
    return `${url}${separator}api_key=${TMDB_API_KEY}`;
  }

  private buildUrl(endpoint: string): string {
    return this.appendApiKey(`${TMDB_BASE_URL}/${endpoint}`);
  }

  private buildRequestHeaders(): HeadersInit {
    if (!this.hasReadAccessToken()) {
      return {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'User-Agent': 'CineMatch/1.0.0'
      };
    }

    return {
      'Accept': 'application/json',
      'Content-Type': 'application/json',
      'User-Agent': 'CineMatch/1.0.0',
      'Authorization': `Bearer ${TMDB_READ_ACCESS_TOKEN}`
    };
  }

  // Public method to make requests (for TMDbRecommendationService)
  async makeRequest<T>(endpoint: string, cacheKey?: string): Promise<T> {
    const url = this.buildUrl(endpoint);
    return this.makeRequestInternal<T>(url, cacheKey);
  }

  private async makeRequestInternal<T>(url: string, cacheKey?: string): Promise<T> {
    // Validate API key before making requests
    this.validateCredentials();

    // Check cache first
    if (cacheKey) {
      const cached = movieDetailsCache.get(cacheKey);
      if (cached) {
        logger.debug('Cache hit for:', cacheKey);
        return cached;
      }
    }

    // Prevent duplicate requests
    if (this.requestQueue.has(url)) {
      logger.debug('Request already in progress, waiting:', url);
      return this.requestQueue.get(url);
    }

    // Create a promise for this request
    const requestPromise = new Promise<T>((resolve, reject) => {
      const queuedRequest: QueuedRequest = {
        url,
        cacheKey,
        resolve,
        reject,
        retryCount: 0
      };

      // Add to queue
      this.pendingRequestQueue.push(queuedRequest);
      this.processPendingQueue();
    });

    this.requestQueue.set(url, requestPromise);

    try {
      const result = await requestPromise;
      
      // Cache successful results
      if (cacheKey && result) {
        movieDetailsCache.set(cacheKey, result);
      }
      
      return result;
    } finally {
      this.requestQueue.delete(url);
    }
  }

  private processPendingQueue(): void {
    // Process pending requests up to the concurrency limit
    while (
      this.currentActiveRequests < this.MAX_CONCURRENT_REQUESTS && 
      this.pendingRequestQueue.length > 0
    ) {
      const queuedRequest = this.pendingRequestQueue.shift();
      if (queuedRequest) {
        this.currentActiveRequests++;
        this.executeQueuedRequest(queuedRequest);
      }
    }
  }

  private async executeQueuedRequest(queuedRequest: QueuedRequest): Promise<void> {
    try {
      const result = await this.executeRequest(queuedRequest.url);
      queuedRequest.resolve(result);
    } catch (error) {
      // Handle retries for specific error types
      if (this.shouldRetry(error, queuedRequest.retryCount)) {
        queuedRequest.retryCount++;
        
        // Calculate exponential backoff delay
        const delay = this.calculateRetryDelay(error, queuedRequest.retryCount);
        
        logger.warn(`Retrying request (attempt ${queuedRequest.retryCount}/${this.MAX_RETRIES}) after ${delay}ms:`, queuedRequest.url);
        
        setTimeout(() => {
          // Add back to queue for retry
          this.pendingRequestQueue.unshift(queuedRequest);
          this.currentActiveRequests--;
          this.processPendingQueue();
        }, delay);
      } else {
        queuedRequest.reject(error);
        this.currentActiveRequests--;
        this.processPendingQueue();
      }
    }
  }

  private shouldRetry(error: any, retryCount: number): boolean {
    if (retryCount >= this.MAX_RETRIES) {
      return false;
    }

    // Retry on network errors, timeouts, and rate limiting
    if (error instanceof TMDbError) {
      return error.isNetworkError || 
             error.statusCode === 429 || 
             error.statusCode === 503 || 
             error.statusCode === 502;
    }

    // Retry on fetch errors
    return error.name === 'AbortError' || 
           error.message === 'Failed to fetch' || 
           error.message.includes('fetch');
  }

  private calculateRetryDelay(error: any, retryCount: number): number {
    // Use longer delay for rate limiting
    if (error instanceof TMDbError && error.statusCode === 429) {
      return this.RATE_LIMIT_DELAY * retryCount;
    }

    // Exponential backoff for other errors
    return this.RETRY_DELAY_BASE * Math.pow(2, retryCount - 1);
  }

  private async executeRequest<T>(url: string): Promise<T> {
    try {
      logger.debug('TMDb API Request:', url);
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
      
      const response = await fetch(url, {
        signal: controller.signal,
        headers: this.buildRequestHeaders()
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        if (response.status === 401) {
          const errorMessage = 'Invalid TMDb API credentials. Please check your API key or Read Access Token:\n\n' +
            '1. Verify your API key or Read Access Token is correct in the .env file\n' +
            '2. Ensure your API key or Read Access Token is active at https://www.themoviedb.org/settings/api\n' +
            '3. Restart the development server after updating the .env file';
          
          logger.error(errorMessage);
          throw new TMDbError(errorMessage, undefined, undefined, false, true);
        }
        
        if (response.status === 404) {
          // Log 404 errors at debug level to reduce console noise
          // Only log in development mode to reduce console clutter
          if (import.meta.env.DEV) {
            logger.debug(`Content not found (404) for URL: ${url}`);
          }
          throw new TMDbError(`Content not found (404)`, response.status, undefined, false, false);
        }
        
        if (response.status === 429) {
          throw new TMDbError('Too many requests. Please wait a moment and try again.', response.status, undefined, true, false);
        }
        
        throw new TMDbError(`HTTP error! status: ${response.status}`, response.status, undefined, response.status >= 500, false);
      }
      
      const data = await response.json();
      
      // Check for TMDb API error response
      if (data.success === false) {
        const errorData = data as TMDbErrorResponse;
        logger.error('TMDb API Error:', errorData.status_message);
        throw new TMDbError(`TMDb API Error: ${errorData.status_message}`, errorData.status_code, errorData.status_message, false, true);
      }
      
      logger.debug('TMDb API Response received for:', url);
      return data;
    } catch (error) {
      if (error instanceof TMDbError) {
        throw error;
      }
      
      if (error instanceof Error && error.name === 'AbortError') {
        logger.warn('Request timeout - will retry if possible');
        throw new TMDbError('Request timeout - please check your internet connection', undefined, undefined, true, false);
      }
      
      // Enhanced error logging for network issues
      if (error instanceof Error && (error.message === 'Failed to fetch' || error.message.includes('fetch'))) {
        logger.warn('Network error detected, this might be a temporary issue');
        throw new TMDbError('Network connection issue. This might be temporary - please try again.', undefined, undefined, true, false);
      }
      
      // Type guard to check if error has a message property
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      throw new TMDbError(`Network error: ${errorMessage}`, undefined, undefined, true, false);
    } finally {
      // Always decrement active requests and process queue
      this.currentActiveRequests--;
      this.processPendingQueue();
    }
  }

  async fetchGenres(): Promise<Genre[]> {
    if (this.genres.length > 0) return this.genres;

    try {
      const data = await this.makeRequestInternal<{ genres: Genre[] }>(
        this.appendApiKey(`${TMDB_BASE_URL}/genre/movie/list?language=tr-TR`),
        'movie_genres'
      );
      this.genres = data.genres;
      return this.genres;
    } catch (error) {
      logger.error('Error fetching genres:', error);
      // Return empty array to prevent app crash
      return [];
    }
  }

  async fetchTVGenres(): Promise<Genre[]> {
    if (this.tvGenres.length > 0) return this.tvGenres;

    try {
      const data = await this.makeRequestInternal<{ genres: Genre[] }>(
        this.appendApiKey(`${TMDB_BASE_URL}/genre/tv/list?language=tr-TR`),
        'tv_genres'
      );
      this.tvGenres = data.genres;
      return this.tvGenres;
    } catch (error) {
      logger.error('Error fetching TV genres:', error);
      return [];
    }
  }

  async searchMovies(query: string, page = 1): Promise<TMDbResponse<Movie>> {
    const cacheKey = generateCacheKey.search(query, 'movie');
    
    try {
      return await this.makeRequestInternal<TMDbResponse<Movie>>(
        this.appendApiKey(`${TMDB_BASE_URL}/search/movie?language=tr-TR&query=${encodeURIComponent(query)}&page=${page}`),
        cacheKey
      );
    } catch (error) {
      logger.error('Error searching movies:', error);
      return { page: 1, results: [], total_pages: 0, total_results: 0 };
    }
  }

  async searchTVShows(query: string, page = 1): Promise<TMDbResponse<TVShow>> {
    const cacheKey = generateCacheKey.search(query, 'tv');
    
    try {
      return await this.makeRequestInternal<TMDbResponse<TVShow>>(
        this.appendApiKey(`${TMDB_BASE_URL}/search/tv?language=tr-TR&query=${encodeURIComponent(query)}&page=${page}`),
        cacheKey
      );
    } catch (error) {
      logger.error('Error searching TV shows:', error);
      return { page: 1, results: [], total_pages: 0, total_results: 0 };
    }
  }

  async searchPeople(query: string, page = 1): Promise<TMDbResponse<Person>> {
    const cacheKey = generateCacheKey.search(query, 'person');
    
    try {
      return await this.makeRequestInternal<TMDbResponse<Person>>(
        this.appendApiKey(`${TMDB_BASE_URL}/search/person?language=tr-TR&query=${encodeURIComponent(query)}&page=${page}`),
        cacheKey
      );
    } catch (error) {
      logger.error('Error searching people:', error);
      return { page: 1, results: [], total_pages: 0, total_results: 0 };
    }
  }

  async searchMulti(query: string, page = 1): Promise<TMDbResponse<Movie | TVShow>> {
    const cacheKey = generateCacheKey.search(query, 'multi');
    
    try {
      const data = await this.makeRequestInternal<TMDbResponse<Movie | TVShow>>(
        this.appendApiKey(`${TMDB_BASE_URL}/search/multi?language=tr-TR&query=${encodeURIComponent(query)}&page=${page}`),
        cacheKey
      );
      
      // Filter only movies and TV shows
      const filteredResults = data.results.filter((item: any) => 
        item.media_type === 'movie' || item.media_type === 'tv'
      );
      
      return {
        ...data,
        results: filteredResults
      };
    } catch (error) {
      logger.error('Error searching multi:', error);
      return { page: 1, results: [], total_pages: 0, total_results: 0 };
    }
  }

  // Enhanced search that combines content and people search
  async enhancedSearch(query: string): Promise<SearchResult> {
    const cacheKey = `enhanced_search_${query.toLowerCase().trim()}`;
    
    try {
      // Check cache first
      const cached = searchCache.get(cacheKey);
      if (cached) {
        logger.debug('Enhanced search cache hit for:', query);
        return cached;
      }

      logger.debug('Enhanced search for:', query);

      // Parallel search for movies, TV shows, and people
      const [movieResults, tvResults, peopleResults] = await Promise.all([
        this.searchMovies(query, 1),
        this.searchTVShows(query, 1),
        this.searchPeople(query, 1)
      ]);

      // Determine search type based on results
      let searchType: 'content' | 'person' | 'mixed' = 'content';
      const totalContentResults = movieResults.total_results + tvResults.total_results;
      const totalPeopleResults = peopleResults.total_results;

      if (totalPeopleResults > 0 && totalContentResults === 0) {
        searchType = 'person';
      } else if (totalPeopleResults > 0 && totalContentResults > 0) {
        searchType = 'mixed';
      }

      // If we found people, get their content
      const additionalContent: (Movie | TVShow)[] = [];
      if (peopleResults.results.length > 0) {
        // Get content from top 3 people found
        const topPeople = peopleResults.results.slice(0, 3);
        
        for (const person of topPeople) {
          try {
            const personCredits = await this.getPersonCombinedCredits(person.id);
            
            // Add top rated content from this person
            const personContent = [...personCredits.cast, ...personCredits.crew]
              .filter(credit => (credit.vote_average ?? 0) >= 6.0 && (credit.vote_count ?? 0) >= 100)
              .sort((a, b) => (b.vote_average ?? 0) - (a.vote_average ?? 0))
              .slice(0, 10)
              .map(credit => credit as unknown as Movie | TVShow);
            
            additionalContent.push(...personContent);
          } catch (error) {
            logger.warn(`Error fetching credits for person ${person.name}:`, error);
          }
        }
      }

      // Combine and deduplicate results
      const allMovies = [...movieResults.results];
      const allTVShows = [...tvResults.results];

      // Add content from people search
      additionalContent.forEach(content => {
        if (content.media_type === 'movie') {
          // Check if not already in results
          if (!allMovies.find(m => m.id === content.id)) {
            allMovies.push(content as Movie);
          }
        } else if (content.media_type === 'tv') {
          if (!allTVShows.find(tv => tv.id === content.id)) {
            allTVShows.push(content as TVShow);
          }
        }
      });

      const result: SearchResult = {
        movies: allMovies.slice(0, 20), // Limit results
        tvShows: allTVShows.slice(0, 20),
        people: peopleResults.results.slice(0, 10),
        totalResults: totalContentResults + totalPeopleResults,
        searchType
      };

      // Cache the result
      searchCache.set(cacheKey, result);

      logger.debug(`Enhanced search completed: ${result.movies.length} movies, ${result.tvShows.length} TV shows, ${result.people.length} people`);
      return result;

    } catch (error) {
      logger.error('Error in enhanced search:', error);
      return {
        movies: [],
        tvShows: [],
        people: [],
        totalResults: 0,
        searchType: 'content'
      };
    }
  }

  async getPopularMovies(page = 1): Promise<TMDbResponse<Movie>> {
    try {
      return await this.makeRequestInternal<TMDbResponse<Movie>>(
        this.appendApiKey(`${TMDB_BASE_URL}/movie/popular?language=tr-TR&page=${page}`),
        `popular_movies_${page}`
      );
    } catch (error) {
      logger.error('Error fetching popular movies:', error);
      return { page: 1, results: [], total_pages: 0, total_results: 0 };
    }
  }

  async getPopularTVShows(page = 1): Promise<TMDbResponse<TVShow>> {
    try {
      return await this.makeRequestInternal<TMDbResponse<TVShow>>(
        this.appendApiKey(`${TMDB_BASE_URL}/tv/popular?language=tr-TR&page=${page}`),
        `popular_tv_${page}`
      );
    } catch (error) {
      logger.error('Error fetching popular TV shows:', error);
      return { page: 1, results: [], total_pages: 0, total_results: 0 };
    }
  }

  async getMovieDetails(movieId: number): Promise<Movie> {
    const cacheKey = generateCacheKey.movieDetails(movieId);
    
    try {
      const response = await this.makeRequestInternal<Movie>(
        this.appendApiKey(`${TMDB_BASE_URL}/movie/${movieId}?append_to_response=credits,videos,keywords,external_ids`),
        cacheKey
      );
      
      // IMDb entegrasyonu kaldırıldı - sadece TMDb verilerini döndür
      return response;
    } catch (error) {
      logger.error(`Error fetching movie details for ID ${movieId}:`, error);
      throw error;
    }
  }

  async getTVShowDetails(tvId: number): Promise<TVShow> {
    const cacheKey = generateCacheKey.tvDetails(tvId);
    
    try {
      const response = await this.makeRequestInternal<TVShow>(
        this.appendApiKey(`${TMDB_BASE_URL}/tv/${tvId}?append_to_response=credits,videos,keywords,external_ids`),
        cacheKey
      );
      
      // IMDb entegrasyonu kaldırıldı - sadece TMDb verilerini döndür
      return response;
    } catch (error) {
      logger.error(`Error fetching TV show details for ID ${tvId}:`, error);
      throw error;
    }
  }

  // Enhanced Person filmography methods with proper error handling
  async getPersonMovieCredits(personId: number): Promise<PersonCreditsResponse> {
    if (!personId || personId <= 0) {
      throw new TMDbError('Invalid person ID provided');
    }

    const cacheKey = `${generateCacheKey.personFilmography(personId)}_movies`;
    
    try {
      logger.debug('Fetch movie credits for personId:', personId);
      const data = await this.makeRequestInternal<PersonCreditsResponse>(
        this.appendApiKey(`${TMDB_BASE_URL}/person/${personId}/movie_credits?language=tr-TR`),
        cacheKey
      );
      
      // Add media_type to all credits
      const processedData = {
        ...data,
        cast: data.cast.map(credit => ({ ...credit, media_type: 'movie' as const })),
        crew: data.crew.map(credit => ({ ...credit, media_type: 'movie' as const }))
      };
      
      logger.debug(`Movie credits response processed: ${processedData.cast.length + processedData.crew.length} items`);
      return processedData;
    } catch (error) {
      logger.error(`Error fetching movie credits for person ${personId}:`, error);
      throw error;
    }
  }

  async getPersonTvCredits(personId: number): Promise<PersonCreditsResponse> {
    if (!personId || personId <= 0) {
      throw new TMDbError('Invalid person ID provided');
    }

    const cacheKey = `${generateCacheKey.personFilmography(personId)}_tv`;
    
    try {
      logger.debug('Fetch TV credits for personId:', personId);
      const data = await this.makeRequestInternal<PersonCreditsResponse>(
        this.appendApiKey(`${TMDB_BASE_URL}/person/${personId}/tv_credits?language=tr-TR`),
        cacheKey
      );
      
      // Add media_type to all credits
      const processedData = {
        ...data,
        cast: data.cast.map(credit => ({ ...credit, media_type: 'tv' as const })),
        crew: data.crew.map(credit => ({ ...credit, media_type: 'tv' as const }))
      };
      
      logger.debug(`TV credits response processed: ${processedData.cast.length + processedData.crew.length} items`);
      return processedData;
    } catch (error) {
      logger.error(`Error fetching TV credits for person ${personId}:`, error);
      throw error;
    }
  }

  async getPersonCombinedCredits(personId: number): Promise<PersonCreditsResponse> {
    if (!personId || personId <= 0) {
      throw new TMDbError('Invalid person ID provided');
    }

    const cacheKey = generateCacheKey.personFilmography(personId);
    
    // Check cache first
    const cached = personFilmographyCache.get(cacheKey);
    if (cached) {
      logger.debug('Cache hit for person filmography:', personId);
      return cached;
    }

    try {
      logger.debug('Fetch combined credits for personId:', personId);
      
      // Use Promise.all for parallel requests
      const [movieCredits, tvCredits] = await Promise.all([
        this.getPersonMovieCredits(personId),
        this.getPersonTvCredits(personId)
      ]);

      // Combine results
      const combinedData: PersonCreditsResponse = {
        id: personId,
        cast: [...movieCredits.cast, ...tvCredits.cast],
        crew: [...movieCredits.crew, ...tvCredits.crew]
      };

      // Cache the combined result
      personFilmographyCache.set(cacheKey, combinedData);
      
      logger.debug(`Combined credits processed: ${combinedData.cast.length + combinedData.crew.length} total items`);
      return combinedData;
    } catch (error) {
      logger.error(`Error fetching combined credits for person ${personId}:`, error);
      throw error;
    }
  }

  async discoverMovies(params: DiscoverMovieParams): Promise<TMDbResponse<Movie>> {
    try {
      const queryParams = new URLSearchParams({
        language: 'tr-TR',
        ...Object.fromEntries(
          Object.entries(params).map(([key, value]) => [key, String(value)])
        )
      });

      if (this.hasValidApiKey()) {
        queryParams.set('api_key', TMDB_API_KEY);
      }

      return await this.makeRequestInternal<TMDbResponse<Movie>>(
        `${TMDB_BASE_URL}/discover/movie?${queryParams}`
      );
    } catch (error) {
      logger.error('Error discovering movies:', error);
      return { page: 1, results: [], total_pages: 0, total_results: 0 };
    }
  }

  async discoverTVShows(params: DiscoverTVParams): Promise<TMDbResponse<TVShow>> {
    try {
      const queryParams = new URLSearchParams({
        language: 'tr-TR',
        ...Object.fromEntries(
          Object.entries(params).map(([key, value]) => [key, String(value)])
        )
      });

      if (this.hasValidApiKey()) {
        queryParams.set('api_key', TMDB_API_KEY);
      }

      return await this.makeRequestInternal<TMDbResponse<TVShow>>(
        `${TMDB_BASE_URL}/discover/tv?${queryParams}`
      );
    } catch (error) {
      logger.error('Error discovering TV shows:', error);
      return { page: 1, results: [], total_pages: 0, total_results: 0 };
    }
  }

  getImageUrl(path: string | null, size: 'w92' | 'w154' | 'w185' | 'w342' | 'w500' | 'w780' | 'original' = 'w500'): string {
    if (!path) return '/placeholder-movie.jpg';
    return `${TMDB_IMAGE_BASE_URL}/${size}${path}`;
  }

  getYouTubeUrl(key: string): string {
    return `https://www.youtube.com/embed/${key}?autoplay=1&rel=0&modestbranding=1`;
  }

  // User content cache management
  async cacheUserContent(contentId: number, mediaType: 'movie' | 'tv'): Promise<void> {
    const cacheKey = generateCacheKey.userContent(contentId, mediaType);
    
    // Check if already cached
    if (userContentCache.has(cacheKey)) {
      logger.debug('User content already cached:', cacheKey);
      return;
    }

    try {
      let content;
      if (mediaType === 'movie') {
        content = await this.getMovieDetails(contentId);
      } else {
        content = await this.getTVShowDetails(contentId);
      }
      
      // Cache the content
      userContentCache.set(cacheKey, content);
      logger.debug('User content cached:', cacheKey);
    } catch (error) {
      logger.warn(`Failed to cache user content (${cacheKey}):`, error);
    }
  }

  async cacheMultipleUserContent(contents: Array<{ id: number; mediaType: 'movie' | 'tv' }>): Promise<void> {
    const cachePromises = contents.map(content => 
      this.cacheUserContent(content.id, content.mediaType)
    );
    
    await Promise.allSettled(cachePromises);
    logger.debug(`Cached ${contents.length} user contents`);
  }

  getUserContentFromCache(contentId: number, mediaType: 'movie' | 'tv'): any | null {
    const cacheKey = generateCacheKey.userContent(contentId, mediaType);
    return userContentCache.get(cacheKey);
  }

  // Cache management methods
  clearCache(): void {
    personFilmographyCache.clear();
    movieDetailsCache.clear();
    searchCache.clear();
    userContentCache.clear();
    logger.info('All caches cleared');
  }

  getCacheStats() {
    return {
      personFilmography: personFilmographyCache.getStats(),
      movieDetails: movieDetailsCache.getStats(),
      search: searchCache.getStats(),
      userContent: userContentCache.getStats()
    };
  }

  // Debug method to check queue status
  getQueueStatus() {
    return {
      activeRequests: this.currentActiveRequests,
      pendingRequests: this.pendingRequestQueue.length,
      maxConcurrent: this.MAX_CONCURRENT_REQUESTS
    };
  }
}

export const tmdbService = TMDbService.getInstance();
