const HybridRecommendationEngine = require('../services/recommendationEngine');
const { TrackingService } = require('../services/trackingService');

// Mock Redis and MongoDB for testing
jest.mock('ioredis', () => {
  return jest.fn().mockImplementation(() => ({
    get: jest.fn().mockResolvedValue(null),
    setex: jest.fn().mockResolvedValue('OK'),
    keys: jest.fn().mockResolvedValue([]),
    del: jest.fn().mockResolvedValue(1),
    hgetall: jest.fn().mockResolvedValue({}),
    incr: jest.fn().mockResolvedValue(1),
    lpush: jest.fn().mockResolvedValue(1),
    ltrim: jest.fn().mockResolvedValue('OK'),
    lrange: jest.fn().mockResolvedValue([])
  }));
});

jest.mock('mongoose', () => ({
  connect: jest.fn(),
  connection: { readyState: 1 },
  Schema: jest.fn(),
  model: jest.fn()
}));

describe('Hybrid Recommendation Engine', () => {
  let recommendationEngine;

  beforeEach(() => {
    recommendationEngine = new HybridRecommendationEngine();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('calculateWeights', () => {
    it('should return rule/content-heavy weights for new users', () => {
      const userProfile = { ratingCount: 2, sessionDepth: 0.1, recencyScore: 0.2 };
      const weights = recommendationEngine.calculateWeights(userProfile);

      expect(weights.content).toBeGreaterThan(weights.collaborative);
      expect(weights.rule).toBeGreaterThan(weights.collaborative);
    });

    it('should include sequence weight for intermediate users', () => {
      const userProfile = { ratingCount: 20, sessionDepth: 0.6, recencyScore: 0.5 };
      const weights = recommendationEngine.calculateWeights(userProfile);

      expect(weights.sequence).toBeGreaterThan(0.2);
      expect(weights.rule).toBeGreaterThan(0);
    });

    it('should return collaborative-heavy weights for experienced users', () => {
      const userProfile = { ratingCount: 100, sessionDepth: 0.4, recencyScore: 0.3 };
      const weights = recommendationEngine.calculateWeights(userProfile);

      expect(weights.collaborative).toBeGreaterThan(weights.content);
    });
  });

  describe('normalizeScore', () => {
    it('should normalize scores correctly', () => {
      expect(recommendationEngine.normalizeScore(1)).toBe(0);
      expect(recommendationEngine.normalizeScore(10)).toBe(1);
      expect(recommendationEngine.normalizeScore(5.5)).toBeCloseTo(0.5);
    });

    it('should handle edge cases', () => {
      expect(recommendationEngine.normalizeScore(0)).toBe(0);
      expect(recommendationEngine.normalizeScore(11)).toBe(1);
    });
  });

  describe('combineScores', () => {
    it('should combine content, collaborative, sequence, and rule scores correctly', () => {
      const contentScores = [
        { movieId: 1, score: 0.8, movie: { id: 1, title: 'Movie 1' } }
      ];

      const collaborativeScores = [
        { movieId: 1, score: 0.6, movie: { id: 1, title: 'Movie 1' } }
      ];

      const sequenceScores = [
        { movieId: 1, score: 0.7, movie: { id: 1, title: 'Movie 1' } }
      ];

      const ruleScores = [
        { movieId: 1, score: 0.5, movie: { id: 1, title: 'Movie 1' } }
      ];

      const weights = { content: 0.4, collaborative: 0.3, sequence: 0.2, rule: 0.1 };

      const hybridScores = recommendationEngine.combineScores(
        contentScores,
        collaborativeScores,
        sequenceScores,
        ruleScores,
        weights
      );

      expect(hybridScores).toHaveLength(1);
      expect(hybridScores[0].score).toBeCloseTo(0.69); // 0.8*0.4 + 0.6*0.3 + 0.7*0.2 + 0.5*0.1
      expect(hybridScores[0].source).toBe('hybrid');
    });
  });

  describe('calculateRatingVariance', () => {
    it('should calculate variance correctly', () => {
      const ratings = [
        { value: 5 },
        { value: 7 },
        { value: 3 }
      ];

      const variance = recommendationEngine.calculateRatingVariance(ratings);
      expect(variance).toBeCloseTo(2.67, 1);
    });

    it('should return 0 for insufficient data', () => {
      const ratings = [{ value: 5 }];
      const variance = recommendationEngine.calculateRatingVariance(ratings);
      expect(variance).toBe(0);
    });
  });

  describe('groupActionsBySessions', () => {
    it('should group actions by sessions correctly', () => {
      const actions = [
        { timestamp: new Date('2023-01-01T10:00:00Z') },
        { timestamp: new Date('2023-01-01T10:15:00Z') },
        { timestamp: new Date('2023-01-01T11:00:00Z') } // New session
      ];

      const sessions = recommendationEngine.groupActionsBySessions(actions);
      expect(sessions).toHaveLength(2);
      expect(sessions[0]).toHaveLength(2);
      expect(sessions[1]).toHaveLength(1);
    });
  });
});

describe('TrackingService', () => {
  let trackingService;

  beforeEach(() => {
    trackingService = new TrackingService();
  });

  describe('validateAction', () => {
    it('should validate correct action', () => {
      const action = {
        userId: 'user123',
        movieId: 456,
        actionType: 'rate',
        value: 8
      };

      const validated = trackingService.validateAction(action);
      expect(validated.userId).toBe('user123');
      expect(validated.movieId).toBe(456);
      expect(validated.actionType).toBe('rate');
      expect(validated.value).toBe(8);
    });

    it('should throw error for missing fields', () => {
      const action = {
        userId: 'user123',
        actionType: 'rate'
        // Missing movieId and value
      };

      expect(() => trackingService.validateAction(action)).toThrow();
    });

    it('should throw error for invalid action type', () => {
      const action = {
        userId: 'user123',
        movieId: 456,
        actionType: 'invalid_action',
        value: 8
      };

      expect(() => trackingService.validateAction(action)).toThrow();
    });

    it('should throw error for invalid rating value', () => {
      const action = {
        userId: 'user123',
        movieId: 456,
        actionType: 'rate',
        value: 15 // Invalid rating
      };

      expect(() => trackingService.validateAction(action)).toThrow();
    });
  });
});

// Integration test placeholder
describe('Integration Tests', () => {
  it('should generate recommendations end-to-end', async () => {
    const engine = new HybridRecommendationEngine();
    expect(engine).toBeDefined();
  });
});
