import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, test, expect, beforeEach } from 'vitest';
import { NeuralModelInfo } from './NeuralModelInfo';
import { NeuralRecommendationService } from '../../recommendation/services/neuralRecommendationService';

// Mock the NeuralRecommendationService
vi.mock('../../recommendation/services/neuralRecommendationService', () => ({
  NeuralRecommendationService: {
    initializeModel: vi.fn(),
    trainModel: vi.fn(),
    evaluateModel: vi.fn(),
  },
}));

const mockProfile = {
  id: 'test-user',
  username: 'testuser',
  email: 'test@example.com',
  totalRatings: 50,
  averageScore: 7.5,
  learningPhase: 'optimizing',
  genreDistribution: {
    '28': 0.3,
    '12': 0.2,
    '16': 0.15,
    '35': 0.2,
    '80': 0.15
  },
  periodPreference: {
    '2020s': 0.4,
    '2010s': 0.3,
    '2000s': 0.2,
    '1990s': 0.1
  },
  favoriteActors: {},
  favoriteDirectors: {},
  favoriteWriters: {},
  qualityTolerance: {
    minRating: 6.0,
    minVoteCount: 1000
  },
  demographics: {
    age: 25,
    gender: 'male',
    language: 'en'
  },
  lastUpdated: Date.now(),
  accuracyScore: 0.85
};

describe('NeuralModelInfo', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Training Button Visibility', () => {
    test('should show training button when user has 20+ ratings', () => {
      const sufficientRatings = Array.from({ length: 25 }, (_, i) => ({
        movieId: i + 1,
        rating: 7,
        mediaType: 'movie' as const,
        timestamp: Date.now()
      }));

      render(<NeuralModelInfo profile={mockProfile} ratings={sufficientRatings} />);

      // Wait for component to load
      waitFor(() => {
        expect(screen.getByText('Modeli Eğit')).toBeTruthy();
      });
    });

    test('should not show training button when user has less than 20 ratings', async () => {
      const insufficientRatings = Array.from({ length: 15 }, (_, i) => ({
        movieId: i + 1,
        rating: 7,
        mediaType: 'movie' as const,
        timestamp: Date.now()
      }));

      render(<NeuralModelInfo profile={mockProfile} ratings={insufficientRatings} />);

      // Should show training requirements message
      await screen.findByText((content) => content.includes('Eğitim Gereksinimleri'));
      expect(screen.queryByText('Modeli Eğit')).toBeFalsy();
    });

    test('should show correct training data count', async () => {
      const ratings = Array.from({ length: 30 }, (_, i) => ({
        movieId: i + 1,
        rating: 7,
        mediaType: 'movie' as const,
        timestamp: Date.now()
      }));

      render(<NeuralModelInfo profile={mockProfile} ratings={ratings} />);

      // Should show 30 as training data count
      await screen.findByText((content) => content.trim() === '30');
    });
  });

  describe('Training Button Functionality', () => {
    test('should call trainModel when training button is clicked', async () => {
      const mockTrainModel = vi.fn().mockResolvedValue(undefined);
      const mockEvaluateModel = vi.fn().mockResolvedValue({
        accuracy: 0.85,
        mae: 0.5,
        rmse: 0.7,
        coverage: 0.9
      });

      (NeuralRecommendationService.trainModel as any) = mockTrainModel;
      (NeuralRecommendationService.evaluateModel as any) = mockEvaluateModel;

      const sufficientRatings = Array.from({ length: 25 }, (_, i) => ({
        movieId: i + 1,
        rating: 7,
        mediaType: 'movie' as const,
        timestamp: Date.now()
      }));

      render(<NeuralModelInfo profile={mockProfile} ratings={sufficientRatings} />);

      // Wait for component to load and find the button
      await waitFor(() => {
        const trainButton = screen.getByText('Modeli Eğit');
        expect(trainButton).toBeTruthy();
        
        // Click the training button
        fireEvent.click(trainButton);
      });

      // Verify that trainModel was called
      await waitFor(() => {
        expect(mockTrainModel).toHaveBeenCalledWith(sufficientRatings, mockProfile);
      });
    });

    test('should show loading state during training', async () => {
      const mockTrainModel = vi.fn().mockImplementation(() => 
        new Promise(resolve => setTimeout(resolve, 100))
      );
      const mockEvaluateModel = vi.fn().mockResolvedValue({
        accuracy: 0.85,
        mae: 0.5,
        rmse: 0.7,
        coverage: 0.9
      });

      (NeuralRecommendationService.trainModel as any) = mockTrainModel;
      (NeuralRecommendationService.evaluateModel as any) = mockEvaluateModel;

      const sufficientRatings = Array.from({ length: 25 }, (_, i) => ({
        movieId: i + 1,
        rating: 7,
        mediaType: 'movie' as const,
        timestamp: Date.now()
      }));

      render(<NeuralModelInfo profile={mockProfile} ratings={sufficientRatings} />);

      // Wait for component to load and find the button
      await waitFor(() => {
        const trainButton = screen.getByText('Modeli Eğit');
        expect(trainButton).toBeTruthy();
        
        // Click the training button
        fireEvent.click(trainButton);
      });

      // Should show loading state
      await waitFor(() => {
        expect(screen.getByText('Eğitiliyor...')).toBeTruthy();
      });

      // Wait for training to complete
      await waitFor(() => {
        expect(screen.getByText('Modeli Eğit')).toBeTruthy();
      }, { timeout: 200 });
    });

    test('should handle training errors gracefully', async () => {
      const mockTrainModel = vi.fn().mockRejectedValue(new Error('Training failed'));
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      (NeuralRecommendationService.trainModel as any) = mockTrainModel;

      const sufficientRatings = Array.from({ length: 25 }, (_, i) => ({
        movieId: i + 1,
        rating: 7,
        mediaType: 'movie' as const,
        timestamp: Date.now()
      }));

      render(<NeuralModelInfo profile={mockProfile} ratings={sufficientRatings} />);

      // Wait for component to load and find the button
      await waitFor(() => {
        const trainButton = screen.getByText('Modeli Eğit');
        expect(trainButton).toBeTruthy();
        
        // Click the training button
        fireEvent.click(trainButton);
      });

      // Should handle error and not crash
      await waitFor(() => {
        expect(consoleSpy).toHaveBeenCalledWith('Training failed:', expect.any(Error));
      });

      consoleSpy.mockRestore();
    });
  });

  describe('Evaluation Button', () => {
    test('should show evaluation button when model is available', async () => {
      // Mock model info to show model is available
      const mockInitializeModel = vi.fn().mockResolvedValue({
        accuracy: 0.85,
        lastTrained: Date.now(),
        trainingData: 25
      });

      (NeuralRecommendationService.initializeModel as any) = mockInitializeModel;

      const sufficientRatings = Array.from({ length: 25 }, (_, i) => ({
        movieId: i + 1,
        rating: 7,
        mediaType: 'movie' as const,
        timestamp: Date.now()
      }));

      render(<NeuralModelInfo profile={mockProfile} ratings={sufficientRatings} />);

      // Wait for component to load
      await waitFor(() => {
        expect(screen.getByText('Değerlendir')).toBeTruthy();
      });
    });

    test('should call evaluateModel when evaluation button is clicked', async () => {
      const mockInitializeModel = vi.fn().mockResolvedValue({
        accuracy: 0.85,
        lastTrained: Date.now(),
        trainingData: 25
      });
      const mockEvaluateModel = vi.fn().mockResolvedValue({
        accuracy: 0.85,
        mae: 0.5,
        rmse: 0.7,
        coverage: 0.9
      });

      (NeuralRecommendationService.initializeModel as any) = mockInitializeModel;
      (NeuralRecommendationService.evaluateModel as any) = mockEvaluateModel;

      const sufficientRatings = Array.from({ length: 25 }, (_, i) => ({
        movieId: i + 1,
        rating: 7,
        mediaType: 'movie' as const,
        timestamp: Date.now()
      }));

      render(<NeuralModelInfo profile={mockProfile} ratings={sufficientRatings} />);

      // Wait for component to load and find the evaluation button
      await waitFor(() => {
        const evalButton = screen.getByText('Değerlendir');
        expect(evalButton).toBeTruthy();
        
        // Click the evaluation button
        fireEvent.click(evalButton);
      });

      // Verify that evaluateModel was called
      await waitFor(() => {
        expect(mockEvaluateModel).toHaveBeenCalledWith(sufficientRatings, mockProfile);
      });
    });
  });

  describe('Model Status Display', () => {
    test('should display model accuracy correctly', async () => {
      const mockInitializeModel = vi.fn().mockResolvedValue({
        isAvailable: true,
        accuracy: 0.85,
        lastTrained: Date.now(),
        trainingData: 25
      });

      (NeuralRecommendationService.initializeModel as any) = mockInitializeModel;

      const sufficientRatings = Array.from({ length: 25 }, (_, i) => ({
        movieId: i + 1,
        rating: 7,
        mediaType: 'movie' as const,
        timestamp: Date.now()
      }));

      render(<NeuralModelInfo profile={mockProfile} ratings={sufficientRatings} />);

      // Wait for component to load and check accuracy display
      await waitFor(() => {
        expect(screen.getByText('85.0%')).toBeTruthy();
      });
    });

    test('should display training data count correctly', async () => {
      const mockInitializeModel = vi.fn().mockResolvedValue({
        isAvailable: false,
        accuracy: 0,
        lastTrained: 0,
        trainingData: 30
      });

      (NeuralRecommendationService.initializeModel as any) = mockInitializeModel;

      const sufficientRatings = Array.from({ length: 30 }, (_, i) => ({
        movieId: i + 1,
        rating: 7,
        mediaType: 'movie' as const,
        timestamp: Date.now()
      }));

      render(<NeuralModelInfo profile={mockProfile} ratings={sufficientRatings} />);

      // Wait for component to load and check training data count
      await waitFor(() => {
        expect(screen.getByText('30')).toBeTruthy();
      });
    });
  });
}); 
