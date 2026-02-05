import React, { useState, useEffect } from 'react';
import { Brain, TrendingUp, Clock, Target, Zap, AlertCircle } from 'lucide-react';
import { NeuralRecommendationService } from '../../recommendation/services/neuralRecommendationService';

interface NeuralModelInfoProps {
  profile: any;
  ratings: any[];
}

export const NeuralModelInfo: React.FC<NeuralModelInfoProps> = ({
  profile,
  ratings
}) => {
  const [modelInfo, setModelInfo] = useState<{
    isAvailable: boolean;
    accuracy: number;
    lastTrained: number;
    trainingData: number;
  } | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [evaluation, setEvaluation] = useState<{
    accuracy: number;
    mae: number;
    rmse: number;
    coverage: number;
  } | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);

  useEffect(() => {
    loadModelInfo();
  }, [profile, ratings]);

  const loadModelInfo = async () => {
    try {
      const model = await NeuralRecommendationService.initializeModel();
      const validRatings = ratings.filter(r => typeof r.rating === 'number');
      
      // Check if model is valid and has been trained
      const isModelValid = model && model.lastTrained > 0 && model.accuracy > 0.25;
      
      console.log('Model validation:', {
        model: !!model,
        lastTrained: model?.lastTrained,
        accuracy: model?.accuracy,
        isModelValid,
        trainingData: validRatings.length
      });
      
      setModelInfo({
        isAvailable: isModelValid,
        accuracy: model?.accuracy || 0,
        lastTrained: model?.lastTrained || 0,
        trainingData: validRatings.length
      });
    } catch (error) {
      console.error('Error loading model info:', error);
      setModelInfo({
        isAvailable: false,
        accuracy: 0,
        lastTrained: 0,
        trainingData: ratings.filter(r => typeof r.rating === 'number').length
      });
    }
  };

  const handleTrainModel = async () => {
    if (!profile || ratings.length < 20) return;

    setIsTraining(true);
    try {
      await NeuralRecommendationService.trainModel(ratings, profile);
      await loadModelInfo();
      
      // Evaluate model
      const evalResult = await NeuralRecommendationService.evaluateModel(ratings, profile);
      setEvaluation(evalResult);
    } catch (error) {
      console.error('Training failed:', error);
    } finally {
      setIsTraining(false);
    }
  };

  const handleEvaluateModel = async () => {
    if (!profile) return;
    const ratingCount = ratings.filter(r => typeof r.rating === 'number').length;
    const canTrainModel = ratingCount >= 20;
    const canEvaluate = ratingCount >= 10;

    if (!canEvaluate) return;

    setIsEvaluating(true);
    try {
      if (!modelInfo?.isAvailable && canTrainModel) {
        setIsTraining(true);
        await NeuralRecommendationService.trainModel(ratings, profile);
        await loadModelInfo();
      }

      const evalResult = await NeuralRecommendationService.evaluateModel(ratings, profile);
      setEvaluation(evalResult);
    } catch (error) {
      console.error('Evaluation failed:', error);
    }
    finally {
      setIsTraining(false);
      setIsEvaluating(false);
    }
  };

  if (!modelInfo) {
    return (
      <div className="bg-slate-800 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Brain className="h-6 w-6 text-purple-400" />
          <h3 className="text-lg font-semibold text-white">Neural Network Model</h3>
        </div>
        <div className="text-slate-400">Loading model information...</div>
      </div>
    );
  }

  const trainingDataCount = ratings.filter(r => typeof r.rating === 'number').length;
  const canTrain = trainingDataCount >= 20;
  const canEvaluate = trainingDataCount >= 10;
  const lastTrainedDate = modelInfo.lastTrained ? new Date(modelInfo.lastTrained) : null;
  const daysSinceTraining = lastTrainedDate ? 
    Math.floor((Date.now() - lastTrainedDate.getTime()) / (1000 * 60 * 60 * 24)) : null;

  return (
    <div className="bg-slate-800 rounded-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="bg-gradient-to-r from-purple-500 to-pink-500 p-2 rounded-lg">
            <Brain className="h-6 w-6 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">Neural Network Model</h3>
            <p className="text-slate-400 text-sm">Deep Learning Tabanlı Öneri Sistemi</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          {modelInfo.isAvailable && (
            <div className="flex items-center space-x-1 text-green-400">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <span className="text-sm font-medium">Aktif</span>
            </div>
          )}
        </div>
      </div>

      {/* Model Status */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-slate-700/50 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Target className="h-4 w-4 text-blue-400" />
            <span className="text-slate-300 text-sm font-medium">Doğruluk</span>
          </div>
          <div className="text-2xl font-bold text-blue-400">
            {modelInfo.isAvailable ? `${(modelInfo.accuracy * 100).toFixed(1)}%` : 'N/A'}
          </div>
        </div>

        <div className="bg-slate-700/50 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Clock className="h-4 w-4 text-amber-400" />
            <span className="text-slate-300 text-sm font-medium">Son Eğitim</span>
          </div>
          <div className="text-sm text-amber-400">
            {lastTrainedDate ? 
              `${daysSinceTraining} gün önce` : 
              'Hiç eğitilmedi'
            }
          </div>
        </div>

        <div className="bg-slate-700/50 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <TrendingUp className="h-4 w-4 text-green-400" />
            <span className="text-slate-300 text-sm font-medium">Eğitim Verisi</span>
          </div>
          <div className="text-2xl font-bold text-green-400">
            {modelInfo.trainingData}
          </div>
        </div>
      </div>

      {/* Training Requirements */}
      {!canTrain && (
        <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-4 mb-6">
          <div className="flex items-center space-x-2 mb-2">
            <AlertCircle className="h-4 w-4 text-amber-400" />
            <span className="text-amber-400 font-medium">Eğitim Gereksinimleri</span>
          </div>
          <p className="text-amber-200 text-sm">
            Neural network modelini eğitmek için en az 20 puanlanmış içerik gereklidir. 
            Şu anda {modelInfo.trainingData} içerik puanlanmış.
          </p>
        </div>
      )}

      {/* Model Evaluation */}
      {evaluation && (
        <div className="bg-slate-700/50 rounded-lg p-4 mb-6">
          <div className="flex items-center space-x-2 mb-4">
            <Zap className="h-4 w-4 text-purple-400" />
            <span className="text-white font-medium">Model Değerlendirmesi</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div className="text-slate-400 text-xs">Doğruluk</div>
              <div className="text-white font-medium">{(evaluation.accuracy * 100).toFixed(1)}%</div>
            </div>
            <div>
              <div className="text-slate-400 text-xs">MAE</div>
              <div className="text-white font-medium">{evaluation.mae.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-slate-400 text-xs">RMSE</div>
              <div className="text-white font-medium">{evaluation.rmse.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-slate-400 text-xs">Kapsam</div>
              <div className="text-white font-medium">{(evaluation.coverage * 100).toFixed(1)}%</div>
            </div>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex items-center space-x-3">
        {canTrain && (
          <button
            onClick={handleTrainModel}
            disabled={isTraining}
            className="flex items-center space-x-2 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 disabled:from-purple-500/50 disabled:to-pink-500/50 text-white px-4 py-2 rounded-lg transition-all duration-200 font-medium"
          >
            <Brain className={`h-4 w-4 ${isTraining ? 'animate-spin' : ''}`} />
            <span>{isTraining ? 'Eğitiliyor...' : 'Modeli Eğit'}</span>
          </button>
        )}
        
        {canEvaluate && (
          <button
            onClick={handleEvaluateModel}
            disabled={isEvaluating || isTraining}
            className="flex items-center space-x-2 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-700/50 text-white px-4 py-2 rounded-lg transition-all duration-200 font-medium"
          >
            <Target className="h-4 w-4" />
            <span>{modelInfo.isAvailable ? 'Değerlendir' : 'Eğit ve değerlendir'}</span>
          </button>
        )}
        
        {/* Debug info for button visibility */}
        {!modelInfo.isAvailable && canEvaluate && (
          <div className="text-xs text-slate-500 mt-2">
            Model henüz eğitilmedi. Değerlendirme için eğitim başlatılacak.
          </div>
        )}
        {!canEvaluate && (
          <div className="text-xs text-slate-500 mt-2">
            Değerlendir butonu aktif değil: Yetersiz eğitim verisi ({modelInfo.trainingData}/10)
          </div>
        )}
      </div>

      {/* Technical Details */}
      <div className="mt-6 pt-6 border-t border-slate-700">
        <h4 className="text-sm font-medium text-slate-300 mb-3">Teknik Detaylar</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs text-slate-400">
          <div>
            <div className="font-medium">Giriş Boyutu</div>
            <div>128 özellik</div>
          </div>
          <div>
            <div className="font-medium">Gizli Katmanlar</div>
            <div>64 → 32 → 16</div>
          </div>
          <div>
            <div className="font-medium">Aktivasyon</div>
            <div>ReLU + Sigmoid</div>
          </div>
          <div>
            <div className="font-medium">Eğitim</div>
            <div>100 epoch</div>
          </div>
        </div>
      </div>
    </div>
  );
}; 
