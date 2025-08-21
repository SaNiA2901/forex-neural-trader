
import { useState, useEffect } from 'react';
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Clock, Brain, BarChart, TrendingUp } from "lucide-react";

interface PredictionGeneratorProps {
  isGenerating: boolean;
}

const PredictionGenerator = ({ isGenerating }: PredictionGeneratorProps) => {
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');

  const steps = [
    { label: 'Загрузка исторических данных...', duration: 800 },
    { label: 'Расчет технических индикаторов...', duration: 1200 },
    { label: 'Анализ паттернов свечей...', duration: 600 },
    { label: 'Применение ML модели...', duration: 1000 },
    { label: 'Генерация рекомендации...', duration: 400 }
  ];

  useEffect(() => {
    if (!isGenerating) {
      setProgress(0);
      setCurrentStep('');
      return;
    }

    let stepIndex = 0;
    let totalProgress = 0;
    const stepSize = 100 / steps.length;

    const runStep = () => {
      if (stepIndex >= steps.length) return;
      
      const step = steps[stepIndex];
      setCurrentStep(step.label);
      
      const stepDuration = step.duration;
      const updateInterval = stepDuration / (stepSize * 0.9); // 90% of step size for smooth animation
      
      let stepProgress = 0;
      const progressInterval = setInterval(() => {
        stepProgress += 1;
        setProgress(totalProgress + (stepProgress / (stepSize * 0.9)) * stepSize);
        
        if (stepProgress >= stepSize * 0.9) {
          clearInterval(progressInterval);
          totalProgress += stepSize;
          stepIndex++;
          
          setTimeout(() => {
            runStep();
          }, stepDuration * 0.1); // Brief pause between steps
        }
      }, updateInterval);
    };

    runStep();
  }, [isGenerating]);

  if (!isGenerating) return null;

  return (
    <Card className="p-6 bg-slate-800/50 border-slate-700">
      <div className="text-center">
        <div className="flex items-center justify-center space-x-2 mb-4">
          <Brain className="h-5 w-5 text-purple-400 animate-pulse" />
          <span className="text-white">Анализ с помощью ИИ</span>
        </div>
        
        <Progress value={progress} className="mb-3 h-3" />
        
        <div className="flex items-center justify-center space-x-2 mb-4">
          <div className="flex space-x-1">
            {progress > 20 && <BarChart className="h-4 w-4 text-green-400" />}
            {progress > 50 && <TrendingUp className="h-4 w-4 text-blue-400" />}
            {progress > 80 && <Clock className="h-4 w-4 text-yellow-400" />}
          </div>
        </div>
        
        <p className="text-slate-300 text-sm font-medium mb-2">{currentStep}</p>
        <p className="text-slate-400 text-xs">
          {progress.toFixed(0)}% завершено • Используется продвинутая ML модель
        </p>
      </div>
    </Card>
  );
};

export default PredictionGenerator;
