import { useEffect, useCallback } from 'react';
import { useToast } from '@/hooks/use-toast';

interface UseKeyboardShortcutsProps {
  onSaveCandle?: () => void;
  onDeleteLastCandle?: () => void;
  onCreateSession?: () => void;
  onExportData?: () => void;
}

export const useKeyboardShortcuts = ({
  onSaveCandle,
  onDeleteLastCandle,
  onCreateSession,
  onExportData
}: UseKeyboardShortcutsProps) => {
  const { toast } = useToast();

  const handleKeyPress = useCallback((event: KeyboardEvent) => {
    // Игнорируем если активен input/textarea
    if (event.target instanceof HTMLInputElement || 
        event.target instanceof HTMLTextAreaElement) {
      return;
    }

    const isCtrl = event.ctrlKey || event.metaKey;
    const isShift = event.shiftKey;

    switch (event.key.toLowerCase()) {
      case 's':
        if (isCtrl) {
          event.preventDefault();
          onSaveCandle?.();
          toast({
            title: "Горячие клавиши",
            description: "Ctrl+S - Сохранить свечу"
          });
        }
        break;
      
      case 'z':
        if (isCtrl && !isShift) {
          event.preventDefault();
          onDeleteLastCandle?.();
          toast({
            title: "Горячие клавиши", 
            description: "Ctrl+Z - Удалить последнюю свечу"
          });
        }
        break;

      case 'n':
        if (isCtrl) {
          event.preventDefault();
          onCreateSession?.();
          toast({
            title: "Горячие клавиши",
            description: "Ctrl+N - Новая сессия"
          });
        }
        break;

      case 'e':
        if (isCtrl && isShift) {
          event.preventDefault();
          onExportData?.();
          toast({
            title: "Горячие клавиши",
            description: "Ctrl+Shift+E - Экспорт данных"
          });
        }
        break;

      case 'h':
        if (isCtrl) {
          event.preventDefault();
          showHelpModal();
        }
        break;
    }
  }, [onSaveCandle, onDeleteLastCandle, onCreateSession, onExportData, toast]);

  const showHelpModal = useCallback(() => {
    toast({
      title: "Горячие клавиши",
      description: "Ctrl+S: Сохранить свечу\nCtrl+Z: Удалить последнюю\nCtrl+N: Новая сессия\nCtrl+Shift+E: Экспорт\nCtrl+H: Справка",
      duration: 5000
    });
  }, [toast]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [handleKeyPress]);

  return { showHelpModal };
};