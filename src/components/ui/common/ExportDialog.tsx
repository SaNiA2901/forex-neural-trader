import { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { Download, Upload, FileText, FileSpreadsheet } from 'lucide-react';
import { TradingSession, CandleData } from '@/types/session';
import { useDataExport } from '@/hooks/useDataExport';

interface ExportDialogProps {
  session: TradingSession | null;
  candles: CandleData[];
  onSessionImported: (session: TradingSession, candles: CandleData[]) => Promise<void>;
  children: React.ReactNode;
}

export const ExportDialog = ({ 
  session, 
  candles, 
  onSessionImported,
  children 
}: ExportDialogProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const { exportSessionData, exportToCsv, importSessionData } = useDataExport();

  const handleExportJson = async () => {
    if (!session) return;
    await exportSessionData(session, candles);
    setIsOpen(false);
  };

  const handleExportCsv = async () => {
    if (!session) return;
    await exportToCsv(session, candles);
    setIsOpen(false);
  };

  const handleImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const success = await importSessionData(file, onSessionImported);
    if (success) {
      setIsOpen(false);
    }
    // Сбрасываем input
    event.target.value = '';
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        {children}
      </DialogTrigger>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Download className="h-5 w-5" />
            Импорт и экспорт данных
          </DialogTitle>
        </DialogHeader>
        
        <div className="space-y-6">
          {/* Экспорт */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium flex items-center gap-2">
              <Download className="h-4 w-4" />
              Экспорт данных
            </h3>
            
            {session ? (
              <div className="grid grid-cols-2 gap-3">
                <Button
                  onClick={handleExportJson}
                  variant="outline"
                  className="w-full flex items-center gap-2"
                >
                  <FileText className="h-4 w-4" />
                  JSON
                </Button>
                <Button
                  onClick={handleExportCsv}
                  variant="outline"
                  className="w-full flex items-center gap-2"
                >
                  <FileSpreadsheet className="h-4 w-4" />
                  CSV
                </Button>
              </div>
            ) : (
              <div className="text-sm text-muted-foreground p-4 bg-muted/50 rounded-md">
                Выберите сессию для экспорта данных
              </div>
            )}
          </div>

          <Separator />

          {/* Импорт */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium flex items-center gap-2">
              <Upload className="h-4 w-4" />
              Импорт данных
            </h3>
            
            <div className="space-y-3">
              <Label htmlFor="import-file" className="text-sm text-muted-foreground">
                Выберите JSON файл с данными сессии
              </Label>
              <Input
                id="import-file"
                type="file"
                accept=".json"
                onChange={handleImport}
                className="cursor-pointer"
              />
            </div>
            
            <div className="text-xs text-muted-foreground p-3 bg-muted/30 rounded-md">
              <strong>Поддерживаемый формат:</strong> JSON файлы, экспортированные из этого приложения
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};