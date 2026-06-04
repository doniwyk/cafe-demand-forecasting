import { useRetrainStatus } from "@/features/forecasts/hooks/use-forecasts";
import { MODEL_TYPES } from "@/contexts/model-context";
import { ModelTrainingCard } from "@/features/settings/components/model-training-card";
import { DataCleanupCard } from "@/features/settings/components/data-cleanup-card";
import { LanguageCard } from "@/features/settings/components/language-card";

export function SettingsPage() {
  const retrainStatus = useRetrainStatus();
  const statuses = retrainStatus.data ?? {};
  const isAnyTraining = MODEL_TYPES.some((mt) => statuses[mt]?.status === "training");

  return (
    <div className="flex flex-1 flex-col gap-6 p-4">
      <ModelTrainingCard />
      <DataCleanupCard disabled={isAnyTraining} />
      <LanguageCard />
    </div>
  );
}
