import { ModelTrainingCard } from "@/features/settings/components/model-training-card";
import { LanguageCard } from "@/features/settings/components/language-card";

export function SettingsPage() {
  return (
    <div className="flex flex-1 flex-col gap-6 p-4">
      <ModelTrainingCard />
      <LanguageCard />
    </div>
  );
}
