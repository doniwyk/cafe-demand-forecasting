import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Trash2Icon } from "lucide-react";
import { useCleanup } from "@/features/forecasts/hooks/use-forecasts";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";

interface DataCleanupCardProps {
  disabled: boolean;
}

export function DataCleanupCard({ disabled }: DataCleanupCardProps) {
  const cleanup = useCleanup();
  const queryClient = useQueryClient();
  const { t } = useTranslation();

  return (
    <Card data-tour="data-cleanup">
      <CardHeader>
        <CardTitle>{t("settings.dataCleanup")}</CardTitle>
        <CardDescription>{t("settings.dataCleanupDesc")}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium">{t("settings.deleteInactiveData")}</p>
            <p className="text-xs text-muted-foreground">{t("settings.deleteInactiveDesc")}</p>
            {cleanup.isSuccess && cleanup.data && (
              <p className="text-xs text-muted-foreground mt-1">
                {t("settings.deletedRuns", {
                  runs: cleanup.data.deleted_runs,
                  forecasts: cleanup.data.deleted_forecasts,
                })}
              </p>
            )}
          </div>
          <Button
            variant="destructive"
            onClick={() => {
              cleanup.mutate(undefined, {
                onSuccess: () => {
                  queryClient.invalidateQueries({ queryKey: ["forecasts"] });
                },
              });
            }}
            disabled={cleanup.isPending || disabled}
          >
            <Trash2Icon data-icon="inline-start" />
            {t("settings.cleanup")}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
