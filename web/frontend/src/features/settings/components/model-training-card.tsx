import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import {
  useRetrain,
  useRetrainStatus,
  useRetrainCancel,
} from "@/features/forecasts/hooks/use-forecasts";
import { MODEL_TYPES, MODEL_LABELS } from "@/contexts/model-context";
import {
  PlayIcon,
  TerminalIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  XIcon,
} from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { StatusIcon } from "@/features/settings/components/status-icon";
import { StatusBadge } from "@/features/settings/components/status-badge";

export function ModelTrainingCard() {
  const retrain = useRetrain();
  const retrainCancel = useRetrainCancel();
  const retrainStatus = useRetrainStatus();
  const queryClient = useQueryClient();
  const [expandedLogs, setExpandedLogs] = useState<Record<string, boolean>>({});
  const [syncHus, setSyncHus] = useState(true);
  const [includeNew, setIncludeNew] = useState(false);
  const { t } = useTranslation();

  const statuses = retrainStatus.data ?? {};
  const isAnyTraining = MODEL_TYPES.some((mt) => statuses[mt]?.status === "training");

  const handleTrain = (modelType: string) => {
    retrain.mutate(
      { model_type: modelType, sync_hus: syncHus, include_new_products: includeNew },
      {
        onSuccess: () => {
          queryClient.invalidateQueries({ queryKey: ["forecasts", "retrain-status"] });
        },
      },
    );
  };

  const handleTrainAll = async () => {
    for (const mt of MODEL_TYPES) {
      if (statuses[mt]?.status === "training") continue;
      await retrain.mutateAsync({
        model_type: mt,
        sync_hus: syncHus,
        include_new_products: includeNew,
      });
      await queryClient.invalidateQueries({ queryKey: ["forecasts", "retrain-status"] });
    }
  };

  return (
    <Card data-tour="model-training">
      <CardHeader>
        <CardTitle>{t("settings.modelTraining")}</CardTitle>
        <CardDescription>{t("settings.modelTrainingDesc")}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <Checkbox
                id="sync-hus"
                checked={syncHus}
                onCheckedChange={(v) => setSyncHus(!!v)}
              />
              <Label htmlFor="sync-hus" className="text-sm">
                {t("settings.syncFromHus")}
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox
                id="include-new"
                checked={includeNew}
                onCheckedChange={(v) => setIncludeNew(!!v)}
                disabled={!syncHus}
              />
              <Label htmlFor="include-new" className="text-sm">
                {t("settings.includeNewProducts")}
              </Label>
            </div>
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">{t("settings.trainAllModels")}</span>
              {isAnyTraining && <Badge variant="secondary">{t("settings.inProgress")}</Badge>}
            </div>
            <Button onClick={handleTrainAll} disabled={isAnyTraining || retrain.isPending}>
              <PlayIcon data-icon="inline-start" />
              {t("settings.trainAllSequential")}
            </Button>
          </div>

          <Separator />

          <div className="flex flex-col gap-3">
            {MODEL_TYPES.map((mt) => {
              const st = statuses[mt];
              const logs = st?.logs ?? [];

              return (
                <div key={mt} className="flex flex-col gap-2 rounded-lg border p-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <StatusIcon status={st?.status ?? "idle"} />
                      <div>
                        <p className="text-sm font-medium">{MODEL_LABELS[mt]}</p>
                        {st?.message && (
                          <p className="text-xs text-muted-foreground">{st.message}</p>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <StatusBadge status={st?.status ?? "idle"} />
                      {st?.status === "training" ? (
                        <Button
                          size="sm"
                          variant="destructive"
                          onClick={() => {
                            retrainCancel.mutate(mt, {
                              onSuccess: () => {
                                queryClient.invalidateQueries({
                                  queryKey: ["forecasts", "retrain-status"],
                                });
                              },
                            });
                          }}
                        >
                          <XIcon data-icon="inline-start" />
                          {t("settings.cancel")}
                        </Button>
                      ) : (
                        <Button
                          size="sm"
                          variant="outline"
                          disabled={retrain.isPending}
                          onClick={() => handleTrain(mt)}
                        >
                          <PlayIcon data-icon="inline-start" />
                          {t("settings.train")}
                        </Button>
                      )}
                    </div>
                  </div>

                  {(logs.length > 0 || (st?.status && st.status !== "idle")) && (
                    <div className="mt-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="w-full justify-start text-xs text-muted-foreground"
                        onClick={() => setExpandedLogs((prev) => ({ ...prev, [mt]: !prev[mt] }))}
                      >
                        <TerminalIcon className="mr-2 size-3" />
                        {expandedLogs[mt] ? t("settings.hideLogs") : t("settings.showLogs")} (
                        {logs.length})
                        {expandedLogs[mt] ? (
                          <ChevronUpIcon className="ml-auto size-3" />
                        ) : (
                          <ChevronDownIcon className="ml-auto size-3" />
                        )}
                      </Button>
                      {expandedLogs[mt] && (
                        <pre className="mt-2 max-h-60 overflow-x-auto rounded-lg bg-muted p-2 text-xs font-mono whitespace-pre-wrap wrap-break-word">
                          {logs.join("\n")}
                        </pre>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
