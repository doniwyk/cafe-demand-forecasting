import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useRetrain, useRetrainStatus, useRetrainCancel, useCleanup } from "@/hooks/use-forecasts";
import { MODEL_TYPES, MODEL_LABELS } from "@/contexts/model-context";
import {
  PlayIcon,
  CheckIcon,
  LoaderIcon,
  AlertCircleIcon,
  Trash2Icon,
  TerminalIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  XIcon,
  GlobeIcon,
} from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";

const LANGUAGES = [
  { value: "id", label: "Bahasa Indonesia" },
  { value: "en", label: "English" },
] as const;

export const Route = createFileRoute("/settings")({
  component: SettingsPage,
});

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case "training":
      return <LoaderIcon className="size-4 animate-spin" />;
    case "success":
      return <CheckIcon className="size-4 text-emerald-600" />;
    case "error":
      return <AlertCircleIcon className="size-4 text-destructive" />;
    default:
      return null;
  }
}

function StatusBadge({ status }: { status: string }) {
  const { t } = useTranslation();
  switch (status) {
    case "training":
      return <Badge variant="secondary">{t("settings.inProgress")}</Badge>;
    case "success":
      return <Badge variant="default">Trained</Badge>;
    case "error":
      return <Badge variant="destructive">Error</Badge>;
    default:
      return <Badge variant="outline">Not Trained</Badge>;
  }
}

function SettingsPage() {
  const retrain = useRetrain();
  const retrainCancel = useRetrainCancel();
  const retrainStatus = useRetrainStatus();
  const cleanup = useCleanup();
  const queryClient = useQueryClient();
  const [expandedLogs, setExpandedLogs] = useState<Record<string, boolean>>({});
  const { t, i18n } = useTranslation();

  const statuses = retrainStatus.data ?? {};

  const isAnyTraining = MODEL_TYPES.some((mt) => statuses[mt]?.status === "training");

  const handleTrain = (modelType: string) => {
    retrain.mutate(
      { model_type: modelType },
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
      await retrain.mutateAsync({ model_type: mt });
      await queryClient.invalidateQueries({ queryKey: ["forecasts", "retrain-status"] });
    }
  };

  return (
    <div className="flex flex-1 flex-col gap-6 p-4">
      <Card data-tour="model-training">
        <CardHeader>
          <CardTitle>{t("settings.modelTraining")}</CardTitle>
          <CardDescription>{t("settings.modelTrainingDesc")}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4">
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
                          <pre className="mt-2 max-h-60 overflow-x-auto rounded-lg bg-muted p-2 text-xs font-mono whitespace-pre-wrap break-words">
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
              disabled={cleanup.isPending || isAnyTraining}
            >
              <Trash2Icon data-icon="inline-start" />
              {t("settings.cleanup")}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{t("settings.language")}</CardTitle>
          <CardDescription>{t("settings.languageDesc")}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-3 max-w-sm">
            <GlobeIcon className="size-4 text-muted-foreground" />
            <Select
              value={i18n.language ?? undefined}
              onValueChange={(v) => i18n.changeLanguage(v ?? undefined)}
            >
              <SelectTrigger className="w-50">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {LANGUAGES.map((lang) => (
                  <SelectItem key={lang.value} value={lang.value}>
                    {lang.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
