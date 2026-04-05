import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
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
} from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";

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
  switch (status) {
    case "training":
      return <Badge variant="secondary">Training</Badge>;
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
      <Card>
        <CardHeader>
          <CardTitle>Model Training</CardTitle>
          <CardDescription>
            Train forecasting models individually or all at once (one after another).
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Train All Models</span>
                {isAnyTraining && <Badge variant="secondary">In Progress</Badge>}
              </div>
              <Button onClick={handleTrainAll} disabled={isAnyTraining || retrain.isPending}>
                <PlayIcon data-icon="inline-start" />
                Train All (Sequential)
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
                            Cancel
                          </Button>
                        ) : (
                          <Button
                            size="sm"
                            variant="outline"
                            disabled={retrain.isPending}
                            onClick={() => handleTrain(mt)}
                          >
                            <PlayIcon data-icon="inline-start" />
                            Train
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
                          {expandedLogs[mt] ? "Hide" : "Show"} Logs ({logs.length} lines)
                          {expandedLogs[mt] ? (
                            <ChevronUpIcon className="ml-auto size-3" />
                          ) : (
                            <ChevronDownIcon className="ml-auto size-3" />
                          )}
                        </Button>
                        {expandedLogs[mt] && (
                          <pre className="mt-2 max-h-60 overflow-auto rounded-lg bg-muted p-2 text-xs font-mono">
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

      <Card>
        <CardHeader>
          <CardTitle>Data Cleanup</CardTitle>
          <CardDescription>
            Remove stale model runs and forecasts from previous training sessions.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">Delete Inactive Model Data</p>
              <p className="text-xs text-muted-foreground">
                Removes old model runs, their forecasts, class metrics, and top items that are no
                longer active.
              </p>
              {cleanup.isSuccess && cleanup.data && (
                <p className="text-xs text-muted-foreground mt-1">
                  Deleted {cleanup.data.deleted_runs} runs, {cleanup.data.deleted_forecasts}{" "}
                  forecasts
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
              Cleanup
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
