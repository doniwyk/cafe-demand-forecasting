import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";
import {
  useRetrain,
  useRetrainStatus,
  useRetrainCancel,
} from "@/features/forecasts/hooks/use-forecasts";
import { useLatestSalesDate } from "@/features/sales/hooks/use-sales";
import { MODEL_LABEL } from "@/contexts/model-context";
import {
  PlayIcon,
  TerminalIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  XIcon,
  CalendarIcon,
} from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { format, parseISO } from "date-fns";
import { StatusIcon } from "@/features/settings/components/status-icon";
import { StatusBadge } from "@/features/settings/components/status-badge";

export function ModelTrainingCard() {
  const retrain = useRetrain();
  const retrainCancel = useRetrainCancel();
  const retrainStatus = useRetrainStatus();
  const latestSalesDate = useLatestSalesDate();
  const queryClient = useQueryClient();
  const [expandedLogs, setExpandedLogs] = useState(false);
  const [endDate, setEndDate] = useState<Date | undefined>(undefined);
  const [dateOpen, setDateOpen] = useState(false);
  const { t } = useTranslation();

  useEffect(() => {
    if (latestSalesDate.data) {
      setEndDate(parseISO(latestSalesDate.data));
    }
  }, [latestSalesDate.data]);

  const statuses = retrainStatus.data ?? {};
  const xgbStatus = statuses["xgboost"];
  const isTraining = xgbStatus?.status === "training";

  const endDateStr = endDate ? format(endDate, "yyyy-MM-dd") : undefined;

  const handleTrain = () => {
    retrain.mutate(
      {
        model_type: "xgboost",
        include_new_products: false,
        end_date: endDateStr,
      },
      {
        onSuccess: () => {
          queryClient.invalidateQueries({ queryKey: ["forecasts", "retrain-status"] });
        },
      },
    );
  };

  const logs = xgbStatus?.logs ?? [];

  return (
    <Card data-tour="model-training">
      <CardHeader>
        <CardTitle>{t("settings.modelTraining")}</CardTitle>
        <CardDescription>{t("settings.modelTrainingDesc")}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-4">
            <div className="flex flex-col gap-1.5">
              <span className="text-sm font-medium">{t("settings.trainingDataCutoff")}</span>
              <Popover open={dateOpen} onOpenChange={setDateOpen}>
                <PopoverTrigger
                  render={
                    <Button variant="outline" className="w-56 justify-start text-left font-normal">
                      <CalendarIcon className="mr-2 size-4 shrink-0" />
                      {endDate ? (
                        format(endDate, "MMM dd, yyyy")
                      ) : (
                        <span className="text-muted-foreground">{t("settings.selectDateRange")}</span>
                      )}
                    </Button>
                  }
                />
                <PopoverContent className="w-auto p-0" align="start">
                  <Calendar
                    mode="single"
                    selected={endDate}
                    onSelect={(date) => {
                      setEndDate(date);
                      setDateOpen(false);
                    }}
                  />
                </PopoverContent>
              </Popover>
            </div>
          </div>

          <Separator />

          <div className="flex flex-col gap-2 rounded-lg border p-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <StatusIcon status={xgbStatus?.status ?? "idle"} />
                <div>
                  <p className="text-sm font-medium">{MODEL_LABEL}</p>
                  {xgbStatus?.message && (
                    <p className="text-xs text-muted-foreground">{xgbStatus.message}</p>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-3">
                <StatusBadge status={xgbStatus?.status ?? "idle"} />
                {isTraining ? (
                  <Button
                    size="sm"
                    variant="destructive"
                    onClick={() => {
                      retrainCancel.mutate("xgboost", {
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
                    onClick={handleTrain}
                  >
                    <PlayIcon data-icon="inline-start" />
                    {t("settings.train")}
                  </Button>
                )}
              </div>
            </div>

            {(logs.length > 0 || (xgbStatus?.status && xgbStatus.status !== "idle")) && (
              <div className="mt-2">
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start text-xs text-muted-foreground"
                  onClick={() => setExpandedLogs((prev) => !prev)}
                >
                  <TerminalIcon className="mr-2 size-3" />
                  {expandedLogs ? t("settings.hideLogs") : t("settings.showLogs")} (
                  {logs.length})
                  {expandedLogs ? (
                    <ChevronUpIcon className="ml-auto size-3" />
                  ) : (
                    <ChevronDownIcon className="ml-auto size-3" />
                  )}
                </Button>
                {expandedLogs && (
                  <pre className="mt-2 max-h-60 overflow-x-auto rounded-lg bg-muted p-2 text-xs font-mono whitespace-pre-wrap wrap-break-word">
                    {logs.join("\n")}
                  </pre>
                )}
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
