import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useABCAnalysis, useModelMetrics } from "@/features/analytics/hooks/use-analytics";
import { useForecastSummary } from "@/features/forecasts/hooks/use-forecasts";
import { useModelType } from "@/contexts/model-context";
import { useTranslation } from "react-i18next";
import { MetricsGrid } from "@/features/analytics/components/metrics-grid";
import { CLASS_COLORS } from "@/features/analytics/lib/constants";

export function DashboardPage() {
  const { modelType } = useModelType();
  const forecastSummary = useForecastSummary(modelType);
  const abc = useABCAnalysis();
  const metrics = useModelMetrics(modelType);
  const { t } = useTranslation();

  const forecastItemCount = useMemo(() => {
    if (!forecastSummary.data) return 0;
    return Object.values(forecastSummary.data.class_metrics).reduce((s, m) => s + m.n_items, 0);
  }, [forecastSummary.data]);

  const abcByClass = useMemo(() => {
    if (!abc.data?.classifications) return { A: [], B: [], C: [] };
    const grouped: Record<string, typeof abc.data.classifications> = { A: [], B: [], C: [] };
    for (const item of abc.data.classifications) {
      const cls = item.class_label;
      if (grouped[cls]) grouped[cls].push(item);
    }
    return grouped;
  }, [abc]);

  return (
    <div className="flex flex-1 flex-col gap-4 p-4">
      {/* Top Bento: KPIs + Accuracy (left) | Model Performance (right) */}
      <div className="grid gap-4 lg:grid-cols-5">
        {/* Left column: KPIs + Accuracy by Class */}
        <div className="lg:col-span-2 flex flex-col gap-4">
          {/* KPIs horizontal */}
          <div className="grid grid-cols-2 gap-4" data-tour="kpi-cards">
            <Card>
              <CardContent>
                <p className="text-xs text-muted-foreground">{t("dashboard.activeItems")}</p>
                <p className="text-2xl font-bold tracking-tight mt-1">{forecastItemCount || "-"}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent>
                <p className="text-xs text-muted-foreground">{t("dashboard.itemsForecasted")}</p>
                <p className="text-2xl font-bold tracking-tight mt-1">{forecastItemCount || "-"}</p>
              </CardContent>
            </Card>
          </div>

          {/* Accuracy by Class */}
          <Card data-tour="accuracy-by-class">
            <CardHeader>
              <CardTitle>{t("dashboard.accuracyByClass")}</CardTitle>
            </CardHeader>
            <CardContent>
              {forecastSummary.data ? (
                <div className="flex flex-col gap-2">
                  {Object.entries(forecastSummary.data.class_metrics).map(([cls, m]) => (
                    <Tooltip key={cls}>
                      <TooltipTrigger
                        render={
                          <div className="flex items-center justify-between rounded-lg border px-3 py-2 cursor-default" />
                        }
                      >
                        <div className="flex items-center gap-2">
                          <div
                            className="flex size-7 items-center justify-center rounded-md text-xs font-bold text-primary-foreground"
                            style={{ backgroundColor: CLASS_COLORS[cls] || "var(--chart-1)" }}
                          >
                            {cls}
                          </div>
                          <span className="text-sm font-medium">
                            {m.n_items} {t("dashboard.items")}
                          </span>
                        </div>
                        <span className="text-sm text-muted-foreground">
                          wMAPE {m.wmape.toFixed(1)}%
                        </span>
                      </TooltipTrigger>
                      <TooltipContent side="right" className="w-48">
                        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
                          <span className="text-muted-foreground">R²</span>
                          <span className="font-medium text-right">{m.r2.toFixed(3)}</span>
                          <span className="text-muted-foreground">wMAPE</span>
                          <span className="font-medium text-right">{m.wmape.toFixed(1)}%</span>
                          <span className="text-muted-foreground">MAE</span>
                          <span className="font-medium text-right">{m.mae.toFixed(2)}</span>
                          <span className="text-muted-foreground">RMSE</span>
                          <span className="font-medium text-right">{m.rmse.toFixed(2)}</span>
                          <span className="text-muted-foreground">±20%</span>
                          <span className="font-medium text-right">{m.periods_within_20pct.toFixed(1)}%</span>
                          <span className="text-muted-foreground">±50%</span>
                          <span className="font-medium text-right">{m.periods_within_50pct.toFixed(1)}%</span>
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  ))}
                </div>
              ) : (
                <Skeleton className="h-20 w-full" />
              )}
            </CardContent>
          </Card>
        </div>

        {/* Right column: Model Performance */}
        <Card data-tour="model-performance" className="lg:col-span-3">
          <CardHeader>
            <CardTitle>{t("analytics.modelPerformance")}</CardTitle>
          </CardHeader>
          <CardContent>
            <MetricsGrid metrics={metrics.data} />
          </CardContent>
        </Card>
      </div>

      {/* ABC Classification */}
      <Card data-tour="abc-classification">
        <CardHeader>
          <CardTitle>{t("analytics.abcClassification")}</CardTitle>
        </CardHeader>
        <CardContent>
          {abc.data ? (
            <Tabs defaultValue="A">
              <TabsList>
                {(["A", "B", "C"] as const).map((cls) => (
                  <TabsTrigger key={cls} value={cls}>
                    {cls}-class {forecastSummary.data?.class_metrics[cls]?.n_items ? `(${forecastSummary.data.class_metrics[cls].n_items})` : ""}
                  </TabsTrigger>
                ))}
              </TabsList>
              {(["A", "B", "C"] as const).map((cls) => (
                <TabsContent key={cls} value={cls} className="max-h-65 overflow-y-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>{t("analytics.item")}</TableHead>
                        <TableHead className="text-right">{t("analytics.volume")}</TableHead>
                        <TableHead className="text-right">
                          {t("analytics.cumulativePct")}
                        </TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {abcByClass[cls].length > 0 ? (
                        abcByClass[cls].slice(0, 50).map((item, idx) => (
                          <TableRow key={`${item.item}-${idx}`}>
                            <TableCell className="font-medium">{item.item}</TableCell>
                            <TableCell className="text-right">
                              {Math.round(item.vol).toLocaleString()}
                            </TableCell>
                            <TableCell className="text-right">
                              {(item.pct * 100).toFixed(1)}%
                            </TableCell>
                          </TableRow>
                        ))
                      ) : (
                        <TableRow>
                          <TableCell colSpan={3} className="text-center text-muted-foreground">
                            {t("common.noData")}
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TabsContent>
              ))}
            </Tabs>
          ) : (
            <Skeleton className="h-65 w-full" />
          )}
        </CardContent>
      </Card>
    </div>
  );
}
