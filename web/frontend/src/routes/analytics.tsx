import { createFileRoute } from "@tanstack/react-router";
import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useABCAnalysis, useModelMetrics } from "@/hooks/use-analytics";
import { useForecastSummary } from "@/hooks/use-forecasts";
import { useModelType } from "@/contexts/model-context";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { useTranslation } from "react-i18next";

export const Route = createFileRoute("/analytics")({
  component: AnalyticsPage,
});

const CLASS_COLORS: Record<string, string> = {
  A: "var(--chart-1)",
  B: "var(--chart-2)",
  C: "var(--chart-3)",
};

function MetricsGrid({ metrics }: { metrics: Record<string, number> | undefined }) {
  const { t } = useTranslation();

  if (!metrics) {
    return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
        <Skeleton className="h-20" />
        <Skeleton className="h-20" />
        <Skeleton className="h-20" />
        <Skeleton className="h-20" />
      </div>
    );
  }

  const items = [
    {
      label: t("analytics.rSquared"),
      value: metrics["r2"]?.toFixed(3) ?? "-",
      description: t("analytics.rSquaredDesc"),
    },
    {
      label: t("analytics.wmape"),
      value: `${(metrics["wmape"] ?? 0).toFixed(1)}%`,
      description: t("analytics.wmapeDesc"),
    },
    { label: t("analytics.mae"), value: metrics["mae"]?.toFixed(1) ?? "-", description: t("analytics.maeDesc") },
    {
      label: t("analytics.medianPeriodAccuracy"),
      value: `${(metrics["median_period_accuracy"] ?? 0).toFixed(1)}%`,
      description: t("analytics.medianPeriodAccuracyDesc"),
    },
    {
      label: t("analytics.periodsWithin20pct"),
      value: `${(metrics["periods_within_20pct"] ?? 0).toFixed(1)}%`,
      description: t("analytics.periodsWithin20pctDesc"),
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      {items.map((m) => (
        <div key={m.label} className="rounded-lg border p-4">
          <p className="text-xs text-muted-foreground">{m.label}</p>
          <p className="text-xl font-bold mt-1">{m.value}</p>
          <p className="text-xs text-muted-foreground mt-1">{m.description}</p>
        </div>
      ))}
    </div>
  );
}

function AnalyticsPage() {
  const { modelType } = useModelType();
  const abc = useABCAnalysis(modelType);
  const metrics = useModelMetrics(modelType);
  const forecastSummary = useForecastSummary(modelType);
  const { t } = useTranslation();

  const classBarData = useMemo(() => {
    if (!forecastSummary.data) return [];
    return Object.entries(forecastSummary.data.class_metrics).map(([cls, m]) => ({
      class: cls,
      items: m.n_items,
      accuracy: +m.median_period_accuracy.toFixed(1),
    }));
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
    <div className="flex flex-1 flex-col gap-6 p-4">
      <Card data-tour="model-performance">
        <CardHeader>
          <CardTitle>{t("analytics.modelPerformance")}</CardTitle>
        </CardHeader>
        <CardContent>
          <MetricsGrid metrics={metrics.data} />
        </CardContent>
      </Card>

      <Card data-tour="abc-classification">
        <CardHeader>
          <CardTitle>{t("analytics.abcClassification")}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 lg:grid-cols-5">
            {classBarData.length > 0 && (
              <div className="lg:col-span-2 flex flex-col">
                <div className="flex-1 min-h-[250px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={classBarData} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis
                        dataKey="class"
                        tick={{ fontSize: 13, fontWeight: 600 }}
                        stroke="var(--muted-foreground)"
                      />
                      <YAxis tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "var(--popover)",
                          border: "1px solid var(--border)",
                          borderRadius: "var(--radius)",
                          fontSize: 12,
                        }}
                      />
                      <Bar dataKey="items" name={t("analytics.volume")} radius={[4, 4, 0, 0]}>
                        {classBarData.map((d) => (
                          <Cell key={d.class} fill={CLASS_COLORS[d.class] || "var(--chart-1)"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            <div className="lg:col-span-3">
              {abc.data ? (
                <Tabs defaultValue="A">
                  <TabsList>
                    {(["A", "B", "C"] as const).map((cls) => (
                      <TabsTrigger key={cls} value={cls}>
                        {cls}-class {abcByClass[cls].length > 0 && `(${abcByClass[cls].length})`}
                      </TabsTrigger>
                    ))}
                  </TabsList>
                  {(["A", "B", "C"] as const).map((cls) => (
                    <TabsContent key={cls} value={cls} className="max-h-65 overflow-y-auto">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Item</TableHead>
                            <TableHead className="text-right">{t("analytics.volume")}</TableHead>
                            <TableHead className="text-right">{t("analytics.cumulativePct")}</TableHead>
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
            </div>
          </div>
        </CardContent>
      </Card>

    </div>
  );
}
